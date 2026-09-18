# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native gateway handlers.

All endpoints are prefixed /twinkle/* and registered via _register_twinkle_routes(app, self_fn).
"""
from __future__ import annotations

from collections.abc import Callable
from fastapi import Depends, FastAPI, HTTPException, Request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import GatewayServer

import twinkle_client.types as types
from twinkle.server.checkpoint import create_checkpoint_manager, create_training_run_manager, validate_user_path
from twinkle.server.lifecycle.envelope import envelope_from_record
from twinkle.server.lifecycle.poll_config import long_poll_window
from twinkle.server.utils.auth import get_token_from_request
from twinkle.utils.logger import get_logger
from .services import create_session as create_session_use_case
from .services import delete_checkpoint
from .services import get_training_run as get_training_run_use_case
from .services import get_weights_info, list_checkpoints, list_training_runs, poll_future, touch_session

logger = get_logger()


def _register_twinkle_routes(app: FastAPI, self_fn: Callable[[], GatewayServer]) -> None:
    """Register all /twinkle/* routes on the given FastAPI app."""

    @app.get('/twinkle/capacity_info', response_model=types.CapacityInfoResponse)
    async def get_capacity_info(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ) -> types.CapacityInfoResponse:
        info = await self.state.get_capacity_info()
        return types.CapacityInfoResponse(**info)

    @app.get('/twinkle/healthz', response_model=types.HealthResponse)
    async def healthz(request: Request) -> types.HealthResponse:
        return types.HealthResponse(status='ok')

    @app.get('/twinkle/healthz/deep')
    async def healthz_deep(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ) -> dict:
        """Deep health check: verifies model actors are alive, not just the gateway.

        Returns 503 if any model deployment's actors are unreachable (e.g. OOM/SIGSEGV).
        The entrypoint watchdog should poll this endpoint to detect silent failures.
        """
        from fastapi.responses import JSONResponse

        results = {}
        all_healthy = True

        for model in self.supported_models:
            model_name = model.model_name
            try:
                resp = await self.proxy.proxy_request(request, 'healthz', model_name, 'model')
                healthy = (resp.status_code == 200)
                if not healthy:
                    all_healthy = False
                results[model_name] = {
                    'healthy': healthy,
                    'status_code': resp.status_code,
                }
            except Exception as e:
                all_healthy = False
                results[model_name] = {
                    'healthy': False,
                    'detail': str(e),
                }

        body = {'healthy': all_healthy, 'models': results}
        if not all_healthy:
            return JSONResponse(status_code=503, content=body)
        return body

    @app.get('/twinkle/get_server_capabilities', response_model=types.GetServerCapabilitiesResponse)
    async def get_server_capabilities(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ) -> types.GetServerCapabilitiesResponse:
        return types.GetServerCapabilitiesResponse(
            supported_models=self.supported_models,
            protocol_version=1,
            features=types.ClientFeatures(
                task_envelope=True,
                cancel=True,
                data_plane=True,
                full_training=True,
                batch_retrieve=False,
            ),
            limits=types.ProtocolLimits(long_poll_timeout_seconds=long_poll_window()),
        )

    @app.post('/twinkle/create_session', response_model=types.CreateSessionResponse)
    async def create_session(
            request: Request,
            body: types.CreateSessionRequest,
            self: GatewayServer = Depends(self_fn),
    ) -> types.CreateSessionResponse:
        session_id = await create_session_use_case(self.state, body.model_dump())
        return types.CreateSessionResponse(session_id=session_id)

    @app.post('/twinkle/session_heartbeat', response_model=types.SessionHeartbeatResponse)
    async def session_heartbeat(
            request: Request,
            body: types.SessionHeartbeatRequest,
            self: GatewayServer = Depends(self_fn),
    ) -> types.SessionHeartbeatResponse:
        alive = await touch_session(self.state, body.session_id)
        if not alive:
            raise HTTPException(status_code=404, detail='Unknown session')
        return types.SessionHeartbeatResponse()

    @app.post('/twinkle/retrieve_future', response_model=types.TaskEnvelope)
    async def retrieve_future(
            request: Request,
            body: types.RetrieveFutureRequest,
            self: GatewayServer = Depends(self_fn),
    ) -> types.TaskEnvelope:
        """Long-poll a twinkle-native task to a terminal state.

        Returns 200 for every outcome except a request_id that stayed invisible for
        a whole window -- the HTTP call succeeded, it successfully reported the
        task's state. Unlike the tinker endpoint next door, ``completed`` with a
        null result is a valid success (step / zero_grad / lr_step all return None),
        so this handler never raises the tinker endpoint's
        ``HTTPException(500, 'Task completed but no result found')``.

        A fixed interval, not exponential backoff: measured on real hardware, a
        0.05->1.0s doubling schedule is ~22% SLOWER per step because its interval
        grows fastest across the 0.5-1.2s band where data-plane tasks actually
        finish. See ``poll_config`` for the numbers.
        """
        request_id = body.request_id
        outcome = await poll_future(self.state, request_id)
        if outcome.record is None:
            raise HTTPException(status_code=404, detail=f'request_id {request_id} not found or expired')
        return envelope_from_record(request_id, outcome.record)

    @app.post('/twinkle/cancel', response_model=types.CancelResponse)
    async def cancel_future(
            request: Request,
            body: types.CancelRequest,
            self: GatewayServer = Depends(self_fn),
    ) -> types.CancelResponse:
        """Best-effort cancel of a not-yet-started task.

        Drops the task from the compute queue only if it has not begun running; a
        running or already-terminal task is reported but never interrupted, so cancel
        can never corrupt in-flight GPU/optimizer state.
        """
        result = await self.state.cancel_future(body.request_id)
        return types.CancelResponse(**result)

    @app.get('/twinkle/training_runs', response_model=types.TrainingRunsResponse)
    async def get_training_runs(request: Request, limit: int = 20, offset: int = 0) -> types.TrainingRunsResponse:
        token = get_token_from_request(request)
        return list_training_runs(token, 'twinkle', limit=limit, offset=offset)

    @app.get('/twinkle/training_runs/{run_id}', response_model=types.TrainingRun)
    async def get_training_run(request: Request, run_id: str) -> types.TrainingRun:
        token = get_token_from_request(request)
        run = get_training_run_use_case(token, 'twinkle', run_id, check_permission=True)
        if not run:
            raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
        return run

    @app.get('/twinkle/training_runs/{run_id}/checkpoints', response_model=types.CheckpointsListResponse)
    async def get_run_checkpoints(request: Request, run_id: str) -> types.CheckpointsListResponse:
        token = get_token_from_request(request)
        response = list_checkpoints(token, 'twinkle', run_id)
        if response is None:
            raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
        return response

    @app.delete(
        '/twinkle/training_runs/{run_id}/checkpoints/{checkpoint_id:path}',
        response_model=types.DeleteCheckpointResponse)
    async def delete_run_checkpoint(request: Request, run_id: str,
                                    checkpoint_id: str) -> types.DeleteCheckpointResponse:
        token = get_token_from_request(request)

        if not validate_user_path(token, checkpoint_id):
            raise HTTPException(status_code=400, detail='Invalid checkpoint path: path traversal not allowed')

        success = delete_checkpoint(token, 'twinkle', run_id, checkpoint_id)
        if not success:
            raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found or access denied')

        return types.DeleteCheckpointResponse(success=True, message=f'Checkpoint {checkpoint_id} deleted successfully')

    @app.post('/twinkle/weights_info', response_model=types.WeightsInfoResponse)
    async def weights_info(request: Request, body: types.WeightsInfoRequest) -> types.WeightsInfoResponse:
        token = get_token_from_request(request)
        response = get_weights_info(token, 'twinkle', body.twinkle_path)
        if response is None:
            raise HTTPException(status_code=404, detail=f'Weights at {body.twinkle_path} not found or access denied')
        return response

    @app.get('/twinkle/checkpoint_path/{run_id}/{checkpoint_id:path}', response_model=types.CheckpointPathResponse)
    async def get_checkpoint_path(request: Request, run_id: str, checkpoint_id: str) -> types.CheckpointPathResponse:
        token = get_token_from_request(request)

        if not validate_user_path(token, checkpoint_id):
            raise HTTPException(status_code=400, detail='Invalid checkpoint path: path traversal not allowed')

        training_run_manager = create_training_run_manager(token, client_type='twinkle')
        checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')

        run = training_run_manager.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')

        checkpoint = checkpoint_manager.get(run_id, checkpoint_id)
        if not checkpoint:
            raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found')

        ckpt_dir = checkpoint_manager.get_ckpt_dir(run_id, checkpoint_id)
        return types.CheckpointPathResponse(path=str(ckpt_dir), twinkle_path=checkpoint.twinkle_path)

    @app.get('/twinkle/status')
    async def status(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ) -> dict:
        cleanup_stats = await self.state.get_cleanup_stats()
        return {
            'resources': cleanup_stats['resource_counts'],
            'cleanup': {
                'running': cleanup_stats['cleanup_running'],
                'expiration_timeout': cleanup_stats['expiration_timeout'],
            },
        }
