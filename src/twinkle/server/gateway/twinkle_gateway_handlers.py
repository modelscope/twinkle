# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native gateway handlers.

All endpoints are prefixed /twinkle/* and registered via _register_twinkle_routes(app, self_fn).
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .server import GatewayServer

import twinkle_client.types as types
from twinkle.server.common.checkpoint_factory import create_checkpoint_manager, create_training_run_manager
from twinkle.server.utils.checkpoint_base import validate_user_path
from twinkle.server.utils.validation import get_token_from_request
from twinkle.utils.logger import get_logger

logger = get_logger()


def _register_twinkle_routes(app: FastAPI, self_fn: Callable[[], GatewayServer]) -> None:
    """Register all /twinkle/* routes on the given FastAPI app."""

    @app.get('/twinkle/healthz', response_model=types.HealthResponse)
    async def healthz(request: Request) -> types.HealthResponse:
        return types.HealthResponse(status='ok')

    @app.get('/twinkle/get_server_capabilities', response_model=types.GetServerCapabilitiesResponse)
    async def get_server_capabilities(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ) -> types.GetServerCapabilitiesResponse:
        return types.GetServerCapabilitiesResponse(supported_models=self.supported_models)

    @app.post('/twinkle/create_session', response_model=types.CreateSessionResponse)
    async def create_session(
            request: Request,
            body: types.CreateSessionRequest,
            self: GatewayServer = Depends(self_fn),
    ) -> types.CreateSessionResponse:
        session_id = await self.state.create_session(body.model_dump())
        return types.CreateSessionResponse(session_id=session_id)

    @app.post('/twinkle/session_heartbeat', response_model=types.SessionHeartbeatResponse)
    async def session_heartbeat(
            request: Request,
            body: types.SessionHeartbeatRequest,
            self: GatewayServer = Depends(self_fn),
    ) -> types.SessionHeartbeatResponse:
        alive = await self.state.touch_session(body.session_id)
        if not alive:
            raise HTTPException(status_code=404, detail='Unknown session')
        return types.SessionHeartbeatResponse()

    @app.get('/twinkle/training_runs', response_model=types.TrainingRunsResponse)
    async def get_training_runs(request: Request, limit: int = 20, offset: int = 0) -> types.TrainingRunsResponse:
        token = get_token_from_request(request)
        training_run_manager = create_training_run_manager(token, client_type='twinkle')
        return training_run_manager.list_runs(limit=limit, offset=offset)

    @app.get('/twinkle/training_runs/{run_id}', response_model=types.TrainingRun)
    async def get_training_run(request: Request, run_id: str) -> types.TrainingRun:
        token = get_token_from_request(request)
        training_run_manager = create_training_run_manager(token, client_type='twinkle')
        run = training_run_manager.get_with_permission(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
        return run

    @app.get('/twinkle/training_runs/{run_id}/checkpoints', response_model=types.CheckpointsListResponse)
    async def get_run_checkpoints(request: Request, run_id: str) -> types.CheckpointsListResponse:
        token = get_token_from_request(request)
        checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
        response = checkpoint_manager.list_checkpoints(run_id)
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

        checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
        success = checkpoint_manager.delete(run_id, checkpoint_id)
        if not success:
            raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found or access denied')

        return types.DeleteCheckpointResponse(success=True, message=f'Checkpoint {checkpoint_id} deleted successfully')

    @app.post('/twinkle/weights_info', response_model=types.WeightsInfoResponse)
    async def weights_info(request: Request, body: types.WeightsInfoRequest) -> types.WeightsInfoResponse:
        token = get_token_from_request(request)
        checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
        response = checkpoint_manager.get_weights_info(body.twinkle_path)
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
