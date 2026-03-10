# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Tinker-compatible gateway router.

Provides all tinker management and proxy endpoints under /tinker/* prefix.
Extracted from tinker/server.py — same endpoint logic, now on an APIRouter.
"""
from __future__ import annotations

import asyncio
import os
from fastapi import APIRouter, HTTPException, Request, Response
from tinker import types
from typing import Any

from twinkle.hub import HubOperation
from twinkle.server.common.io_utils import create_checkpoint_manager, create_training_run_manager
from twinkle.server.utils.task_queue import QueueState
from twinkle.server.utils.validation import get_token_from_request
from twinkle.utils.logger import get_logger

logger = get_logger()

tinker_router = APIRouter()


@tinker_router.get('/healthz')
async def healthz(request: Request) -> types.HealthResponse:
    return types.HealthResponse(status='ok')


@tinker_router.get('/get_server_capabilities')
async def get_server_capabilities(request: Request) -> types.GetServerCapabilitiesResponse:
    # GatewayServer injects self.supported_models via request.state in middleware
    supported_models = getattr(request.state, 'supported_models', [])
    return types.GetServerCapabilitiesResponse(supported_models=supported_models)


@tinker_router.post('/telemetry')
async def telemetry(request: Request, body: types.TelemetrySendRequest) -> types.TelemetryResponse:
    return types.TelemetryResponse(status='accepted')


@tinker_router.post('/create_session')
async def create_session(request: Request, body: types.CreateSessionRequest) -> types.CreateSessionResponse:
    state = request.state.server_state
    session_id = state.create_session(body.model_dump())
    return types.CreateSessionResponse(session_id=session_id)


@tinker_router.post('/session_heartbeat')
async def session_heartbeat(request: Request, body: types.SessionHeartbeatRequest) -> types.SessionHeartbeatResponse:
    state = request.state.server_state
    alive = state.touch_session(body.session_id)
    if not alive:
        raise HTTPException(status_code=404, detail='Unknown session')
    return types.SessionHeartbeatResponse()


@tinker_router.post('/create_sampling_session')
async def create_sampling_session(request: Request,
                                  body: types.CreateSamplingSessionRequest) -> types.CreateSamplingSessionResponse:
    state = request.state.server_state
    sampling_session_id = state.create_sampling_session(body.model_dump())
    return types.CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)


@tinker_router.post('/retrieve_future')
async def retrieve_future(request: Request, body: types.FutureRetrieveRequest) -> Any:
    """Retrieve the result of an async task with long polling."""
    state = request.state.server_state
    request_id = body.request_id
    max_wait = float(os.environ.get('TWINKLE_LONG_POLL_TIMEOUT', '30'))
    poll_interval = float(os.environ.get('TWINKLE_POLL_INTERVAL', '0.5'))
    start = asyncio.get_event_loop().time()

    while True:
        record = state.get_future(request_id)

        if record is None:
            return {'type': 'try_again'}

        status = record.get('status')
        if status not in ('pending', 'queued', 'running', 'rate_limited'):
            break

        if asyncio.get_event_loop().time() - start >= max_wait:
            response_data = {'type': 'try_again'}
            if queue_state := record.get('queue_state'):
                response_data['queue_state'] = queue_state
            if queue_state_reason := record.get('queue_state_reason'):
                response_data['queue_state_reason'] = queue_state_reason
            return response_data

        await asyncio.sleep(poll_interval)

    record = state.get_future(request_id)
    if not record:
        return {'type': 'try_again'}

    status = record.get('status')

    if status == 'rate_limited':
        return {
            'type': 'try_again',
            'queue_state': QueueState.PAUSED_RATE_LIMIT.value,
            'queue_state_reason': record.get('reason', 'Rate limit exceeded')
        }

    if status == 'failed':
        result = record.get('result', {})
        return {'error': result.get('error', 'Unknown error'), 'category': result.get('category', 'Server')}

    result = record.get('result')
    if result is None:
        raise HTTPException(status_code=500, detail='Task completed but no result found')

    if hasattr(result, 'model_dump'):
        return result.model_dump()
    return result


# --- Training Runs Endpoints ---


@tinker_router.get('/training_runs')
async def get_training_runs(request: Request, limit: int = 20, offset: int = 0) -> types.TrainingRunsResponse:
    token = get_token_from_request(request)
    training_run_manager = create_training_run_manager(token, client_type='tinker')
    return training_run_manager.list_runs(limit=limit, offset=offset)


@tinker_router.get('/training_runs/{run_id}')
async def get_training_run(request: Request, run_id: str) -> types.TrainingRun:
    token = get_token_from_request(request)
    training_run_manager = create_training_run_manager(token, client_type='tinker')
    run = training_run_manager.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f'Training run {run_id} not found')
    return run


@tinker_router.get('/training_runs/{run_id}/checkpoints')
async def get_run_checkpoints(request: Request, run_id: str) -> types.CheckpointsListResponse:
    token = get_token_from_request(request)
    checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
    response = checkpoint_manager.list_checkpoints(run_id)
    if not response:
        raise HTTPException(status_code=404, detail=f'Training run {run_id} not found')
    return response


@tinker_router.delete('/training_runs/{run_id}/checkpoints/{checkpoint_id:path}')
async def delete_run_checkpoint(request: Request, run_id: str, checkpoint_id: str) -> Any:
    token = get_token_from_request(request)
    checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
    success = checkpoint_manager.delete(run_id, checkpoint_id)
    if not success:
        raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found for run {run_id}')
    return None


@tinker_router.post('/weights_info')
async def weights_info(request: Request, body: dict[str, Any]) -> types.WeightsInfoResponse:
    token = get_token_from_request(request)
    checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
    tinker_path = body.get('tinker_path')
    response = checkpoint_manager.get_weights_info(tinker_path)
    if not response:
        raise HTTPException(status_code=404, detail=f'Weights at {tinker_path} not found')
    return response


@tinker_router.post('/training_runs/{run_id}/checkpoints/{checkpoint_id:path}/publish')
async def publish_checkpoint(request: Request, run_id: str, checkpoint_id: str) -> Response:
    token = get_token_from_request(request)
    modelscope_config_lock = request.state.modelscope_config_lock

    training_run_manager = create_training_run_manager(token, client_type='tinker')
    checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')

    run = training_run_manager.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')

    checkpoint = checkpoint_manager.get(run_id, checkpoint_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found')

    checkpoint_dir = str(checkpoint_manager.get_ckpt_dir(run_id, checkpoint_id))

    async with modelscope_config_lock:
        try:
            from modelscope.hub.api import HubApi, ModelScopeConfig
            hub_api = HubApi(token=token)
            hub_api.login()
            username = ModelScopeConfig.get_user_info()[0]
        except Exception as e:
            logger.error(f'Failed to get username from ModelScope: {e}')
            raise HTTPException(
                status_code=401, detail='Failed to get username from ModelScope. Please ensure your token is valid.')

    checkpoint_name = checkpoint_id.split('/')[-1]
    hub_model_id = f'{username}/{run_id}_{checkpoint_name}'
    HubOperation.async_push_to_hub(repo_id=hub_model_id, folder_path=checkpoint_dir, token=token, private=True)

    return Response(status_code=204)


# --- Model Proxy Endpoints ---


@tinker_router.post('/create_model')
async def create_model(request: Request, body: types.CreateModelRequest) -> Any:
    proxy = request.state.proxy
    validate_base_model = request.state.validate_base_model
    validate_base_model(body.base_model)
    return await proxy.proxy_to_model(request, 'create_model', body.base_model)


@tinker_router.post('/get_info')
async def get_info(request: Request, body: types.GetInfoRequest) -> Any:
    proxy = request.state.proxy
    get_base_model = request.state.get_base_model
    return await proxy.proxy_to_model(request, 'get_info', get_base_model(body.model_id))


@tinker_router.post('/unload_model')
async def unload_model(request: Request, body: types.UnloadModelRequest) -> Any:
    proxy = request.state.proxy
    get_base_model = request.state.get_base_model
    return await proxy.proxy_to_model(request, 'unload_model', get_base_model(body.model_id))


@tinker_router.post('/forward')
async def forward(request: Request, body: types.ForwardRequest) -> Any:
    proxy = request.state.proxy
    get_base_model = request.state.get_base_model
    return await proxy.proxy_to_model(request, 'forward', get_base_model(body.model_id))


@tinker_router.post('/forward_backward')
async def forward_backward(request: Request, body: types.ForwardBackwardRequest) -> Any:
    proxy = request.state.proxy
    get_base_model = request.state.get_base_model
    return await proxy.proxy_to_model(request, 'forward_backward', get_base_model(body.model_id))


@tinker_router.post('/optim_step')
async def optim_step(request: Request, body: types.OptimStepRequest) -> Any:
    proxy = request.state.proxy
    get_base_model = request.state.get_base_model
    return await proxy.proxy_to_model(request, 'optim_step', get_base_model(body.model_id))


@tinker_router.post('/save_weights')
async def save_weights(request: Request, body: types.SaveWeightsRequest) -> Any:
    proxy = request.state.proxy
    get_base_model = request.state.get_base_model
    return await proxy.proxy_to_model(request, 'save_weights', get_base_model(body.model_id))


@tinker_router.post('/load_weights')
async def load_weights(request: Request, body: types.LoadWeightsRequest) -> Any:
    proxy = request.state.proxy
    get_base_model = request.state.get_base_model
    return await proxy.proxy_to_model(request, 'load_weights', get_base_model(body.model_id))


# --- Sampler Proxy Endpoints ---


@tinker_router.post('/asample')
async def asample(request: Request, body: types.SampleRequest) -> Any:
    proxy = request.state.proxy
    state = request.state.server_state
    base_model = body.base_model
    if not base_model and body.sampling_session_id:
        session = state.get_sampling_session(body.sampling_session_id)
        if session:
            base_model = session.get('base_model')
    return await proxy.proxy_to_sampler(request, 'asample', base_model)


@tinker_router.post('/save_weights_for_sampler')
async def save_weights_for_sampler(request: Request, body: types.SaveWeightsForSamplerRequest) -> Any:
    proxy = request.state.proxy
    get_base_model = request.state.get_base_model
    return await proxy.proxy_to_model(request, 'save_weights_for_sampler', get_base_model(body.model_id))
