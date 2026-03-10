# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native gateway router.

Provides all twinkle management endpoints under /twinkle/* prefix.
Extracted from twinkle/server.py — same endpoint logic, now on an APIRouter.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Any

from twinkle.server.common.io_utils import create_checkpoint_manager, create_training_run_manager, validate_user_path
from twinkle.server.utils.validation import get_token_from_request
from twinkle.utils.logger import get_logger
from twinkle_client.types.server import DeleteCheckpointResponse, HealthResponse
from twinkle_client.types.training import (CheckpointsListResponse, TrainingRun, TrainingRunsResponse,
                                           WeightsInfoResponse)

logger = get_logger()

twinkle_router = APIRouter()


class WeightsInfoRequest(BaseModel):
    twinkle_path: str


@twinkle_router.get('/healthz', response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    return HealthResponse(status='ok')


@twinkle_router.get('/training_runs', response_model=TrainingRunsResponse)
async def get_training_runs(request: Request, limit: int = 20, offset: int = 0) -> TrainingRunsResponse:
    token = get_token_from_request(request)
    training_run_manager = create_training_run_manager(token, client_type='twinkle')
    return training_run_manager.list_runs(limit=limit, offset=offset)


@twinkle_router.get('/training_runs/{run_id}', response_model=TrainingRun)
async def get_training_run(request: Request, run_id: str) -> TrainingRun:
    token = get_token_from_request(request)
    training_run_manager = create_training_run_manager(token, client_type='twinkle')
    run = training_run_manager.get_with_permission(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
    return run


@twinkle_router.get('/training_runs/{run_id}/checkpoints', response_model=CheckpointsListResponse)
async def get_run_checkpoints(request: Request, run_id: str) -> CheckpointsListResponse:
    token = get_token_from_request(request)
    checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
    response = checkpoint_manager.list_checkpoints(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
    return response


@twinkle_router.delete('/training_runs/{run_id}/checkpoints/{checkpoint_id:path}')
async def delete_run_checkpoint(request: Request, run_id: str, checkpoint_id: str) -> DeleteCheckpointResponse:
    token = get_token_from_request(request)

    if not validate_user_path(token, checkpoint_id):
        raise HTTPException(status_code=400, detail='Invalid checkpoint path: path traversal not allowed')

    checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
    success = checkpoint_manager.delete(run_id, checkpoint_id)
    if not success:
        raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found or access denied')

    return DeleteCheckpointResponse(success=True, message=f'Checkpoint {checkpoint_id} deleted successfully')


@twinkle_router.post('/weights_info', response_model=WeightsInfoResponse)
async def weights_info(request: Request, body: WeightsInfoRequest) -> WeightsInfoResponse:
    token = get_token_from_request(request)
    checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
    response = checkpoint_manager.get_weights_info(body.twinkle_path)
    if response is None:
        raise HTTPException(status_code=404, detail=f'Weights at {body.twinkle_path} not found or access denied')
    return response


@twinkle_router.get('/checkpoint_path/{run_id}/{checkpoint_id:path}')
async def get_checkpoint_path(request: Request, run_id: str, checkpoint_id: str) -> dict[str, str]:
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
    return {'path': str(ckpt_dir), 'twinkle_path': checkpoint.twinkle_path}
