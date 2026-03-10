# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native gateway handler mixin.

All endpoints are prefixed /twinkle/* and registered via _register_twinkle_routes(app).
Route closures use self.* directly (no request.state injection needed).
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from twinkle.server.common.io_utils import create_checkpoint_manager, create_training_run_manager, validate_user_path
from twinkle.server.utils.validation import get_token_from_request
from twinkle.utils.logger import get_logger
from twinkle_client.types.server import DeleteCheckpointResponse, HealthResponse, WeightsInfoRequest
from twinkle_client.types.training import (CheckpointsListResponse, TrainingRun, TrainingRunsResponse,
                                           WeightsInfoResponse)

logger = get_logger()


class TwinkleGatewayHandlers:
    """
    Mixin providing Twinkle-native gateway management endpoints.

    Expects the combined class to have: self.state
    """

    @staticmethod
    def _register_twinkle_routes(app: FastAPI):
        """Register all /twinkle/* routes on the given FastAPI app."""

        @app.get('/twinkle/healthz', response_model=HealthResponse)
        async def healthz(self, request: Request) -> HealthResponse:
            return HealthResponse(status='ok')

        @app.get('/twinkle/training_runs', response_model=TrainingRunsResponse)
        async def get_training_runs(self, request: Request, limit: int = 20, offset: int = 0) -> TrainingRunsResponse:
            token = get_token_from_request(request)
            training_run_manager = create_training_run_manager(token, client_type='twinkle')
            return training_run_manager.list_runs(limit=limit, offset=offset)

        @app.get('/twinkle/training_runs/{run_id}', response_model=TrainingRun)
        async def get_training_run(self, request: Request, run_id: str) -> TrainingRun:
            token = get_token_from_request(request)
            training_run_manager = create_training_run_manager(token, client_type='twinkle')
            run = training_run_manager.get_with_permission(run_id)
            if not run:
                raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
            return run

        @app.get('/twinkle/training_runs/{run_id}/checkpoints', response_model=CheckpointsListResponse)
        async def get_run_checkpoints(self, request: Request, run_id: str) -> CheckpointsListResponse:
            token = get_token_from_request(request)
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            response = checkpoint_manager.list_checkpoints(run_id)
            if response is None:
                raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
            return response

        @app.delete('/twinkle/training_runs/{run_id}/checkpoints/{checkpoint_id:path}')
        async def delete_run_checkpoint(self, request: Request, run_id: str,
                                        checkpoint_id: str) -> DeleteCheckpointResponse:
            token = get_token_from_request(request)

            if not validate_user_path(token, checkpoint_id):
                raise HTTPException(status_code=400, detail='Invalid checkpoint path: path traversal not allowed')

            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            success = checkpoint_manager.delete(run_id, checkpoint_id)
            if not success:
                raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found or access denied')

            return DeleteCheckpointResponse(success=True, message=f'Checkpoint {checkpoint_id} deleted successfully')

        @app.post('/twinkle/weights_info', response_model=WeightsInfoResponse)
        async def weights_info(self, request: Request, body: WeightsInfoRequest) -> WeightsInfoResponse:
            token = get_token_from_request(request)
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            response = checkpoint_manager.get_weights_info(body.twinkle_path)
            if response is None:
                raise HTTPException(
                    status_code=404, detail=f'Weights at {body.twinkle_path} not found or access denied')
            return response

        @app.get('/twinkle/checkpoint_path/{run_id}/{checkpoint_id:path}')
        async def get_checkpoint_path(self, request: Request, run_id: str, checkpoint_id: str) -> dict[str, str]:
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
