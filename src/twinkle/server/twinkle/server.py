# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle REST API Server

This module provides a FastAPI server with REST API endpoints for:
- Training run management (list, get, update)
- Checkpoint management (list, delete)
- Weights info retrieval

All endpoints include permission control to ensure users can only
access their own resources.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ray import serve
from typing import Any

from twinkle.server.utils.state import ServerStateProxy, get_server_state
from twinkle.server.utils.validation import get_token_from_request, verify_request_token
from .common.io_utils import (CheckpointsListResponse, TrainingRun, TrainingRunsResponse, WeightsInfoResponse,
                              create_checkpoint_manager, create_training_run_manager, validate_user_path)

# ----- Request/Response Models -----


class HealthResponse(BaseModel):
    status: str


class WeightsInfoRequest(BaseModel):
    twinkle_path: str


class DeleteCheckpointResponse(BaseModel):
    success: bool
    message: str


class ErrorResponse(BaseModel):
    detail: str


def build_server_app(deploy_options: dict[str, Any], **kwargs):
    """
    Build the Twinkle REST API server application.

    This function creates a FastAPI application wrapped in a Ray Serve deployment
    that provides REST API endpoints for managing training runs and checkpoints.

    Args:
        deploy_options: Ray Serve deployment options (num_replicas, etc.)
        **kwargs: Additional configuration options

    Returns:
        A Ray Serve deployment handle
    """
    app = FastAPI(
        title='Twinkle Server', description='REST API for managing training runs and checkpoints', version='1.0.0')

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        """Verify authentication token for all requests."""
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name='TwinkleServer')
    @serve.ingress(app)
    class TwinkleServer:
        """
        Twinkle REST API Server.

        This server provides endpoints for:
        - Health checks
        - Training run management
        - Checkpoint management
        - Weights info retrieval

        All modifying operations (delete, etc.) are protected by permission checks
        to ensure users can only modify their own resources.
        """

        def __init__(self, **kwargs) -> None:
            self.state: ServerStateProxy = get_server_state()
            self.route_prefix = kwargs.get('route_prefix', '/api/v1')

        def _get_user_token(self, request: Request) -> str:
            """Extract user token from request state."""
            return get_token_from_request(request)

        # ----- Health Check -----

        @app.get('/healthz', response_model=HealthResponse)
        async def healthz(self, request: Request) -> HealthResponse:
            """
            Health check endpoint.

            Returns:
                HealthResponse with status "ok" if server is healthy
            """
            return HealthResponse(status='ok')

        # ----- Training Runs Endpoints -----

        @app.get('/training_runs', response_model=TrainingRunsResponse)
        async def get_training_runs(self, request: Request, limit: int = 20, offset: int = 0) -> TrainingRunsResponse:
            """
            List training runs.

            Returns training runs owned by the current user.

            Args:
                limit: Maximum number of results (default: 20)
                offset: Offset for pagination (default: 0)

            Returns:
                TrainingRunsResponse with list of training runs and pagination info
            """
            token = self._get_user_token(request)
            training_run_manager = create_training_run_manager(token)
            return training_run_manager.list_runs(limit=limit, offset=offset)

        @app.get('/training_runs/{run_id}', response_model=TrainingRun)
        async def get_training_run(self, request: Request, run_id: str) -> TrainingRun:
            """
            Get details of a specific training run.

            Users can only view their own training runs.

            Args:
                run_id: The training run identifier

            Returns:
                TrainingRun details

            Raises:
                HTTPException 404 if run not found or not owned by user
            """
            token = self._get_user_token(request)
            training_run_manager = create_training_run_manager(token)
            run = training_run_manager.get_with_permission(run_id)
            if not run:
                raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
            return run

        @app.get('/training_runs/{run_id}/checkpoints', response_model=CheckpointsListResponse)
        async def get_run_checkpoints(self, request: Request, run_id: str) -> CheckpointsListResponse:
            """
            List checkpoints for a training run.

            Users can only view checkpoints for their own training runs.

            Args:
                run_id: The training run identifier

            Returns:
                CheckpointsListResponse with list of checkpoints

            Raises:
                HTTPException 404 if run not found or not owned by user
            """
            token = self._get_user_token(request)
            checkpoint_manager = create_checkpoint_manager(token)
            response = checkpoint_manager.list_checkpoints(run_id)
            if response is None:
                raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')
            return response

        @app.delete('/training_runs/{run_id}/checkpoints/{checkpoint_id:path}')
        async def delete_run_checkpoint(self, request: Request, run_id: str,
                                        checkpoint_id: str) -> DeleteCheckpointResponse:
            """
            Delete a checkpoint from a training run.

            Users can only delete checkpoints from their own training runs.
            Path traversal (using ..) is not allowed.

            Args:
                run_id: The training run identifier
                checkpoint_id: The checkpoint identifier (can include path like weights/checkpoint_name)

            Returns:
                DeleteCheckpointResponse indicating success or failure

            Raises:
                HTTPException 400 for invalid paths
                HTTPException 403 if not owned by user
                HTTPException 404 if checkpoint not found
            """
            token = self._get_user_token(request)

            # Validate path safety
            if not validate_user_path(token, checkpoint_id):
                raise HTTPException(status_code=400, detail='Invalid checkpoint path: path traversal not allowed')

            checkpoint_manager = create_checkpoint_manager(token)
            success = checkpoint_manager.delete(run_id, checkpoint_id)
            if not success:
                raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found or access denied')

            return DeleteCheckpointResponse(success=True, message=f'Checkpoint {checkpoint_id} deleted successfully')

        @app.post('/weights_info', response_model=WeightsInfoResponse)
        async def weights_info(self, request: Request, body: WeightsInfoRequest) -> WeightsInfoResponse:
            """
            Get information about saved weights.

            Users can only view info for their own weights.

            Args:
                body: Request containing the twinkle_path

            Returns:
                WeightsInfoResponse with weight details

            Raises:
                HTTPException 404 if weights not found or not owned by user
            """
            token = self._get_user_token(request)
            checkpoint_manager = create_checkpoint_manager(token)
            response = checkpoint_manager.get_weights_info(body.twinkle_path)
            if response is None:
                raise HTTPException(
                    status_code=404, detail=f'Weights at {body.twinkle_path} not found or access denied')
            return response

        # ----- Checkpoint Path Resolution -----

        @app.get('/checkpoint_path/{run_id}/{checkpoint_id:path}')
        async def get_checkpoint_path(self, request: Request, run_id: str, checkpoint_id: str) -> dict[str, str]:
            """
            Get the filesystem path for a checkpoint.

            This endpoint resolves a checkpoint ID to its actual filesystem path,
            which can be used for loading weights during resume training.

            Args:
                run_id: The training run identifier
                checkpoint_id: The checkpoint identifier

            Returns:
                Dict with 'path' key containing the filesystem path

            Raises:
                HTTPException 403/404 for permission/not found errors
            """
            token = self._get_user_token(request)

            # Validate path safety
            if not validate_user_path(token, checkpoint_id):
                raise HTTPException(status_code=400, detail='Invalid checkpoint path: path traversal not allowed')

            training_run_manager = create_training_run_manager(token)
            checkpoint_manager = create_checkpoint_manager(token)

            # Check ownership
            run = training_run_manager.get(run_id)
            if not run:
                raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')

            # Get checkpoint with token-based path
            checkpoint = checkpoint_manager.get(run_id, checkpoint_id)
            if not checkpoint:
                raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found')

            # Return the filesystem path
            ckpt_dir = checkpoint_manager.get_ckpt_dir(run_id, checkpoint_id)
            return {'path': str(ckpt_dir), 'twinkle_path': checkpoint.twinkle_path}

    return TwinkleServer.options(**deploy_options).bind(**kwargs)
