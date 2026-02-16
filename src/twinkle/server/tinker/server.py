# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Tinker-compatible server implementation.

This module provides a Ray Serve-based server that implements the Tinker API for distributed
training and inference. It acts as a routing layer that:
1. Handles client requests and validates tokens
2. Manages training runs and checkpoints with user isolation
3. Proxies requests to appropriate model or sampler deployments based on base_model
"""

from __future__ import annotations

import asyncio
import dataclasses
import httpx
import logging
import os
from fastapi import FastAPI, HTTPException, Request, Response
from ray import serve
from tinker import types
from typing import Any, Dict, List, Optional

from twinkle.hub import HubOperation
from twinkle.server.utils.state import get_server_state
from twinkle.server.utils.task_queue import QueueState
from twinkle.server.utils.validation import get_token_from_request, verify_request_token
from .common.io_utils import create_checkpoint_manager, create_training_run_manager

logger = logging.getLogger(__name__)


def build_server_app(deploy_options: dict[str, Any],
                     supported_models: list[types.SupportedModel] | None = None,
                     server_config: dict[str, Any] = {},
                     **kwargs):
    """Build and configure the Tinker-compatible server application.

    This factory function creates a FastAPI application with Ray Serve deployment
    that handles routing, authentication, and proxying for training and inference.

    Args:
        deploy_options: Ray Serve deployment configuration (num_replicas, etc.)
        supported_models: List of supported base models for validation
        server_config: Server configuration options (per_token_adapter_limit, etc.)
        **kwargs: Additional keyword arguments (route_prefix, etc.)

    Returns:
        Configured Ray Serve deployment bound with options
    """
    app = FastAPI()

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        """Middleware to verify authentication token for all requests."""
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name='TinkerCompatServer')
    @serve.ingress(app)
    class TinkerCompatServer:
        """Main server class handling Tinker API endpoints and request routing.

        This class manages:
        - Server state and session management
        - Request validation and authentication
        - Proxying to model/sampler deployments
        - Training run and checkpoint CRUD operations
        """

        def __init__(self,
                     supported_models: list[types.SupportedModel] | None = None,
                     server_config: dict[str, Any] = {},
                     **kwargs) -> None:
            """Initialize the Tinker-compatible server.

            Args:
                supported_models: List of supported base models for validation
                **kwargs: Additional configuration (route_prefix, etc.)
            """
            # Get per_token_adapter_limit from kwargs or use default
            self.state = get_server_state(**server_config)
            # Disable proxy for internal requests to avoid routing through external proxies
            self.client = httpx.AsyncClient(timeout=None, trust_env=False)
            self.route_prefix = kwargs.get('route_prefix', '/api/v1')
            self.supported_models = self.normalize_models(supported_models) or [
                types.SupportedModel(model_name='Qwen/Qwen3-30B-A3B-Instruct-2507'),
            ]
            # Lock for ModelScope config file operations (login writes, get_user_info reads)
            self._modelscope_config_lock = asyncio.Lock()

        def normalize_models(self, supported_models):
            # Normalize supported_models to objects; passing raw dicts can trigger internal errors
            # when creating LoRA training clients via the tinker API.
            if supported_models:
                normalized = []
                for item in supported_models:
                    if isinstance(item, types.SupportedModel):
                        normalized.append(item)
                    elif isinstance(item, dict):
                        normalized.append(types.SupportedModel(**item))
                    else:
                        normalized.append(types.SupportedModel(name=item))
                return normalized

        def _validate_base_model(self, base_model: str) -> None:
            """Validate that base_model is in supported_models list.

            Args:
                base_model: The base model name to validate

            Raises:
                HTTPException: If base_model is not supported
            """
            supported_model_names = [m.model_name for m in self.supported_models]
            if base_model not in supported_model_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"Base model '{base_model}' is not supported. "
                    f"Supported models: {', '.join(supported_model_names)}")

        def _get_base_model(self, model_id: str) -> str:
            """Get base_model for a model_id from state metadata.

            Args:
                model_id: The model identifier to lookup

            Returns:
                The base model name

            Raises:
                HTTPException: If model_id not found in state
            """
            metadata = self.state.get_model_metadata(model_id)
            if metadata and metadata.get('base_model'):
                return metadata['base_model']
            raise HTTPException(status_code=404, detail=f'Model {model_id} not found')

        async def _proxy_request(self, request: Request, endpoint: str, base_model: str, service_type: str) -> Response:
            """Generic proxy method to forward requests to model or sampler services.

            This method consolidates the common proxy logic for both model and sampler endpoints.

            Args:
                request: The incoming FastAPI request
                endpoint: The target endpoint name (e.g., 'create_model', 'asample')
                base_model: The base model name for routing
                service_type: Either 'model' or 'sampler' to determine the target service

            Returns:
                Proxied response from the target service
            """
            body_bytes = await request.body()

            # Construct target URL: /{service_type}/{base_model}/{endpoint}
            prefix = self.route_prefix.rstrip('/') if self.route_prefix else ''
            base_url = f'{request.url.scheme}://{request.url.netloc}'
            target_url = f'{base_url}{prefix}/{service_type}/{base_model}/{endpoint}'

            headers = dict(request.headers)
            headers.pop('host', None)
            headers.pop('content-length', None)

            try:
                if os.environ.get('TWINKLE_DEBUG_PROXY', '0') == '1':
                    logger.info('proxy_to_model endpoint=%s target_url=%s x-ray-serve-request-id=%s', endpoint,
                                target_url, headers.get('x-ray-serve-request-id'))
                rp_ = await self.client.request(
                    method=request.method,
                    url=target_url,
                    content=body_bytes,
                    headers=headers,
                    params=request.query_params,
                )
                if os.environ.get('TWINKLE_DEBUG_PROXY', '0') == '1':
                    logger.info('proxy_to_model response status=%s body=%s', rp_.status_code, rp_.text[:200])
                return Response(
                    content=rp_.content,
                    status_code=rp_.status_code,
                    headers=dict(rp_.headers),
                    media_type=rp_.headers.get('content-type'),
                )
            except Exception as e:
                return Response(content=f'Proxy Error: {str(e)}', status_code=502)

        async def _proxy_to_model(self, request: Request, endpoint: str, base_model: str) -> Response:
            """Proxy request to model endpoint.

            Routes the request to the appropriate model deployment based on base_model.

            Args:
                request: The incoming FastAPI request
                endpoint: The target endpoint name (e.g., 'create_model', 'forward')
                base_model: The base model name for routing

            Returns:
                Proxied response from the model service
            """
            return await self._proxy_request(request, endpoint, base_model, 'model')

        async def _proxy_to_sampler(self, request: Request, endpoint: str, base_model: str) -> Response:
            """Proxy request to sampler endpoint.

            Routes the request to the appropriate sampler deployment based on base_model.

            Args:
                request: The incoming FastAPI request
                endpoint: The target endpoint name (e.g., 'asample')
                base_model: The base model name for routing

            Returns:
                Proxied response from the sampler service
            """
            return await self._proxy_request(request, endpoint, base_model, 'sampler')

        # --- Endpoints ---------------------------------------------------------

        @app.get('/healthz')
        async def healthz(self, request: Request) -> types.HealthResponse:
            """Health check endpoint.

            Returns:
                HealthResponse indicating server is operational
            """
            return types.HealthResponse(status='ok')

        @app.get('/get_server_capabilities')
        async def get_server_capabilities(self, request: Request) -> types.GetServerCapabilitiesResponse:
            """Get server capabilities including supported models.

            Returns:
                GetServerCapabilitiesResponse with list of supported models
            """
            return types.GetServerCapabilitiesResponse(supported_models=self.supported_models)

        @app.post('/telemetry')
        async def telemetry(self, request: Request, body: types.TelemetrySendRequest) -> types.TelemetryResponse:
            """Accept telemetry data from clients.

            Note: Telemetry is accepted but not persisted; this endpoint is intentionally lightweight.

            Returns:
                TelemetryResponse indicating data was accepted
            """
            return types.TelemetryResponse(status='accepted')

        @app.post('/create_session')
        async def create_session(self, request: Request,
                                 body: types.CreateSessionRequest) -> types.CreateSessionResponse:
            """Create a new training session.

            Args:
                body: Session creation parameters

            Returns:
                CreateSessionResponse with new session_id
            """
            session_id = self.state.create_session(body.model_dump())
            return types.CreateSessionResponse(session_id=session_id)

        @app.post('/session_heartbeat')
        async def session_heartbeat(self, request: Request,
                                    body: types.SessionHeartbeatRequest) -> types.SessionHeartbeatResponse:
            """Keep a session alive via heartbeat.

            Args:
                body: Heartbeat request with session_id

            Returns:
                SessionHeartbeatResponse if session is alive

            Raises:
                HTTPException: If session not found
            """
            alive = self.state.touch_session(body.session_id)
            if not alive:
                raise HTTPException(status_code=404, detail='Unknown session')
            return types.SessionHeartbeatResponse()

        @app.post('/create_sampling_session')
        async def create_sampling_session(
                self, request: Request,
                body: types.CreateSamplingSessionRequest) -> types.CreateSamplingSessionResponse:
            """Create a new sampling (inference) session.

            Args:
                body: Sampling session creation parameters

            Returns:
                CreateSamplingSessionResponse with new sampling_session_id
            """
            sampling_session_id = self.state.create_sampling_session(body.model_dump())
            return types.CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)

        @app.post('/retrieve_future')
        async def retrieve_future(self, request: Request, body: types.FutureRetrieveRequest) -> Any:
            """Retrieve the result of an async task with long polling.

            Server waits up to 30s for task completion instead of immediately returning try_again.
            This reduces client polling frequency from ~100 req/s to ~1 req/30s.
            """
            request_id = body.request_id
            max_wait = float(os.environ.get('TWINKLE_LONG_POLL_TIMEOUT', '30'))
            poll_interval = float(os.environ.get('TWINKLE_POLL_INTERVAL', '0.5'))
            start = asyncio.get_event_loop().time()

            # Long poll: wait for task completion or timeout
            while True:
                record = self.state.get_future(request_id)

                if record is None:
                    return {'type': 'try_again'}

                status = record.get('status')

                # Task finished, return immediately
                if status not in ('pending', 'queued', 'running', 'rate_limited'):
                    break

                # Timeout, let client retry
                if asyncio.get_event_loop().time() - start >= max_wait:
                    response_data = {'type': 'try_again'}
                    if queue_state := record.get('queue_state'):
                        response_data['queue_state'] = queue_state
                    if queue_state_reason := record.get('queue_state_reason'):
                        response_data['queue_state_reason'] = queue_state_reason
                    return response_data

                await asyncio.sleep(poll_interval)

            # Handle final result
            record = self.state.get_future(request_id)
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

        # --- Restful Endpoints ------------------------------------------

        @app.get('/training_runs')
        async def get_training_runs(self,
                                    request: Request,
                                    limit: int = 20,
                                    offset: int = 0) -> types.TrainingRunsResponse:
            """
            List training runs for the current user.

            Uses token-based isolation to only show runs owned by the requesting user.

            Args:
                request: FastAPI request with token in state
                limit: Maximum number of results
                offset: Pagination offset

            Returns:
                TrainingRunsResponse with user's training runs
            """
            token = get_token_from_request(request)
            training_run_manager = create_training_run_manager(token)
            return training_run_manager.list_runs(limit=limit, offset=offset)

        @app.get('/training_runs/{run_id}')
        async def get_training_run(self, request: Request, run_id: str) -> types.TrainingRun:
            """
            Get a specific training run.

            Uses token-based isolation to verify user owns the run.

            Args:
                request: FastAPI request with token in state
                run_id: The training run identifier

            Returns:
                TrainingRun details

            Raises:
                HTTPException 404 if run not found in user's token directory
            """
            token = get_token_from_request(request)
            training_run_manager = create_training_run_manager(token)
            run = training_run_manager.get(run_id)
            if not run:
                raise HTTPException(status_code=404, detail=f'Training run {run_id} not found')
            return run

        @app.get('/training_runs/{run_id}/checkpoints')
        async def get_run_checkpoints(self, request: Request, run_id: str) -> types.CheckpointsListResponse:
            """
            List checkpoints for a training run.

            Uses token-based isolation to verify user owns the run.

            Args:
                request: FastAPI request with token in state
                run_id: The training run identifier

            Returns:
                CheckpointsListResponse with list of checkpoints

            Raises:
                HTTPException 404 if run not found in user's token directory
            """
            token = get_token_from_request(request)
            checkpoint_manager = create_checkpoint_manager(token)
            response = checkpoint_manager.list_checkpoints(run_id)
            if not response:
                raise HTTPException(status_code=404, detail=f'Training run {run_id} not found')
            return response

        @app.delete('/training_runs/{run_id}/checkpoints/{checkpoint_id:path}')
        async def delete_run_checkpoint(self, request: Request, run_id: str, checkpoint_id: str) -> Any:
            """
            Delete a checkpoint from a training run.

            Uses token-based isolation to verify user owns the checkpoint.

            Args:
                request: FastAPI request with token in state
                run_id: The training run identifier
                checkpoint_id: The checkpoint identifier (path)

            Returns:
                None (200 OK) if successful

            Raises:
                HTTPException 404 if checkpoint not found in user's token directory
            """
            token = get_token_from_request(request)
            checkpoint_manager = create_checkpoint_manager(token)
            success = checkpoint_manager.delete(run_id, checkpoint_id)
            if not success:
                raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found for run {run_id}')
            return None

        @app.post('/weights_info')
        async def weights_info(self, request: Request, body: dict[str, Any]) -> types.WeightsInfoResponse:
            """
            Get weights information from a tinker path.

            Uses token-based isolation to verify user owns the weights.

            Args:
                request: FastAPI request with token in state
                body: Dict with 'tinker_path' key

            Returns:
                WeightsInfoResponse with weight details

            Raises:
                HTTPException 404 if weights not found in user's token directory
            """
            token = get_token_from_request(request)
            checkpoint_manager = create_checkpoint_manager(token)
            tinker_path = body.get('tinker_path')
            response = checkpoint_manager.get_weights_info(tinker_path)
            if not response:
                raise HTTPException(status_code=404, detail=f'Weights at {tinker_path} not found')
            return response

        @app.post('/training_runs/{run_id}/checkpoints/{checkpoint_id:path}/publish')
        async def publish_checkpoint(self, request: Request, run_id: str, checkpoint_id: str) -> Response:
            """
            Publish a checkpoint to the hub.

            This endpoint uploads a checkpoint to a hub repository. The hub_model_id
            is automatically generated from the checkpoint content and user token.
            The upload is performed asynchronously by default.

            Args:
                request: FastAPI request object (contains token in state)
                run_id: The training run identifier
                checkpoint_id: The checkpoint identifier (can include path like weights/checkpoint_name)

            Returns:
                Response with 204 No Content status

            Raises:
                HTTPException 404 if checkpoint not found or access denied
            """
            token = get_token_from_request(request)

            training_run_manager = create_training_run_manager(token)
            checkpoint_manager = create_checkpoint_manager(token)

            # Check ownership and get training run info
            run = training_run_manager.get(run_id)
            if not run:
                raise HTTPException(status_code=404, detail=f'Training run {run_id} not found or access denied')

            # Get checkpoint with token-based path
            checkpoint = checkpoint_manager.get(run_id, checkpoint_id)
            if not checkpoint:
                raise HTTPException(status_code=404, detail=f'Checkpoint {checkpoint_id} not found')

            # Get the filesystem path for the checkpoint
            checkpoint_dir = str(checkpoint_manager.get_ckpt_dir(run_id, checkpoint_id))

            # Generate hub_model_id from checkpoint content and user token
            # Format: {username}/{run_id}_{checkpoint_name}
            # Use lock to prevent race conditions when multiple requests access ModelScope config file
            async with self._modelscope_config_lock:
                try:
                    from modelscope.hub.api import HubApi, ModelScopeConfig
                    hub_api = HubApi(token=token)
                    hub_api.login()  # Save user info to local
                    username = ModelScopeConfig.get_user_info()[0]
                except Exception as e:
                    logger.error(f'Failed to get username from ModelScope: {e}')
                    raise HTTPException(
                        status_code=401,
                        detail='Failed to get username from ModelScope. Please ensure your token is valid.')

            # Extract checkpoint name from checkpoint_id (e.g., "weights/step-8" -> "step-8")
            checkpoint_name = checkpoint_id.split('/')[-1]
            hub_model_id = f'{username}/{run_id}_{checkpoint_name}'

            # Upload to hub asynchronously with default async_upload=True
            HubOperation.async_push_to_hub(repo_id=hub_model_id, folder_path=checkpoint_dir, token=token, private=True)

            # Return 204 No Content (successful with no response body)
            return Response(status_code=204)

    # --- Proxy Endpoints ---------------------------------------------------------

    # --- Model Proxy Endpoints ----------------------------------------

        @app.post('/create_model')
        async def create_model(self, request: Request, body: types.CreateModelRequest) -> Any:
            """Create a new model (adapter) for training.

            Args:
                body: Model creation request with base_model and config

            Returns:
                Proxied response from model service
            """
            self._validate_base_model(body.base_model)
            return await self._proxy_to_model(request, 'create_model', body.base_model)

        @app.post('/get_info')
        async def get_info(self, request: Request, body: types.GetInfoRequest) -> Any:
            """Get information about a model.

            Args:
                body: Info request with model_id

            Returns:
                Proxied response from model service
            """
            return await self._proxy_to_model(request, 'get_info', self._get_base_model(body.model_id))

        @app.post('/unload_model')
        async def unload_model(self, request: Request, body: types.UnloadModelRequest) -> Any:
            """Unload a model adapter from memory.

            Args:
                body: Unload request with model_id

            Returns:
                Proxied response from model service
            """
            return await self._proxy_to_model(request, 'unload_model', self._get_base_model(body.model_id))

        @app.post('/forward')
        async def forward(self, request: Request, body: types.ForwardRequest) -> Any:
            """Execute forward pass without backward.

            Args:
                body: Forward request with inputs

            Returns:
                Proxied response from model service
            """
            return await self._proxy_to_model(request, 'forward', self._get_base_model(body.model_id))

        @app.post('/forward_backward')
        async def forward_backward(self, request: Request, body: types.ForwardBackwardRequest) -> Any:
            """Execute forward and backward pass for training.

            Args:
                body: Forward-backward request with inputs

            Returns:
                Proxied response from model service
            """
            return await self._proxy_to_model(request, 'forward_backward', self._get_base_model(body.model_id))

        @app.post('/optim_step')
        async def optim_step(self, request: Request, body: types.OptimStepRequest) -> Any:
            """Execute optimizer step to update model weights.

            Args:
                body: Optimizer step request with parameters

            Returns:
                Proxied response from model service
            """
            return await self._proxy_to_model(request, 'optim_step', self._get_base_model(body.model_id))

        @app.post('/save_weights')
        async def save_weights(self, request: Request, body: types.SaveWeightsRequest) -> Any:
            """Save model weights to storage.

            Args:
                body: Save weights request with path

            Returns:
                Proxied response from model service
            """
            return await self._proxy_to_model(request, 'save_weights', self._get_base_model(body.model_id))

        @app.post('/load_weights')
        async def load_weights(self, request: Request, body: types.LoadWeightsRequest) -> Any:
            """Load model weights from storage.

            Args:
                body: Load weights request with path

            Returns:
                Proxied response from model service
            """
            return await self._proxy_to_model(request, 'load_weights', self._get_base_model(body.model_id))

    # --- Sampler Proxy Endpoints ----------------------------------------

        @app.post('/asample')
        async def asample(self, request: Request, body: types.SampleRequest) -> Any:
            """Execute text generation (inference).

            Proxies the request to the sampler service based on base_model.
            The sampler handles model_path resolution from sampling session.

            Args:
                body: Sample request with prompt and sampling parameters

            Returns:
                Proxied response from sampler service
            """
            base_model = body.base_model

            # If base_model not provided, look up from sampling session
            if not base_model and body.sampling_session_id:
                session = self.state.get_sampling_session(body.sampling_session_id)
                if session:
                    base_model = session.get('base_model')

            return await self._proxy_to_sampler(request, 'asample', base_model)

        @app.post('/save_weights_for_sampler')
        async def save_weights_for_sampler(self, request: Request, body: types.SaveWeightsForSamplerRequest) -> Any:
            """Save/convert weights for inference use.

            This endpoint proxies to the model service to save weights for sampler.

            Args:
                body: Save weights request with model_id

            Returns:
                Proxied response from model service
            """
            # Proxy to model service for save_weights_for_sampler
            base_model = self._get_base_model(body.model_id)
            return await self._proxy_to_model(request, 'save_weights_for_sampler', base_model)

    return TinkerCompatServer.options(**deploy_options).bind(
        supported_models=supported_models, server_config=server_config, **kwargs)
