# Copyright (c) ModelScope Contributors. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from ray import serve

from tinker import types

from twinkle.server.twinkle.common.validation import verify_request_token, get_token_from_request
from twinkle.server.twinkle.common.state import get_server_state, schedule_task
from .common.io_utils import create_training_run_manager, create_checkpoint_manager


def build_server_app(
    deploy_options: Dict[str, Any],
    supported_models: Optional[List[types.SupportedModel]] = None,
    **kwargs
):
    app = FastAPI()

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name="TinkerCompatServer")
    @serve.ingress(app)
    class TinkerCompatServer:
        def __init__(self, supported_models: Optional[List[types.SupportedModel]] = None, **kwargs) -> None:
            self.state = get_server_state()
            self.client = httpx.AsyncClient(timeout=None)
            self.route_prefix = kwargs.get("route_prefix", "/api/v1")
            self.supported_models = supported_models or [
                types.SupportedModel(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
                types.SupportedModel(model_name="Qwen/Qwen2.5-7B-Instruct"),
                types.SupportedModel(model_name="Qwen/Qwen2.5-72B-Instruct"),
            ]

        def _validate_base_model(self, base_model: str) -> None:
            """Validate that base_model is in supported_models list."""
            supported_model_names = [
                m.model_name for m in self.supported_models]
            if base_model not in supported_model_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"Base model '{base_model}' is not supported. "
                    f"Supported models: {', '.join(supported_model_names)}"
                )

        def _get_base_model(self, model_id: str) -> str:
            """Get base_model for a model_id from state metadata."""
            metadata = self.state.get_model_metadata(model_id)
            if metadata and metadata.get('base_model'):
                return metadata['base_model']
            raise HTTPException(
                status_code=404, detail=f"Model {model_id} not found")

        async def _proxy_to_model(self, request: Request, endpoint: str, base_model: str) -> Response:
            """Proxy request to model endpoint."""
            body_bytes = await request.body()

            # Construct target URL
            prefix = self.route_prefix.rstrip("/") if self.route_prefix else ""
            base_url = f"{request.url.scheme}://{request.url.netloc}"
            target_url = f"{base_url}{prefix}/model/{base_model}/{endpoint}"

            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)

            try:
                rp_ = await self.client.request(
                    method=request.method,
                    url=target_url,
                    content=body_bytes,
                    headers=headers,
                    params=request.query_params,
                )
                return Response(
                    content=rp_.content,
                    status_code=rp_.status_code,
                    headers=dict(rp_.headers),
                    media_type=rp_.headers.get("content-type"),
                )
            except Exception as e:
                return Response(content=f"Proxy Error: {str(e)}", status_code=502)

        async def _proxy_to_sampler(self, request: Request, endpoint: str, base_model: str) -> Response:
            """Proxy request to sampler endpoint."""
            body_bytes = await request.body()

            # Construct target URL: /sampler/{base_model}/{endpoint}
            prefix = self.route_prefix.rstrip("/") if self.route_prefix else ""
            base_url = f"{request.url.scheme}://{request.url.netloc}"
            target_url = f"{base_url}{prefix}/sampler/{base_model}/{endpoint}"

            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)

            try:
                rp_ = await self.client.request(
                    method=request.method,
                    url=target_url,
                    content=body_bytes,
                    headers=headers,
                    params=request.query_params,
                )
                return Response(
                    content=rp_.content,
                    status_code=rp_.status_code,
                    headers=dict(rp_.headers),
                    media_type=rp_.headers.get("content-type"),
                )
            except Exception as e:
                return Response(content=f"Proxy Error: {str(e)}", status_code=502)

        @staticmethod
        def _sample_output() -> types.SampleResponse:
            sequence = types.SampledSequence(stop_reason="stop", tokens=[
                                             1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
            return types.SampleResponse(sequences=[sequence])

        # --- Endpoints ---------------------------------------------------------

        @app.get("/healthz")
        async def healthz(self, request: Request) -> types.HealthResponse:
            return types.HealthResponse(status="ok")

        @app.get("/get_server_capabilities")
        async def get_server_capabilities(self, request: Request) -> types.GetServerCapabilitiesResponse:
            return types.GetServerCapabilitiesResponse(supported_models=self.supported_models)

        @app.post("/telemetry")
        async def telemetry(self, request: Request, body: types.TelemetrySendRequest) -> types.TelemetryResponse:
            # Telemetry is accepted but not persisted; this endpoint is intentionally lightweight.
            return types.TelemetryResponse(status="accepted")

        @app.post("/create_session")
        async def create_session(self, request: Request, body: types.CreateSessionRequest) -> types.CreateSessionResponse:
            session_id = self.state.create_session(body.model_dump())
            return types.CreateSessionResponse(session_id=session_id)

        @app.post("/session_heartbeat")
        async def session_heartbeat(self, request: Request, body: types.SessionHeartbeatRequest) -> types.SessionHeartbeatResponse:
            alive = self.state.touch_session(body.session_id)
            if not alive:
                raise HTTPException(status_code=404, detail="Unknown session")
            return types.SessionHeartbeatResponse()

        @app.post("/create_sampling_session")
        async def create_sampling_session(
            self, request: Request, body: types.CreateSamplingSessionRequest
        ) -> types.CreateSamplingSessionResponse:
            sampling_session_id = self.state.create_sampling_session(
                body.model_dump())
            return types.CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)

        @app.post("/asample")
        async def asample(self, request: Request, body: types.SampleRequest) -> Any:
            """Execute text generation (inference).
            
            This endpoint first tries to use a local sampler if available.
            Otherwise, it proxies the request to the sampler service.
            """
            model_path = body.model_path
            base_model = body.base_model
            
            # If both are None, look up from sampling session
            if not model_path and not base_model and body.sampling_session_id:
                session = self.state.get_sampling_session(body.sampling_session_id)
                if session:
                    model_path = session.get('model_path')
                    base_model = session.get('base_model')
            
            # Extract base_model from model_path if needed
            if model_path and not base_model:
                # Format: twinkle://Qwen/Qwen2.5-0.5B-Instruct/lora/xxx -> Qwen/Qwen2.5-0.5B-Instruct
                path = model_path.replace("twinkle://", "").replace("tinker://", "")
                parts = path.split("/")
                if len(parts) >= 2:
                    base_model = f"{parts[0]}/{parts[1]}"
            
            return await self._proxy_to_sampler(request, "asample", base_model)
            
        @app.post("/save_weights_for_sampler")
        async def save_weights_for_sampler(
            self, request: Request, body: types.SaveWeightsForSamplerRequest
        ) -> Any:
            """Save/convert weights for inference use.
            
            This endpoint proxies to the model service to save weights for sampler.
            """
            # Proxy to model service for save_weights_for_sampler
            base_model = self._get_base_model(body.model_id)
            return await self._proxy_to_model(request, "save_weights_for_sampler", base_model)

        @app.post("/retrieve_future")
        async def retrieve_future(self, request: Request, body: types.FutureRetrieveRequest) -> Any:
            record = self.state.get_future(body.request_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Future not found")
            result = record["result"]
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result

        # --- Training Runs Endpoints ------------------------------------------

        @app.get("/training_runs")
        async def get_training_runs(self, request: Request, limit: int = 20, offset: int = 0) -> types.TrainingRunsResponse:
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

        @app.get("/training_runs/{run_id}")
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
                raise HTTPException(
                    status_code=404, detail=f"Training run {run_id} not found")
            return run

        @app.get("/training_runs/{run_id}/checkpoints")
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
                raise HTTPException(
                    status_code=404, detail=f"Training run {run_id} not found")
            return response

        @app.delete("/training_runs/{run_id}/checkpoints/{checkpoint_id:path}")
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
                raise HTTPException(
                    status_code=404,
                    detail=f"Checkpoint {checkpoint_id} not found for run {run_id}"
                )
            return None

        @app.post("/weights_info")
        async def weights_info(self, request: Request, body: Dict[str, Any]) -> types.WeightsInfoResponse:
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
            tinker_path = body.get("tinker_path")
            response = checkpoint_manager.get_weights_info(tinker_path)
            if not response:
                raise HTTPException(
                    status_code=404, detail=f"Weights at {tinker_path} not found")
            return response

    # --- Proxy Endpoints ---------------------------------------------------------

    # --- Model Proxy Endpoints ----------------------------------------

        @app.post("/create_model")
        async def create_model(self, request: Request, body: types.CreateModelRequest) -> Any:
            self._validate_base_model(body.base_model)
            return await self._proxy_to_model(request, "create_model", body.base_model)

        @app.post("/get_info")
        async def get_info(self, request: Request, body: types.GetInfoRequest) -> Any:
            return await self._proxy_to_model(request, "get_info", self._get_base_model(body.model_id))

        @app.post("/unload_model")
        async def unload_model(self, request: Request, body: types.UnloadModelRequest) -> Any:
            return await self._proxy_to_model(request, "unload_model", self._get_base_model(body.model_id))

        @app.post("/forward")
        async def forward(self, request: Request, body: types.ForwardRequest) -> Any:
            return await self._proxy_to_model(request, "forward", self._get_base_model(body.model_id))

        @app.post("/forward_backward")
        async def forward_backward(self, request: Request, body: types.ForwardBackwardRequest) -> Any:
            return await self._proxy_to_model(request, "forward_backward", self._get_base_model(body.model_id))

        @app.post("/optim_step")
        async def optim_step(self, request: Request, body: types.OptimStepRequest) -> Any:
            return await self._proxy_to_model(request, "optim_step", self._get_base_model(body.model_id))

        @app.post("/save_weights")
        async def save_weights(self, request: Request, body: types.SaveWeightsRequest) -> Any:
            return await self._proxy_to_model(request, "save_weights", self._get_base_model(body.model_id))

        @app.post("/load_weights")
        async def load_weights(self, request: Request, body: types.LoadWeightsRequest) -> Any:
            return await self._proxy_to_model(request, "load_weights", self._get_base_model(body.model_id))

    return TinkerCompatServer.options(**deploy_options).bind(supported_models=supported_models, **kwargs)
