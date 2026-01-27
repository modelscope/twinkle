# Copyright (c) ModelScope Contributors. All rights reserved.

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from ray import serve

from tinker import types

from twinkle.server.twinkle.validation import verify_request_token
from .state import get_server_state, schedule_task
from .common.io_utils import TrainingRunManager, CheckpointManager


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

        async def _proxy(self, request: Request, target_path: str) -> Response:
            # Construct target URL on the same host
            # Ensure we respect the current route prefix (e.g. /api/v1)
            # when forwarding to sub-routes like /api/v1/model/...
            prefix = self.route_prefix.rstrip("/") if self.route_prefix else ""
            base_url = f"{request.url.scheme}://{request.url.netloc}"
            target_url = f"{base_url}{prefix}{target_path}"

            # Prepare headers
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)

            try:
                rp_ = await self.client.request(
                    method=request.method,
                    url=target_url,
                    content=await request.body(),
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
        async def asample(self, request: Request, body: types.SampleRequest) -> types.UntypedAPIFuture:
            async def _do_sample():
                return self._sample_output()

            return await schedule_task(self.state, _do_sample())

        @app.post("/save_weights_for_sampler")
        async def save_weights_for_sampler(
            self, request: Request, body: types.SaveWeightsForSamplerRequest
        ) -> types.UntypedAPIFuture:
            suffix = body.path or f"sampler-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            path = f"tinker://{body.model_id}/{suffix}"
            sampling_session_id = None
            if body.sampling_session_seq_id is not None:
                sampling_session_id = f"sampling_{body.sampling_session_seq_id}"

            async def _do_save():
                return types.SaveWeightsForSamplerResponseInternal(
                    path=path, sampling_session_id=sampling_session_id
                )

            return await schedule_task(self.state, _do_save(), model_id=body.model_id)

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
            return TrainingRunManager.list_runs(limit=limit, offset=offset)

        @app.get("/training_runs/{run_id}")
        async def get_training_run(self, request: Request, run_id: str) -> types.TrainingRun:
            run = TrainingRunManager.get(run_id)
            if not run:
                raise HTTPException(
                    status_code=404, detail=f"Training run {run_id} not found")
            return run

        @app.get("/training_runs/{run_id}/checkpoints")
        async def get_run_checkpoints(self, request: Request, run_id: str) -> types.CheckpointsListResponse:
            response = CheckpointManager.list_checkpoints(run_id)
            if not response:
                raise HTTPException(
                    status_code=404, detail=f"Training run {run_id} not found")
            return response

        @app.delete("/training_runs/{run_id}/checkpoints/{checkpoint_id:path}")
        async def delete_run_checkpoint(self, request: Request, run_id: str, checkpoint_id: str) -> Any:
            CheckpointManager.delete(run_id, checkpoint_id)
            # We return 200 (null) even if not found to be idempotent, or could raise 404
            return None

        @app.post("/weights_info")
        async def weights_info(self, request: Request, body: Dict[str, Any]) -> types.WeightsInfoResponse:
            tinker_path = body.get("tinker_path")
            response = CheckpointManager.get_weights_info(tinker_path)
            if not response:
                raise HTTPException(
                    status_code=404, detail=f"Weights at {tinker_path} not found")
            return response

    # --- Proxy Endpoints ---------------------------------------------------------

    # --- Model Proxy Endpoints ----------------------------------------

        @app.post("/create_model")
        async def create_model(self, request: Request) -> Any:
            return await self._proxy(request, "/model/create_model")

        @app.post("/get_info")
        async def get_info(self, request: Request) -> Any:
            return await self._proxy(request, "/model/get_info")

        @app.post("/unload_model")
        async def unload_model(self, request: Request) -> Any:
            return await self._proxy(request, "/model/unload_model")

        @app.post("/forward")
        async def forward(self, request: Request) -> Any:
            return await self._proxy(request, "/model/forward")

        @app.post("/forward_backward")
        async def forward_backward(self, request: Request) -> Any:
            return await self._proxy(request, "/model/forward_backward")

        @app.post("/optim_step")
        async def optim_step(self, request: Request) -> Any:
            return await self._proxy(request, "/model/optim_step")

        @app.post("/save_weights")
        async def save_weights(self, request: Request) -> Any:
            return await self._proxy(request, "/model/save_weights")

        @app.post("/load_weights")
        async def load_weights(self, request: Request) -> Any:
            return await self._proxy(request, "/model/load_weights")

    return TinkerCompatServer.options(**deploy_options).bind(supported_models=supported_models, **kwargs)
