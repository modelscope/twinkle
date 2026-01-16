"""
Minimal Ray Serve + FastAPI implementation of the Tinker server API.
The endpoints mirror the Python SDK expectations so the client can talk to
this server without further glue code. This is a lightweight reference
implementation meant for local development, integration tests, or as a
starting point for a production backend.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import ray
from fastapi import FastAPI, HTTPException, Request
from ray import serve

from tinker import types
from .state import get_server_state, schedule_task


# ----- FastAPI + Serve deployment ------------------------------------------


app = FastAPI(title="Tinker Ray Server", version="0.1.0")


@serve.deployment
@serve.ingress(app)
class TinkerCompatServer:
    def __init__(self, supported_models: Optional[List[types.SupportedModel]] = None) -> None:
        self.state = get_server_state()
        self.supported_models = supported_models or [
            types.SupportedModel(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
            types.SupportedModel(model_name="Qwen/Qwen2.5-7B-Instruct"),
            types.SupportedModel(model_name="Qwen/Qwen2.5-72B-Instruct"),
        ]

    # --- Helpers -----------------------------------------------------------

    @staticmethod
    def _sample_output() -> types.SampleResponse:
        sequence = types.SampledSequence(stop_reason="stop", tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
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
        session_id = await self.state.create_session(body)
        return types.CreateSessionResponse(session_id=session_id)

    @app.post("/session_heartbeat")
    async def session_heartbeat(self, request: Request, body: types.SessionHeartbeatRequest) -> types.SessionHeartbeatResponse:
        alive = await self.state.touch_session(body.session_id)
        if not alive:
            raise HTTPException(status_code=404, detail="Unknown session")
        return types.SessionHeartbeatResponse()


    @app.post("/create_sampling_session")
    async def create_sampling_session(
        self, request: Request, body: types.CreateSamplingSessionRequest
    ) -> types.CreateSamplingSessionResponse:
        sampling_session_id = await self.state.create_sampling_session(body)
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
        record = await self.state.get_future(body.request_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Future not found")
        result = record["result"]
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result


def build_graph() -> Any:
    """Helper to bind the deployment for serve.run."""
    return TinkerCompatServer.bind()


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("TINKER_PORT", "8000"))
    
    # Start serve with specific HTTP options
    serve.start(http_options={"host": "0.0.0.0", "port": port})
    
    serve.run(build_graph(), route_prefix="/api/v1")
    input("\nPress Enter to stop the server...")