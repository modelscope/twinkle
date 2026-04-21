# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified Gateway Server.

A single Ray Serve deployment that serves both Tinker (/tinker/*) and
Twinkle (/twinkle/*) management and proxy endpoints.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from ray import serve
from typing import Any

import twinkle_client.types as types
from twinkle.server.utils.metrics import create_metrics_middleware
from twinkle.server.utils.state import get_server_state
from twinkle.server.utils.validation import verify_request_token
from twinkle.utils.logger import get_logger
from .proxy import ServiceProxy
from .tinker_gateway_handlers import _register_tinker_routes
from .twinkle_gateway_handlers import _register_twinkle_routes

logger = get_logger()


class GatewayServer:
    """Unified gateway server handling both Tinker and Twinkle API clients."""

    def __init__(self,
                 supported_models: list | None = None,
                 server_config: dict[str, Any] | None = None,
                 http_options: dict[str, Any] | None = None,
                 **kwargs) -> None:
        server_config = server_config or {}
        self.state = get_server_state(**server_config)
        self.route_prefix = kwargs.get('route_prefix', '/api/v1')
        self.http_options = http_options or {}
        self.proxy = ServiceProxy(http_options=http_options, route_prefix=self.route_prefix)
        self.supported_models = self._normalize_models(supported_models) or [
            types.SupportedModel(model_name='Qwen/Qwen3.6-35B-A3B'),
        ]
        self._modelscope_config_lock = asyncio.Lock()

    def _normalize_models(self, supported_models):
        if not supported_models:
            return []
        normalized = []
        for item in supported_models:
            if isinstance(item, types.SupportedModel):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(types.SupportedModel(**item))
            elif isinstance(item, str):
                normalized.append(types.SupportedModel(model_name=item))
        return normalized

    def _validate_base_model(self, base_model: str) -> None:
        supported_model_names = [m.model_name for m in self.supported_models]
        if base_model not in supported_model_names:
            raise HTTPException(
                status_code=400,
                detail=f"Base model '{base_model}' is not supported. "
                f"Supported models: {', '.join(supported_model_names)}")

    async def _get_base_model(self, model_id: str) -> str:
        metadata = await self.state.get_model_metadata(model_id)
        if metadata and metadata.get('base_model'):
            return metadata['base_model']
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')


def build_server_app(deploy_options: dict[str, Any],
                     supported_models: list | None = None,
                     server_config: dict[str, Any] | None = None,
                     http_options: dict[str, Any] | None = None,
                     **kwargs):
    """Build and configure the unified gateway server application.

    Serves Tinker endpoints at /* and Twinkle endpoints at /twinkle/*.

    Args:
        deploy_options: Ray Serve deployment configuration
        supported_models: List of supported base models for tinker validation
        server_config: Server configuration options
        http_options: HTTP server options (host, port) for internal proxy routing
        **kwargs: Additional keyword arguments (route_prefix, etc.)

    Returns:
        Configured Ray Serve deployment bound with options
    """

    def get_self() -> GatewayServer:
        return serve.get_replica_context().servable_object

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        try:
            await get_self().proxy.close()
        except Exception:
            pass

    app = FastAPI(lifespan=lifespan)

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    app.middleware('http')(create_metrics_middleware('Gateway'))

    _register_tinker_routes(app, get_self)
    _register_twinkle_routes(app, get_self)

    GatewayServerWithIngress = serve.ingress(app)(GatewayServer)
    DeploymentClass = serve.deployment(name='GatewayServer')(GatewayServerWithIngress)
    return DeploymentClass.options(**deploy_options).bind(
        supported_models=supported_models, server_config=server_config, http_options=http_options, **kwargs)
