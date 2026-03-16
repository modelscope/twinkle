# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified Gateway Server.

A single Ray Serve deployment that serves both Tinker (/tinker/*) and
Twinkle (/twinkle/*) management and proxy endpoints.
"""
from __future__ import annotations

import asyncio
from fastapi import FastAPI, HTTPException, Request
from ray import serve
from tinker import types as tinker_types
from typing import Any

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
                 server_config: dict[str, Any] = {},
                 http_options: dict[str, Any] | None = None,
                 **kwargs) -> None:
        self.state = get_server_state(**server_config)
        self.route_prefix = kwargs.get('route_prefix', '/api/v1')
        self.http_options = http_options or {}
        self.proxy = ServiceProxy(http_options=http_options, route_prefix=self.route_prefix)
        self.supported_models = self._normalize_models(supported_models) or [
            tinker_types.SupportedModel(model_name='Qwen/Qwen3-30B-A3B-Instruct-2507'),
        ]
        self._modelscope_config_lock = asyncio.Lock()

    def _normalize_models(self, supported_models):
        if not supported_models:
            return []
        normalized = []
        for item in supported_models:
            if isinstance(item, tinker_types.SupportedModel):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(tinker_types.SupportedModel(**item))
            elif isinstance(item, str):
                normalized.append(tinker_types.SupportedModel(model_name=item))
        return normalized

    def _validate_base_model(self, base_model: str) -> None:
        supported_model_names = [m.model_name for m in self.supported_models]
        if base_model not in supported_model_names:
            raise HTTPException(
                status_code=400,
                detail=f"Base model '{base_model}' is not supported. "
                f"Supported models: {', '.join(supported_model_names)}")

    def _get_base_model(self, model_id: str) -> str:
        metadata = self.state.get_model_metadata(model_id)
        if metadata and metadata.get('base_model'):
            return metadata['base_model']
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')


def build_server_app(deploy_options: dict[str, Any],
                     supported_models: list | None = None,
                     server_config: dict[str, Any] = {},
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
    app = FastAPI()

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    def get_self() -> GatewayServer:
        return serve.get_replica_context().servable_object

    _register_tinker_routes(app, get_self)
    _register_twinkle_routes(app, get_self)

    GatewayServerWithIngress = serve.ingress(app)(GatewayServer)
    DeploymentClass = serve.deployment(name='GatewayServer')(GatewayServerWithIngress)
    return DeploymentClass.options(**deploy_options).bind(
        supported_models=supported_models, server_config=server_config, http_options=http_options, **kwargs)
