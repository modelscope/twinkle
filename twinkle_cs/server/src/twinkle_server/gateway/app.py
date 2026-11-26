# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified Gateway Server.

A single Ray Serve deployment that serves both Tinker (/tinker/*) and
Twinkle (/twinkle/*) management and proxy endpoints.
"""
from __future__ import annotations

import asyncio
from fastapi import FastAPI, HTTPException
from typing import Any

import twinkle.protocol.types as types
from twinkle_server.deployment import LazyCleanupMixin, bind_deployment, build_deployment_app
from twinkle_server.lifecycle.proxy import ServiceProxy
from twinkle_server.state import get_server_state
from twinkle.utils.logger import get_logger
from .openai_handlers import _register_api_key_gate, _register_openai_routes
from .tinker_handlers import _register_gateway_tinker_routes
from .twinkle_handlers import _register_gateway_twinkle_routes

logger = get_logger()


class GatewayServer(LazyCleanupMixin):
    """Unified gateway server handling both Tinker and Twinkle API clients."""

    def __init__(self,
                 supported_models: list | None = None,
                 server_config: Any = None,
                 http_options: dict[str, Any] | None = None,
                 **kwargs) -> None:
        # ``server_config`` may be a typed ServerStateArgs (from the launcher), a
        # raw dict (direct callers), or None. Normalize to the keyword args
        # ``get_server_state`` expects, dropping unset (None) values so they fall
        # back to ServerState's own defaults.
        state_kwargs = self._normalize_server_state_args(server_config)
        self.state = get_server_state(**state_kwargs)
        self.route_prefix = kwargs.get('route_prefix', '/api/v1')
        self.http_options = http_options or {}
        self.training_models = self._normalize_models(kwargs.get('training_models', supported_models))
        self._training_model_names = frozenset(m.model_name for m in self.training_models)
        self.supported_models: list[types.SupportedModel] = []
        self.openai_model_aliases: dict[str, str] = {}
        self._template_initialized: set[str] = set()
        self.reconfigure({
            'supported_models': supported_models,
            'openai_model_aliases': kwargs.get('openai_model_aliases') or {},
        })
        self.proxy = ServiceProxy(http_options=http_options, route_prefix=self.route_prefix)
        # OpenAI-compat surface knobs: ``api_key`` gates every inference route with a Bearer check
        # (None = unauthenticated), ``owned_by`` is the ``owned_by`` field of ``/models`` entries.
        self.api_key = kwargs.get('api_key')
        self.owned_by = kwargs.get('owned_by') or 'twinkle'
        self._modelscope_config_lock = asyncio.Lock()
        self._state_cleanup_started = False

    def reconfigure(self, config: dict | None) -> None:
        """Apply the persisted discovery snapshot to every replica without replacing training/session state."""
        if not config:
            return
        models = self._normalize_models(config.get('supported_models', self.supported_models))
        aliases = config.get('openai_model_aliases', self.openai_model_aliases)
        conflicts = aliases.keys() & {model.model_name for model in models}
        if conflicts:
            raise ValueError(f'OpenAI aliases conflict with registered model names: {", ".join(sorted(conflicts))}')
        self.supported_models = models
        self.openai_model_aliases = dict(aliases)
        self._supported_model_names = frozenset(model.model_name for model in models if not model.base_model)
        self._template_initialized.intersection_update(self._supported_model_names)

    @property
    def supported_model_names(self) -> frozenset[str]:
        """Registered public names of base samplers."""
        return self._supported_model_names

    def get_openai_model_catalog(self) -> dict[str, Any]:
        """Read the configured model names for a deployment update via a Serve handle."""
        return {
            'models': [model.model_dump(exclude_none=True) for model in self.supported_models],
            'aliases': dict(self.openai_model_aliases),
        }

    @staticmethod
    def _normalize_server_state_args(server_config: Any) -> dict[str, Any]:
        """Normalize the gateway ``server_config`` into ``get_server_state`` kwargs.

        Accepts a typed ``ServerStateArgs`` (Pydantic), a raw dict, or ``None``.
        Unset (``None``) values are dropped so ``ServerState`` applies its own
        defaults rather than receiving an explicit ``None``.
        """
        if server_config is None:
            return {}
        if hasattr(server_config, 'model_dump'):
            data = server_config.model_dump()
        elif isinstance(server_config, dict):
            data = dict(server_config)
        else:
            data = {}
        return {k: v for k, v in data.items() if v is not None}

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
        if base_model not in self._training_model_names:
            raise HTTPException(
                status_code=400,
                detail=f"Base model '{base_model}' is not supported. "
                f"Supported models: {', '.join(sorted(self._training_model_names))}")

    async def _get_base_model(self, model_id: str) -> str:
        metadata = await self.state.get_model_metadata(model_id)
        if metadata and metadata.get('base_model'):
            return metadata['base_model']
        raise HTTPException(status_code=404, detail=f'Model {model_id} not found')


def build_gateway_app(deploy_options: dict[str, Any],
                      supported_models: list | None = None,
                      server_config: Any = None,
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

    # Build the FastAPI app + middleware stack + routes via the shared scaffold,
    # then bind the Ray Serve deployment. The Gateway passes its ``proxy.close()``
    # teardown via ``on_shutdown`` and sets ``attach_cleanup_middleware=True``
    # because it has no per-handler request hook, so the lazy-cleanup middleware
    # must cover every route (and stays innermost).
    def register_routes(app: FastAPI, get_self: Any) -> None:
        _register_gateway_tinker_routes(app, get_self)
        _register_gateway_twinkle_routes(app, get_self)
        _register_openai_routes(app, get_self)
        # Registered last so the api_key gate wraps everything installed before it.
        _register_api_key_gate(app, get_self)

    async def _on_shutdown(servable: Any) -> None:
        await servable.proxy.close()

    app = build_deployment_app(
        'Gateway',
        register_routes,
        on_shutdown=_on_shutdown,
        attach_cleanup_middleware=True,
    )

    return bind_deployment(
        app,
        GatewayServer,
        deploy_options,
        deployment_name='GatewayServer',
        bind_kwargs=dict(
            supported_models=supported_models,
            server_config=server_config,
            http_options=http_options,
            **kwargs,
        ),
    )
