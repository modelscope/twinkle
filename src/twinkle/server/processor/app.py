# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Processor management application.

Provides a Ray Serve deployment for managing distributed processors
(datasets, dataloaders, preprocessors, rewards, templates, weight loaders, etc.).

Follows the same structural pattern as model/app.py:
- ProcessorManagement is a top-level class inheriting ProcessorManagerMixin
- Routes are registered in build_processor_app() via _register_processor_routes()
- serve.ingress(app)(ProcessorManagement) applied before deployment
- Sticky session routing via @serve.multiplexed keyed on session ID
"""
from __future__ import annotations

import os
from fastapi import FastAPI
from ray import serve
from typing import Any

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.server.deployment import LazyCleanupMixin, bind_deployment, build_deployment_app
from twinkle.server.state import ServerState, get_server_state
from twinkle.server.utils.lifecycle import ProcessorManagerMixin
from .twinkle_handlers import _register_processor_routes

logger = get_logger()


class ProcessorManagement(LazyCleanupMixin, ProcessorManagerMixin):
    """Processor management service.

    Manages lifecycle and invocation of distributed processor objects
    (datasets, dataloaders, rewards, templates, etc.).

    Lifecycle is handled by ProcessorManagerMixin:
    - Processors are registered with a session ID on creation.
    - A background thread expires processors whose session has timed out.
    - Per-user processor limit is enforced at registration.
    - Sticky session routing ensures session requests hit the same replica.
    """

    def __init__(self,
                 ncpu_proc_per_node: int,
                 device_group: dict[str, Any],
                 device_mesh: dict[str, Any],
                 nproc_per_node: int = 1,
                 processor_config: dict[str, Any] | None = None):
        self.device_group = DeviceGroup(**device_group)
        twinkle.initialize(
            mode='ray',
            nproc_per_node=nproc_per_node,
            groups=[self.device_group],
            lazy_collect=False,
            ncpu_proc_per_node=ncpu_proc_per_node)
        if 'mesh_dim_names' in device_mesh:
            self.device_mesh = DeviceMesh(**device_mesh)
        else:
            self.device_mesh = DeviceMesh.from_sizes(**device_mesh)

        # processor objects keyed by processor_id
        self.resource_dict: dict[str, Any] = {}
        self.state: ServerState = get_server_state()

        _cfg = processor_config or {}
        _env_limit = int(os.environ.get('TWINKLE_PER_USER_PROCESSOR_LIMIT', 20))
        self._init_processor_manager(
            processor_timeout=float(_cfg.get('processor_timeout', 1800.0)),
            per_token_processor_limit=int(_cfg.get('per_token_processor_limit', _env_limit)),
        )
        # Note: countdown task is started lazily in _ensure_sticky()

    @serve.multiplexed(max_num_models_per_replica=100)
    async def _sticky_entry(self, sticky_key: str):
        return sticky_key

    async def _ensure_sticky(self):
        sticky_key = serve.get_multiplexed_model_id()
        await self._sticky_entry(sticky_key)
        # Lazy-start countdown task on first request (requires running event loop)
        self._ensure_countdown_started()
        await self._ensure_state_cleanup_started()

    def _on_processor_expired(self, processor_id: str) -> None:
        """Called by the countdown thread when a processor's session expires."""
        self.resource_dict.pop(processor_id, None)
        self.unregister_resource(processor_id)


def build_processor_app(ncpu_proc_per_node: int,
                        device_group: dict[str, Any],
                        device_mesh: dict[str, Any],
                        deploy_options: dict[str, Any],
                        nproc_per_node: int = 1,
                        processor_config: dict[str, Any] | None = None,
                        **kwargs):
    """Build the processor management application.

    Follows the same pattern as build_model_app(): FastAPI app and routes are
    built here BEFORE serve.ingress so that the frozen app contains the full
    route table visible to ProxyActor.

    Args:
        ncpu_proc_per_node: Number of CPU processes per node.
        device_group: Device group configuration dict.
        device_mesh: Device mesh configuration dict.
        deploy_options: Ray Serve deployment options.
        nproc_per_node: Number of GPU processes per node (default 1).
        processor_config: Optional lifecycle configuration dict.
            Supported keys:
            - ``processor_timeout`` (float): Session inactivity timeout seconds. Default 1800.0.
            - ``per_token_processor_limit`` (int): Max processors per user.
              Overrides ``TWINKLE_PER_USER_PROCESSOR_LIMIT`` env var when provided.
        **kwargs: Additional arguments.

    Returns:
        Ray Serve deployment bound with configuration.
    """

    # Build the FastAPI app + middleware stack + routes via the shared scaffold,
    # then bind the Ray Serve deployment. No per-builder copy of the lifespan /
    # middleware construction remains here.
    def register_routes(app: FastAPI, get_self: Any) -> None:
        _register_processor_routes(app, get_self)

    app = build_deployment_app('Processor', register_routes)

    return bind_deployment(
        app,
        ProcessorManagement,
        deploy_options,
        deployment_name='ProcessorManagement',
        bind_args=(ncpu_proc_per_node, device_group, device_mesh, nproc_per_node, processor_config),
    )
