# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified model management application.

Builds a single Ray Serve deployment (ModelManagement) that simultaneously handles
both Tinker (/tinker/*) and Twinkle (/twinkle/*) model endpoints.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from ray import serve
from ray.serve.config import RequestRouterConfig
from typing import Any, Dict, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.server.exceptions import ConfigError
from twinkle.server.utils.lifecycle import AdapterManagerMixin
from twinkle.server.telemetry.tracing import create_tracing_middleware
from twinkle.server.utils.metrics import create_metrics_middleware
from twinkle.server.state import ServerStateProxy, get_server_state
from twinkle.server.utils.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.utils.validation import get_token_from_request, verify_request_token
from twinkle.utils.logger import get_logger
from ..common.router import StickyLoraRequestRouter
from ..utils import wrap_builder_with_device_group_env
from .tinker_handlers import _register_tinker_routes
from .twinkle_handlers import _register_twinkle_routes

logger = get_logger()


_MODEL_BACKENDS: tuple[str, ...] = ('mock', 'transformers', 'megatron')


def _validate_model_backend(backend: Any) -> str:
    """Pure validation of the ``backend`` selector (R3.9).

    Raises :class:`ConfigError` (naming the field, value, and allowed set)
    when ``backend`` is missing, empty, non-string, or not exactly one of
    the permitted values. No imports or side effects.
    """
    if not isinstance(backend, str) or backend == '' or backend not in _MODEL_BACKENDS:
        raise ConfigError(field='backend', value=backend, allowed=list(_MODEL_BACKENDS))
    return backend


def _dispatch_model_backend(backend: str, ctor_kwargs: dict[str, Any]) -> Any:
    """Instantiate the model backend selected by an already-validated ``backend``."""
    if backend == 'mock':
        from .backends.mock_model import TwinkleCompatMockModel

        return TwinkleCompatMockModel(**ctor_kwargs)
    if backend == 'megatron':
        from .backends.megatron_model import TwinkleCompatMegatronModel

        return TwinkleCompatMegatronModel(**ctor_kwargs)
    from .backends.transformers_model import TwinkleCompatTransformersModel

    return TwinkleCompatTransformersModel(**ctor_kwargs)


class ModelManagement(TaskQueueMixin, AdapterManagerMixin):
    """Unified model management service.

    Handles:
    - Base model and multiple LoRA adapters (multi-user)
    - Tinker training operations via /tinker/* endpoints (async/polling)
    - Twinkle training operations via /twinkle/* endpoints (synchronous)
    - Adapter lifecycle via AdapterManagerMixin
    - Per-user rate limiting via TaskQueueMixin
    """

    def __init__(self,
                 model_id: str,
                 nproc_per_node: int,
                 device_group: dict[str, Any],
                 device_mesh: dict[str, Any],
                 backend: str,
                 adapter_config: dict[str, Any] | None = None,
                 queue_config: dict[str, Any] | None = None,
                 **kwargs):
        # R3.9: validate ``backend`` BEFORE any side effect (twinkle.initialize,
        # DeviceGroup construction, replica registration). An invalid value
        # never produces a partial backend nor reaches a ready state.
        backend = _validate_model_backend(backend)
        self.backend = backend
        # Skip twinkle.initialize for the mock backend (R3.7) — the largest
        # startup-time saving and the only way to start without CUDA/torch.
        if backend != 'mock':
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(
                mode='ray',
                nproc_per_node=nproc_per_node,
                groups=[self.device_group],
                lazy_collect=False,
            )
            if 'mesh_dim_names' in device_mesh:
                self.device_mesh = DeviceMesh(**device_mesh)
            else:
                self.device_mesh = DeviceMesh.from_sizes(**device_mesh)
        else:
            self.device_group = None
            self.device_mesh = None
        self.replica_id = serve.get_replica_context().replica_id.unique_id
        self.max_loras = kwargs.get('max_loras', 5)
        self.base_model = model_id

        ctor_kwargs: dict[str, Any] = {'model_id': model_id, **kwargs}
        if backend != 'mock':
            ctor_kwargs.update(
                device_mesh=self.device_mesh,
                remote_group=self.device_group.name,
                instance_id=self.replica_id,
            )
        self.model = _dispatch_model_backend(backend, ctor_kwargs)

        self.state: ServerStateProxy = get_server_state()
        self._replica_registered = False

        # Initialize mixins
        self._init_task_queue(TaskQueueConfig.from_dict(queue_config), deployment_name='Model')
        self._init_adapter_manager(**(adapter_config or {}))
        # Note: countdown task is started lazily in _ensure_sticky()

    async def _ensure_replica_registered(self):
        """Lazily register replica on first async request."""
        if not self._replica_registered:
            await self.state.register_replica(self.replica_id, self.max_loras)
            self._replica_registered = True

    @serve.multiplexed(max_num_models_per_replica=5)
    async def _sticky_entry(self, sticky_key: str):
        return sticky_key

    async def _ensure_sticky(self):
        sticky_key = serve.get_multiplexed_model_id()
        await self._sticky_entry(sticky_key)
        # Lazy-start countdown task on first request (requires running event loop)
        self._ensure_countdown_started()

    async def _on_request_start(self, request: Request) -> str:
        await self._ensure_sticky()
        await self._ensure_replica_registered()
        token = get_token_from_request(request)
        return token

    async def shutdown(self) -> None:
        """Explicit async cleanup — called via FastAPI shutdown event."""
        try:
            await self.state.unregister_replica(self.replica_id)
        except Exception:
            pass

    async def _cleanup_adapter(self, adapter_name: str) -> None:
        if self.get_resource_info(adapter_name):
            self.clear_resource_state(adapter_name)
            self.model.remove_adapter(adapter_name)
            self.unregister_resource(adapter_name)
            await self.state.unload_model(adapter_name)

    async def _on_adapter_expired(self, adapter_name: str) -> None:
        self.fail_pending_tasks_for_model(adapter_name, reason='Adapter expired')
        await self._cleanup_adapter(adapter_name)


def build_model_app(model_id: str,
                    nproc_per_node: int,
                    device_group: dict[str, Any],
                    device_mesh: dict[str, Any],
                    deploy_options: dict[str, Any],
                    backend: str,
                    adapter_config: dict[str, Any] | None = None,
                    queue_config: dict[str, Any] | None = None,
                    **kwargs):
    """Build a unified model management application for distributed training.

    Supports both Tinker (polling-style) and Twinkle (synchronous) clients.

    Args:
        model_id: Base model identifier (e.g., "Qwen/Qwen3.5-4B")
        nproc_per_node: Number of processes per node for distributed training
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for tensor parallelism
        deploy_options: Ray Serve deployment options
        backend: Model backend selector — ``mock`` | ``transformers`` | ``megatron``
            (R3.1-3.3, R3.9). Validated up front; bad values raise
            :class:`ConfigError` before any side effect.
        adapter_config: Adapter lifecycle config (timeout, per-token limits)
        queue_config: Task queue configuration (rate limiting, etc.)
        **kwargs: Additional model initialization arguments

    Returns:
        Configured Ray Serve deployment bound with parameters
    """
    # Fail fast on bad backend values at builder time (the launcher imports
    # this builder at startup, so the error surfaces before deployment).
    backend = _validate_model_backend(backend)

    # Build the FastAPI app and register all routes BEFORE serve.ingress so that
    # the frozen app contains the complete route table (visible to ProxyActor).

    def get_self() -> ModelManagement:
        return serve.get_replica_context().servable_object

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize telemetry in worker process (after deserialization)
        from twinkle.server.telemetry.worker_init import ensure_telemetry_initialized
        ensure_telemetry_initialized()
        # Start the ServerState cleanup loop now that we have a running loop;
        # idempotent across replicas in the same process.
        try:
            await get_self().state.start_cleanup_task()
        except Exception as e:
            logger.warning(f'Failed to start ServerState cleanup task: {e}')
        try:
            await get_self()._ensure_replica_registered()
        except Exception as e:
            logger.warning(f'Failed to register replica at startup: {e}')
        yield
        try:
            await get_self().shutdown()
        except Exception:
            pass

    app = FastAPI(lifespan=lifespan)

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    # Registration order: FastAPI runs middleware LIFO. Tracing first → metrics
    # last makes metrics the outermost wrapper, so its latency observation
    # covers the full request path including tracing overhead.
    app.middleware('http')(create_tracing_middleware('Model'))
    app.middleware('http')(create_metrics_middleware('Model'))

    _register_tinker_routes(app, get_self)
    _register_twinkle_routes(app, get_self)

    ModelManagementWithIngress = serve.ingress(app)(ModelManagement)
    DeploymentClass = serve.deployment(
        name='ModelManagement',
        request_router_config=RequestRouterConfig(request_router_class=StickyLoraRequestRouter),
    )(
        ModelManagementWithIngress)
    return DeploymentClass.options(**deploy_options).bind(
        model_id, nproc_per_node, device_group, device_mesh, backend,
        adapter_config, queue_config, **kwargs,
    )


build_model_app = wrap_builder_with_device_group_env(build_model_app)
