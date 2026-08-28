# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified model management application.

Builds a single Ray Serve deployment (ModelManagement) that simultaneously handles
both Tinker (/tinker/*) and Twinkle (/twinkle/*) model endpoints.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from ray import serve
from ray.serve.config import RequestRouterConfig
from typing import Any

from twinkle import DeviceGroup
from twinkle.server.common.router import StickyLoraRequestRouter
from twinkle.server.deployment import LazyCleanupMixin, bind_deployment, build_deployment_app, init_twinkle_runtime
from twinkle.server.exceptions import FullModeBusyError
from twinkle.server.state import ServerState, get_server_state
from twinkle.server.utils import wrap_builder_with_device_group_env
from twinkle.server.utils.backend_dispatch import BackendSelector
from twinkle.server.utils.lifecycle import AdapterManagerMixin
from twinkle.server.utils.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.utils.validation import get_token_from_request
from twinkle.utils.logger import get_logger
from .tinker_handlers import _register_tinker_routes
from .twinkle_handlers import _register_twinkle_routes

logger = get_logger()

# ``FullModeBusyError`` lives in ``twinkle.server.exceptions``; re-exported here
# for backwards compatibility with callers importing it from this module.
__all__ = ['FullModeBusyError', 'ModelManagement', 'build_model_app']

# Ctor kwargs consumed by the MultiLora wrappers' signatures but unknown to the
# plain (full-parameter) model classes, where **kwargs flows into HF
# ``from_pretrained``. Dropped when constructing a full-mode backend.
_LORA_ONLY_CTOR_KWARGS = ('max_loras', 'max_r', 'max_length', 'target_modules', 'lora_config')


def _make_mock_model(kw: dict[str, Any]) -> Any:
    from .backends.mock_model import TwinkleCompatMockModel

    kw.pop('train_mode', None)  # mock backend behaves the same for lora/full
    return TwinkleCompatMockModel(**kw)


def _make_transformers_model(kw: dict[str, Any]) -> Any:
    train_mode = kw.pop('train_mode', 'lora')
    if train_mode == 'full':
        from .backends.transformers_model import TwinkleCompatFullTransformersModel
        for key in _LORA_ONLY_CTOR_KWARGS:
            kw.pop(key, None)
        return TwinkleCompatFullTransformersModel(**kw)
    from .backends.transformers_model import TwinkleCompatTransformersModel
    return TwinkleCompatTransformersModel(**kw)


def _make_megatron_model(kw: dict[str, Any]) -> Any:
    train_mode = kw.pop('train_mode', 'lora')
    if train_mode == 'full':
        from .backends.megatron_model import TwinkleCompatFullMegatronModel
        for key in _LORA_ONLY_CTOR_KWARGS:
            kw.pop(key, None)
        return TwinkleCompatFullMegatronModel(**kw)
    from .backends.megatron_model import TwinkleCompatMegatronModel
    return TwinkleCompatMegatronModel(**kw)


# Single validate-then-dispatch selector for the model backend.
MODEL_SELECTOR = BackendSelector(
    'backend',
    {
        'mock': _make_mock_model,
        'transformers': _make_transformers_model,
        'megatron': _make_megatron_model,
    },
)


class ModelManagement(LazyCleanupMixin, TaskQueueMixin, AdapterManagerMixin):
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
                 queue_config: TaskQueueConfig | None = None,
                 data_plane_url: str | None = None,
                 **kwargs):
        self.backend = backend
        self.device_group = DeviceGroup(**device_group)
        self.device_mesh = init_twinkle_runtime(
            is_mock=(backend == 'mock'),
            nproc_per_node=nproc_per_node,
            device_group=self.device_group,
            device_mesh_dict=device_mesh,
        )
        self.replica_id = serve.get_replica_context().replica_id.unique_id
        self.max_loras = kwargs.get('max_loras', 5)
        self.base_model = model_id
        # ``train_mode`` selects LoRA multi-tenant (default) or exclusive
        # full-parameter training. Popped here so the underlying model
        # constructor never receives it; re-injected into ctor_kwargs so the
        # backend builder can pick the full vs LoRA wrapper class.
        self.train_mode = kwargs.pop('train_mode', 'lora')

        ctor_kwargs: dict[str, Any] = {
            'model_id': model_id,
            'remote_group': self.device_group.name,
            'instance_id': self.replica_id,
            'train_mode': self.train_mode,
            **kwargs,
        }
        if self.device_mesh is not None:
            ctor_kwargs['device_mesh'] = self.device_mesh
        self.model = MODEL_SELECTOR.construct(backend, ctor_kwargs)

        self.state: ServerState = get_server_state()
        from twinkle.server.data_plane import DataPlaneProxy
        self.data_plane = DataPlaneProxy(data_plane_url)
        self._replica_registered = False

        # Initialize mixins
        self._init_task_queue(queue_config, deployment_name='Model')
        self._init_adapter_manager(**(adapter_config or {}))
        # Note: countdown task is started lazily in _ensure_sticky()

    @property
    def data_world_size(self) -> int:
        """Effective data-parallel world size.

        Returns the real ``device_mesh.data_world_size`` for distributed
        backends; falls back to 1 on the ``mock`` backend where
        ``device_mesh`` is None.
        """
        return self.device_mesh.data_world_size if self.device_mesh is not None else 1

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
        await self._ensure_state_cleanup_started()
        token = get_token_from_request(request)
        return token

    async def shutdown(self) -> None:
        """Explicit async cleanup — called via FastAPI shutdown event."""
        try:
            await self.state.unregister_replica(self.replica_id)
        except Exception:
            pass
        await self.data_plane.close()

    def check_model_health(self) -> dict:
        """Probe model actors liveness via a lightweight ping.

        Returns a dict with 'healthy' (bool) and 'detail' (str).
        If the model actors are dead (e.g. OOM/SIGSEGV), the ping call
        will raise RayActorError, signalling the watchdog to restart.
        """
        try:
            result = self.model.ping()
            if result is True:
                return {'healthy': True, 'detail': 'model actors alive'}
            return {'healthy': False, 'detail': f'unexpected ping result: {result}'}
        except Exception as e:
            return {'healthy': False, 'detail': f'model actor unreachable: {e}'}

    async def _cleanup_adapter(self, adapter_name: str) -> None:
        if self.get_resource_info(adapter_name):
            self.clear_resource_state(adapter_name)
            if self.train_mode == 'full':
                # No PEFT adapter to remove; restore clean base weights so the
                # next tenant does not inherit this tenant's trained weights.
                self.model.reload_initial_weights()
            else:
                self.model.remove_adapter(adapter_name)
            self.unregister_resource(adapter_name)
            await self.state.unload_model(adapter_name)

    @property
    def is_full_mode(self) -> bool:
        """True when this deployment runs exclusive full-parameter training."""
        return self.train_mode == 'full'

    def resolve_model_adapter_name(self, adapter_name: str | None) -> str | None:
        """Map a per-tenant adapter name to the name the model expects.

        In full-parameter mode every request targets the empty-string default
        optimizer group; in LoRA mode the per-tenant adapter name is used as-is.
        """
        if self.is_full_mode:
            return ''
        return adapter_name

    def assert_full_mode_available(self, adapter_name: str | None = None) -> None:
        """In full mode, ensure no other tenant currently holds this deployment.

        Call before registering any state so a rejected request leaves nothing
        behind. ``adapter_name`` (when given) is excluded from the check so a
        tenant re-issuing against its own resource is not blocked.

        Raises:
            FullModeBusyError: another (non-expiring) tenant already holds it.
        """
        if not self.is_full_mode:
            return
        for rid, info in self._resource_records.items():
            if rid != adapter_name and not info.get('expiring'):
                raise FullModeBusyError(rid)

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
                    queue_config: TaskQueueConfig | None = None,
                    data_plane_url: str | None = None,
                    **kwargs):
    """Build a unified model management application for distributed training.

    Supports both Tinker (polling-style) and Twinkle (synchronous) clients.

    Args:
        model_id: Base model identifier (e.g., "Qwen/Qwen3.5-4B")
        nproc_per_node: Number of processes per node for distributed training
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for tensor parallelism
        deploy_options: Ray Serve deployment options
        backend: Model backend selector — ``mock`` | ``transformers`` | ``megatron``.
            Validated up front; bad values raise :class:`ConfigError` before
            any side effect.
        adapter_config: Adapter lifecycle config (timeout, per-token limits)
        queue_config: Task queue configuration (rate limiting, etc.)
        **kwargs: Additional model initialization arguments

    Returns:
        Configured Ray Serve deployment bound with parameters
    """
    # Fail fast on bad backend values at builder time (the launcher imports
    # this builder at startup, so the error surfaces before deployment).
    backend = MODEL_SELECTOR.validate(backend)

    # Build the FastAPI app + middleware stack + routes via the shared scaffold,
    # then bind the Ray Serve deployment. The Model passes its ``shutdown()``
    # teardown via ``on_shutdown`` and its sticky-LoRA router via
    # ``request_router_config``.
    def register_routes(app: FastAPI, get_self: Any) -> None:
        _register_tinker_routes(app, get_self)
        _register_twinkle_routes(app, get_self)

    async def _on_shutdown(servable: Any) -> None:
        await servable.shutdown()

    app = build_deployment_app('Model', register_routes, on_shutdown=_on_shutdown, attach_replica_id_header=True)

    return bind_deployment(
        app,
        ModelManagement,
        deploy_options,
        deployment_name='ModelManagement',
        request_router_config=RequestRouterConfig(request_router_class=StickyLoraRequestRouter),
        bind_args=(model_id, nproc_per_node, device_group, device_mesh, backend, adapter_config, queue_config,
                   data_plane_url),
        bind_kwargs=kwargs,
    )


build_model_app = wrap_builder_with_device_group_env(build_model_app)
