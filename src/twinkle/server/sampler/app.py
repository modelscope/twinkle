# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified sampler management application.

Builds a single Ray Serve deployment (SamplerManagement) that simultaneously handles
both Tinker (/tinker/asample) and Twinkle (/twinkle/*) sampler endpoints.
"""
from __future__ import annotations

import asyncio
import os
from fastapi import FastAPI, Request
from ray import serve
from typing import Any

from twinkle import DeviceGroup
from twinkle.server.config.backend_dispatch import BackendSelector
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.deployment import LazyCleanupMixin, bind_deployment, build_deployment_app
from twinkle.server.exceptions import RequestRejectedError
from twinkle.server.middleware.auth import get_token_from_request
from twinkle.server.runtime import init_twinkle_runtime
from twinkle.server.state import ServerState, get_server_state
from twinkle.server.state.models import RuntimeContext
from twinkle.server.task_queue.mixin import TaskQueueMixin
from twinkle.server.utils import get_template_for_model, wrap_builder_with_device_group_env
from twinkle.utils.logger import get_logger
from .concurrency import SamplerConcurrency
from .runtime import SamplerRuntime
from .tinker_handlers import _register_tinker_sampler_routes
from .twinkle_handlers import _register_twinkle_sampler_routes

logger = get_logger()


def _make_mock_sampler(kw: dict[str, Any]) -> Any:
    from .backends.mock_sampler import MockSampler

    return MockSampler(**kw)


def _make_vllm_sampler(kw: dict[str, Any]) -> Any:
    from twinkle.sampler import vLLMSampler

    return vLLMSampler(**kw)


def _make_vllm_async_sampler(kw: dict[str, Any]) -> Any:
    """Construct the vLLM backend with non-blocking generation admission."""
    from .backends.vllm_async import ServerVLLMSampler

    return ServerVLLMSampler(**kw, context_manager=None)


# Single validate-then-dispatch selector for the sampler backend.
SAMPLER_SELECTOR = BackendSelector(
    'sampler_type',
    {
        'mock': _make_mock_sampler,
        'vllm': _make_vllm_sampler,
        'vllm_async': _make_vllm_async_sampler,
    },
)


def _construct_sampler_backend(
    sampler_type: str,
    sampler_kwargs: dict[str, Any],
    data_plane_url: str | None,
) -> Any:
    # Backend selection is explicit. DataPlane config controls where results
    # are stored, not which sampler implementation is instantiated.
    return SAMPLER_SELECTOR.construct(sampler_type, sampler_kwargs)


class SamplerManagement(LazyCleanupMixin, TaskQueueMixin):
    """Unified sampler management service.

    Manages:
    - mock or vLLM sampler initialization and lifecycle
    - Tinker inference requests (/tinker/asample) with rate limiting via TaskQueueMixin
    - Concurrent Twinkle inference requests (/twinkle/*) with future polling
    - Template configuration for trajectory encoding
    """

    async def __init__(self,
                       model_id: str,
                       nproc_per_node: int,
                       device_group: dict[str, Any],
                       device_mesh: dict[str, Any],
                       sampler_type: str,
                       engine_args: dict[str, Any] | None = None,
                       queue_config: TaskQueueConfig | None = None,
                       data_plane_url: str | None = None,
                       **kwargs):
        self.request_runtime = SamplerRuntime(self)
        self.device_group = DeviceGroup(**device_group)
        context = kwargs.pop('sampling_context', None)
        if context is not None:
            context = RuntimeContext.model_validate(context)
            from twinkle.infra._ray.ray_helper import RayHelper
            from twinkle.infra._ray.resource_manager import ResourceManager
            os.environ['TWINKLE_RAY_NAMESPACE'] = context.namespace
            os.environ['TWINKLE_RAY_REGISTRY_NAME'] = context.registry_name
            RayHelper.resource_manager = ResourceManager.from_allocation(self.device_group,
                                                                         kwargs.pop('placement_group'),
                                                                         kwargs.pop('allocation_devices'))
        self.device_mesh = init_twinkle_runtime(
            is_mock=(sampler_type == 'mock'),
            nproc_per_node=nproc_per_node,
            device_group=self.device_group,
            device_mesh_dict=device_mesh,
        )
        self.sampler_type = sampler_type
        self.model_id = model_id
        if context is not None:
            self.replica_id = context.instance_id
        else:
            self.replica_id = serve.get_replica_context().replica_id.unique_id

        sampler_kwargs: dict[str, Any] = {
            'model_id': model_id,
            'remote_group': self.device_group.name,
            'instance_id': self.replica_id,
        }
        if sampler_type != 'mock':
            sampler_kwargs.update(
                engine_args=engine_args or {},
                device_mesh=self.device_mesh,
                **{
                    k: v
                    for k, v in kwargs.items() if k not in ('engine_args', )
                },
            )
        else:
            sampler_kwargs.update(kwargs)
        self.sampler = _construct_sampler_backend(sampler_type, sampler_kwargs, data_plane_url)
        actors = getattr(self.sampler, '_actors', ())
        await asyncio.gather(*(actor.__ray_ready__.remote() for actor in actors))

        self.state: ServerState = get_server_state()
        from twinkle.server.data_plane import DataPlaneProxy
        self.data_plane = DataPlaneProxy(data_plane_url)

        actors = getattr(self.sampler, '_actors', None)
        self._init_task_queue(
            queue_config,
            deployment_name='Sampler',
            collect_width=len(actors) if actors else 1,
        )
        # ComputeWorker bounds the client future. Keep collecting the actual
        # backend result so a timed-out request cannot release live weights.
        self.sampler._ray_get_timeout = None
        self.sampling_guard = SamplerConcurrency()
        if sampler_type != 'mock':
            await self.call_backend(self.sampler.set_template, get_template_for_model(model_id), model_id=model_id)

    async def shutdown(self) -> None:
        self.sampling_guard.close()
        try:
            cancel_all = getattr(self.sampler, 'cancel_all_generations', None)
            if callable(cancel_all):
                await self.call_backend(cancel_all, admit=False)
        finally:
            try:
                await self.state.unregister_replica(self.replica_id)
            finally:
                await self.shutdown_task_queue()
                await self.sampling_guard.wait_closed()
                await self.request_runtime.shutdown()
                await self.data_plane.close()

    async def invoke(self, ticket, payload, stream_queue=None):
        return await self.request_runtime.invoke(ticket, payload, stream_queue)

    async def wait_operation(self, ticket):
        await self.request_runtime.wait_operation(ticket)

    async def install_base_weights(self, path):
        await self.request_runtime.install_base_weights(path)

    async def pending_futures(self):
        return await self.request_runtime.pending_futures()

    async def runtime_identity(self):
        return await self.request_runtime.runtime_identity()

    async def check_health(self) -> None:
        self.sampling_guard.check_available()

    async def _on_request_start(self, request: Request) -> str:
        await self.state.touch_replica_last_seen(self.replica_id)
        await self._ensure_state_cleanup_started()
        return get_token_from_request(request)


def build_sampler_app(model_id: str,
                      nproc_per_node: int,
                      device_group: dict[str, Any],
                      device_mesh: dict[str, Any],
                      deploy_options: dict[str, Any],
                      sampler_type: str,
                      engine_args: dict[str, Any] | None = None,
                      queue_config: TaskQueueConfig | None = None,
                      data_plane_url: str | None = None,
                      **kwargs):
    """Build a unified sampler application for text generation inference.

    Supports both Tinker (polling-style /tinker/asample) and
    Twinkle (synchronous /twinkle/*) sampler clients.

    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen3.5-4B")
        nproc_per_node: Number of processes per node
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for parallelism
        deploy_options: Ray Serve deployment options
        sampler_type: Sampler selector — ``mock`` | ``vllm`` | ``vllm_async``.
            Validated up front; bad values raise :class:`ConfigError` before
            any side effect.
        engine_args: Additional engine arguments for the sampler
        queue_config: Validated :class:`TaskQueueConfig` (rps_limit, tps_limit, etc.)
        **kwargs: Additional arguments passed to the sampler

    Returns:
        Ray Serve deployment bound with configuration
    """
    # Fail fast at builder time on bad sampler_type values.
    sampler_type = SAMPLER_SELECTOR.validate(sampler_type)

    # Build the FastAPI app + middleware stack + routes via the shared scaffold,
    # then bind the Ray Serve deployment. The Sampler passes its FastAPI
    # title/description/version through ``fastapi_kwargs``.
    managed = kwargs.get('sampling_context') is not None

    def register_routes(app: FastAPI, get_self: Any) -> None:
        if managed:

            def managed_self():
                # Internal compute URLs must not bypass the controller's tickets.
                raise RequestRejectedError('Sampling requests must use the stable sampler endpoint')

            dependency = managed_self
        else:
            dependency = get_self
        _register_tinker_sampler_routes(app, dependency)
        _register_twinkle_sampler_routes(app, dependency)

    app = build_deployment_app(
        'Sampler',
        register_routes,
        fastapi_kwargs={
            'title': 'Unified Sampler',
            'description': 'REST API for distributed text generation inference (Tinker + Twinkle)',
            'version': '1.0.0',
        },
        attach_replica_id_header=True,
        on_shutdown=lambda servable: servable.shutdown(),
    )

    return bind_deployment(
        app,
        SamplerManagement,
        deploy_options,
        deployment_name='SamplerManagement',
        bind_args=(model_id, nproc_per_node, device_group, device_mesh, sampler_type, engine_args, queue_config,
                   data_plane_url),
        bind_kwargs=kwargs,
    )


build_sampler_app = wrap_builder_with_device_group_env(build_sampler_app)
