# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified sampler management application.

Builds a single Ray Serve deployment (SamplerManagement) that simultaneously handles
both Tinker (/tinker/asample) and Twinkle (/twinkle/*) sampler endpoints.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from ray import serve
from typing import Any, Dict, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.server.exceptions import ConfigError
from twinkle.server.state import ServerState, get_server_state
from twinkle.server.telemetry.tracing import create_tracing_middleware
from twinkle.server.utils.metrics import create_metrics_middleware
from twinkle.server.utils.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.utils.validation import get_token_from_request, verify_request_token
from twinkle.utils.logger import get_logger
from ..utils import wrap_builder_with_device_group_env
from .tinker_handlers import _register_tinker_sampler_routes
from .twinkle_handlers import _register_twinkle_sampler_routes

logger = get_logger()

_SAMPLER_TYPES: tuple[str, ...] = ('mock', 'vllm', 'torch')


def _validate_sampler_type(sampler_type: Any) -> str:
    """Pure validation of the ``sampler_type`` selector (R3.10).

    Raises :class:`ConfigError` (naming the field, value, and allowed set)
    when the value is missing, empty, non-string, or not exactly one of the
    permitted values. No imports or side effects.
    """
    if (not isinstance(sampler_type, str) or sampler_type == '' or sampler_type not in _SAMPLER_TYPES):
        raise ConfigError(field='sampler_type', value=sampler_type, allowed=list(_SAMPLER_TYPES))
    return sampler_type


def _dispatch_sampler_backend(sampler_type: str, ctor_kwargs: dict[str, Any]) -> Any:
    """Instantiate the sampler selected by an already-validated ``sampler_type``."""
    if sampler_type == 'mock':
        from .backends.mock_sampler import MockSampler

        # MockSampler accepts only model_id/seed/vocab_size — strip extras silently.
        return MockSampler(
            model_id=ctor_kwargs.get('model_id'),
            seed=ctor_kwargs.get('seed', 0),
            vocab_size=ctor_kwargs.get('vocab_size', 32),
        )
    if sampler_type == 'torch':
        from twinkle.sampler import TorchSampler  # type: ignore[attr-defined]

        return TorchSampler(**ctor_kwargs)
    from twinkle.sampler import vLLMSampler

    return vLLMSampler(**ctor_kwargs)


class SamplerManagement(TaskQueueMixin):
    """Unified sampler management service.

    Manages:
    - vLLM or Torch sampler initialization and lifecycle
    - Tinker inference requests (/tinker/asample) with rate limiting via TaskQueueMixin
    - Twinkle inference requests (/twinkle/*) calling sampler directly
    - Template configuration for trajectory encoding
    """

    def __init__(self,
                 model_id: str,
                 nproc_per_node: int,
                 device_group: dict[str, Any],
                 device_mesh: dict[str, Any],
                 sampler_type: str,
                 engine_args: dict[str, Any] | None = None,
                 queue_config: dict[str, Any] | None = None,
                 **kwargs):
        # R3.10: validate ``sampler_type`` BEFORE any side effect.
        sampler_type = _validate_sampler_type(sampler_type)
        # Skip twinkle.initialize for the mock backend (R3.8) — start without
        # CUDA/torch/vllm.
        if sampler_type != 'mock':
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
        self.sampler_type = sampler_type
        self.model_id = model_id
        replica_context = serve.get_replica_context()
        replica_id = replica_context.replica_id.unique_id

        sampler_kwargs: dict[str, Any] = {'model_id': model_id}
        if sampler_type != 'mock':
            sampler_kwargs.update(
                engine_args=engine_args or {},
                device_mesh=self.device_mesh,
                remote_group=self.device_group.name,
                instance_id=replica_id,
                **{
                    k: v
                    for k, v in kwargs.items() if k not in ('engine_args', )
                },
            )
        self.sampler = _dispatch_sampler_backend(sampler_type, sampler_kwargs)

        self.state: ServerState = get_server_state()

        # Initialize task queue mixin
        self._init_task_queue(TaskQueueConfig.from_dict(queue_config), deployment_name='Sampler')

    @serve.multiplexed(max_num_models_per_replica=5)
    async def _sticky_entry(self, sticky_key: str):
        return sticky_key

    async def _ensure_sticky(self):
        sticky_key = serve.get_multiplexed_model_id()
        await self._sticky_entry(sticky_key)

    async def _on_request_start(self, request: Request) -> str:
        await self._ensure_sticky()
        token = get_token_from_request(request)
        return token


def build_sampler_app(model_id: str,
                      nproc_per_node: int,
                      device_group: dict[str, Any],
                      device_mesh: dict[str, Any],
                      deploy_options: dict[str, Any],
                      sampler_type: str,
                      engine_args: dict[str, Any] | None = None,
                      queue_config: dict[str, Any] | None = None,
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
        sampler_type: Sampler selector — ``mock`` | ``vllm`` | ``torch`` (R3.4-3.6,
            R3.10). Validated up front; bad values raise :class:`ConfigError`
            before any side effect.
        engine_args: Additional engine arguments for the sampler
        queue_config: Task queue configuration dict (rps_limit, tps_limit, etc.)
        **kwargs: Additional arguments passed to the sampler

    Returns:
        Ray Serve deployment bound with configuration
    """
    # Fail fast at builder time on bad sampler_type values.
    sampler_type = _validate_sampler_type(sampler_type)

    # Build the FastAPI app and register all routes BEFORE serve.ingress so that
    # the frozen app contains the complete route table (visible to ProxyActor).

    def get_self() -> SamplerManagement:
        return serve.get_replica_context().servable_object

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize telemetry in worker process (after deserialization)
        from twinkle.server.telemetry.worker_init import ensure_telemetry_initialized
        ensure_telemetry_initialized()
        # Start the ServerState cleanup loop now that we have a running loop.
        try:
            await get_self().state.start_cleanup_task()
        except Exception as e:
            logger.warning(f'Failed to start ServerState cleanup task: {e}')
        yield

    app = FastAPI(
        title='Unified Sampler',
        description='REST API for distributed text generation inference (Tinker + Twinkle)',
        version='1.0.0',
        lifespan=lifespan)

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    # Registration order: FastAPI runs middleware LIFO. Tracing first → metrics
    # last makes metrics the outermost wrapper, so its latency observation
    # covers the full request path including tracing overhead.
    app.middleware('http')(create_tracing_middleware('Sampler'))
    app.middleware('http')(create_metrics_middleware('Sampler'))

    # Register routes BEFORE @serve.ingress so Ray Serve captures them at decoration time
    _register_tinker_sampler_routes(app, get_self)
    _register_twinkle_sampler_routes(app, get_self)

    SamplerManagementWithIngress = serve.ingress(app)(SamplerManagement)
    DeploymentClass = serve.deployment(name='SamplerManagement')(SamplerManagementWithIngress)
    return DeploymentClass.options(**deploy_options).bind(model_id, nproc_per_node, device_group, device_mesh,
                                                          sampler_type, engine_args, queue_config, **kwargs)


build_sampler_app = wrap_builder_with_device_group_env(build_sampler_app)
