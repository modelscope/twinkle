# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified sampler management application.

Builds a single Ray Serve deployment (SamplerManagement) that simultaneously handles
both Tinker (/tinker/asample) and Twinkle (/twinkle/*) sampler endpoints.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from ray import serve
from typing import Any, Dict, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.server.app_scaffold import LazyCleanupMixin, bind_deployment, build_deployment_app
from twinkle.server.state import ServerState, get_server_state
from twinkle.server.utils.backend_dispatch import BackendSelector
from twinkle.server.utils.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.utils.validation import get_token_from_request
from twinkle.utils.logger import get_logger
from ..utils import wrap_builder_with_device_group_env
from .tinker_handlers import _register_tinker_sampler_routes
from .twinkle_handlers import _register_twinkle_sampler_routes

logger = get_logger()


def _make_mock_sampler(kw: dict[str, Any]) -> Any:
    from .backends.mock_sampler import MockSampler

    # Forward ctor kwargs verbatim; MockSampler keeps model_id/seed/vocab_size
    # and logs any unknown keys at DEBUG (so a real-backend signature drift is
    # visible in the mock e2e instead of being silently stripped here).
    return MockSampler(**kw)


def _make_vllm_sampler(kw: dict[str, Any]) -> Any:
    from twinkle.sampler import vLLMSampler

    return vLLMSampler(**kw)


def _make_torch_sampler(kw: dict[str, Any]) -> Any:
    from twinkle.sampler import TorchSampler  # type: ignore[attr-defined]

    return TorchSampler(**kw)


# Single validate-then-dispatch selector for the sampler backend. Insertion
# order defines the reported permitted set; lazy imports stay local to the
# callbacks so a CPU-only host never pulls in vllm/torch.
SAMPLER_SELECTOR = BackendSelector(
    'sampler_type',
    {
        'mock': _make_mock_sampler,
        'vllm': _make_vllm_sampler,
        'torch': _make_torch_sampler,
    },
)


class SamplerManagement(LazyCleanupMixin, TaskQueueMixin):
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
                 queue_config: TaskQueueConfig | None = None,
                 **kwargs):
        # Validate ``sampler_type`` BEFORE any side effect.
        sampler_type = SAMPLER_SELECTOR.validate(sampler_type)
        # Skip twinkle.initialize for the mock backend — start without
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
        else:
            # Forward extra ctor kwargs verbatim to the mock backend so a real-
            # backend signature drift surfaces as a visible DEBUG log there
            # instead of being silently stripped at the dispatch boundary.
            sampler_kwargs.update(kwargs)
        self.sampler = SAMPLER_SELECTOR.construct(sampler_type, sampler_kwargs)

        self.state: ServerState = get_server_state()

        # Initialize task queue mixin
        self._init_task_queue(queue_config, deployment_name='Sampler')

    @serve.multiplexed(max_num_models_per_replica=5)
    async def _sticky_entry(self, sticky_key: str):
        return sticky_key

    async def _ensure_sticky(self):
        sticky_key = serve.get_multiplexed_model_id()
        await self._sticky_entry(sticky_key)

    async def _on_request_start(self, request: Request) -> str:
        await self._ensure_sticky()
        await self._ensure_state_cleanup_started()
        token = get_token_from_request(request)
        return token


def build_sampler_app(model_id: str,
                      nproc_per_node: int,
                      device_group: dict[str, Any],
                      device_mesh: dict[str, Any],
                      deploy_options: dict[str, Any],
                      sampler_type: str,
                      engine_args: dict[str, Any] | None = None,
                      queue_config: TaskQueueConfig | None = None,
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
    def register_routes(app: FastAPI, get_self: Any) -> None:
        _register_tinker_sampler_routes(app, get_self)
        _register_twinkle_sampler_routes(app, get_self)

    app = build_deployment_app(
        'Sampler',
        register_routes,
        fastapi_kwargs={
            'title': 'Unified Sampler',
            'description': 'REST API for distributed text generation inference (Tinker + Twinkle)',
            'version': '1.0.0',
        },
    )

    @app.middleware('http')
    async def inject_replica_id(request: Request, call_next):
        response = await call_next(request)
        try:
            ctx = serve.get_replica_context()
            response.headers['X-Twinkle-Replica-Id'] = ctx.replica_id.unique_id
        except Exception:
            pass
        return response

    return bind_deployment(
        app,
        SamplerManagement,
        deploy_options,
        deployment_name='SamplerManagement',
        bind_args=(model_id, nproc_per_node, device_group, device_mesh, sampler_type, engine_args, queue_config),
        bind_kwargs=kwargs,
    )


build_sampler_app = wrap_builder_with_device_group_env(build_sampler_app)
