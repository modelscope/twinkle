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
from twinkle.server.utils.state import ServerStateProxy, get_server_state
from twinkle.server.utils.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.utils.validation import get_token_from_request, verify_request_token
from twinkle.utils.logger import get_logger
from ..utils import wrap_builder_with_device_group_env
from .tinker_handlers import _register_tinker_sampler_routes
from .twinkle_handlers import _register_twinkle_sampler_routes

logger = get_logger()


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
                 sampler_type: str = 'vllm',
                 engine_args: dict[str, Any] | None = None,
                 queue_config: dict[str, Any] | None = None,
                 **kwargs):
        self.device_group = DeviceGroup(**device_group)
        twinkle.initialize(mode='ray', nproc_per_node=nproc_per_node, groups=[self.device_group], lazy_collect=False)
        if 'mesh_dim_names' in device_mesh:
            self.device_mesh = DeviceMesh(**device_mesh)
        else:
            self.device_mesh = DeviceMesh.from_sizes(**device_mesh)
        self.sampler_type = sampler_type
        replica_context = serve.get_replica_context()
        replica_id = replica_context.replica_id.unique_id

        # Initialize sampler based on type
        if sampler_type == 'vllm':
            from twinkle.sampler import vLLMSampler
            sampler_kwargs = engine_args or {}
            self.sampler = vLLMSampler(
                model_id=model_id,
                engine_args=sampler_kwargs,
                device_mesh=self.device_mesh,
                remote_group=self.device_group.name,
                instance_id=replica_id,
                **{
                    k: v
                    for k, v in kwargs.items() if k not in ['engine_args']
                })
        else:
            from twinkle.sampler import TorchSampler
            self.sampler = TorchSampler(
                model_id=model_id,
                device_mesh=self.device_mesh,
                instance_id=replica_id,
                remote_group=self.device_group.name,
                **kwargs)

        self.sampler.set_template('Template', model_id=model_id)
        self.state: ServerStateProxy = get_server_state()

        # Initialize task queue mixin
        self._init_task_queue(TaskQueueConfig.from_dict(queue_config))

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
                      sampler_type: str = 'vllm',
                      engine_args: dict[str, Any] | None = None,
                      queue_config: dict[str, Any] | None = None,
                      **kwargs):
    """Build a unified sampler application for text generation inference.

    Supports both Tinker (polling-style /tinker/asample) and
    Twinkle (synchronous /twinkle/*) sampler clients.

    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        nproc_per_node: Number of processes per node
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for parallelism
        deploy_options: Ray Serve deployment options
        sampler_type: Type of sampler to use ('vllm' or 'torch')
        engine_args: Additional engine arguments for the sampler
        queue_config: Task queue configuration dict (rps_limit, tps_limit, etc.)
        **kwargs: Additional arguments passed to the sampler

    Returns:
        Ray Serve deployment bound with configuration
    """
    # Build the FastAPI app and register all routes BEFORE serve.ingress so that
    # the frozen app contains the complete route table (visible to ProxyActor).
    app = FastAPI(
        title='Unified Sampler',
        description='REST API for distributed text generation inference (Tinker + Twinkle)',
        version='1.0.0')

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    def get_self() -> SamplerManagement:
        return serve.get_replica_context().servable_object

    # Register routes BEFORE @serve.ingress so Ray Serve captures them at decoration time
    _register_tinker_sampler_routes(app, get_self)
    _register_twinkle_sampler_routes(app, get_self)

    SamplerManagementWithIngress = serve.ingress(app)(SamplerManagement)
    DeploymentClass = serve.deployment(name='SamplerManagement')(SamplerManagementWithIngress)
    return DeploymentClass.options(**deploy_options).bind(model_id, nproc_per_node, device_group, device_mesh,
                                                          sampler_type, engine_args, queue_config, **kwargs)


build_sampler_app = wrap_builder_with_device_group_env(build_sampler_app)
