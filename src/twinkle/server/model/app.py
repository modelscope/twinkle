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
from typing import Any, Dict, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.server.utils.adapter_manager import AdapterManagerMixin
from twinkle.server.utils.state import ServerStateProxy, get_server_state
from twinkle.server.utils.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.utils.validation import get_token_from_request, verify_request_token
from twinkle.utils.logger import get_logger
from ..common.router import StickyLoraRequestRouter
from ..utils import wrap_builder_with_device_group_env
from .tinker_handlers import _register_tinker_routes
from .twinkle_handlers import _register_twinkle_routes

logger = get_logger()


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
                 use_megatron: bool = False,
                 adapter_config: dict[str, Any] = {},
                 queue_config: dict[str, Any] | None = None,
                 **kwargs):
        self.device_group = DeviceGroup(**device_group)
        twinkle.initialize(mode='ray', nproc_per_node=nproc_per_node, groups=[self.device_group], lazy_collect=False)
        if 'mesh_dim_names' in device_mesh:
            self.device_mesh = DeviceMesh(**device_mesh)
        else:
            self.device_mesh = DeviceMesh.from_sizes(**device_mesh)
        self.use_megatron = use_megatron
        self.replica_id = serve.get_replica_context().replica_id.unique_id
        self.max_loras = kwargs.get('max_loras', 5)
        self.base_model = model_id

        # Choose model backend
        if use_megatron:
            from ..model.backends.megatron_model import TwinkleCompatMegatronModel

            self.model = TwinkleCompatMegatronModel(
                model_id=model_id,
                device_mesh=self.device_mesh,
                remote_group=self.device_group.name,
                instance_id=self.replica_id,
                **kwargs)
        else:
            from ..model.backends.transformers_model import TwinkleCompatTransformersModel
            self.model = TwinkleCompatTransformersModel(
                model_id=model_id,
                device_mesh=self.device_mesh,
                remote_group=self.device_group.name,
                instance_id=self.replica_id,
                **kwargs)

        self.state: ServerStateProxy = get_server_state()
        self.state.register_replica(self.replica_id, self.max_loras)

        # Initialize mixins
        self._init_task_queue(TaskQueueConfig.from_dict(queue_config))
        self._init_adapter_manager(**adapter_config)
        self.start_adapter_countdown()

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

    def __del__(self):
        self.state.unregister_replica(self.replica_id)

    def _cleanup_adapter(self, adapter_name: str) -> None:
        if self.get_adapter_info(adapter_name):
            self.clear_adapter_state(adapter_name)
            self.model.remove_adapter(adapter_name)
            self.unregister_adapter(adapter_name)
            self.state.unload_model(adapter_name)

    def _on_adapter_expired(self, adapter_name: str) -> None:
        self.fail_pending_tasks_for_model(adapter_name, reason='Adapter expired')
        self._cleanup_adapter(adapter_name)


def build_model_app(model_id: str,
                    nproc_per_node: int,
                    device_group: dict[str, Any],
                    device_mesh: dict[str, Any],
                    deploy_options: dict[str, Any],
                    use_megatron: bool = False,
                    adapter_config: dict[str, Any] = {},
                    queue_config: dict[str, Any] | None = None,
                    **kwargs):
    """Build a unified model management application for distributed training.

    Supports both Tinker (polling-style) and Twinkle (synchronous) clients.

    Args:
        model_id: Base model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        nproc_per_node: Number of processes per node for distributed training
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for tensor parallelism
        deploy_options: Ray Serve deployment options
        use_megatron: Whether to use Megatron backend (vs Transformers)
        adapter_config: Adapter lifecycle config (timeout, per-token limits)
        queue_config: Task queue configuration (rate limiting, etc.)
        **kwargs: Additional model initialization arguments

    Returns:
        Configured Ray Serve deployment bound with parameters
    """
    # Build the FastAPI app and register all routes BEFORE serve.ingress so that
    # the frozen app contains the complete route table (visible to ProxyActor).
    app = FastAPI()

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    def get_self() -> ModelManagement:
        return serve.get_replica_context().servable_object

    _register_tinker_routes(app, get_self)
    _register_twinkle_routes(app, get_self)

    ModelManagementWithIngress = serve.ingress(app)(ModelManagement)
    DeploymentClass = serve.deployment(
        name='ModelManagement',
        request_router_config=RequestRouterConfig(request_router_class=StickyLoraRequestRouter),
    )(
        ModelManagementWithIngress)
    return DeploymentClass.options(**deploy_options).bind(model_id, nproc_per_node, device_group, device_mesh,
                                                          use_megatron, adapter_config, queue_config, **kwargs)


build_model_app = wrap_builder_with_device_group_env(build_model_app)
