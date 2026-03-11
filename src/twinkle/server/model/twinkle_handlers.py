# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native model handler mixin.

All endpoints are prefixed /twinkle/... and use schedule_task_and_wait() returning
results directly (synchronous from the client's perspective).
self_fn is injected via FastAPI Depends to obtain the ModelManagement instance at request time.
"""
from __future__ import annotations

import traceback
from fastapi import Depends, FastAPI, Request
from peft import LoraConfig
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .app import ModelManagement

from twinkle.data_format import InputFeature, Trajectory
from twinkle.server.common.checkpoint_factory import create_checkpoint_manager, create_training_run_manager
from twinkle.server.common.serialize import deserialize_object
from twinkle.utils.logger import get_logger
from twinkle_client.types.model import (AdapterRequest, AddAdapterRequest, CalculateMetricRequest, CreateRequest,
                                        ForwardOnlyRequest, ForwardRequest, GetStateDictRequest, HeartbeatRequest,
                                        LoadRequest, SaveRequest, SetLossRequest, SetLrSchedulerRequest,
                                        SetOptimizerRequest, SetProcessorRequest, SetTemplateRequest,
                                        UploadToHubRequest)

logger = get_logger()


def _parse_inputs(inputs: Any):
    """Convert raw dict/list inputs to InputFeature or Trajectory objects."""
    if isinstance(inputs, list) and inputs:
        first = inputs[0]
        if isinstance(first, dict) and 'input_ids' in first:
            return [InputFeature(**item) for item in inputs]
        else:
            return [Trajectory(**item) for item in inputs]
    elif isinstance(inputs, dict):
        if 'input_ids' in inputs:
            return [InputFeature(**inputs)]
        else:
            return [Trajectory(**inputs)]
    return inputs


def _get_twinkle_adapter_name(request: Request, adapter_name: str | None) -> str | None:
    """Build the per-request adapter name from the request_id prefix."""
    if adapter_name is None or adapter_name == '':
        return None
    return request.state.request_id + '-' + adapter_name


def _register_twinkle_routes(app: FastAPI, self_fn: Callable[[], ModelManagement]) -> None:
    """Register all /twinkle/* routes on the given FastAPI app.

    self_fn is a zero-argument callable that returns the current ModelManagement
    replica instance. It is wired in via Depends so it is resolved lazily at request time.
    """

    @app.post('/twinkle/create')
    async def create(request: Request, body: CreateRequest, self: ModelManagement = Depends(self_fn)):
        return {'status': 'ok'}

    @app.post('/twinkle/forward')
    async def forward(request: Request, body: ForwardRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            inputs = _parse_inputs(body.inputs)
            ret = self.model.forward(inputs=inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='forward')

    @app.post('/twinkle/forward_only')
    async def forward_only(request: Request, body: ForwardOnlyRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            inputs = _parse_inputs(body.inputs)
            ret = self.model.forward_only(inputs=inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='forward_only')

    @app.post('/twinkle/calculate_loss')
    async def calculate_loss(request: Request, body: AdapterRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.calculate_loss(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='calculate_loss')

    @app.post('/twinkle/backward')
    async def backward(request: Request, body: AdapterRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.backward(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='backward')

    @app.post('/twinkle/forward_backward')
    async def forward_backward(request: Request, body: ForwardRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            inputs = _parse_inputs(body.inputs)
            ret = self.model.twinkle_forward_backward(inputs=inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='forward_backward')

    @app.post('/twinkle/clip_grad_norm')
    async def clip_grad_norm(request: Request, body: AdapterRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.clip_grad_norm(adapter_name=adapter_name, **extra_kwargs)
            return {'result': str(ret)}

        return await self.schedule_task_and_wait(_task, task_type='clip_grad_norm')

    @app.post('/twinkle/step')
    async def step(request: Request, body: AdapterRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.step(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='step')

    @app.post('/twinkle/zero_grad')
    async def zero_grad(request: Request, body: AdapterRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.zero_grad(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='zero_grad')

    @app.post('/twinkle/lr_step')
    async def lr_step(request: Request, body: AdapterRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.lr_step(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='lr_step')

    @app.post('/twinkle/get_train_configs')
    async def get_train_configs(request: Request, body: AdapterRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.get_train_configs(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='get_train_configs')

    @app.post('/twinkle/set_loss')
    async def set_loss(request: Request, body: SetLossRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_loss(body.loss_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='set_loss')

    @app.post('/twinkle/set_optimizer')
    async def set_optimizer(request: Request, body: SetOptimizerRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_optimizer(body.optimizer_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='set_optimizer')

    @app.post('/twinkle/set_lr_scheduler')
    async def set_lr_scheduler(request: Request, body: SetLrSchedulerRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_lr_scheduler(body.scheduler_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='set_lr_scheduler')

    @app.post('/twinkle/save')
    async def save(request: Request, body: SaveRequest, self: ModelManagement = Depends(self_fn)):
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            checkpoint_name = checkpoint_manager.get_ckpt_name(body.name)
            save_dir = checkpoint_manager.get_save_dir(model_id=adapter_name, is_sampler=False)
            checkpoint_dir = self.model.save(
                name=checkpoint_name,
                output_dir=save_dir,
                adapter_name=adapter_name,
                save_optimizer=body.save_optimizer,
                **extra_kwargs)
            twinkle_path = checkpoint_manager.save(model_id=adapter_name, name=checkpoint_name, is_sampler=False)
            return {'result': twinkle_path, 'checkpoint_dir': checkpoint_dir}

        return await self.schedule_task_and_wait(_task, task_type='save')

    @app.post('/twinkle/load')
    async def load(request: Request, body: LoadRequest, self: ModelManagement = Depends(self_fn)):
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            resolved = checkpoint_manager.resolve_load_path(body.name)
            ret = self.model.load(
                name=resolved.checkpoint_name,
                output_dir=resolved.checkpoint_dir,
                adapter_name=adapter_name,
                load_optimizer=body.load_optimizer,
                token=token,
                **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='load')

    @app.post('/twinkle/upload_to_hub')
    async def upload_to_hub(request: Request, body: UploadToHubRequest, self: ModelManagement = Depends(self_fn)):
        token = await self._on_request_start(request)

        async def _task():
            if body.checkpoint_dir.startswith('twinkle://'):
                checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
                parsed = checkpoint_manager.parse_twinkle_path(body.checkpoint_dir)
                if not parsed:
                    raise ValueError(f'Invalid twinkle path format: {body.checkpoint_dir}')
                checkpoint_id = parsed.checkpoint_id
                model_id_to_load = parsed.training_run_id
                checkpoint = checkpoint_manager.get(model_id_to_load, checkpoint_id)
                if not checkpoint:
                    raise ValueError(f'Checkpoint not found or access denied: {body.checkpoint_dir}')
                checkpoint_dir = str(
                    checkpoint_manager.get_ckpt_dir(model_id=model_id_to_load, checkpoint_id=checkpoint_id))
            else:
                checkpoint_dir = body.checkpoint_dir
            self.model.upload_to_hub(
                checkpoint_dir=checkpoint_dir,
                hub_model_id=body.hub_model_id,
                hub_token=body.hub_token or token,
                async_upload=body.async_upload)
            return {'result': body.hub_model_id}

        return await self.schedule_task_and_wait(_task, task_type='upload_to_hub')

    @app.post('/twinkle/add_adapter_to_model')
    async def add_adapter_to_model(request: Request, body: AddAdapterRequest, self: ModelManagement = Depends(self_fn)):
        assert body.adapter_name, 'You need to specify a valid `adapter_name`'
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            config = deserialize_object(body.config)
            extra_kwargs = body.model_extra or {}
            training_run_manager = create_training_run_manager(token, client_type='twinkle')
            self.register_adapter(adapter_name, token)
            self.model.add_adapter_to_model(adapter_name, config, **extra_kwargs)
            from twinkle_client.types.training import CreateModelRequest
            from twinkle_client.types.training import LoraConfig as IoLoraConfig
            lora_config = None
            if isinstance(config, LoraConfig):
                lora_config = IoLoraConfig(rank=config.r, train_unembed=False, train_mlp=True, train_attn=True)
            run_config = CreateModelRequest(
                base_model=self.base_model, lora_config=lora_config, user_metadata={'adapter_name': body.adapter_name})
            training_run_manager.save(adapter_name, run_config)
            return {'status': 'ok', 'adapter_name': adapter_name}

        return await self.schedule_task_and_wait(_task, task_type='add_adapter_to_model')

    @app.post('/twinkle/set_template')
    async def set_template(request: Request, body: SetTemplateRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_template(body.template_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='set_template')

    @app.post('/twinkle/set_processor')
    async def set_processor(request: Request, body: SetProcessorRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_processor(body.processor_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='set_processor')

    @app.post('/twinkle/heartbeat')
    async def heartbeat(request: Request, body: HeartbeatRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)
        self.assert_adapter_exists(adapter_name=adapter_name)
        self.touch_adapter(adapter_name)
        return {'status': 'ok'}

    @app.post('/twinkle/calculate_metric')
    async def calculate_metric(request: Request, body: CalculateMetricRequest,
                               self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.calculate_metric(is_training=body.is_training, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='calculate_metric')

    @app.post('/twinkle/get_state_dict')
    async def get_state_dict(request: Request, body: GetStateDictRequest, self: ModelManagement = Depends(self_fn)):
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.get_state_dict(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await self.schedule_task_and_wait(_task, task_type='get_state_dict')
