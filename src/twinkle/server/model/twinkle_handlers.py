# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native model handler mixin.

All endpoints are prefixed /twinkle/... and use schedule_task_and_wait() returning
results directly (synchronous from the client's perspective).
self_fn is injected via FastAPI Depends to obtain the ModelManagement instance at request time.
"""
from __future__ import annotations

import asyncio
import torch
import traceback
from fastapi import Depends, FastAPI, HTTPException, Request
from pathlib import Path
from peft import LoraConfig
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .app import ModelManagement

import twinkle_client.types as types
from twinkle.data_format import InputFeature, Trajectory
from twinkle.server.common.checkpoint_factory import create_checkpoint_manager, create_training_run_manager
from twinkle.server.utils.validation import get_session_id_from_request
from twinkle.utils.logger import get_logger
from twinkle_client.common.serialize import deserialize_object

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

    async def run_task(coro):
        """Await a schedule_task_and_wait coroutine and surface any exception as a
        structured HTTP 500 response so the client receives the full traceback instead
        of an opaque connection-level error.

        Note: HTTPException is re-raised directly to preserve its status code and detail.
        """
        try:
            return await coro
        except HTTPException:
            raise  # Re-raise HTTPException directly to preserve status code
        except Exception:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post('/twinkle/create', response_model=types.CreateResponse)
    async def create(request: Request, body: types.CreateRequest,
                     self: ModelManagement = Depends(self_fn)) -> types.CreateResponse:
        await self._on_request_start(request)
        return types.CreateResponse()

    @app.post('/twinkle/forward', response_model=types.ForwardResponse)
    async def forward(request: Request, body: types.ForwardRequest,
                      self: ModelManagement = Depends(self_fn)) -> types.ForwardResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            inputs = _parse_inputs(body.inputs)
            ret = self.model.forward(inputs=inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        inputs_list = body.inputs if isinstance(body.inputs, list) else [body.inputs]
        input_tokens = sum(len(inp.get('input_ids', [])) if isinstance(inp, dict) else 0 for inp in inputs_list)
        batch_size = len(inputs_list)
        return await run_task(
            self.schedule_task_and_wait(
                _task,
                model_id=adapter_name,
                token=token,
                input_tokens=input_tokens,
                batch_size=batch_size,
                data_world_size=self.device_mesh.data_world_size,
                task_type='forward',
            ))

    @app.post('/twinkle/forward_only', response_model=types.ForwardResponse)
    async def forward_only(
            request: Request,
            body: types.ForwardOnlyRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.ForwardResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            inputs = _parse_inputs(body.inputs)
            ret = self.model.forward_only(inputs=inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        inputs_list = body.inputs if isinstance(body.inputs, list) else [body.inputs]
        input_tokens = sum(len(inp.get('input_ids', [])) if isinstance(inp, dict) else 0 for inp in inputs_list)
        return await run_task(
            self.schedule_task_and_wait(
                _task,
                model_id=adapter_name,
                token=token,
                input_tokens=input_tokens,
                task_type='forward_only',
            ))

    @app.post('/twinkle/calculate_loss', response_model=types.CalculateLossResponse)
    async def calculate_loss(
            request: Request,
            body: types.AdapterRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.CalculateLossResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.calculate_loss(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='calculate_loss'))

    @app.post('/twinkle/backward')
    async def backward(request: Request, body: types.AdapterRequest, self: ModelManagement = Depends(self_fn)) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.backward(adapter_name=adapter_name, **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='backward'))

    @app.post('/twinkle/forward_backward', response_model=types.ForwardBackwardResponse)
    async def forward_backward(
            request: Request,
            body: types.ForwardRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.ForwardBackwardResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        def first_element(data):
            while isinstance(data, list):
                if len(data) == 0:
                    return None
                data = data[0]
            return data

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            all_inputs = _parse_inputs(body.inputs)
            for inputs in all_inputs:
                for key in inputs:
                    if isinstance(inputs[key], list) and isinstance(first_element(inputs[key]), (int, float)):
                        inputs[key] = torch.tensor(inputs[key])
            ret = self.model.forward_backward(inputs=all_inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        inputs_list = body.inputs if isinstance(body.inputs, list) else [body.inputs]
        input_tokens = sum(len(inp.get('input_ids', [])) if isinstance(inp, dict) else 0 for inp in inputs_list)
        batch_size = len(inputs_list)
        return await run_task(
            self.schedule_task_and_wait(
                _task,
                model_id=adapter_name,
                token=token,
                input_tokens=input_tokens,
                batch_size=batch_size,
                data_world_size=self.device_mesh.data_world_size,
                task_type='forward_backward',
            ))

    @app.post('/twinkle/clip_grad_norm', response_model=types.ClipGradNormResponse)
    async def clip_grad_norm(
            request: Request,
            body: types.AdapterRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.ClipGradNormResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.clip_grad_norm(adapter_name=adapter_name, **extra_kwargs)
            return {'result': str(ret)}

        return await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='clip_grad_norm'))

    @app.post('/twinkle/step')
    async def step(request: Request, body: types.AdapterRequest, self: ModelManagement = Depends(self_fn)) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.step(adapter_name=adapter_name, **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='step'))

    @app.post('/twinkle/zero_grad')
    async def zero_grad(request: Request, body: types.AdapterRequest, self: ModelManagement = Depends(self_fn)) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.zero_grad(adapter_name=adapter_name, **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='zero_grad'))

    @app.post('/twinkle/lr_step')
    async def lr_step(request: Request, body: types.AdapterRequest, self: ModelManagement = Depends(self_fn)) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.lr_step(adapter_name=adapter_name, **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='lr_step'))

    @app.post('/twinkle/clip_grad_and_step')
    async def clip_grad_and_step(
            request: Request,
            body: types.ClipGradAndStepRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.clip_grad_and_step(
                max_grad_norm=body.max_grad_norm,
                norm_type=body.norm_type,
                adapter_name=adapter_name,
                **extra_kwargs,
            )

        await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='clip_grad_and_step'))

    @app.post('/twinkle/get_train_configs', response_model=types.GetTrainConfigsResponse)
    async def get_train_configs(
            request: Request,
            body: types.AdapterRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.GetTrainConfigsResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.get_train_configs(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='get_train_configs'))

    @app.post('/twinkle/set_loss')
    async def set_loss(request: Request, body: types.SetLossRequest, self: ModelManagement = Depends(self_fn)) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.set_loss(body.loss_cls, adapter_name=adapter_name, **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='set_loss'))

    @app.post('/twinkle/set_optimizer')
    async def set_optimizer(
            request: Request,
            body: types.SetOptimizerRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.set_optimizer(body.optimizer_cls, adapter_name=adapter_name, **extra_kwargs)

        await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='set_optimizer'))

    @app.post('/twinkle/set_lr_scheduler')
    async def set_lr_scheduler(
            request: Request,
            body: types.SetLrSchedulerRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.set_lr_scheduler(body.scheduler_cls, adapter_name=adapter_name, **extra_kwargs)

        await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='set_lr_scheduler'))

    @app.post('/twinkle/save', response_model=types.SaveResponse)
    async def save(request: Request, body: types.SaveRequest,
                   self: ModelManagement = Depends(self_fn)) -> types.SaveResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            checkpoint_name = checkpoint_manager.get_ckpt_name(body.name)
            save_dir = checkpoint_manager.get_save_dir(model_id=adapter_name, is_sampler=body.is_sampler)
            # Must save the checkpoint in the twinkle format before calling model.save()
            twinkle_path = checkpoint_manager.save(
                model_id=adapter_name, name=checkpoint_name, is_sampler=body.is_sampler)
            # For sampler weights the actual data is always written to 'latest/'.
            model_save_name = 'latest' if body.is_sampler else checkpoint_name
            checkpoint_dir = self.model.save(
                name=model_save_name,
                output_dir=save_dir,
                adapter_name=adapter_name,
                save_optimizer=body.save_optimizer,
                **extra_kwargs)
            return {'twinkle_path': twinkle_path, 'checkpoint_dir': checkpoint_dir}

        return await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='save'))

    @app.post('/twinkle/load')
    async def load(request: Request, body: types.LoadRequest, self: ModelManagement = Depends(self_fn)) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            resolved = checkpoint_manager.resolve_load_path(body.name)
            self.model.load(
                name=resolved.checkpoint_name,
                output_dir=resolved.checkpoint_dir,
                adapter_name=adapter_name,
                load_optimizer=body.load_optimizer,
                token=token,
                **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='load'))

    @app.post('/twinkle/resume_from_checkpoint', response_model=types.TrainingProgressResponse)
    async def resume_from_checkpoint(
            request: Request,
            body: types.ResumeFromCheckpointRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.TrainingProgressResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            resolved = checkpoint_manager.resolve_load_path(body.name)
            checkpoint_dir = (
                Path(resolved.checkpoint_dir, resolved.checkpoint_name).as_posix()
                if resolved.checkpoint_dir else body.name)
            ret = self.model.resume_from_checkpoint(
                checkpoint_dir,
                resume_only_model=body.resume_only_model,
                adapter_name=adapter_name,
            )
            return {'result': ret}

        return await run_task(self.schedule_task_and_wait(_task, task_type='resume'))

    @app.post('/twinkle/upload_to_hub', response_model=types.UploadToHubResponse)
    async def upload_to_hub(
            request: Request,
            body: types.UploadToHubRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UploadToHubResponse:
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
            # Run blocking upload in thread pool so the event loop is not blocked.
            # async_upload is intentionally ignored here: the task queue + client polling
            # already provide the fire-and-forget / wait semantics without holding the
            # HTTP connection open for the full duration of the upload.
            await asyncio.to_thread(
                self.model.upload_to_hub,
                checkpoint_dir=checkpoint_dir,
                hub_model_id=body.hub_model_id,
                hub_token=body.hub_token or token,
                async_upload=False,
            )

        future_ref = await self.schedule_background_task(_task, task_type='upload_to_hub')
        request_id = future_ref.get('request_id')
        if request_id is None:
            raise HTTPException(status_code=500, detail=f'Upload task scheduling failed: {future_ref}')
        return types.UploadToHubResponse(request_id=request_id)

    @app.get('/twinkle/upload_status/{request_id}', response_model=types.UploadStatusResponse)
    async def upload_status(
            request: Request,
            request_id: str,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UploadStatusResponse:
        await self._on_request_start(request)
        record = await self.state.get_future(request_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f'Upload task not found: {request_id}')
        status = record.get('status', 'unknown')
        error = None
        if status == 'failed':
            error = record.get('result', {}).get('error', 'Unknown error')
        return types.UploadStatusResponse(request_id=request_id, status=status, error=error)

    @app.post('/twinkle/add_adapter_to_model', response_model=types.AddAdapterResponse)
    async def add_adapter_to_model(
            request: Request,
            body: types.AddAdapterRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.AddAdapterResponse:
        assert body.adapter_name, 'You need to specify a valid `adapter_name`'
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)
        session_id = get_session_id_from_request(request)

        async def _task():
            config = deserialize_object(body.config)
            extra_kwargs = body.model_extra or {}
            training_run_manager = create_training_run_manager(token, client_type='twinkle')
            self.register_resource(adapter_name, token, session_id)
            self.model.add_adapter_to_model(adapter_name, config, **extra_kwargs)

            lora_config = None
            if isinstance(config, LoraConfig):
                lora_config = types.LoraConfig(rank=config.r, train_unembed=False, train_mlp=True, train_attn=True)
            run_config = types.CreateModelRequest(
                base_model=self.base_model, lora_config=lora_config, user_metadata={'adapter_name': body.adapter_name})
            training_run_manager.save(adapter_name, run_config)
            return {'status': 'ok', 'adapter_name': adapter_name}

        return await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='add_adapter_to_model'))

    @app.post('/twinkle/apply_patch')
    async def apply_patch(
            request: Request,
            body: types.ApplyPatchRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            patch_cls = deserialize_object(body.patch_cls)
            self.model.apply_patch(patch_cls, adapter_name=adapter_name, **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='apply_patch'))

    @app.post('/twinkle/add_metric')
    async def add_metric(
            request: Request,
            body: types.AddMetricRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            metric_cls = deserialize_object(body.metric_cls)
            self.model.add_metric(metric_cls, is_training=body.is_training, adapter_name=adapter_name, **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='add_metric'))

    @app.post('/twinkle/set_template')
    async def set_template(
            request: Request,
            body: types.SetTemplateRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.set_template(body.template_cls, adapter_name=adapter_name, **extra_kwargs)

        await run_task(self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='set_template'))

    @app.post('/twinkle/set_processor')
    async def set_processor(
            request: Request,
            body: types.SetProcessorRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> None:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            self.model.set_processor(body.processor_cls, adapter_name=adapter_name, **extra_kwargs)

        await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='set_processor'))

    @app.post('/twinkle/calculate_metric', response_model=types.CalculateMetricResponse)
    async def calculate_metric(
            request: Request,
            body: types.CalculateMetricRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.CalculateMetricResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.calculate_metric(is_training=body.is_training, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='calculate_metric'))

    @app.post('/twinkle/get_state_dict', response_model=types.GetStateDictResponse)
    async def get_state_dict(
            request: Request,
            body: types.GetStateDictRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.GetStateDictResponse:
        token = await self._on_request_start(request)
        adapter_name = _get_twinkle_adapter_name(request, body.adapter_name)

        async def _task():
            self.assert_resource_exists(adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.get_state_dict(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        return await run_task(
            self.schedule_task_and_wait(_task, model_id=adapter_name, token=token, task_type='get_state_dict'))
