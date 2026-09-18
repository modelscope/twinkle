# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native model handler mixin.

All queued endpoints are prefixed /twinkle/... and return a Task_Envelope via the
shared ``run_submit`` judgment sequence: the handler submits work and returns
immediately, and the client's Client_Future_Layer resolves the envelope to a
terminal state. self_fn is injected via FastAPI Depends to obtain the
ModelManagement instance at request time.
"""
from __future__ import annotations

import torch
from collections.abc import Callable
from fastapi import Depends, FastAPI, HTTPException, Request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import ModelManagement

import twinkle_client.types as types
from twinkle.server.checkpoint import (_resolve_client_save_dir, create_checkpoint_manager, create_training_run_manager,
                                       validate_user_path)
from twinkle.server.exceptions import RequestRejectedError, TrainModeMismatchError
from twinkle.server.lifecycle.submit import (backend_kwargs, input_metrics, resolve_twinkle_adapter_name, run_submit,
                                             to_backend_inputs)
from twinkle.server.model.utils import (data_plane_request_shape, merge_forward_kwargs, resolve_data_plane_model_inputs,
                                        select_output_rows)
from twinkle.server.utils.validation import get_session_id_from_request
from twinkle.server.validation import BackendCapability
from twinkle.utils.logger import get_logger

logger = get_logger()


def _dp_metrics(self, body):
    """Scheduling metrics for inline data-parallel endpoints (forward / forward_backward)."""
    return input_metrics(self, body, data_parallel=True)


def _tokens_only_metrics(self, body):
    """Scheduling metrics for inline non-data-parallel endpoints (forward_only)."""
    return input_metrics(self, body, data_parallel=False)


def _data_plane_metrics(self, body):
    """Scheduling metrics derived from DataRef shape for *_from_data_plane endpoints."""
    input_tokens, batch_size = data_plane_request_shape(body)
    return {'input_tokens': input_tokens, 'batch_size': batch_size, 'data_world_size': self.data_world_size}


def _register_twinkle_routes(app: FastAPI, self_fn: Callable[[], ModelManagement]) -> None:
    """Register all /twinkle/* routes on the given FastAPI app.

    self_fn is a zero-argument callable that returns the current ModelManagement
    replica instance. It is wired in via Depends so it is resolved lazily at request time.
    """

    @app.get('/healthz')
    async def model_healthz(
            request: Request,
            self: ModelManagement = Depends(self_fn),
    ) -> dict:
        """Deep health probe: pings underlying model actors to verify liveness."""
        result = await self.check_model_health()
        if self._model_unhealthy or not result['healthy']:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=503, content=result)
        return result

    @app.post('/twinkle/create', response_model=types.CreateResponse)
    async def create(request: Request, body: types.CreateRequest,
                     self: ModelManagement = Depends(self_fn)) -> types.CreateResponse:
        await self._on_request_start(request)
        return types.CreateResponse()

    # ------------------------------------------------------------------ #
    # Inline data / forward family
    # ------------------------------------------------------------------ #

    @app.post('/twinkle/forward', response_model=types.TaskEnvelope)
    async def forward(request: Request, body: types.ForwardRequest,
                      self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            ret = await self.call_backend(
                self.model.forward,
                inputs=to_backend_inputs(body.inputs),
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))
            return {'result': ret}

        return await run_submit(
            self,
            request,
            body,
            task_type='forward',
            backend_call=_call,
            metrics=_dp_metrics,
            capability=BackendCapability.Forward)

    @app.post('/twinkle/forward_only', response_model=types.TaskEnvelope)
    async def forward_only(
        request: Request, body: types.ForwardOnlyRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            ret = await self.call_backend(
                self.model.forward_only,
                inputs=to_backend_inputs(body.inputs),
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))
            return {'result': ret}

        return await run_submit(
            self, request, body, task_type='forward_only', backend_call=_call, metrics=_tokens_only_metrics)

    @app.post('/twinkle/forward_backward', response_model=types.TaskEnvelope)
    async def forward_backward(
        request: Request, body: types.ForwardBackwardTaskRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):

            def first_element(data):
                while isinstance(data, list):
                    if len(data) == 0:
                        return None
                    data = data[0]
                return data

            all_inputs = to_backend_inputs(body.inputs)
            for inputs in all_inputs:
                for key in inputs:
                    if isinstance(inputs[key], list) and isinstance(first_element(inputs[key]), (int, float)):
                        inputs[key] = torch.tensor(inputs[key])
            ret = await self.call_backend(
                self.model.forward_backward,
                inputs=all_inputs,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))
            return {'result': ret}

        return await run_submit(
            self, request, body, task_type='forward_backward', backend_call=_call, metrics=_dp_metrics)

    @app.post('/twinkle/calculate_loss', response_model=types.TaskEnvelope)
    async def calculate_loss(
        request: Request, body: types.AdapterRequest, self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            ret = await self.call_backend(
                self.model.calculate_loss,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))
            return {'result': ret}

        return await run_submit(
            self,
            request,
            body,
            task_type='calculate_loss',
            backend_call=_call,
            capability=BackendCapability.CalculateLoss)

    @app.post('/twinkle/backward', response_model=types.TaskEnvelope)
    async def backward(request: Request, body: types.AdapterRequest,
                       self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.backward, adapter_name=self.resolve_model_adapter_name(adapter_name), **backend_kwargs(body))

        return await run_submit(
            self, request, body, task_type='backward', backend_call=_call, capability=BackendCapability.Backward)

    # ------------------------------------------------------------------ #
    # Data-plane forward family (DataRef inputs; response only enters the contract)
    # ------------------------------------------------------------------ #

    @app.post('/twinkle/forward_from_data_plane', response_model=types.TaskEnvelope)
    async def forward_from_data_plane(
        request: Request, body: types.DataPlaneForwardRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            raw_inputs, field_kwargs = await resolve_data_plane_model_inputs(body, self.data_plane)
            kwargs = merge_forward_kwargs(backend_kwargs(body), field_kwargs)
            ret = await self.call_backend(
                self.model.forward,
                inputs=to_backend_inputs(raw_inputs),
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **kwargs)
            return {'result': ret}

        return await run_submit(
            self,
            request,
            body,
            task_type='forward_from_data_plane',
            backend_call=_call,
            metrics=_data_plane_metrics,
            capability=BackendCapability.Forward)

    @app.post('/twinkle/forward_only_from_data_plane', response_model=types.TaskEnvelope)
    async def forward_only_from_data_plane(
        request: Request, body: types.DataPlaneForwardOnlyRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            raw_inputs, field_kwargs = await resolve_data_plane_model_inputs(body, self.data_plane)
            inputs = to_backend_inputs(raw_inputs)
            kwargs = merge_forward_kwargs(backend_kwargs(body), field_kwargs)
            ret = await self.call_backend(
                self.model.forward_only,
                inputs=inputs,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **kwargs)
            if body.output_ref is not None:
                rows = select_output_rows(ret, batch_size=len(inputs), output_fields=body.output_fields)
                output_ref = await self.data_plane.append(body.output_ref, rows)
                return {'result': output_ref.model_dump()}
            return {'result': ret}

        return await run_submit(
            self,
            request,
            body,
            task_type='forward_only_from_data_plane',
            backend_call=_call,
            metrics=_data_plane_metrics)

    @app.post('/twinkle/forward_backward_from_data_plane', response_model=types.TaskEnvelope)
    async def forward_backward_from_data_plane(
        request: Request, body: types.DataPlaneForwardRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            raw_inputs, field_kwargs = await resolve_data_plane_model_inputs(body, self.data_plane)
            kwargs = merge_forward_kwargs(backend_kwargs(body), field_kwargs)
            ret = await self.call_backend(
                self.model.forward_backward,
                inputs=to_backend_inputs(raw_inputs),
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **kwargs)
            return {'result': ret}

        return await run_submit(
            self,
            request,
            body,
            task_type='forward_backward_from_data_plane',
            backend_call=_call,
            metrics=_data_plane_metrics)

    # ------------------------------------------------------------------ #
    # Optimizer / control plane
    # ------------------------------------------------------------------ #

    @app.post('/twinkle/clip_grad_norm', response_model=types.TaskEnvelope)
    async def clip_grad_norm(
        request: Request, body: types.ClipGradNormRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            ret = await self.call_backend(
                self.model.clip_grad_norm,
                max_grad_norm=body.max_grad_norm,
                norm_type=body.norm_type,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))
            return {'result': str(ret)}

        return await run_submit(self, request, body, task_type='clip_grad_norm', backend_call=_call)

    @app.post('/twinkle/step', response_model=types.TaskEnvelope)
    async def step(request: Request, body: types.StepRequest,
                   self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.step, adapter_name=self.resolve_model_adapter_name(adapter_name), **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='step', backend_call=_call)

    @app.post('/twinkle/zero_grad', response_model=types.TaskEnvelope)
    async def zero_grad(request: Request, body: types.AdapterRequest,
                        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.zero_grad,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='zero_grad', backend_call=_call)

    @app.post('/twinkle/lr_step', response_model=types.TaskEnvelope)
    async def lr_step(request: Request, body: types.LrStepRequest,
                      self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.lr_step, adapter_name=self.resolve_model_adapter_name(adapter_name), **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='lr_step', backend_call=_call)

    @app.post('/twinkle/clip_grad_and_step', response_model=types.TaskEnvelope)
    async def clip_grad_and_step(
        request: Request, body: types.ClipGradAndStepRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.clip_grad_and_step,
                max_grad_norm=body.max_grad_norm,
                norm_type=body.norm_type,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='clip_grad_and_step', backend_call=_call)

    @app.post('/twinkle/get_train_configs', response_model=types.TaskEnvelope)
    async def get_train_configs(
        request: Request, body: types.AdapterRequest, self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            ret = await self.call_backend(
                self.model.get_train_configs,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))
            return {'result': ret}

        return await run_submit(self, request, body, task_type='get_train_configs', backend_call=_call)

    @app.post('/twinkle/set_loss', response_model=types.TaskEnvelope)
    async def set_loss(request: Request, body: types.SetLossRequest,
                       self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.set_loss,
                body.loss_cls,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='set_loss', backend_call=_call)

    @app.post('/twinkle/set_optimizer', response_model=types.TaskEnvelope)
    async def set_optimizer(
        request: Request, body: types.SetOptimizerRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.set_optimizer,
                body.optimizer_cls,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='set_optimizer', backend_call=_call)

    @app.post('/twinkle/set_lr_scheduler', response_model=types.TaskEnvelope)
    async def set_lr_scheduler(
        request: Request, body: types.SetLrSchedulerRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.set_lr_scheduler,
                body.scheduler_cls,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='set_lr_scheduler', backend_call=_call)

    @app.post('/twinkle/set_template', response_model=types.TaskEnvelope)
    async def set_template(
        request: Request, body: types.SetTemplateRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.set_template,
                body.template_cls,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='set_template', backend_call=_call)

    @app.post('/twinkle/set_processor', response_model=types.TaskEnvelope)
    async def set_processor(
        request: Request, body: types.SetProcessorRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            await self.call_backend(
                self.model.set_processor,
                body.processor_cls,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='set_processor', backend_call=_call)

    @app.post('/twinkle/add_metric', response_model=types.TaskEnvelope)
    async def add_metric(request: Request, body: types.AddMetricRequest,
                         self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            from twinkle_client.common.serialize import deserialize_object
            metric_cls = deserialize_object(body.metric_cls)
            await self.call_backend(
                self.model.add_metric,
                metric_cls,
                is_training=body.is_training,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='add_metric', backend_call=_call)

    @app.post('/twinkle/apply_patch', response_model=types.TaskEnvelope)
    async def apply_patch(
        request: Request, body: types.ApplyPatchRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            from twinkle_client.common.serialize import deserialize_object
            patch_cls = deserialize_object(body.patch_cls)
            await self.call_backend(
                self.model.apply_patch,
                patch_cls,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='apply_patch', backend_call=_call)

    @app.post('/twinkle/calculate_metric', response_model=types.TaskEnvelope)
    async def calculate_metric(
        request: Request, body: types.CalculateMetricRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            ret = await self.call_backend(
                self.model.calculate_metric,
                is_training=body.is_training,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                **backend_kwargs(body))
            return {'result': ret}

        return await run_submit(self, request, body, task_type='calculate_metric', backend_call=_call)

    # ------------------------------------------------------------------ #
    # Checkpoint I/O (need the caller token)
    # ------------------------------------------------------------------ #

    @app.post('/twinkle/save', response_model=types.TaskEnvelope)
    async def save(request: Request, body: types.SaveRequest,
                   self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            checkpoint_name = checkpoint_manager.get_ckpt_name(body.name)
            save_dir = checkpoint_manager.get_save_dir(model_id=adapter_name, is_sampler=body.is_sampler)
            # Must save the checkpoint in the twinkle format before calling model.save()
            twinkle_path = checkpoint_manager.save(
                model_id=adapter_name, name=checkpoint_name, is_sampler=body.is_sampler)
            # For sampler weights the actual data is always written to 'latest/'.
            model_save_name = 'latest' if body.is_sampler else checkpoint_name
            checkpoint_dir = await self.call_backend(
                self.model.save,
                name=model_save_name,
                output_dir=save_dir,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                save_optimizer=body.save_optimizer,
                **backend_kwargs(body))
            return {'twinkle_path': twinkle_path, 'checkpoint_dir': checkpoint_dir}

        return await run_submit(self, request, body, task_type='save', backend_call=_call)

    @app.post('/twinkle/load', response_model=types.TaskEnvelope)
    async def load(request: Request, body: types.LoadRequest,
                   self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            resolved = checkpoint_manager.resolve_load_path(body.name)
            await self.call_backend(
                self.model.load,
                name=resolved.checkpoint_name,
                output_dir=resolved.checkpoint_dir,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                load_optimizer=body.load_optimizer,
                token=token,
                **backend_kwargs(body))

        return await run_submit(self, request, body, task_type='load', backend_call=_call)

    @app.post('/twinkle/resume_from_checkpoint', response_model=types.TaskEnvelope)
    async def resume_from_checkpoint(
        request: Request, body: types.ResumeFromCheckpointRequest,
        self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:

        async def _call(self, body, adapter_name, token):
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            resolved = checkpoint_manager.resolve_load_path(body.name)
            checkpoint_dir = (
                Path(resolved.checkpoint_dir, resolved.checkpoint_name).as_posix()
                if resolved.checkpoint_dir else body.name)
            ret = await self.call_backend(
                self.model.resume_from_checkpoint,
                checkpoint_dir,
                resume_only_model=body.resume_only_model,
                adapter_name=self.resolve_model_adapter_name(adapter_name))
            return {'result': ret}

        return await run_submit(self, request, body, task_type='resume', backend_call=_call)

    # ------------------------------------------------------------------ #
    # Adapter lifecycle (create / drop the adapter itself: no resource assert)
    # ------------------------------------------------------------------ #

    @app.post('/twinkle/remove_adapter', response_model=types.TaskEnvelope)
    async def remove_adapter(
        request: Request, body: types.AdapterRequest, self: ModelManagement = Depends(self_fn)) -> types.TaskEnvelope:
        """Release a drained tenant's in-memory training adapter."""

        async def _call(self, body, adapter_name, token):
            await self._cleanup_adapter(adapter_name)
            return {'status': 'ok'}

        return await run_submit(
            self, request, body, task_type='remove_adapter', backend_call=_call, assert_resource=False)

    @app.post('/twinkle/add_adapter_to_model', response_model=types.TaskEnvelope)
    async def add_adapter_to_model(
            request: Request,
            body: types.AddAdapterRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.TaskEnvelope:
        # This endpoint creates the adapter, so it cannot use the standard resource
        # assertion. The Decision_Boundary left checks (train_mode 400 / full-mode
        # 409) run here, before any state write, raising RequestRejectedError
        # subclasses (zero future writes).
        #
        # Raised, not asserted: a missing adapter_name is decidable from the request body
        # alone, so it owes the caller a real 400. A bare `assert` would surface as a 500
        # ('the server broke') and would vanish entirely under `python -O`, letting an
        # empty adapter_name through to the backend.
        if not body.adapter_name:
            raise RequestRejectedError('`adapter_name` is required and must be non-empty.')
        token = await self._on_request_start(request)
        if not validate_user_path(token, body.adapter_name):
            raise RequestRejectedError(f'Invalid adapter_name: {body.adapter_name}')
        adapter_name = resolve_twinkle_adapter_name(request, body.adapter_name)
        session_id = get_session_id_from_request(request)
        try:
            resolved_save_dir = _resolve_client_save_dir(body.save_dir).as_posix() if body.save_dir else None
        except ValueError as exc:
            raise RequestRejectedError(str(exc)) from exc

        from twinkle_client.common.serialize import deserialize_object
        config = deserialize_object(body.config)

        # ---- Decision_Boundary left: validate against the deployment's train_mode ----
        if self.is_full_mode and config is not None:
            raise TrainModeMismatchError('This deployment runs in full-parameter (exclusive) mode; pass '
                                         'config=None (do not send a LoraConfig).')
        if (not self.is_full_mode) and config is None:
            raise TrainModeMismatchError('This deployment runs in LoRA mode; a LoraConfig is required.')
        if self.is_full_mode:
            # Raises FullModeBusyError (409) if another tenant holds the exclusive deployment.
            self.assert_full_mode_available(adapter_name)

        async def _task():
            from peft import LoraConfig
            extra_kwargs = backend_kwargs(body)
            training_run_manager = create_training_run_manager(token, client_type='twinkle')
            lora_config = None
            if isinstance(config, LoraConfig):
                lora_config = types.LoraConfig(rank=config.r, train_unembed=False, train_mlp=True, train_attn=True)
            run_config = types.CreateModelRequest(
                base_model=self.base_model,
                lora_config=lora_config,
                save_dir=resolved_save_dir,
                user_metadata={'adapter_name': body.adapter_name})
            await self.state.register_model(
                run_config.model_dump(),
                token=token,
                model_id=adapter_name,
                replica_id=self.replica_id,
                session_id=session_id,
            )
            try:
                self.register_resource(adapter_name, token, session_id)
                if self.is_full_mode:
                    # No PEFT adapter to add; the default optimizer group is used.
                    self.set_resource_state(adapter_name, 'grad_ready', False)
                else:
                    await self.call_backend(self.model.add_adapter_to_model, adapter_name, config, **extra_kwargs)
            except Exception:
                self.unregister_resource(adapter_name)
                await self.state.unload_model(adapter_name)
                raise
            training_run_manager.save(adapter_name, run_config)
            return {'status': 'ok', 'adapter_name': adapter_name}

        return await self.submit_and_peek(_task, model_id=adapter_name, token=token, task_type='add_adapter_to_model')

    # ------------------------------------------------------------------ #
    # Hub upload (pure I/O -> background task; state-tracked via Retrieve_Endpoint)
    # ------------------------------------------------------------------ #

    @app.post('/twinkle/upload_to_hub', response_model=types.TaskEnvelope)
    async def upload_to_hub(
            request: Request,
            body: types.UploadToHubRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.TaskEnvelope:
        token = await self._on_request_start(request)

        async def _task():
            if body.checkpoint_dir.startswith('twinkle://'):
                checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
                parsed = checkpoint_manager.parse_twinkle_path(body.checkpoint_dir)
                if not parsed:
                    raise ValueError(f'Invalid twinkle path format: {body.checkpoint_dir}')
                checkpoint = checkpoint_manager.get(parsed.training_run_id, parsed.checkpoint_id)
                if not checkpoint:
                    raise ValueError(f'Checkpoint not found or access denied: {body.checkpoint_dir}')
                checkpoint_dir = str(
                    checkpoint_manager.get_ckpt_dir(
                        model_id=parsed.training_run_id, checkpoint_id=parsed.checkpoint_id))
            else:
                checkpoint_dir = body.checkpoint_dir
            await self.call_backend(
                self.model.upload_to_hub,
                checkpoint_dir=checkpoint_dir,
                hub_model_id=body.hub_model_id,
                hub_token=body.hub_token or token,
                async_upload=False,
            )

        return await self.submit_background_and_peek(_task, task_type='upload_to_hub')
