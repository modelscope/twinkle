# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tinker-compatible routes for the Model deployment.

Registered by ``_register_model_tinker_routes(app, self_fn)`` -- module-level route
registration closing over ``self_fn`` via ``Depends``, not a mixin: there is no
inheritance relationship with the deployment class. All endpoints are prefixed
/tinker/... and use schedule_task() returning UntypedAPIFuture. ``self_fn`` is injected
via FastAPI Depends to obtain the ModelManagement instance at request time.
"""
from __future__ import annotations

from collections.abc import Callable
from fastapi import Depends, FastAPI, Request
from functools import partial
from tinker import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import ModelManagement

from twinkle.server.checkpoint import create_checkpoint_manager, create_training_run_manager
from twinkle.server.exceptions import RequestRejectedError
from twinkle.server.lifecycle.submit import submit_training_task
from twinkle.server.validation.tinker_loss import validate_tinker_loss
from twinkle.utils.logger import get_logger

logger = get_logger()


def _register_model_tinker_routes(app: FastAPI, self_fn: Callable[[], ModelManagement]) -> None:
    """Register all /tinker/* routes on the given FastAPI app.

    self_fn is a zero-argument callable that returns the current ModelManagement
    replica instance. It is wired in via Depends so it is resolved lazily at request time.
    """

    @app.post('/tinker/create_model')
    async def create_model(
            request: Request,
            body: types.CreateModelRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UntypedAPIFuture:
        raise RequestRejectedError('Training lifecycle operations must use the catalog endpoint')

    @app.post('/tinker/get_info')
    async def get_info(request: Request, body: types.GetInfoRequest,
                       self: ModelManagement = Depends(self_fn)) -> types.GetInfoResponse:
        token = await self._on_request_start(request)
        training_run_manager = create_training_run_manager(token, client_type='tinker')
        metadata = training_run_manager.get(str(body.model_id))
        model_name = metadata.base_model if metadata else self.base_model
        lora_rank = None
        is_lora = False
        if metadata and hasattr(metadata, 'lora_rank') and metadata.lora_rank:
            lora_rank = metadata.lora_rank
            is_lora = metadata.is_lora
        return types.GetInfoResponse(
            model_data=types.ModelData(model_name=model_name),
            model_id=body.model_id,
            is_lora=is_lora,
            lora_rank=lora_rank,
            model_name=model_name,
        )

    @app.post('/tinker/unload_model')
    async def unload_model(
            request: Request,
            body: types.UnloadModelRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UntypedAPIFuture:
        raise RequestRejectedError('Training lifecycle operations must use the catalog endpoint')

    @app.post('/tinker/forward')
    async def forward(request: Request, body: types.ForwardRequest,
                      self: ModelManagement = Depends(self_fn)) -> types.UntypedAPIFuture:
        token = await self._on_request_start(request)

        async def _do_forward():
            adapter_name = body.model_id
            self.assert_resource_exists(adapter_name)
            model_adapter = self.resolve_model_adapter_name(adapter_name)
            datum_list = body.forward_input.data
            loss_fn_config = body.forward_input.loss_fn_config or {}
            output, loss = await self.call_backend(
                self.model.tinker_forward_only, inputs=datum_list, adapter_name=model_adapter, **loss_fn_config)
            return types.ForwardBackwardOutput(
                loss_fn_output_type='CrossEntropyLossReturn',
                loss_fn_outputs=output,
                metrics={'loss:mean': loss},
            )

        datum_list = body.forward_input.data
        input_tokens = sum(len(d.model_input.to_ints()) for d in datum_list)
        batch_size = len(datum_list)
        return await submit_training_task(
            self,
            _do_forward,
            training_id=body.model_id,
            seq_id=body.seq_id,
            stream='tinker::forward',
            token=token,
            input_tokens=input_tokens,
            batch_size=batch_size,
            data_world_size=self.data_world_size,
            task_type='forward',
        )

    @app.post('/tinker/forward_backward')
    async def forward_backward(
            request: Request,
            body: types.ForwardBackwardRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UntypedAPIFuture:
        token = await self._on_request_start(request)

        async def _do_forward_backward():
            adapter_name = body.model_id
            self.assert_resource_exists(adapter_name)
            model_adapter = self.resolve_model_adapter_name(adapter_name)
            datum_list = body.forward_backward_input.data
            loss_fn = body.forward_backward_input.loss_fn
            loss_fn_config = body.forward_backward_input.loss_fn_config or {}
            preflight = getattr(self.model, 'preflight_loss_inputs', None)
            if preflight is not None and loss_fn == 'importance_sampling' and not validate_tinker_loss(
                    loss_fn, datum_list, data_world_size=self.data_world_size):
                # GRPO datums carry ragged per-sample targets; the loss check inside
                # the sliced backend would deadlock the peer ranks (BUG-05), so the
                # full batch is validated on every rank first.
                await self.call_backend(
                    preflight, inputs=datum_list, adapter_name=model_adapter, loss_fn=loss_fn, **loss_fn_config)
            output, loss = await self.call_backend(
                self.model.tinker_forward_backward,
                inputs=datum_list,
                adapter_name=model_adapter,
                loss_fn=loss_fn,
                **loss_fn_config)
            output_type = ('ImportanceSamplingLossReturn'
                           if loss_fn == 'importance_sampling' else 'CrossEntropyLossReturn')
            self.get_training_state(adapter_name).grad_ready = True
            return types.ForwardBackwardOutput(
                loss_fn_output_type=output_type,
                loss_fn_outputs=output,
                metrics={'loss:mean': loss},
            )

        datum_list = body.forward_backward_input.data
        input_tokens = sum(len(d.model_input.to_ints()) for d in datum_list)
        batch_size = len(datum_list)
        loss_fn = body.forward_backward_input.loss_fn
        is_dpo = validate_tinker_loss(loss_fn, datum_list, data_world_size=self.data_world_size)
        return await submit_training_task(
            self,
            _do_forward_backward,
            training_id=body.model_id,
            seq_id=body.seq_id,
            stream='tinker::forward_backward',
            token=token,
            input_tokens=input_tokens,
            batch_size=batch_size,
            data_world_size=self.data_world_size,
            batch_size_multiple=2 if is_dpo else None,
            task_type='forward_backward',
        )

    @app.post('/tinker/optim_step')
    async def optim_step(
            request: Request,
            body: types.OptimStepRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UntypedAPIFuture:
        token = await self._on_request_start(request)

        async def _do_optim():
            adapter_name = body.model_id
            self.assert_resource_exists(adapter_name)
            model_adapter = self.resolve_model_adapter_name(adapter_name)
            if not self.get_training_state(adapter_name).grad_ready:
                raise RuntimeError(f'No accumulated gradients for adapter={adapter_name}; '
                                   'call forward_backward before optim_step')
            await self.call_backend(self.model.tinker_step, adam_params=body.adam_params, adapter_name=model_adapter)
            self.get_training_state(adapter_name).grad_ready = False
            metrics = await self.call_backend(
                self.model.tinker_calculate_metric, is_training=True, adapter_name=model_adapter)
            return types.OptimStepResponse(metrics=metrics)

        return await submit_training_task(
            self,
            _do_optim,
            training_id=body.model_id,
            seq_id=body.seq_id,
            stream='tinker::optim_step',
            token=token,
            task_type='optim_step')

    @app.post('/tinker/save_weights')
    async def save_weights(
            request: Request,
            body: types.SaveWeightsRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UntypedAPIFuture:
        token = await self._on_request_start(request)

        async def _do_save():
            adapter_name = body.model_id
            self.assert_resource_exists(adapter_name)
            checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
            save_weights = partial(
                self.call_backend,
                self.model.save,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                save_optimizer=True)
            tinker_path, _ = await checkpoint_manager.save_training(body.model_id, body.path, save_weights)
            return types.SaveWeightsResponse(path=tinker_path, type='save_weights')

        return await submit_training_task(
            self,
            _do_save,
            training_id=body.model_id,
            seq_id=body.seq_id,
            stream='tinker::save_weights',
            token=token,
            task_type='save_weights')

    @app.post('/tinker/save_weights_for_sampler')
    async def save_weights_for_sampler(
            request: Request,
            body: types.SaveWeightsForSamplerRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UntypedAPIFuture:
        token = await self._on_request_start(request)

        async def _do_save_for_sampler():
            adapter_name = body.model_id
            self.assert_resource_exists(adapter_name)
            checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')

            save_weights = partial(
                self.call_backend,
                self.model.save,
                adapter_name=self.resolve_model_adapter_name(adapter_name),
                save_optimizer=False)

            tinker_path, _ = await checkpoint_manager.save_sampler(body.model_id, body.path, save_weights)
            payload = body.model_dump()
            payload['model_path'] = tinker_path
            metadata = await self.state.get_model_metadata(body.model_id) or {}
            if metadata.get('base_model'):
                payload['base_model'] = metadata['base_model']
            sampling_session_id = await self.state.create_sampling_session(payload)
            # Tinker SDK distinguishes two modes by whether sampling_session_seq_id is set:
            # 1. save_weights_for_sampler(name): expects path != None
            # 2. save_weights_and_get_sampling_client(): expects path == None, sampling_session_id != None
            if body.sampling_session_seq_id is not None:
                return types.SaveWeightsForSamplerResponseInternal(path=None, sampling_session_id=sampling_session_id)
            else:
                return types.SaveWeightsForSamplerResponseInternal(
                    path=tinker_path, sampling_session_id=sampling_session_id)

        return await submit_training_task(
            self,
            _do_save_for_sampler,
            training_id=body.model_id,
            seq_id=body.seq_id,
            stream='tinker::save_weights_for_sampler',
            token=token,
            task_type='save_weights_for_sampler')

    @app.post('/tinker/load_weights')
    async def load_weights(
            request: Request,
            body: types.LoadWeightsRequest,
            self: ModelManagement = Depends(self_fn),
    ) -> types.UntypedAPIFuture:
        token = await self._on_request_start(request)

        async def _do_load():
            assert self.model is not None, 'Model not loaded, please load model first'
            adapter_name = body.model_id
            self.assert_resource_exists(adapter_name)
            # Path resolution (token isolation, twinkle-vs-external path shapes) is a
            # server storage policy and belongs in the handler, not in the GPU-side
            # backend actor. The backend receives an already-resolved location.
            checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
            resolved = checkpoint_manager.resolve_load_path(body.path)
            await self.call_backend(
                self.model.tinker_load,
                checkpoint_name=resolved.checkpoint_name,
                output_dir=resolved.checkpoint_dir if resolved.is_twinkle_path else None,
                load_optimizer=body.optimizer,
                adapter_name=self.resolve_model_adapter_name(adapter_name))
            self.get_training_state(adapter_name).grad_ready = False
            return types.LoadWeightsResponse(path=body.path, type='load_weights')

        return await submit_training_task(
            self,
            _do_load,
            training_id=body.model_id,
            seq_id=body.seq_id,
            stream='tinker::load_weights',
            token=token,
            task_type='load_weights')
