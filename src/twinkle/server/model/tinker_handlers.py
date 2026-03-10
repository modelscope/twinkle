# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Tinker-compatible model handler mixin.

All endpoints are prefixed /tinker/... and use schedule_task() returning UntypedAPIFuture.
"""
import traceback
from fastapi import FastAPI, Request
from peft import LoraConfig
from tinker import types
from typing import Any

from twinkle.server.common.io_utils import create_checkpoint_manager, create_training_run_manager
from twinkle.server.utils.validation import get_token_from_request
from twinkle.utils.logger import get_logger

logger = get_logger()


class TinkerModelHandlers:
    """
    Mixin providing Tinker-compatible model management endpoints.

    Expects the combined class to also inherit TaskQueueMixin and AdapterManagerMixin,
    and to have:
      self.model, self.state, self.device_mesh, self.base_model, self.replica_id
    """

    @staticmethod
    def _register_tinker_routes(app: FastAPI, model_id: str):
        """Register all tinker routes on the given FastAPI app.

        This is called once during build_model_app to wire routes.
        """

        @app.post('/tinker/create_model')
        async def create_model(self, request: Request, body: types.CreateModelRequest) -> types.UntypedAPIFuture:
            token = await self._on_request_start(request)

            async def _create_adapter():
                _model_id = None
                try:
                    _model_id = self.state.register_model(body.model_dump(), token=token, replica_id=self.replica_id)
                    if body.lora_config:
                        lora_cfg = LoraConfig(r=body.lora_config.rank, target_modules='all-linear')
                        adapter_name = self.get_adapter_name(adapter_name=_model_id)
                        self.register_adapter(adapter_name, token, session_id=body.session_id)
                        self.model.add_adapter_to_model(adapter_name=adapter_name, config_or_dir=lora_cfg)
                        self.model.set_template('Template', adapter_name=adapter_name, model_id=model_id)
                        self.model.set_processor('InputProcessor', adapter_name=adapter_name)
                        self.model.set_optimizer('Adam', adapter_name=adapter_name)
                        self.set_adapter_state(adapter_name, 'grad_ready', False)
                    training_run_manager = create_training_run_manager(token, client_type='tinker')
                    training_run_manager.save(_model_id, body)
                    return types.CreateModelResponse(model_id=_model_id)
                except Exception:
                    if _model_id:
                        adapter_name = self.get_adapter_name(adapter_name=_model_id)
                        self._cleanup_adapter(adapter_name)
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(_create_adapter, token=token, task_type='create_model')

        @app.post('/tinker/get_info')
        async def get_info(self, request: Request, body: types.GetInfoRequest) -> types.GetInfoResponse:
            token = await self._on_request_start(request)
            training_run_manager = create_training_run_manager(token, client_type='tinker')
            metadata = training_run_manager.get(str(body.model_id))
            model_name = metadata.base_model if metadata else model_id
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
        async def unload_model(self, request: Request, body: types.UnloadModelRequest) -> types.UntypedAPIFuture:
            token = await self._on_request_start(request)

            async def _do_unload():
                adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                self._cleanup_adapter(adapter_name)
                return types.UnloadModelResponse(model_id=body.model_id)

            return await self.schedule_task(_do_unload, model_id=body.model_id, token=token, task_type='unload_model')

        @app.post('/tinker/forward')
        async def forward(self, request: Request, body: types.ForwardRequest) -> types.UntypedAPIFuture:
            token = await self._on_request_start(request)

            async def _do_forward():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)
                    self.touch_adapter(adapter_name)
                    datum_list = body.forward_input.data
                    loss_fn_config = body.forward_input.loss_fn_config or {}
                    output = self.model.forward_only(inputs=datum_list, adapter_name=adapter_name)
                    loss = self.model.calculate_loss(adapter_name=adapter_name, **loss_fn_config)
                    return types.ForwardBackwardOutput(
                        loss_fn_output_type='CrossEntropyLossReturn',
                        loss_fn_outputs=output,
                        metrics={'loss:sum': loss},
                    )
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            datum_list = body.forward_input.data
            input_tokens = sum(len(d.model_input.to_ints()) for d in datum_list)
            batch_size = len(datum_list)
            return await self.schedule_task(
                _do_forward,
                model_id=body.model_id,
                token=token,
                input_tokens=input_tokens,
                batch_size=batch_size,
                data_world_size=self.device_mesh.data_world_size,
                task_type='forward',
            )

        @app.post('/tinker/forward_backward')
        async def forward_backward(self, request: Request,
                                   body: types.ForwardBackwardRequest) -> types.UntypedAPIFuture:
            token = await self._on_request_start(request)

            async def _do_forward_backward():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)
                    self.touch_adapter(adapter_name)
                    datum_list = body.forward_backward_input.data
                    loss_fn = body.forward_backward_input.loss_fn
                    loss_fn_config = body.forward_backward_input.loss_fn_config or {}
                    output, loss = self.model.forward_backward(
                        inputs=datum_list, adapter_name=adapter_name, loss_fn=loss_fn, **loss_fn_config)
                    if loss_fn == 'importance_sampling':
                        output_type = 'ImportanceSamplingLossReturn'
                    else:
                        output_type = 'CrossEntropyLossReturn'
                    self.set_adapter_state(adapter_name, 'grad_ready', True)
                    return types.ForwardBackwardOutput(
                        loss_fn_output_type=output_type,
                        loss_fn_outputs=output,
                        metrics={'loss:avg': loss},
                    )
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            datum_list = body.forward_backward_input.data
            input_tokens = sum(len(d.model_input.to_ints()) for d in datum_list)
            batch_size = len(datum_list)
            return await self.schedule_task(
                _do_forward_backward,
                model_id=body.model_id,
                token=token,
                input_tokens=input_tokens,
                batch_size=batch_size,
                data_world_size=self.device_mesh.data_world_size,
                task_type='forward_backward',
            )

        @app.post('/tinker/optim_step')
        async def optim_step(self, request: Request, body: types.OptimStepRequest) -> types.UntypedAPIFuture:
            token = await self._on_request_start(request)

            async def _do_optim():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)
                    if not self.get_adapter_state(adapter_name, 'grad_ready', False):
                        raise RuntimeError(
                            f'No accumulated gradients for adapter={adapter_name}; call forward_backward before optim_step'  # noqa: E501
                        )
                    self.touch_adapter(adapter_name)
                    self.model.step(adam_params=body.adam_params, adapter_name=adapter_name)
                    self.set_adapter_state(adapter_name, 'grad_ready', False)
                    metrics = self.model.calculate_metric(is_training=True, adapter_name=adapter_name)
                    return types.OptimStepResponse(metrics=metrics)
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(_do_optim, model_id=body.model_id, token=token, task_type='optim_step')

        @app.post('/tinker/save_weights')
        async def save_weights(self, request: Request, body: types.SaveWeightsRequest) -> types.UntypedAPIFuture:
            token = await self._on_request_start(request)

            async def _do_save():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)
                    self.touch_adapter(adapter_name)
                    checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
                    checkpoint_name = checkpoint_manager.get_ckpt_name(body.path)
                    save_dir = checkpoint_manager.get_save_dir(model_id=body.model_id, is_sampler=False)
                    self.model.save(
                        name=checkpoint_name, output_dir=save_dir, adapter_name=adapter_name, save_optimizer=True)
                    tinker_path = checkpoint_manager.save(body.model_id, name=checkpoint_name, is_sampler=False)
                    return types.SaveWeightsResponse(path=tinker_path, type='save_weights')
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(_do_save, model_id=body.model_id, token=token, task_type='save_weights')

        @app.post('/tinker/save_weights_for_sampler')
        async def save_weights_for_sampler(self, request: Request,
                                           body: types.SaveWeightsForSamplerRequest) -> types.UntypedAPIFuture:
            token = await self._on_request_start(request)

            async def _do_save_for_sampler():
                try:
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)
                    self.touch_adapter(adapter_name)
                    checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
                    checkpoint_name = checkpoint_manager.get_ckpt_name(body.path)
                    save_dir = checkpoint_manager.get_save_dir(model_id=body.model_id, is_sampler=True)
                    tinker_path = checkpoint_manager.save(body.model_id, name=checkpoint_name, is_sampler=True)
                    logger.info(f'Saving weights to {save_dir}')
                    self.model.save(
                        name=checkpoint_name, output_dir=save_dir, adapter_name=adapter_name, save_optimizer=False)
                    payload = body.model_dump()
                    payload['model_path'] = tinker_path
                    metadata = self.state.get_model_metadata(body.model_id) or {}
                    if metadata.get('base_model'):
                        payload['base_model'] = metadata['base_model']
                    sampling_session_id = self.state.create_sampling_session(payload)
                    return types.SaveWeightsForSamplerResponseInternal(
                        path=None, sampling_session_id=sampling_session_id)
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(
                _do_save_for_sampler, model_id=body.model_id, token=token, task_type='save_weights_for_sampler')

        @app.post('/tinker/load_weights')
        async def load_weights(self, request: Request, body: types.LoadWeightsRequest) -> types.UntypedAPIFuture:
            token = await self._on_request_start(request)

            async def _do_load():
                try:
                    assert self.model is not None, 'Model not loaded, please load model first'
                    adapter_name = self.get_adapter_name(adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)
                    self.touch_adapter(adapter_name)
                    self.model.load(
                        checkpoint_dir=body.path, load_optimizer=body.optimizer, adapter_name=adapter_name, token=token)
                    self.set_adapter_state(adapter_name, 'grad_ready', False)
                    return types.LoadWeightsResponse(path=body.path, type='load_weights')
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await self.schedule_task(_do_load, model_id=body.model_id, token=token, task_type='load_weights')

    # Tinker uses {request_id}-{adapter_name} prefix via self.get_adapter_name()
    # which is inherited from AdapterManagerMixin (no-op here; method kept for clarity).
    @staticmethod
    def get_adapter_name(adapter_name: Any) -> Any:
        """Returns adapter_name as-is; overridden by AdapterManagerMixin in the combined class."""
        return adapter_name
