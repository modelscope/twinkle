# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import threading
import time
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from peft import LoraConfig
from ray import serve
from tinker import types

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.server.twinkle.validation import (ConfigRegistryProxy,
                                               init_config_registry,
                                               verify_request_token)
from twinkle.utils.logger import get_logger

from .common import TwinkleCompatTransformersModel
from .common.io_utils import (TrainingRunManager, CheckpointManager)
from .state import get_server_state, schedule_task

logger = get_logger()


def build_model_app(nproc_per_node: int, device_group: Dict[str, Any],
                    device_mesh: Dict[str, Any],
                    deploy_options: Dict[str, Any], **kwargs):
    app = FastAPI()

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name='ModelManagement')
    @serve.ingress(app)
    class ModelManagement():

        COUNT_DOWN = 60 * 30

        def __init__(self, nproc_per_node: int, device_group: Dict[str, Any],
                     device_mesh: Dict[str, Any], **kwargs):
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(mode='ray',
                               nproc_per_node=nproc_per_node,
                               groups=[self.device_group],
                               lazy_collect=False)
            self.device_mesh = DeviceMesh(**device_mesh)
            self.model: Optional[TwinkleCompatTransformersModel] = None
            self.model_id: Optional[str] = None
            self.kwargs = kwargs
            self.adapter_records: Dict[str, int] = {}
            self.hb_thread = threading.Thread(target=self.countdown,
                                              daemon=True)
            self.hb_thread.start()
            self.adapter_lock = threading.Lock()
            self.config_registry: ConfigRegistryProxy = init_config_registry()
            self.state = get_server_state()
            self.per_token_model_limit = int(
                os.environ.get('TWINKLE_PER_USER_MODEL_LIMIT', 30))
            self.key_token_dict = {}

        def countdown(self):
            while True:
                time.sleep(1)
                if not self.model:
                    continue
                for key in list(self.adapter_records.keys()):
                    self.adapter_records[key] += 1
                    if self.adapter_records[key] > self.COUNT_DOWN:
                        with self.adapter_lock:
                            self.model.remove_adapter(key)
                        self.adapter_records.pop(key, None)
                        token = self.key_token_dict.pop(key, None)
                        if token:
                            self.handle_adapter_count(token, False)

        def handle_adapter_count(self, token: str, add: bool):
            user_key = token + '_' + 'model_adapter'
            cur_count: int = self.config_registry.get_config(user_key) or 0
            if add:
                if cur_count < self.per_token_model_limit:
                    self.config_registry.add_config(user_key, cur_count + 1)
                else:
                    raise RuntimeError(
                        f'Model adapter count limitation reached: {self.per_token_model_limit}'
                    )
            else:
                if cur_count > 0:
                    cur_count -= 1
                    self.config_registry.add_config(user_key, cur_count)
                if cur_count <= 0:
                    self.config_registry.pop(user_key)

        @staticmethod
        def get_adapter_name(request: Request, adapter_name: str) -> str:
            return adapter_name

        def assert_adapter_exists(self, adapter_name: str):
            assert adapter_name and adapter_name in self.adapter_records, f"Adapter {adapter_name} not found"

        def assert_adapter_valid(self, adapter_name: Optional[str]):
            assert adapter_name is None or adapter_name == '' or adapter_name in self.adapter_records, \
                f"Adapter {adapter_name} is invalid"

        @app.post('/create_model')
        async def create_model(
                self, request: Request,
                body: types.CreateModelRequest) -> types.UntypedAPIFuture:
            # In case create_model called multiple times, we reuse the same model_id
            model_id = self.model_id or self.state.register_model(
                body.model_dump())

            self.model_id = model_id

            async def _load_model():
                if self.model is not None and self.model_id is not None:
                    return types.CreateModelResponse(model_id=self.model_id)

                extra_kwargs = body.user_metadata or {}
                self.model = TwinkleCompatTransformersModel(
                    model_id=body.base_model,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    **extra_kwargs,
                )

                if body.lora_config:
                    # TODO: support more lora config parameters, train_unembed, etc.
                    lora_cfg = LoraConfig(r=body.lora_config.rank, )

                    with self.adapter_lock:
                        adapter_name = self.get_adapter_name(
                            request=request, adapter_name=model_id)
                        self.model.add_adapter_to_model(
                            adapter_name=adapter_name, config_or_dir=lora_cfg)
                        self.adapter_records[adapter_name] = 0
                        self.key_token_dict[adapter_name] = request.state.token
                        self.handle_adapter_count(request.state.token, True)
                    self.model.set_processor('InputProcessor',
                                             adapter_name=adapter_name)
                    self.model.set_loss('CrossEntropyLoss',
                                        adapter_name=adapter_name)
                    self.model.set_optimizer('AdamW',
                                             adapter_name=adapter_name)

                TrainingRunManager.save(model_id, body, request.state.token)

                return types.CreateModelResponse(model_id=model_id)

            return await schedule_task(self.state,
                                       _load_model(),
                                       model_id=model_id)

        @app.post('/get_info')
        async def get_info(
                self, request: Request,
                body: types.GetInfoRequest) -> types.GetInfoResponse:
            metadata = TrainingRunManager.get(str(body.model_id))
            model_name = metadata.base_model if metadata else str(
                body.model_id)
            lora_rank = None
            is_lora = False
            if metadata and metadata.get('lora_config'):
                lora_rank = metadata['lora_config'].get('rank')
                is_lora = True
            return types.GetInfoResponse(
                model_data=types.ModelData(model_name=model_name),
                model_id=body.model_id,
                is_lora=is_lora,
                lora_rank=lora_rank,
                model_name=model_name,
            )

        @app.post('/unload_model')
        async def unload_model(
                self, request: Request,
                body: types.UnloadModelRequest) -> types.UntypedAPIFuture:

            async def _do_unload():
                self.model = None
                self.model_id = None
                with self.adapter_lock:
                    adapter_name = self.get_adapter_name(
                        request=request, adapter_name=body.model_id)
                    del self.adapter_records[adapter_name]
                    del self.key_token_dict[adapter_name]
                    self.handle_adapter_count(request.state.token, add=False)
                self.state.unload_model(body.model_id)
                return types.UnloadModelResponse(model_id=body.model_id)

            return await schedule_task(self.state,
                                       _do_unload(),
                                       model_id=body.model_id)

        @app.post('/forward')
        async def forward(
                self, request: Request,
                body: types.ForwardRequest) -> types.UntypedAPIFuture:

            async def _do_forward():
                if not self.model:
                    return types.RequestFailedResponse(
                        error='Model not loaded, please load model first',
                        category=types.RequestErrorCategory.User,
                    )

                try:
                    adapter_name = self.get_adapter_name(
                        request, adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    datum_list = body.forward_input.data
                    loss_fn_config = body.forward_input.loss_fn_config or {}

                    output = self.model.forward_only(inputs=datum_list,
                                                     adapter_name=adapter_name)
                    loss = self.model.calculate_loss(adapter_name=adapter_name,
                                                     **loss_fn_config)
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

            return await schedule_task(self.state,
                                       _do_forward(),
                                       model_id=body.model_id)

        @app.post('/forward_backward')
        async def forward_backward(
                self, request: Request,
                body: types.ForwardBackwardRequest) -> types.UntypedAPIFuture:

            async def _do_forward_backward():
                if not self.model:
                    return types.RequestFailedResponse(
                        error='Model not loaded, please load model first',
                        category=types.RequestErrorCategory.User,
                    )

                try:
                    adapter_name = self.get_adapter_name(
                        request, adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    datum_list = body.forward_backward_input.data
                    loss_fn_config = body.forward_backward_input.loss_fn_config or {}

                    output = self.model.forward(inputs=datum_list,
                                                adapter_name=adapter_name)
                    loss = self.model.calculate_loss(adapter_name=adapter_name,
                                                     **loss_fn_config)
                    self.model.backward(adapter_name=adapter_name)
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

            return await schedule_task(self.state,
                                       _do_forward_backward(),
                                       model_id=body.model_id)

        @app.post('/optim_step')
        async def optim_step(
                self, request: Request,
                body: types.OptimStepRequest) -> types.UntypedAPIFuture:

            async def _do_optim():
                try:
                    if not self.model:
                        return types.RequestFailedResponse(
                            error='Model not loaded, please load model first',
                            category=types.RequestErrorCategory.User,
                        )
                    adapter_name = self.get_adapter_name(
                        request, adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    self.model.step(adam_params=body.adam_params,
                                    adapter_name=adapter_name)
                    return types.OptimStepResponse(metrics=None)
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await schedule_task(self.state,
                                       _do_optim(),
                                       model_id=body.model_id)

        @app.post('/save_weights')
        async def save_weights(
                self, request: Request,
                body: types.SaveWeightsRequest) -> types.UntypedAPIFuture:

            async def _do_save():
                try:
                    if self.model is None or self.model_id is None:
                        return types.RequestFailedResponse(
                            error='Model not loaded, please load model first',
                            category=types.RequestErrorCategory.User,
                        )

                    adapter_name = self.get_adapter_name(
                        request, adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)

                    # get save dir
                    checkpoint_name = CheckpointManager.get_ckpt_name(
                        body.path)
                    save_dir = CheckpointManager.get_save_dir(
                        model_id=self.model_id,
                        is_sampler=False
                    )

                    self.model.save(name=checkpoint_name,
                                    output_dir=save_dir,
                                    adapter_name=adapter_name,
                                    save_optimizer=True)

                    tinker_path = CheckpointManager.save(
                        self.model_id, name=checkpoint_name, is_sampler=False)

                    return types.SaveWeightsResponse(path=tinker_path,
                                                     type='save_weights')
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await schedule_task(self.state,
                                       _do_save(),
                                       model_id=body.model_id)

        @app.post('/load_weights')
        async def load_weights(
                self, request: Request,
                body: types.LoadWeightsRequest) -> types.UntypedAPIFuture:

            async def _do_load():
                try:
                    adapter_name = self.get_adapter_name(
                        request, adapter_name=body.model_id)
                    self.assert_adapter_exists(adapter_name=adapter_name)
                    weight_path = body.path
                    load_optimizer = body.optimizer
                    self.model.load(checkpoint_dir=weight_path,
                                    load_optimizer=load_optimizer,
                                    adapter_name=adapter_name)
                    return types.LoadWeightsResponse(path=body.path,
                                                     type='load_weights')
                except Exception:
                    logger.error(traceback.format_exc())
                    return types.RequestFailedResponse(
                        error=traceback.format_exc(),
                        category=types.RequestErrorCategory.Server,
                    )

            return await schedule_task(self.state,
                                       _do_load(),
                                       model_id=body.model_id)

    return ModelManagement.options(**deploy_options).bind(
        nproc_per_node, device_group, device_mesh, **kwargs)
