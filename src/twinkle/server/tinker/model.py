# Copyright (c) ModelScope Contributors. All rights reserved.
from datetime import datetime
import os
import threading
import time
from typing import Dict, Any, Optional
import uuid

from fastapi import FastAPI, Request
from peft import LoraConfig
from pydantic import BaseModel
from ray import serve
from tinker import types

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.model import TwinkleModel, MultiLoraTransformersModel
from twinkle.model.base import TwinkleModel
from twinkle.data_format import InputFeature, Trajectory
from twinkle.server.twinkle.validation import verify_request_token, init_config_registry, ConfigRegistryProxy
from .state import get_server_state, schedule_task

def build_model_app(nproc_per_node: int,
                    device_group: Dict[str, Any],
                    device_mesh: Dict[str, Any],
                    deploy_options: Dict[str, Any],
                    **kwargs):
    app = FastAPI()

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name="ModelManagement")
    @serve.ingress(app)
    class ModelManagement():

        COUNT_DOWN = 60 * 30

        def __init__(self, nproc_per_node: int, device_group: Dict[str, Any], device_mesh: Dict[str, Any], **kwargs):
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(mode='ray', nproc_per_node=nproc_per_node, groups=[self.device_group], lazy_collect=False)
            self.device_mesh = DeviceMesh(**device_mesh)
            self.model: TwinkleModel = None
            self.kwargs = kwargs
            self.adapter_records: Dict[str, int] = {}
            self.hb_thread = threading.Thread(target=self.countdown, daemon=True)
            self.hb_thread.start()
            self.adapter_lock = threading.Lock()
            self.config_registry: ConfigRegistryProxy = init_config_registry()
            self.state = get_server_state()
            self.per_token_model_limit = int(os.environ.get("TWINKLE_PER_USER_MODEL_LIMIT", 3))
            self.key_token_dict = {}

        def countdown(self):
            while True:
                time.sleep(1)
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
            cur_count = self.config_registry.get_config(user_key) or 0
            if add:
                if cur_count < self.per_token_model_limit:
                    self.config_registry.add_config(user_key, cur_count + 1)
                else:
                    raise RuntimeError(f'Model adapter count limitation reached: {self.per_token_model_limit}')
            else:
                if cur_count > 0:
                    cur_count -= 1
                    self.config_registry.add_config(user_key, cur_count)
                if cur_count <= 0:
                    self.config_registry.pop(user_key)
        
        @staticmethod
        def get_adapter_name(request: Request, adapter_name: Optional[str]) -> Optional[str]:
            if adapter_name is None or adapter_name == '':
                return None
            return request.state.request_id + '-' + adapter_name
        
        def assert_adapter_exists(self, adapter_name: str):
            assert adapter_name and adapter_name in self.adapter_records, f"Adapter {adapter_name} not found"

        def assert_adapter_valid(self, adapter_name: Optional[str]):
            assert adapter_name is None or adapter_name == '' or adapter_name in self.adapter_records, \
                f"Adapter {adapter_name} is invalid"

        @app.post("/create_model")
        async def create_model(self, request: Request, body: types.CreateModelRequest) -> types.UntypedAPIFuture:
            model_id = await self.state.register_model(body)

            async def _load_model():
                self.model = MultiLoraTransformersModel(
                    model_id=body.base_model,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    **body.user_metadata,
                )
                if body.lora_config:
                    # TODO: support more lora config parameters
                    lora_cfg = LoraConfig(
                        r=body.lora_config.rank,
                    )
                    with self.adapter_lock:
                        adapter_name = self.get_adapter_name(request=request, adapter_name=model_id)
                        self.model.add_adapter_to_model(adapter_name=adapter_name, config_or_dir=lora_cfg)
                    self.adapter_records[adapter_name] = 0
                    self.key_token_dict[adapter_name] = request.state.token
                    self.handle_adapter_count(request.state.token, True)
                return types.CreateModelResponse(model_id=model_id)

            return await schedule_task(self.state, _load_model(), model_id=model_id)

        @app.post("/get_info")
        async def get_info(self, request: Request, body: types.GetInfoRequest) -> types.GetInfoResponse:
            metadata = await self.state.get_model_metadata(str(body.model_id))
            model_name = metadata.get("base_model") if metadata else str(body.model_id)
            lora_rank = None
            is_lora = False
            if metadata and metadata.get("lora_config"):
                lora_rank = metadata["lora_config"].get("rank")
                is_lora = True
            return types.GetInfoResponse(
                model_data=types.ModelData(model_name=model_name),
                model_id=body.model_id,
                is_lora=is_lora,
                lora_rank=lora_rank,
                model_name=model_name,
            )

        @app.post("/unload_model")
        async def unload_model(self, request: Request, body: types.UnloadModelRequest) -> types.UntypedAPIFuture:
            async def _do_unload():
                await self.state.unload_model(body.model_id)
                return types.UnloadModelResponse(model_id=body.model_id)

            return await schedule_task(self.state, _do_unload(), model_id=body.model_id)


        @app.post("/forward_backward")
        async def forward_backward(self, request: Request, body: types.ForwardBackwardRequest) -> types.UntypedAPIFuture:
            async def _do_forward_backward():
                adapter_name = self.get_adapter_name(request, adapter_name=body.model_id)
                self.assert_adapter_exists(adapter_name=adapter_name)
                
                datum_list = body.forward_backward_input.data
                loss_fn = body.forward_backward_input.loss_fn
                loss_fn_config = body.forward_backward_input.loss_fn_config
                
                if isinstance(inputs, list):
                    _input = inputs[0]
                    if 'input_ids' in _input:
                        inputs = [InputFeature(**_input) for _input in inputs]
                    else:
                        inputs = [Trajectory(**_input) for _input in inputs]
                else:
                    assert isinstance(inputs, dict)
                    inputs = InputFeature(**inputs) if 'input_ids' in inputs else Trajectory(**inputs)
                ret = self.model.forward_backward(inputs=inputs, adapter_name=adapter_name)
                return types.F(
                    loss_fn_output_type="TensorData",
                    loss_fn_outputs=[loss_output],
                    metrics={"loss:avg": 0.0, "tokens_per_second:avg": 0.0},
                )

            return await schedule_task(
                self.state, _do_forward_backward(), model_id=body.model_id
            )

        @app.post("/optim_step")
        async def optim_step(self, request: Request, body: types.OptimStepRequest) -> types.UntypedAPIFuture:
            async def _do_optim():
                return types.OptimStepResponse(
                    metrics={
                        "grad_norm": 0.0,
                        "weight_norm": 0.0,
                        "update_norm": 0.0,
                    }
                )

            return await schedule_task(self.state, _do_optim(), model_id=body.model_id)

        @app.post("/save_weights")
        async def save_weights(self, request: Request, body: types.SaveWeightsRequest) -> types.UntypedAPIFuture:
            async def _do_save():
                suffix = body.path or f"checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                path = f"tinker://{body.model_id}/{suffix}"
                return types.SaveWeightsResponse(path=path)

            return await schedule_task(self.state, _do_save(), model_id=body.model_id)

        @app.post("/load_weights")
        async def load_weights(self, request: Request, body: types.LoadWeightsRequest) -> types.UntypedAPIFuture:
            async def _do_load():
                return types.LoadWeightsResponse(path=body.path)

            return await schedule_task(self.state, _do_load(), model_id=body.model_id)

    return ModelManagement.options(**deploy_options).bind(nproc_per_node, device_group, device_mesh, **kwargs)