# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import threading
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request
from peft import LoraConfig
from pydantic import BaseModel
from ray import serve

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.model import MultiLoraTransformersModel, MultiLoraMegatronModel
from twinkle.model.base import TwinkleModel
from twinkle.data_format import InputFeature, Trajectory
from .common.validation import verify_request_token
from .common.state import get_server_state, ServerStateProxy
from .common.serialize import deserialize_object
from .common.io_utils import (
    CreateModelRequest,
    LoraConfig as IoLoraConfig,
    create_training_run_manager,
    create_checkpoint_manager,
)

# 请求体模型定义
class CreateRequest(BaseModel):
    class Config:
        extra = "allow"

class ForwardRequest(BaseModel):
    inputs: Any
    adapter_name: str

    class Config:
        extra = "allow"

class ForwardOnlyRequest(BaseModel):
    inputs: Any
    adapter_name: Optional[str] = None

    class Config:
        extra = "allow"

class AdapterRequest(BaseModel):
    adapter_name: str

    class Config:
        extra = "allow"

class SetLossRequest(BaseModel):
    loss_cls: str
    adapter_name: str

    class Config:
        extra = "allow"

class SetOptimizerRequest(BaseModel):
    optimizer_cls: str
    adapter_name: str

    class Config:
        extra = "allow"

class SetLrSchedulerRequest(BaseModel):
    scheduler_cls: str
    adapter_name: str

    class Config:
        extra = "allow"

class SaveRequest(BaseModel):
    adapter_name: str
    save_optimizer: bool = False
    name: Optional[str] = None

    class Config:
        extra = "allow"

class LoadRequest(BaseModel):
    adapter_name: str
    load_optimizer: bool = False
    name: Optional[str] = None

    class Config:
        extra = "allow"

class AddAdapterRequest(BaseModel):
    adapter_name: str
    config: str

    class Config:
        extra = "allow"

class SetTemplateRequest(BaseModel):
    template_cls: str
    adapter_name: str

    class Config:
        extra = "allow"

class SetProcessorRequest(BaseModel):
    processor_cls: str
    adapter_name: str

    class Config:
        extra = "allow"

class HeartbeatRequest(BaseModel):
    adapter_name: str

class CalculateMetricRequest(BaseModel):
    adapter_name: str
    is_training: bool = True

    class Config:
        extra = "allow"

class GetStateDictRequest(BaseModel):
    adapter_name: str

    class Config:
        extra = "allow"


def build_model_app(model_id: str,
                    nproc_per_node: int,
                    device_group: Dict[str, Any],
                    device_mesh: Dict[str, Any],
                    deploy_options: Dict[str, Any],
                    use_megatron: bool = False,
                    **kwargs):
    app = FastAPI()

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name="ModelManagement")
    @serve.ingress(app)
    class ModelManagement(TwinkleModel):

        COUNT_DOWN = 60 * 30

        def __init__(self, nproc_per_node: int, device_group: Dict[str, Any], device_mesh: Dict[str, Any]):
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(mode='ray', nproc_per_node=nproc_per_node, groups=[self.device_group], lazy_collect=False)
            self.device_mesh = DeviceMesh(**device_mesh)
            if use_megatron:
                self.model = MultiLoraMegatronModel(
                    model_id=model_id,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    **kwargs
                )
            else:
                self.model = MultiLoraTransformersModel(
                    model_id=model_id,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    **kwargs
                )
            self.adapter_records: Dict[str, int] = {}
            self.hb_thread = threading.Thread(target=self.countdown, daemon=True)
            self.hb_thread.start()
            self.adapter_lock = threading.Lock()
            self.state: ServerStateProxy = get_server_state()
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
            cur_count = self.state.get_config(user_key) or 0
            if add:
                if cur_count < self.per_token_model_limit:
                    self.state.add_config(user_key, cur_count + 1)
                else:
                    raise RuntimeError(f'Model adapter count limitation reached: {self.per_token_model_limit}')
            else:
                if cur_count > 0:
                    cur_count -= 1
                    self.state.add_config(user_key, cur_count)
                if cur_count <= 0:
                    self.state.pop_config(user_key)

        @app.post("/create")
        def create(self, request: Request, body: CreateRequest):
            return {'status': 'ok'}

        def assert_adapter_exists(self, adapter_name: str):
            assert adapter_name and adapter_name in self.adapter_records, f"Adapter {adapter_name} not found"

        def assert_adapter_valid(self, adapter_name: Optional[str]):
            assert adapter_name is None or adapter_name == '' or adapter_name in self.adapter_records, \
                f"Adapter {adapter_name} is invalid"

        @staticmethod
        def get_adapter_name(request: Request, adapter_name: Optional[str]) -> Optional[str]:
            if adapter_name is None or adapter_name == '':
                return None
            return request.state.request_id + '-' + adapter_name

        @app.post("/forward")
        def forward(self, request: Request, body: ForwardRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            inputs = body.inputs
            if isinstance(inputs, list):
                _input = inputs[0]
                if 'input_ids' in _input:
                    inputs = [InputFeature(**_input) for _input in inputs]
                else:
                    inputs = [Trajectory(**_input) for _input in inputs]
            else:
                assert isinstance(inputs, dict)
                inputs = InputFeature(**inputs) if 'input_ids' in inputs else Trajectory(**inputs)
            ret = self.model.forward(inputs=inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/forward_only")
        def forward_only(self, request: Request, body: ForwardOnlyRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            inputs = body.inputs
            if isinstance(inputs, list):
                _input = inputs[0]
                if 'input_ids' in _input:
                    inputs = [InputFeature(**_input) for _input in inputs]
                else:
                    inputs = [Trajectory(**_input) for _input in inputs]
            else:
                assert isinstance(inputs, dict)
                inputs = InputFeature(**inputs) if 'input_ids' in inputs else Trajectory(**inputs)
            ret = self.model.forward_only(inputs=inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/calculate_loss")
        def calculate_loss(self, request: Request, body: AdapterRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.calculate_loss(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/backward")
        def backward(self, request: Request, body: AdapterRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.backward(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/forward_backward")
        def forward_backward(self, request: Request, body: ForwardRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            inputs = body.inputs
            if isinstance(inputs, list):
                _input = inputs[0]
                if 'input_ids' in _input:
                    inputs = [InputFeature(**_input) for _input in inputs]
                else:
                    inputs = [Trajectory(**_input) for _input in inputs]
            else:
                assert isinstance(inputs, dict)
                inputs = InputFeature(**inputs) if 'input_ids' in inputs else Trajectory(**inputs)
            ret = self.model.forward_backward(inputs=inputs, adapter_name=adapter_name, **extra_kwargs)
            return {'result': str(ret)}

        @app.post("/get_train_configs")
        def get_train_configs(self, request: Request, body: AdapterRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.get_train_configs(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/clip_grad_norm")
        def clip_grad_norm(self, request: Request, body: AdapterRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.clip_grad_norm(adapter_name=adapter_name, **extra_kwargs)
            return {'result': str(ret)}

        @app.post("/step")
        def step(self, request: Request, body: AdapterRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.step(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/zero_grad")
        def zero_grad(self, request: Request, body: AdapterRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.zero_grad(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/lr_step")
        def lr_step(self, request: Request, body: AdapterRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.lr_step(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/set_loss")
        def set_loss(self, request: Request, body: SetLossRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_loss(body.loss_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/set_optimizer")
        def set_optimizer(self, request: Request, body: SetOptimizerRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_optimizer(body.optimizer_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/set_lr_scheduler")
        def set_lr_scheduler(self, request: Request, body: SetLrSchedulerRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_lr_scheduler(body.scheduler_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/save")
        def save(self, request: Request, body: SaveRequest):
            """
            Save adapter weights with token-based isolation.
            
            This endpoint:
            1. Saves adapter weights to token-specific directory
            2. Saves checkpoint metadata with ownership tracking
            
            Args:
                request: FastAPI request object (contains token in state)
                body: SaveRequest with adapter_name, name, and save_optimizer flag
                
            Returns:
                Dict with result containing the twinkle:// path to saved checkpoint
            """
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            
            # Extract token for directory isolation
            token = request.state.token
            checkpoint_manager = create_checkpoint_manager(token)
            
            # Get checkpoint name and save directory with token-based path
            checkpoint_name = checkpoint_manager.get_ckpt_name(body.name)
            save_dir = checkpoint_manager.get_save_dir(
                model_id=adapter_name,
                is_sampler=False
            )
            
            # Save the model weights
            self.model.save(
                name=checkpoint_name,
                output_dir=save_dir,
                adapter_name=adapter_name, 
                save_optimizer=body.save_optimizer, 
                **extra_kwargs
            )
            
            # Save checkpoint metadata
            twinkle_path = checkpoint_manager.save(
                adapter_name,
                name=checkpoint_name,
                is_sampler=False
            )
            
            return {'result': twinkle_path}
        
        @app.post("/load")
        def load(self, request: Request, body: LoadRequest):
            """
            Load adapter weights with token-based access validation.
            
            This endpoint:
            1. Validates user has access to the checkpoint
            2. Loads weights from token-specific directory
            
            Args:
                request: FastAPI request object (contains token in state)
                body: LoadRequest with adapter_name, name, and load_optimizer flag
                
            Returns:
                Dict with result indicating load status
            """
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            
            # Extract token for directory isolation
            token = request.state.token
            checkpoint_manager = create_checkpoint_manager(token)
            
            # Construct the checkpoint path and validate access
            if body.name:
                # Check if body.name is a twinkle:// path or a simple checkpoint name
                if body.name.startswith("twinkle://"):
                    # Parse twinkle:// path
                    parsed = checkpoint_manager.parse_twinkle_path(body.name)
                    if not parsed:
                        raise ValueError(f"Invalid twinkle path format: {body.name}")
                    
                    # Extract the checkpoint name from the parsed path
                    # parsed.checkpoint_id is like "weights/step-8"
                    checkpoint_id = parsed.checkpoint_id
                    checkpoint_name = parsed.checkpoint_id.split('/')[-1]  # Extract "step-8"
                    
                    # Use the training_run_id from the path as the model_id
                    model_id_to_load = parsed.training_run_id
                else:
                    # User provided a simple checkpoint name
                    checkpoint_id = f"weights/{body.name}"
                    checkpoint_name = body.name
                    model_id_to_load = adapter_name
                
                # Verify checkpoint exists and user has access
                checkpoint = checkpoint_manager.get(model_id_to_load, checkpoint_id)
                if not checkpoint:
                    raise ValueError(
                        f"Checkpoint not found or access denied: {body.name}"
                    )
                
                # Get the actual directory path
                output_dir = checkpoint_manager.get_save_dir(
                    model_id=model_id_to_load,
                    is_sampler=False
                )
                
                ret = self.model.load(
                    name=checkpoint_name,
                    output_dir=output_dir,
                    adapter_name=adapter_name, 
                    load_optimizer=body.load_optimizer, 
                    **extra_kwargs
                )
            else:
                # No checkpoint name provided - use default behavior
                ret = self.model.load(
                    name=body.name,
                    adapter_name=adapter_name, 
                    load_optimizer=body.load_optimizer, 
                    **extra_kwargs
                )
            
            return {'result': ret}

        @app.post("/add_adapter_to_model")
        def add_adapter_to_model(self, request: Request, body: AddAdapterRequest):
            """
            Add a new adapter to the model.
            
            This endpoint:
            1. Creates a new adapter with the specified configuration
            2. Registers it in the adapter tracking system
            3. Saves training run metadata with token-based isolation
            
            Args:
                request: FastAPI request object (contains token in state)
                body: AddAdapterRequest with adapter_name and config
                
            Returns:
                Dict with status and adapter_name
            """
            assert body.adapter_name, 'You need to specify a valid `adapter_name`'
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            config = deserialize_object(body.config)
            extra_kwargs = body.model_extra or {}
            
            # Extract token for metadata storage
            token = request.state.token
            training_run_manager = create_training_run_manager(token)
            
            with self.adapter_lock:
                self.model.add_adapter_to_model(adapter_name, config, **extra_kwargs)
            
            self.adapter_records[adapter_name] = 0
            self.key_token_dict[adapter_name] = token
            self.handle_adapter_count(token, True)
            
            # Save training run metadata (similar to tinker's create_model)
            # Create a training run config from the adapter configuration
            lora_config = None
            if isinstance(config, LoraConfig):
                lora_config = IoLoraConfig(
                    rank=config.r,
                    train_unembed=False,  # Default values
                    train_mlp=True,
                    train_attn=True
                )
            
            run_config = CreateModelRequest(
                base_model=model_id,  # Use the model_id from build_model_app
                lora_config=lora_config,
                user_metadata={'adapter_name': body.adapter_name}
            )
            
            # Save training run metadata with token-based isolation
            training_run_manager.save(adapter_name, run_config)
            
            return {'status': 'ok', 'adapter_name': adapter_name}

        @app.post("/set_template")
        def set_template(self, request: Request, body: SetTemplateRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_template(body.template_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/set_processor")
        def set_processor(self, request: Request, body: SetProcessorRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.set_processor(body.processor_cls, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/heartbeat")
        def heartbeat(self, request: Request, body: HeartbeatRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            self.adapter_records[adapter_name] = 0
            return {'status': 'ok'}

        @app.post("/calculate_metric")
        def calculate_metric(self, request: Request, body: CalculateMetricRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.calculate_metric(is_training=body.is_training, adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

        @app.post("/get_state_dict")
        def get_state_dict(self, request: Request, body: GetStateDictRequest):
            adapter_name = self.get_adapter_name(request, adapter_name=body.adapter_name)
            self.assert_adapter_exists(adapter_name=adapter_name)
            extra_kwargs = body.model_extra or {}
            ret = self.model.get_state_dict(adapter_name=adapter_name, **extra_kwargs)
            return {'result': ret}

    return ModelManagement.options(**deploy_options).bind(nproc_per_node, device_group, device_mesh)
