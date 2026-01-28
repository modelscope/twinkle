# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import Dict, Any, List, Literal
from typing import Type, Optional, Union

from peft import PeftConfig, LoraConfig, load_peft_weights, PeftModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM

from twinkle import remote_class, remote_function, template, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss
from twinkle.metric import Metric
from twinkle.processor import InputProcessor
from .strategy import AccelerateStrategy
from .transformers import TransformersModel, OptimizerGroup
from ..multi_lora import MultiLora


@remote_class()
class MultiLoraTransformersModel(TransformersModel, PreTrainedModel):

    def __init__(self,  # noqa
                 model_cls = AutoModelForCausalLM,
                 model_id: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 grad_scaler_config: Dict[str, Any] = None,
                 **kwargs):
        assert device_mesh.fsdp_world_size <= 0, f'MultiLora does not support FSDP, current is: {str(device_mesh)}'
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        super(PreTrainedModel, self).__init__()
        model_id = HubOperation.download_model(model_id)
        self.model = model_cls.from_pretrained(model_id, config=config, **kwargs)
        self.model_id = model_id
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self.grad_scaler_config = grad_scaler_config
        self._model_wrapped = False
        self.optimizer_group: Dict[str, OptimizerGroup] = {}
        self.multi_adapter = MultiLora()
        self.model = self.multi_adapter.patch(self.model)
        self.strategy = AccelerateStrategy(mixed_precision=mixed_precision, device_mesh=None)
        self.model = self.strategy.wrap_model(self.model)
        self.multi_adapter.save_initial_weights()

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name and adapter_name in self.optimizer_group, f'Use a valid adapter_name first, current is: {adapter_name}'

    def _lazy_wrap_model(self):
        pass

    @remote_function(dispatch='slice_dp', collect='mean')
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        optimizer_config = self.optimizer_group[kwargs.get("adapter_name")]
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list) and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs) # noqa
        self.multi_adapter.check_length(inputs)
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().forward(inputs=inputs, **kwargs)

    @remote_function(dispatch='slice_dp', collect='flatten')
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        optimizer_config = self.optimizer_group[kwargs.get("adapter_name")]
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list) and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs) # noqa
        self.multi_adapter.check_length(inputs)
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().forward_only(inputs=inputs, **kwargs)

    @remote_function()
    def calculate_loss(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().calculate_loss(**kwargs)

    @remote_function()
    def backward(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            super().backward(**kwargs)

    @remote_function()
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            return super().clip_grad_norm(max_grad_norm, norm_type=norm_type, **kwargs)

    def _create_param_group(self, adapter_name: str, lr: float = 1e-5, weight_decay: float = 0.01, **kwargs):
        with self.multi_adapter.adapter(adapter_name) as adapter_name:
            return super()._create_param_group(adapter_name=adapter_name, lr=lr, weight_decay=weight_decay, **kwargs)

    @remote_function()
    def step(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            super().step(**kwargs)

    @remote_function()
    def zero_grad(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            super().zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            super().lr_step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().set_loss(loss_cls, **kwargs)

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.adapter(kwargs.get("adapter_name")):
            super().set_optimizer(optimizer_cls, **kwargs)

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        # prevent opening requires_grad of the base model
        # prevent loading malicious code
        assert not isinstance(config_or_dir,
                              str), 'config_or_dir does not support str, because loading config from modelhub may causing unexpected behavior'
        assert isinstance(config_or_dir, LoraConfig), 'config_or_dir must be a LoraConfig instance'
        # Limit the max peft version in pyproject.toml, in case any newer version opens some untested module grad.
        config_or_dir.modules_to_save = None
        config_or_dir.bias = 'none'
        config_or_dir.init_lora_weights = False
        config_or_dir.modules_to_save = None
        config_or_dir.trainable_token_indices = None
        self.optimizer_group[adapter_name] = self._construct_default_optimizer_group()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config_or_dir
        _gas_default = kwargs.get('gradient_accumulation_steps', 1)
        self.optimizer_group[adapter_name].gradient_accumulation_steps = _gas_default
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor
        self.multi_adapter.acquire_lora(tenant_adapter_name=adapter_name, config=config_or_dir)

    @remote_function()
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().set_lr_scheduler(scheduler_cls, **kwargs)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().set_template(template_cls, **kwargs)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().set_processor(processor_cls, **kwargs)

    @remote_function()
    def get_state_dict(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        return self.multi_adapter.get_state_dict(kwargs.get("adapter_name"))

    @remote_function()
    def save(self, name, output_dir=None, interval=1, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        with self.multi_adapter.save_context(kwargs.get("adapter_name")):
            return super().save(name, output_dir, interval, **kwargs)

    @remote_function()
    def load(self, name: Optional[str] = None, output_dir: Optional[str] = None, **kwargs):
        adapter_name = kwargs.get("adapter_name")
        self._check_adapter_valid(adapter_name)
        with self.multi_adapter.save_context(kwargs.get("adapter_name")):
            load_optimizer = kwargs.get('load_optimizer', False)
            if output_dir is None:
                output_dir = 'output'
            checkpoint_dir = os.path.join(output_dir, name)
            model = self.strategy.unwrap_model(self.model)
            if isinstance(model, PeftModel):
                # Load to CPU to avoid safetensors device issues in Ray environment
                adapter_weights = load_peft_weights(checkpoint_dir, device="cpu")
                self.multi_adapter.set_state_dict(adapter_name, adapter_weights)

            if load_optimizer:
                self._load_optimizer(checkpoint_dir, adapter_name=adapter_name)

    @remote_function()
    def set_grad_scaler(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().set_grad_scaler(**kwargs)

    def add_metric(self, metric_cls: Union[Metric, str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        super().add_metric(metric_cls, **kwargs)

    @remote_function()
    def remove_adapter(self, adapter_name: str):
        if adapter_name in self.optimizer_group:
            self.optimizer_group.pop(adapter_name)
        self.multi_adapter.release_lora(adapter_name)

    def _get_nb_trainable_parameters(self, adapter_name, model):
        return self.multi_adapter.get_nb_trainable_parameters(adapter_name)

    def _get_trainable_parameters_example(self, adapter_name, model):
        return self.multi_adapter.get_trainable_parameters_example(adapter_name)
