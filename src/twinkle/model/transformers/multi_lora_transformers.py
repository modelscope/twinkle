# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch.distributed as dist
import transformers
from peft import LoraConfig, PeftConfig, PeftModel, load_peft_weights
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

from twinkle import DeviceMesh, remote_class, remote_function, template
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.infra import collect_tensor_dict
from twinkle.loss import Loss
from twinkle.metric import Metric
from twinkle.processor import InputProcessor
from ..multi_lora import MultiLora
from .transformers import OptimizerGroup, TransformersModel


@remote_class()
class MultiLoraTransformersModel(TransformersModel, PreTrainedModel):

    def __init__(
            self,  # noqa
            model_cls=None,
            model_id: Optional[str] = None,
            config: Optional[PretrainedConfig] = None,
            device_mesh: Optional[DeviceMesh] = None,
            mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
            strategy: Literal['accelerate', 'native_fsdp'] = 'accelerate',
            ddp_config: Dict[str, Any] = None,
            fsdp_config: Dict[str, Any] = None,
            grad_scaler_config: Dict[str, Any] = None,
            memory_efficient_init: bool = False,
            max_loras: int = 5,
            max_r: int = 32,
            max_length: int = 8192,
            target_modules: Union[List[str], str] = 'all-linear',
            **kwargs):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self._try_init_process_group()
        super(PreTrainedModel, self).__init__()
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self._fsdp_config = dict(fsdp_config or {})
        self._ddp_config = ddp_config or {}
        self._memory_efficient_init = memory_efficient_init
        self._decide_strategy(strategy)
        self.grad_scaler_config = grad_scaler_config
        if model_id is not None:
            model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        if config is None:
            from transformers import AutoConfig
            self.hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        else:
            self.hf_config = config
        if model_cls is None and hasattr(self.hf_config, 'architectures'):
            model_cls = self.hf_config.architectures[0]
        if model_cls is None:
            model_cls = AutoModelForCausalLM
        if isinstance(model_cls, str):
            model_cls = getattr(transformers, model_cls)
        if model_id is None:
            self.model = model_cls.from_config(self.hf_config, **kwargs)
        else:
            with self.strategy.pretrained_load_context():
                self.model = model_cls.from_pretrained(model_id, config=self.hf_config, **kwargs)
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        self._default_tokenizer = None
        self._model_wrapped = False
        self.sp_strategy = None
        # Initialize expert parallel attributes (required by set_optimizer in TransformersModel)
        self.optimizer_group: Dict[str, OptimizerGroup] = {}
        self.multi_adapter = MultiLora(max_loras=max_loras, max_r=max_r, max_length=max_length)
        self.model.gradient_checkpointing_enable()
        self.model = self.multi_adapter.patch(self.model, target_modules=target_modules)
        self.multi_adapter.save_initial_weights()
        # Active group for compatibility with single adapter
        self.active_group = None
        self.handler = self.register_global_mm_forward_hook()
        self.multi_adapter.reset_adapter_status()

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name and adapter_name in self.optimizer_group, (f'Use a valid adapter_name first, '
                                                                       f'current is: {adapter_name}')

    def register_global_mm_forward_hook(self):

        def forward_hook(model, args, kwargs):
            active_adapter = model.active_adapters[0]
            active_adapter = self.multi_adapter.find_lora(active_adapter).tenant_adapter_name
            optimizer_group = self.optimizer_group[active_adapter]
            template = optimizer_group.template
            assert template is not None
            return template.pre_forward_hook(model, args, kwargs)

        model = self.strategy.unwrap_model(self.model)
        return model.register_forward_pre_hook(forward_hook, with_kwargs=True)

    def register_mm_forward_hook(self, optimizer_group: OptimizerGroup):
        pass

    def unregister_mm_forward_hook(self, optimizer_group: OptimizerGroup):
        pass

    def _lazy_wrap_model(self):
        return super()._lazy_wrap_model()

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        optimizer_config = self.optimizer_group[kwargs.get('adapter_name')]
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        self.multi_adapter.check_length(inputs)
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            return super().forward(inputs=inputs, **kwargs)

    @remote_function(dispatch='slice_dp', collect=collect_tensor_dict)
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        adapter_name = kwargs.get('adapter_name')
        disable_lora = kwargs.get('disable_lora', False)
        self._check_adapter_valid(adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        self.multi_adapter.check_length(inputs)
        with self.multi_adapter.adapter(adapter_name, disable_lora=disable_lora):
            return super().forward_only(inputs=inputs, **kwargs)

    @remote_function(collect='mean')
    def calculate_loss(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            return super().calculate_loss(**kwargs)

    @remote_function()
    def backward(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            super().backward(**kwargs)

    @remote_function()
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            return super().clip_grad_norm(max_grad_norm, norm_type=norm_type, **kwargs)

    def _create_param_group(self, adapter_name: str, lr: float = 1e-5, weight_decay: float = 0.01, **kwargs):
        return super()._create_param_group(adapter_name=adapter_name, lr=lr, weight_decay=weight_decay, **kwargs)

    @remote_function()
    def step(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            super().step(**kwargs)

    @remote_function()
    def zero_grad(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            super().zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            super().lr_step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().set_loss(loss_cls, **kwargs)

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.adapter(kwargs.get('adapter_name')):
            super().set_optimizer(optimizer_cls, **kwargs)

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        # prevent opening requires_grad of the base model
        # prevent loading malicious code
        assert not isinstance(
            config_or_dir, str
        ), 'config_or_dir does not support str, because loading config from modelhub may causing unexpected behavior'
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
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().set_lr_scheduler(scheduler_cls, **kwargs)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().set_template(template_cls, **kwargs)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, Callable], **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().set_processor(processor_cls, **kwargs)

    @remote_function(collect='first')
    def get_state_dict(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        return self.multi_adapter.get_state_dict(kwargs.get('adapter_name'))

    @remote_function(collect='first')
    def save(self, name, output_dir: Optional[str] = None, interval=1, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        with self.multi_adapter.save_context(kwargs.get('adapter_name')):
            checkpoint_dir = super().save(name, output_dir, interval, **kwargs)
        if dist.is_initialized():
            dist.barrier()
        return checkpoint_dir

    @remote_function()
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        adapter_name = kwargs.get('adapter_name')
        self._check_adapter_valid(adapter_name)
        with self.multi_adapter.save_context(kwargs.get('adapter_name')):
            load_optimizer = kwargs.get('load_optimizer', False)
            if output_dir is None:
                if os.path.exists(name):
                    checkpoint_dir = name
                else:
                    token = kwargs.pop('token', None)
                    checkpoint_dir = HubOperation.download_model(name, token=token)
            else:
                checkpoint_dir = os.path.join(output_dir, name)
            model = self.strategy.unwrap_model(self.model)
            if isinstance(model, PeftModel):
                # Load to CPU to avoid safetensors device issues in Ray environment
                adapter_weights = load_peft_weights(checkpoint_dir, device='cpu')
                self.multi_adapter.set_state_dict(adapter_name, adapter_weights)

            if load_optimizer:
                self._restore_training_state(checkpoint_dir, adapter_name=adapter_name)
        if dist.is_initialized():
            dist.barrier()

    @remote_function()
    def set_grad_scaler(self, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().set_grad_scaler(**kwargs)

    @remote_function()
    def add_metric(self, metric_cls: Union[Metric, str], is_training: Optional[bool] = None, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        super().add_metric(metric_cls, is_training, **kwargs)

    @remote_function(collect='first', lazy_collect=False)
    def calculate_metric(self, is_training, **kwargs):
        self._check_adapter_valid(kwargs.get('adapter_name'))
        return super().calculate_metric(is_training, **kwargs)

    @remote_function()
    def remove_adapter(self, adapter_name: str):
        if adapter_name in self.optimizer_group:
            self.optimizer_group.pop(adapter_name)
        self.multi_adapter.release_lora(adapter_name)

    def _get_nb_trainable_parameters(self, adapter_name, model):
        with self.multi_adapter.adapter(adapter_name):
            return self.multi_adapter.get_nb_trainable_parameters(adapter_name)

    def _get_trainable_parameters_example(self, adapter_name, model):
        with self.multi_adapter.adapter(adapter_name):
            return self.multi_adapter.get_trainable_parameters_example(adapter_name)

    def _get_trainable_parameters(self, adapter_name):
        with self.multi_adapter.adapter(adapter_name) as real_adapter_name:
            return super()._get_trainable_parameters(real_adapter_name)
