# Copyright (c) ModelScope Contributors. All rights reserved.
import re
import os
from contextlib import contextmanager
from typing import Dict, Any, List, Literal
from typing import Type, Optional, Union

from peft import PeftConfig, LoraConfig
from peft import PeftModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM
import torch.distributed as dist
from twinkle import remote_class, remote_function, template, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.loss import Loss
from twinkle.patch.multi_adapter import MultiAdapter
from twinkle.processor import InputProcessor
from .strategy import AccelerateStrategy
from .transformers import TransformersModel, OptimizerGroup
from twinkle.hub import HubOperation
from twinkle.metric import Metric


@remote_class()
class MultiLoraTransformersModel(TransformersModel, PreTrainedModel):
    DUMMY_ADAPTER_NAME = '__dummy_adapter__'

    def __init__(self,  # noqa
                 model_cls = AutoModelForCausalLM,
                 model_id: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 grad_scaler_config: Dict[str, Any] = None,
                 **kwargs):
        assert device_mesh.fsdp_world_size is None, f'MultiLora does not support FSDP, current is: {str(device_mesh)}'
        dist.init_process_group('nccl')
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
        self._inited = False

        self.multi_adapter = MultiAdapter()
        self.model: PreTrainedModel = self.multi_adapter.patch(self.model)
        with self._no_ddp_context():
            self.strategy = AccelerateStrategy(mixed_precision=mixed_precision, device_mesh=None)
            self.model = self.strategy.wrap_model(self.model)
        self.add_adapter_to_model(MultiLoraTransformersModel.DUMMY_ADAPTER_NAME, LoraConfig(r=1, target_modules='all-linear'))
        self._inited = True

    @contextmanager
    def _no_ddp_context(self):
        origin_local_rank = os.environ['LOCAL_RANK']
        origin_world_size = os.environ['WORLD_SIZE']
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '-1'
        yield
        os.environ['LOCAL_RANK'] = origin_local_rank
        os.environ['WORLD_SIZE'] = origin_world_size

    def _check_adapter_valid(self, adapter_name: str):
        if self._inited:
            assert adapter_name and adapter_name != MultiLoraTransformersModel.DUMMY_ADAPTER_NAME and adapter_name in self.optimizer_group, f'Use a valid adapter_name first, current is: {adapter_name}'

    def _activate_adapter(self, adapter_name: str):
        self.multi_adapter.set_current_adapter_name(adapter_name)

    def _lazy_wrap_model(self):
        pass

    @remote_function(dispatch='slice_dp', collect='mean')
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().forward(inputs=inputs, **kwargs)

    @remote_function(dispatch='slice_dp', collect='flatten')
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().forward_only(inputs=inputs, **kwargs)

    @remote_function()
    def calculate_loss(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().calculate_loss(**kwargs)

    @remote_function()
    def backward(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().backward(**kwargs)
        self._reduce_adapter_grad(adapter_name=kwargs.get("adapter_name"))

    def _reduce_adapter_grad(self, adapter_name: str):
        from torch.distributed.tensor import DTensor
        if adapter_name:
            import torch.distributed as dist
            for p in self._get_trainable_parameters(adapter_name).values():
                if p.grad is None:
                    continue

                grad = p.grad
                if isinstance(grad, DTensor):
                    full_grad = grad.full_tensor()
                    p.grad = full_grad
                    grad = full_grad

                if self.device_mesh.dp_world_size > 1:
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=self.optimizer_group[adapter_name]._dp_group)

    @remote_function()
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        return super().clip_grad_norm(max_grad_norm, norm_type=norm_type, **kwargs)

    @remote_function()
    def step(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().step(**kwargs)

    @remote_function()
    def zero_grad(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().lr_step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().set_loss(loss_cls, **kwargs)

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
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
        self._patch_adapter(adapter_name, config_or_dir, adapter_name, **kwargs)
        self._activate_adapter(adapter_name)
        self._prepare_adapter(adapter_name)

    def _prepare_adapter(self, adapter_name: str):
        self._check_adapter_valid(adapter_name)
        pattern = re.compile(r'\.lora_\w+\.[^.]+\.')
        unwrapped_model = self.strategy.unwrap_model(self.model)
        for name, param in unwrapped_model.named_parameters():
            if pattern.search(name):
                param.requires_grad = True

    @remote_function()
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().set_lr_scheduler(scheduler_cls, **kwargs)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().set_template(template_cls, **kwargs)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().set_processor(processor_cls, **kwargs)

    @remote_function()
    def set_grad_scaler(self, **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().set_grad_scaler(**kwargs)

    def add_metric(self, metric_cls: Union[Metric, str], **kwargs):
        self._check_adapter_valid(kwargs.get("adapter_name"))
        self._activate_adapter(kwargs.get("adapter_name"))
        super().add_metric(metric_cls, **kwargs)

    @remote_function()
    def remove_adapter(self, adapter_name: str):
        if adapter_name in self.optimizer_group:
            self.optimizer_group.pop(adapter_name)
        model = self.strategy.unwrap_model(self.model)
        if isinstance(model, PeftModel):
            model.base_model.delete_adapter(adapter_name=adapter_name)
