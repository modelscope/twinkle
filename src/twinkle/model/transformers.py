import re
from dataclasses import dataclass
from typing import Dict, Any
from typing import overload, Type, Optional, Union, Callable

from peft import PeftConfig
from peft import get_peft_model, PeftModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig

import twinkle
from twinkle import remote_class, remote_function, template
from twinkle.loss.base import Loss
from twinkle.loss.base import Loss
from .base import TwinkleModel
from twinkle.patch import MultiAdapter
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils.plugin import Plugin


@dataclass
class OptimizerGroup:
    adapter_name: str = None
    adapter_config: PeftConfig = None
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None
    inputs: Dict[str, Any] = None
    outputs: Dict[str, Any] = None
    loss_instance: Loss = None
    loss_value: Any = None
    template: Template = None
    processor: InputProcessor = None


@remote_class()
class TransformersModel(TwinkleModel, PreTrainedModel):

    _default_adapter_name = ''

    @overload
    def __init__(self, *, model_cls: Type[PreTrainedModel], config: PretrainedConfig, remote_group, **kwargs) -> None:
        ...

    @overload
    def __init__(self, *, pretrained_model_name_or_path: str, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        ...

    def __init__(self, # noqa
                 model_cls: Optional[Type[PreTrainedModel]] = None,
                 pretrained_model_name_or_path: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 **kwargs):
        if pretrained_model_name_or_path is None:
            self.model = model_cls(config, **kwargs)
        elif model_cls:
            self.model = model_cls.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        self.model_id = pretrained_model_name_or_path
        self.model: PreTrainedModel = MultiAdapter()(self.model) # patch multiple loras
        self.optimizer_group: Dict[str, OptimizerGroup] = {self._default_adapter_name: OptimizerGroup()}

    @remote_function()
    def forward(self, *, inputs: Dict[str, Any], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        processor: InputProcessor = self.optimizer_group[adapter_name].processor
        assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
        inputs: Dict[str, Any] = processor(inputs)
        if adapter_name:
            self.model.set_current_adapter_name(adapter_name)
        outputs = self.model(**inputs)
        if adapter_name:
            self.optimizer_group[adapter_name].inputs = inputs
            self.optimizer_group[adapter_name].outputs = outputs

    @remote_function()
    def forward_only(self, *, inputs: Dict[str, Any], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        import torch
        with torch.no_grad():
            processor: InputProcessor = self.optimizer_group[adapter_name].processor
            assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
            inputs: Dict[str, Any] = processor(inputs)
            if adapter_name:
                self.model.set_current_adapter_name(adapter_name)
            outputs = self.model(**inputs)
        if adapter_name:
            self.optimizer_group[adapter_name].inputs = inputs
            self.optimizer_group[adapter_name].outputs = outputs

    @remote_function()
    def calculate_loss(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        loss_instance = self.optimizer_group[adapter_name].loss_instance
        assert isinstance(loss_instance, Loss), 'Set loss_instance correctly before forwarding'
        inputs = self.optimizer_group[adapter_name].inputs
        outputs = self.optimizer_group[adapter_name].outputs
        assert inputs is not None and outputs is not None, 'Cannot calculate loss of null inputs and outputs'
        loss_value = loss_instance(inputs, outputs, **kwargs)
        self.optimizer_group[adapter_name].loss_value = loss_value
        return loss_value

    @remote_function()
    def backward(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        loss_value = self.optimizer_group[adapter_name].loss_value
        assert loss_value is not None, 'Forward and calculate loss before backward pass.'
        if adapter_name:
            self.model.set_current_adapter_name(adapter_name)
        loss_value.backward()

    @remote_function()
    def forward_backward(self, *, inputs: Dict[str, Any], **kwargs):
        self.forward(inputs=inputs, **kwargs)
        self.calculate_loss(**kwargs)
        self.backward(**kwargs)

    @remote_function()
    def step(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        optimizer = self.optimizer_group[adapter_name].optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'
        optimizer.step()

    @remote_function()
    def zero_grad(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        optimizer = self.optimizer_group[adapter_name].optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'
        optimizer.zero_grad()

    @remote_function()
    def lr_step(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        lr_scheduler = self.optimizer_group[adapter_name].lr_scheduler
        assert isinstance(lr_scheduler, LRScheduler), 'Set lr_scheduler correctly before forwarding'
        lr_scheduler.step()

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        if isinstance(loss_cls, str):
            if hasattr(twinkle.loss, loss_cls):
                loss_cls = getattr(twinkle.loss, loss_cls)
            else:
                loss_cls = Plugin.load_plugin(loss_cls, Loss)
        self.optimizer_group[adapter_name].loss_instance = loss_cls()

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        if isinstance(optimizer_cls, str):
            import torch
            if hasattr(torch.optim, optimizer_cls):
                optimizer_cls = getattr(torch.optim, optimizer_cls)
            else:
                optimizer_cls = Plugin.load_plugin(optimizer_cls, Optimizer)
        self.optimizer_group[adapter_name].optimizer = optimizer_cls(self._get_trainable_parameters(
            adapter_name=adapter_name).values(), **kwargs)

    def _get_trainable_parameters(self, adapter_name=''):
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        is_default = adapter_name == self._default_adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and (pattern.search(name) or is_default):
                params[name] = param
        return params

    @remote_function()
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        if isinstance(scheduler_cls, str):
            import torch
            if hasattr(torch.optim.lr_scheduler, scheduler_cls):
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cls)
            else:
                scheduler_cls = Plugin.load_plugin(scheduler_cls, LRScheduler)
        optimizer = self.optimizer_group[adapter_name].optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before setting lr_scheduler'
        self.optimizer_group[adapter_name].lr_scheduler = scheduler_cls(optimizer, **kwargs)

    @remote_function(execute='first')
    def save_state_dict(self, output_dir, **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        if not adapter_name:
            self.model.save_pretrained(output_dir)
        else:
            state_dict = self.get_state_dict(adapter_name)
            self.model.save_pretrained(output_dir, state_dict=state_dict)

    @remote_function(execute='first')
    def get_state_dict(self, adapter_name: str = ''):
        return self._get_trainable_parameters(adapter_name=adapter_name)

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config: PeftConfig):
        assert adapter_name not in self.optimizer_group, f'{adapter_name} already exists.'
        assert adapter_name, 'Use a different adapter_name, current is empty.'
        self.optimizer_group[adapter_name] = OptimizerGroup()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config
        if isinstance(self.model, PeftModel):
            self.model.add_adapter(adapter_name, config)
        else:
            self.model = get_peft_model(self.model, config, adapter_name=adapter_name)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        if isinstance(template_cls, str):
            if hasattr(template, template_cls):
                template_cls = getattr(template, template_cls)
            else:
                template_cls = Plugin.load_plugin(template_cls, template.Template)
        self.optimizer_group[adapter_name].template = template_cls(self.model_id, **kwargs)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", '')
        assert adapter_name in self.optimizer_group, f'Add {adapter_name} first before training.'
        if isinstance(processor_cls, str):
            if hasattr(__file__.__module__, processor_cls):
                processor_cls = getattr(__file__.__module__, processor_cls)
            else:
                processor_cls = Plugin.load_plugin(processor_cls, InputProcessor)
        self.optimizer_group[adapter_name].processor = processor_cls(self.model_id, **kwargs)