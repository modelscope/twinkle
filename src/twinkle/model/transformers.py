from typing import overload, Type, Optional, Union, Callable, Dict, Any

from peft import PeftConfig, get_peft_model, PeftModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig
import twinkle
from twinkle import remote_class, remote_function
from twinkle.processor import DataProcessorMixin
from twinkle.loss.base import Loss
from twinkle.utils.plugin import Plugin


@remote_class()
class TransformersModel(PreTrainedModel, DataProcessorMixin):

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
        self.loss_instance = None
        self.loss_value = None
        self.inputs: Optional[Dict[str, Any]] = None
        self.outputs = None
        self.optimizer = None
        self.lr_scheduler = None
        self.processor = None

    @remote_function()
    def forward(self, *, inputs: Dict[str, Any], adapter_name: str = None):
        self.inputs: Dict[str, Any] = self.processor(inputs)
        self.outputs = self.model(**self.inputs)

    @remote_function()
    def forward_only(self, *, inputs: Dict[str, Any], adapter_name: str = None):
        import torch
        with torch.no_grad():
            self.inputs: Dict[str, Any] = self.processor(inputs)
            self.outputs = self.model(**self.inputs)

    @remote_function()
    def calculate_loss(self, **kwargs):
        self.loss_value = self.loss_instance(self.input, self.output, **kwargs)
        return self.loss_value

    @remote_function()
    def backward(self):
        self.loss_value.backward()

    @remote_function()
    def forward_backward(self, *, inputs: Dict[str, Any], adapter_name: str, **kwargs):
        self.forward(inputs, adapter_name=adapter_name)
        self.calculate_loss(**kwargs)
        self.backward()

    @remote_function()
    def step(self):
        self.optimizer.step()

    @remote_function()
    def zero_grad(self):
        self.optimizer.zero_grad()

    @remote_function()
    def lr_step(self):
        self.lr_scheduler.step()

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str]):
        if isinstance(loss_cls, str):
            if hasattr(twinkle.loss, loss_cls):
                loss_cls = getattr(twinkle.loss, loss_cls)
            else:
                loss_cls = Plugin.load_plugin(loss_cls, Loss)
        self.loss_instance = loss_cls()

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        if isinstance(optimizer_cls, str):
            import torch
            if hasattr(torch.optim, optimizer_cls):
                optimizer_cls = getattr(torch.optim, optimizer_cls)
            else:
                optimizer_cls = Plugin.load_plugin(optimizer_cls, Optimizer)
        self.optimizer = optimizer_cls(self.model.trainable_parameters(), **kwargs)

    @remote_function()
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str], **kwargs):
        if isinstance(scheduler_cls, str):
            import torch
            if hasattr(torch.optim.lr_scheduler, scheduler_cls):
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cls)
            else:
                scheduler_cls = Plugin.load_plugin(scheduler_cls, LRScheduler)
        self.lr_scheduler = scheduler_cls(self.optimizer, **kwargs)

    def save_state(self, adapter_name: str):
        pass

    def add_adapter_to_model(self, adapter_name: str, config: Union[PeftConfig, Callable]):
        if isinstance(config, PeftConfig):
            if isinstance(self.model, PeftModel):
                self.model = self.model.add_adapter(adapter_name, config)
            else:
                self.model = get_peft_model(self.model, config)
        else:
            self.model = config(self.model)
