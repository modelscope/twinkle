# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import abstractmethod, ABC
from typing import Dict, Any, Union, Type, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from twinkle.loss.base import Loss
from twinkle.processor import InputProcessor
from twinkle.template import Template


class TwinkleModel(ABC):

    @abstractmethod
    def forward(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    @abstractmethod
    def forward_only(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    @abstractmethod
    def calculate_loss(self, **kwargs):
        ...

    @abstractmethod
    def backward(self, **kwargs):
        ...

    @abstractmethod
    def forward_backward(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    @abstractmethod
    def step(self, **kwargs):
        ...

    @abstractmethod
    def zero_grad(self, **kwargs):
        ...

    @abstractmethod
    def lr_step(self, **kwargs):
        ...

    @abstractmethod
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str], **kwargs):
        ...

    @abstractmethod
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str], **kwargs):
        ...

    @abstractmethod
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str], **kwargs):
        ...

    @abstractmethod
    def save(self, name: str, output_dir: str, interval: str = 1, **kwargs):
        ...

    @abstractmethod
    def load(self, name: Optional[str], output_dir: Optional[str] = None, **kwargs):
        ...

    @abstractmethod
    def get_state_dict(self, **kwargs):
        ...

    @abstractmethod
    def calculate_metric(self, is_training: bool, **kwargs):
        ...

    @abstractmethod
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        ...

    @abstractmethod
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        ...

    @abstractmethod
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs):
        ...

    @abstractmethod
    def get_train_configs(self, **kwargs):
        ...
