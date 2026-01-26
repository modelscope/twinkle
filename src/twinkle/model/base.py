# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Any, Union, Type

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from twinkle.loss.base import Loss
from twinkle.processor import InputProcessor
from twinkle.template import Template


class TwinkleModel:

    def forward(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    def forward_only(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    def calculate_loss(self, **kwargs):
        ...

    def backward(self, **kwargs):
        ...

    def forward_backward(self, *, inputs: Dict[str, Any], **kwargs):
        ...

    def step(self, **kwargs):
        ...

    def zero_grad(self, **kwargs):
        ...

    def lr_step(self, **kwargs):
        ...

    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str], **kwargs):
        ...

    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str], **kwargs):
        ...

    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str], **kwargs):
        ...

    def save(self, name, output_dir, interval=1, **kwargs):
        ...

    def get_state_dict(self, **kwargs):
        ...

    def calculate_metric(self, is_training: bool, **kwargs):
        ...

    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        ...

    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        ...

    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs):
        ...

    def get_train_configs(self, **kwargs):
        ...
