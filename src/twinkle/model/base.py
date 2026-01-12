# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Any, Union, Type

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from twinkle import template, processor
from twinkle.loss.base import Loss


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

    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        ...

    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        ...

    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str], **kwargs):
        ...

    def save(self, output_dir, **kwargs):
        ...

    def get_state_dict(self, **kwargs):
        ...

    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        ...

    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        ...

    def set_processor(self, processor_cls: Union[Type[processor.InputProcessor], str], **kwargs):
        ...

    def get_train_configs(self, **kwargs):
        ...
