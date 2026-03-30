# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Any, Dict, List, Optional

from twinkle import DeviceMesh
from twinkle.data_format import InputFeature, ModelOutput
from twinkle.loss import Loss
from twinkle.metric import Metric
from twinkle.processor import InputProcessor
from twinkle.template import Template


@dataclass
class TrainStatus:
    """Status for training or evaluation.

    Encapsulates inputs, outputs, loss, tokens count and metrics for a training/eval step.
    """
    inputs: List[InputFeature] = None
    outputs: ModelOutput = None
    loss_value: Any = None
    num_tokens: int = 0
    metrics: List[Metric] = field(default_factory=list)
    forward_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseOptimizerGroup:
    """Base optimizer group with common fields for training.

    Subclasses: OptimizerGroup (Transformers), MegatronOptimizerGroup (Megatron)
    """
    adapter_name: str = None
    adapter_config: Any = None
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None
    loss_instance: Loss = None
    train_status: TrainStatus = None
    eval_status: TrainStatus = None
    template: Template = None
    processor: InputProcessor = None
    gradient_accumulation_steps: int = 1
    cur_step: int = 0
    _dp_group: Any = None
    _device_mesh: DeviceMesh = None
    _last_grad_norm: float = 0.0

    def do_grad_sync(self, gradient_accumulation_steps: Optional[int] = None) -> bool:
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        else:
            self.gradient_accumulation_steps = gradient_accumulation_steps
        return gradient_accumulation_steps == 1 or ((self.cur_step - 1) % gradient_accumulation_steps == 0
                                                    and self.cur_step > 1)

    def _get_lr(self):
        """Get learning rates from optimizer. Override in subclass."""
        return []

    def accumulate_metrics(self, is_training):
        """Accumulate metrics for train/eval status. Override in subclass if needed."""
        status = self.train_status if is_training else self.eval_status
        if len(status.metrics) > 0 and status.inputs is not None and status.outputs is not None:
            for metric in status.metrics:
                metric.accumulate(
                    status.inputs,
                    status.outputs,
                    lr=self._get_lr(),
                    step=self.cur_step - 1,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    grad_norm=self._last_grad_norm,
                    **status.forward_kwargs)

    def calculate_metrics(self, is_training):
        """Calculate and return metrics."""
        self.accumulate_metrics(is_training)
        status = self.train_status if is_training else self.eval_status
        results = {}
        for metric in status.metrics:
            results.update(metric.calculate())
        status.inputs = None
        status.outputs = None
        return results
