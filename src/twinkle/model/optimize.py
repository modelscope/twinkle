from dataclasses import dataclass
from typing import Dict, Any

from peft import PeftConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from twinkle.loss.base import Loss
from twinkle.processor import InputProcessor
from twinkle.template import Template


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

