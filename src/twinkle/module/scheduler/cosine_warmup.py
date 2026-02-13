# Copyright (c) ModelScope Contributors. All rights reserved.
# Some code borrowed from transformers
import math
from torch.optim.lr_scheduler import LambdaLR


class CosineWarmupScheduler(LambdaLR):

    def __init__(self,
                 optimizer,
                 num_warmup_steps: int,
                 num_training_steps: int,
                 num_cycles: float = 0.5,
                 last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles

        super().__init__(optimizer, lr_lambda=self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, cur_step):
        if cur_step < self.num_warmup_steps:
            return float(cur_step) / float(max(1, self.num_warmup_steps))
        progress = float(cur_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))
