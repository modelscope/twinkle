# Copyright (c) ModelScope Contributors. All rights reserved.
# Some code borrowed from transformers
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupScheduler(LambdaLR):

    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        super().__init__(optimizer, lr_lambda=self._lr_lambda, last_epoch=last_epoch)

    def _lr_lambda(self, cur_step):
        if cur_step < self.num_warmup_steps:
            return float(cur_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - cur_step) / float(max(1, self.num_training_steps - self.num_warmup_steps)))
