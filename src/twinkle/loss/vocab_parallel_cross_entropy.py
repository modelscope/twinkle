# Copyright (c) ModelScope Contributors. All rights reserved.
from ..data_format import LossOutput
from .base import Loss


class VocabParallelCrossEntropyLoss(Loss):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, inputs, outputs, **kwargs):
        labels = inputs['labels']
        logps = outputs.get('logps')

        loss_mask = (labels != self.ignore_index).float()
        return LossOutput(
            loss=(-logps * loss_mask).sum(),
            num_tokens=loss_mask.sum().clamp(min=1),
        )
