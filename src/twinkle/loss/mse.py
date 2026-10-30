# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import LossOutput
from .base import Loss


class MSELoss(Loss):

    require_logits = True

    def micro_batch_scale(self, inputs, indices):
        return len(indices) / len(inputs)

    def __call__(self, inputs, outputs, **kwargs):
        import torch
        preds = outputs['logits']
        labels = inputs['labels']
        return LossOutput(loss=torch.nn.MSELoss()(preds, labels))
