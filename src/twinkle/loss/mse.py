# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
import torch


class MSELoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        preds = outputs['logits']
        labels = inputs['labels']
        return torch.nn.MSELoss()(preds, labels)