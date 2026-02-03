# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss

class MSELoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        import torch
        preds = outputs['logits']
        labels = inputs['labels']
        return torch.nn.MSELoss()(preds, labels)