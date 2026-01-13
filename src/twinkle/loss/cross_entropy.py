# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
import torch


class CrossEntropyLoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        logits = outputs['logits'].view(-1, outputs['logits'].shape[-1])
        labels = inputs['labels'].view(-1)
        return torch.nn.CrossEntropyLoss()(logits, labels)