# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
import torch


class CrossEntropyLoss(Loss):

    def __init__(self, **kwargs):
        self.reduction = kwargs.get('reduction', 'mean')

    def __call__(self, inputs, outputs, **kwargs):
        logits = outputs['logits'].view(-1, outputs['logits'].shape[-1])
        labels = inputs['labels'].view(-1)
        return torch.nn.CrossEntropyLoss(reduction=self.reduction)(logits, labels)