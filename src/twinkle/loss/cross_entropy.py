# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import LossOutput
from twinkle.utils import selective_log_softmax
from .base import Loss


class CrossEntropyLoss(Loss):

    def __init__(self, **kwargs):
        self.reduction = kwargs.get('reduction', 'mean')

    def __call__(self, inputs, outputs, **kwargs):
        import torch
        logits = outputs['logits'].view(-1, outputs['logits'].shape[-1])
        labels = inputs['labels'].view(-1)
        loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)(logits, labels)
        if self.reduction != 'sum':
            return LossOutput(loss=loss, num_tokens=0)
        else:
            return LossOutput(loss=loss, num_tokens=(labels != -100).sum())
