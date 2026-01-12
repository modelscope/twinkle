# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss
import torch


class RerankerLoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        logits = outputs['logits']
        labels = inputs['labels']
        logits = logits.squeeze(1)
        labels = labels.to(logits.dtype)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return loss