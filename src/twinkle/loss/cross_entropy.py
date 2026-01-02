from .base import Loss
import torch


class CrossEntropyLoss(Loss):

    def __call__(self, inputs, outputs, **kwargs):
        logits = outputs['logits']
        labels = inputs['labels']
        return torch.nn.CrossEntropyLoss()(logits, labels)