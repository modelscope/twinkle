# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import LossOutput
from .base import Loss


class CrossEntropyLoss(Loss):
    """Calculate CE from logps"""

    def __init__(self, ignore_index: int = -100, reduction='mean', **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, inputs, outputs, **kwargs):
        labels = inputs['labels']
        logps = outputs.get('logps')
        logits = outputs.get('logits')

        if logps is not None:
            loss_mask = (labels != self.ignore_index).float()
            if self.reduction != 'sum':
                return LossOutput(
                    loss=(-logps * loss_mask).sum() / loss_mask.sum().clamp(min=1),
                    num_tokens=0,
                )
            else:
                return LossOutput(
                    loss=(-logps * loss_mask).sum(),
                    num_tokens=loss_mask.sum().clamp(min=1),
                )
        else:
            import torch
            assert logits is not None
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)(logits, labels)
            if self.reduction != 'sum':
                return LossOutput(loss=loss, num_tokens=0)
            else:
                return LossOutput(loss=loss, num_tokens=(labels != self.ignore_index).sum())
