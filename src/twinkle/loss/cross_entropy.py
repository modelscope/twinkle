# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import LossOutput
from .base import Loss


class CrossEntropyLoss(Loss):
    """Calculate CE from logps, with optional DFT (arxiv 2508.05629) entropy weighting."""

    def __init__(self, ignore_index: int = -100, reduction='mean', dft: bool = False, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dft = dft

    def __call__(self, inputs, outputs, **kwargs):
        _, per_token, mask = self.get_per_token_loss(inputs, outputs)
        return self._reduce(per_token, mask)

    def get_per_token_loss(self, inputs, outputs):
        labels = inputs['labels']
        logps = outputs.get('logps')

        if logps is None:
            import torch.nn.functional as F
            logits = outputs['logits'].view(-1, outputs['logits'].shape[-1])
            original_shape = labels.shape
            labels = labels.view(-1)
            logps = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            labels = labels.reshape(original_shape)
            logps = logps.reshape(original_shape)

        mask = (labels != self.ignore_index).float()
        # DFT: -p·log(p) instead of -log(p)
        per_token = -logps * logps.exp() if self.dft else -logps
        loss_scale = inputs.get('loss_scale')
        if loss_scale is not None:
            import torch
            loss_scale = torch.as_tensor(loss_scale, device=per_token.device, dtype=per_token.dtype)
            if loss_scale.numel() != per_token.numel():
                raise ValueError(
                    f'loss_scale has {loss_scale.numel()} elements, expected {per_token.numel()} to match labels.')
            per_token = per_token * loss_scale.reshape_as(per_token)
        return labels, per_token, mask

    def _reduce(self, per_token, mask):
        if self.reduction != 'sum':
            return LossOutput(loss=(per_token * mask).sum() / mask.sum().clamp(min=1), num_tokens=0)
        return LossOutput(loss=(per_token * mask).sum(), num_tokens=mask.sum().clamp(min=1))
