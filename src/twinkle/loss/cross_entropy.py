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

    def micro_batch_scale(self, inputs, indices):
        if self.reduction == 'sum':
            return 1.0
        token_counts = []
        for model_input in inputs:
            labels = model_input['labels']
            if hasattr(labels, 'ne'):
                token_counts.append(int(labels.ne(self.ignore_index).sum().item()))
            else:
                token_counts.append(sum(int(token != self.ignore_index) for token in labels))
        total_tokens = sum(token_counts)
        if total_tokens == 0:
            return 0.0
        return sum(token_counts[index] for index in indices) / total_tokens

    def __call__(self, inputs, outputs, **kwargs):
        labels = inputs['labels']
        logps = outputs.get('logps')

        if logps is None:
            import torch.nn.functional as F
            logits = outputs['logits'].view(-1, outputs['logits'].shape[-1])
            labels = labels.view(-1)
            logps = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        mask = (labels != self.ignore_index).float()
        # DFT: -p·log(p) instead of -log(p)
        per_token = -logps * logps.exp() if self.dft else -logps

        if self.reduction != 'sum':
            return LossOutput(loss=(per_token * mask).sum() / mask.sum().clamp(min=1), num_tokens=0)
        return LossOutput(loss=(per_token * mask).sum(), num_tokens=mask.sum().clamp(min=1))
