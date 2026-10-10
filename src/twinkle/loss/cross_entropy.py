# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import LossOutput
from .base import Loss


class CrossEntropyLoss(Loss):
    """Calculate CE from logps, with optional DFT (arxiv 2508.05629) entropy weighting."""

    def __init__(self,
                 ignore_index: int = -100,
                 reduction='mean',
                 dft: bool = False,
                 label_smoothing: float = 0.0,
                 **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dft = dft
        self.label_smoothing = label_smoothing

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
        _, per_token, mask = self.get_per_token_loss(inputs, outputs)
        return self._reduce(per_token, mask)

    def get_per_token_loss(self, inputs, outputs):
        labels = inputs['labels']
        logps = outputs.get('logps')
        smooth = None

        if logps is None:
            import torch.nn.functional as F
            logits = outputs['logits'].view(-1, outputs['logits'].shape[-1])
            original_shape = labels.shape
            labels = labels.view(-1)
            log_softmax = F.log_softmax(logits, dim=-1)
            logps = log_softmax.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            if self.label_smoothing:
                # Uniform-target term -(1/V)*sum_c log p_c, blended with the NLL exactly as
                # F.cross_entropy(label_smoothing=eps) does. It needs the full vocab distribution, so
                # it is only derivable on this branch (a caller that pre-gathers logps cannot smooth).
                smooth = -log_softmax.mean(dim=-1).reshape(original_shape)
            labels = labels.reshape(original_shape)
            logps = logps.reshape(original_shape)
        elif self.label_smoothing:
            raise ValueError('label_smoothing needs the full vocab distribution, but this forward supplied '
                             'pre-gathered logps. Drop label_smoothing, or run a task that returns logits.')

        mask = (labels != self.ignore_index).float()
        per_token = -logps
        if smooth is not None:
            per_token = (1 - self.label_smoothing) * per_token + self.label_smoothing * smooth
        if self.dft:
            # DFT: weight each token's loss by its own probability p = exp(logp), i.e. -p*log(p).
            per_token = per_token * logps.exp()
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
