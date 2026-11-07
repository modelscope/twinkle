# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch

from .base import Loss


class DecisionLoss(Loss):
    """Question-normalized proper scoring losses for ragged option sets.

    Return a sum and a question count so Twinkle normalizes over all data
    ranks and accumulated micro-batches, including uneven question counts.
    """

    require_logits = True
    require_logps = False

    def __init__(self, mode='scoring', rps_weight=0.5, brier_weight=1.0, label_smoothing=0.1):
        if mode not in ('scoring', 'distill_rps', 'smooth_brier', 'ce_rps'):
            raise ValueError(f'Unknown decision loss: {mode}')
        if not all(math.isfinite(x) and x >= 0 for x in (rps_weight, brier_weight)):
            raise ValueError('Loss weights must be finite and nonnegative')
        if not 0 <= label_smoothing < 1:
            raise ValueError('label_smoothing must be in [0, 1)')
        self.mode = mode
        self.rps_weight = rps_weight
        self.brier_weight = brier_weight
        self.label_smoothing = label_smoothing

    def micro_batch_scale(self, inputs, indices):
        return 1.0

    def __call__(self, inputs, outputs, **kwargs):
        raw_logits = outputs['logits']
        # Preserve double precision for numerical/gradient reference tests.
        logits = raw_logits if raw_logits.dtype == torch.float64 else raw_logits.float()
        mask = outputs['option_mask'].bool()
        target = inputs['labels'].to(device=logits.device, dtype=logits.dtype)
        if logits.ndim != 2 or logits.shape != mask.shape or logits.shape != target.shape:
            raise ValueError('Expected matching [questions, options] logits, mask and targets')
        if not logits.shape[0] or not bool(mask.any(-1).all()):
            raise ValueError('Every decision must have at least one legal option')
        if not bool(torch.isfinite(logits[mask]).all()):
            raise ValueError('Non-finite legal option logits')
        if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
            raise ValueError('Targets must be finite and nonnegative')
        if bool((target[~mask] != 0).any()) or not torch.allclose(
                target.sum(-1), torch.ones_like(target[:, 0]), atol=1e-6, rtol=0):
            raise ValueError('Target distributions must sum to one over legal options')
        n_options = mask.sum(-1)
        log_probs = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), -1).masked_fill(~mask, 0)
        probs = log_probs.exp().masked_fill(~mask, 0)
        smooth = target
        if self.mode == 'smooth_brier':
            smooth = (1 - self.label_smoothing) * target + self.label_smoothing * mask.to(
                logits.dtype) / n_options[:, None]
        per_question = -(smooth * log_probs).sum(-1)
        if self.mode == 'smooth_brier':
            per_question = per_question + self.brier_weight * ((probs - target).square() * mask).sum(-1) / n_options
        if self.mode in ('distill_rps', 'ce_rps'):
            kinds = outputs.get('kinds')
            if kinds is None or kinds.shape != (logits.shape[0], ):
                raise ValueError('RPS requires one kind per question')
            # Include only CDF boundaries preceding the final real option.
            boundaries = torch.arange(logits.shape[1], device=logits.device)[None] < n_options[:, None] - 1
            rps = ((probs.cumsum(-1) - target.cumsum(-1)).square() * boundaries).sum(-1)
            if self.mode == 'distill_rps':
                rps = rps / (n_options - 1).clamp_min(1)
            per_question = per_question + self.rps_weight * rps * (kinds == 2)
        return {'loss': per_question.sum(), 'num_tokens': logits.new_tensor(logits.shape[0])}
