# Copyright (c) ModelScope Contributors. All rights reserved.
"""GeoMean (GMPO) loss — sequence-level geometric mean policy optimization.

Matches verl's ``@register_policy_loss("geo_mean")``.
Reference: https://arxiv.org/abs/2507.20673
"""
from typing import TYPE_CHECKING, Dict, Optional

from twinkle.data_format import LossOutput
from .grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class GeoMeanLoss(GRPOLoss):
    """Geometric-Mean Policy Optimization (GMPO).

    Instead of token-level importance ratios, GMPO computes a sequence-level
    geometric mean of the clipped per-token ratios:
        ratio_seq = exp( mean_t( clamp(log_ratio_t, -eps_low, eps_high) * sign(adv) ) )
        advantage_seq = mean_t(advantage_t)
        L = -advantage_seq * ratio_seq

    The pessimistic clipping (take min of signed log-ratio and its clamped version)
    ensures the surrogate never overestimates the true objective.
    """

    def _reduce_loss(
        self,
        logps: 'torch.Tensor',
        old_logps: 'torch.Tensor',
        ref_logps: Optional['torch.Tensor'],
        advantages: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        log_importance_weights: 'torch.Tensor',
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        import torch

        # Unclamped token-level log ratio
        negative_approx_kl = logps - old_logps

        # Pessimistic clipping: take the min of signed log-ratio and its clamped version
        sgn_advantage = torch.sign(advantages)
        negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -self.epsilon, self.epsilon_high)
        negative_approx_kl_min = torch.min(
            sgn_advantage * negative_approx_kl,
            sgn_advantage * negative_approx_kl_clamp,
        )
        negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

        # Sequence-level geometric mean ratio. verl divides by (response_mask_sum + 1e-8)
        # (core_algos.py:1984,1987); match it exactly so the oracle agrees bit-for-bit.
        mask_sum = loss_mask.sum(dim=-1) + 1e-8
        seq_ratio = torch.exp((negative_approx_kl_min * loss_mask).sum(dim=-1) / mask_sum)

        # Sequence-level mean advantage
        seq_advantage = (advantages * loss_mask).sum(dim=-1) / mask_sum

        # Per-sequence loss, then batch mean
        per_seq_loss = -seq_advantage * seq_ratio
        loss = per_seq_loss.mean()

        # Optional KL penalty (global token-mean, consistent with base class)
        if self.beta > 0.0 and ref_logps is not None:
            per_token_kl = (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
            kl_loss = (per_token_kl * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = loss + self.beta * kl_loss

        return LossOutput(loss=loss, num_tokens=self._loss_num_tokens(loss_mask))
