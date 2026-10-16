# Copyright (c) ModelScope Contributors. All rights reserved.
"""Clip-Cov loss — PPO clipping with covariance-based token masking.

Matches verl's ``@register_policy_loss("clip_cov")``.
Adapted from https://github.com/PRIME-RL/Entropy-Mechanism-of-RL
"""
from typing import TYPE_CHECKING, Dict, Optional

from twinkle.data_format import LossOutput
from .grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class ClipCovLoss(GRPOLoss):
    """PPO clipping with covariance-based token zeroing (Clip-Cov).

    After standard PPO clipping, identifies tokens with high covariance between
    advantages and log-probabilities (indicating potentially harmful gradient signals),
    and zeros out their loss contribution. This prevents the policy from overfitting
    to spurious advantage-logprob correlations.

    Args:
        clip_cov_ratio: Fraction of valid tokens to zero out (default 0.0002).
        clip_cov_lb: Lower bound for covariance selection window (default 1.0).
        clip_cov_ub: Upper bound for covariance selection window (default 5.0).
    """

    def __init__(self, clip_cov_ratio: float = 0.0002, clip_cov_lb: float = 1.0, clip_cov_ub: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        if clip_cov_ratio <= 0:
            raise ValueError(f'clip_cov_ratio must be positive, got {clip_cov_ratio}')
        self.clip_cov_ratio = clip_cov_ratio
        self.clip_cov_lb = clip_cov_lb
        self.clip_cov_ub = clip_cov_ub

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

        # verl's clip_cov uses the UNCLAMPED log ratio exp(log_prob - old_log_prob) (core_algos.py:1797-1798).
        # The base class clamps log_importance_weights to ±5, which would diverge from that oracle, so
        # recompute the ratio from logps/old_logps here (same precedent as REALLoss._reduce_loss).
        ratio = torch.exp(logps - old_logps)

        # Standard PPO clipping
        pg_losses1 = -advantages * ratio
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon_high)
        pg_losses2 = -advantages * clipped_ratio
        clip_by_origin = (pg_losses2 > pg_losses1) & (loss_mask > 0)

        # Compute covariance between advantages and log-probs
        mask_total = loss_mask.sum().clamp(min=1.0)
        adv_mean = (advantages * loss_mask).sum() / mask_total
        logp_detached = logps.detach()
        logp_mean = (logp_detached * loss_mask).sum() / mask_total
        cov_all = (advantages - adv_mean) * (logp_detached - logp_mean)

        # Exclude already-clipped and masked positions from selection
        cov_all = cov_all.clone()
        cov_all[loss_mask == 0] = -torch.inf
        cov_all[clip_by_origin] = -torch.inf

        # Select tokens within the covariance window
        clip_num = max(int(self.clip_cov_ratio * loss_mask.sum().item()), 1)
        in_window = (cov_all < self.clip_cov_ub) & (cov_all > self.clip_cov_lb) & (loss_mask > 0)
        candidates = torch.nonzero(in_window)

        # Random subsample from candidates
        corr = torch.ones_like(advantages)
        if len(candidates) > 0:
            perm = torch.randperm(len(candidates), device=candidates.device)
            selected = candidates[perm[:min(clip_num, len(candidates))]]
            corr[selected[:, 0], selected[:, 1]] = 0

        # Apply PPO loss with covariance masking
        pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

        # Optional KL penalty
        if self.beta > 0.0 and ref_logps is not None:
            per_token_kl = (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
            pg_losses = pg_losses + self.beta * per_token_kl

        loss = self._aggregate_loss(pg_losses, loss_mask, **kwargs)
        return LossOutput(loss=loss, num_tokens=self._loss_num_tokens(loss_mask))
