# Copyright (c) ModelScope Contributors. All rights reserved.
"""DPPO (Divergence-bounded Policy Optimization) losses — TV and KL variants.

Matches verl's ``@register_policy_loss("dppo_tv")`` and ``@register_policy_loss("dppo_kl")``.
Reference: https://arxiv.org/pdf/2602.04879
"""
from typing import TYPE_CHECKING, Dict, Optional

from twinkle.data_format import LossOutput
from .grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class DPPOTVLoss(GRPOLoss):
    """DPPO with Total Variation divergence threshold.

    Instead of clipping the importance ratio, DPPO-TV masks out tokens where the
    probability difference ``|π_θ - π_old|`` exceeds a divergence threshold:
        valid_positive = (prob - old_prob) <= clip_divergence_high
        valid_negative = (prob - old_prob) >= -clip_divergence_low
        valid_mask = where(advantage > 0, valid_positive, valid_negative)
        L = -advantage * truncated_ratio * log_prob * valid_mask

    verl reuses ``clip_ratio_high``/``clip_ratio_low`` as ``clip_divergence_high``/``_low``
    (core_algos.py:1414-1416), so the threshold here is the inherited ``epsilon_high``/``epsilon``
    -- there is deliberately no separate ``clip_divergence`` knob.

    The ``truncated_ratio = clamp(ratio, max=clip_ratio_c).detach()`` provides
    additional stability (Section 5.4 of the paper recommends a large threshold).
    """

    def __init__(self, clip_ratio_c: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_ratio_c = clip_ratio_c

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

        negative_approx_kl = logps - old_logps
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        # Truncated IS weight (detached — no gradient through the ratio)
        truncated_ratio = torch.clamp(ratio, max=self.clip_ratio_c).detach()

        # TV divergence valid mask
        prob = torch.exp(logps)
        old_prob = torch.exp(old_logps)
        valid_positive_mask = (prob - old_prob) <= self.epsilon_high
        valid_negative_mask = (prob - old_prob) >= -self.epsilon
        valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
        valid_mask = valid_mask.detach().float()

        per_token_loss = -advantages * truncated_ratio * logps * valid_mask

        if self.beta > 0.0 and ref_logps is not None:
            per_token_kl = (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = self._aggregate_loss(per_token_loss, loss_mask, **kwargs)
        return LossOutput(loss=loss, num_tokens=self._loss_num_tokens(loss_mask))


class DPPOKLLoss(GRPOLoss):
    """DPPO with Binary KL divergence threshold.

    Similar to DPPO-TV but uses binary KL divergence for the validity mask:
        binary_kl = old_prob * (old_log_prob - log_prob)
                  + (1 - old_prob) * log((1 - old_prob) / (1 - prob))
        valid_positive = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
        valid_negative = (binary_kl <= clip_divergence_low) | (prob >= old_prob)

    As in DPPO-TV, ``clip_divergence_high``/``_low`` are the inherited ``epsilon_high``/``epsilon``
    (verl reuses ``clip_ratio_high``/``clip_ratio_low``; core_algos.py:1495-1497).
    """

    def __init__(self, clip_ratio_c: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_ratio_c = clip_ratio_c

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

        negative_approx_kl = logps - old_logps
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        # Truncated IS weight
        truncated_ratio = torch.clamp(ratio, max=self.clip_ratio_c).detach()

        # Binary KL divergence mask
        prob = torch.exp(logps)
        old_prob = torch.exp(old_logps)
        binary_kl = old_prob * (old_logps - logps) + (1 - old_prob) * torch.log(
            (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8))
        valid_positive_mask = (binary_kl <= self.epsilon_high) | (prob <= old_prob)
        valid_negative_mask = (binary_kl <= self.epsilon) | (prob >= old_prob)
        valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
        valid_mask = valid_mask.detach().float()

        per_token_loss = -advantages * truncated_ratio * logps * valid_mask

        if self.beta > 0.0 and ref_logps is not None:
            per_token_kl = (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = self._aggregate_loss(per_token_loss, loss_mask, **kwargs)
        return LossOutput(loss=loss, num_tokens=self._loss_num_tokens(loss_mask))
