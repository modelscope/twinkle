# Copyright (c) ModelScope Contributors. All rights reserved.
"""DRO (Direct Reward Optimization) loss with quadratic log-ratio penalty.

Matches verl's ``@register_policy_loss("dro")``.
"""
from typing import TYPE_CHECKING, Dict, Optional

from twinkle.data_format import LossOutput
from .grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class DROLoss(GRPOLoss):
    """Direct Reward Optimization with a quadratic log-ratio penalty.

    Formula:
        L = -(log_prob * advantage - 0.5 * beta * (log_prob - old_log_prob)^2)

    The quadratic penalty regularizes the policy update magnitude without explicit
    KL computation against a reference model. ``dro_beta`` controls the penalty strength.
    """

    def __init__(self, dro_beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if dro_beta <= 0:
            raise ValueError(f'dro_beta must be positive, got {dro_beta}')
        self.dro_beta = dro_beta

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
        """Override to use unclamped log-ratio for the quadratic penalty."""
        import torch

        # Unclamped log ratio (verl uses log_prob - old_log_prob directly)
        log_ratio = logps - old_logps
        per_token_loss = -(logps * advantages - 0.5 * self.dro_beta * log_ratio.square())

        # Standard KL penalty from base class (if beta > 0 and ref_logps given)
        if self.beta > 0.0 and ref_logps is not None:
            per_token_kl = (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = self._aggregate_loss(per_token_loss, loss_mask, **kwargs)
        return LossOutput(loss=loss, num_tokens=self._loss_num_tokens(loss_mask))
