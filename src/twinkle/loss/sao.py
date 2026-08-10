# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from .grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class SAOLoss(GRPOLoss):
    """SAO direct double-sided importance-sampling policy loss.

    Tokens whose current/rollout policy ratio lies outside the strict trust
    interval are assigned zero weight instead of being clipped to a boundary.
    """

    def __init__(
        self,
        epsilon_low: float = 0.3,
        epsilon_high: float = 5.0,
        detach_importance_weight: bool = True,
        **kwargs,
    ):
        if not 0.0 <= epsilon_low < 1.0:
            raise ValueError('epsilon_low must be in [0, 1)')
        if epsilon_high < 0.0:
            raise ValueError('epsilon_high must be non-negative')
        super().__init__(epsilon=epsilon_low, epsilon_high=epsilon_high, **kwargs)
        self.epsilon_low = epsilon_low
        self.detach_importance_weight = detach_importance_weight

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        import torch

        trusted = (ratio > 1.0 - self.epsilon_low) & (ratio < 1.0 + self.epsilon_high)
        weight = torch.where(trusted, ratio, torch.zeros_like(ratio))
        if self.detach_importance_weight:
            weight = weight.detach()
        return -weight * advantages.detach() * per_token_logps.float()

    def _aggregate_loss(self, per_token_loss, loss_mask, **kwargs):
        mask = loss_mask.to(per_token_loss.dtype)
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
