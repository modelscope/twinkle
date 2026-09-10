# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from .grpo import GRPOLoss
from .policy_objective import DISPolicyObjective

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
        self.policy_objective = DISPolicyObjective(
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            detach_importance_weight=detach_importance_weight,
        )
        super().__init__(epsilon=epsilon_low, epsilon_high=epsilon_high, **kwargs)

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        return self.policy_objective(ratio, advantages, per_token_logps)

    def _aggregate_loss(self, per_token_loss, loss_mask, **kwargs):
        mask = loss_mask.to(per_token_loss.dtype)
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
