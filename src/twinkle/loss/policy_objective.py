# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class PolicyObjective:
    """Base interface for a per-token policy optimization objective."""

    def __call__(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        raise NotImplementedError


class DISPolicyObjective(PolicyObjective):
    """Direct double-sided importance-sampling objective used by SAO."""

    def __init__(
        self,
        epsilon_low: float = 0.3,
        epsilon_high: float = 5.0,
        detach_importance_weight: bool = True,
    ):
        if not 0.0 <= epsilon_low < 1.0:
            raise ValueError('epsilon_low must be in [0, 1)')
        if epsilon_high < 0.0:
            raise ValueError('epsilon_high must be non-negative')
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.detach_importance_weight = detach_importance_weight

    def __call__(
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
