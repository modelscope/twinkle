# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPG (Group Policy Gradient) loss — pure REINFORCE without importance ratio.

Adapted from https://github.com/AMAP-ML/GPG (VisualThinker-R1-Zero).
Matches verl's ``@register_policy_loss("gpg")``.
"""
from typing import TYPE_CHECKING

from .grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class GPGLoss(GRPOLoss):
    """REINFORCE-style policy gradient loss: ``-log_prob × advantage``.

    Unlike PPO/GRPO which uses the importance ratio ``π_θ/π_old``, GPG uses the
    raw log probability directly. This is the core of verl's ``bypass_mode`` with
    ``loss_type="reinforce"`` and no IS correction.

    No clipping is applied — the advantage sign alone determines the gradient direction.
    """

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Pure REINFORCE: -log_prob * advantage (ratio is ignored)."""
        return -per_token_logps * advantages
