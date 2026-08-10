# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from .base import Advantage

if TYPE_CHECKING:
    import torch


class SAOGAEAdvantage(Advantage):
    """Batched skip-observation GAE with terminal/truncation semantics.

    True positions in ``action_masks`` form the Bellman chain. Consequently,
    prompt, padding, and environment-observation tokens are skipped.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.5,
        gae_lambda: Optional[float] = None,
        normalize: bool = True,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('gamma must be in [0, 1]')
        if alpha <= 0.0:
            raise ValueError('alpha must be positive')
        if gae_lambda is not None and not 0.0 <= gae_lambda <= 1.0:
            raise ValueError('gae_lambda must be in [0, 1]')
        self.gamma = gamma
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.normalize = normalize

    def __call__(
        self,
        rewards: Union['torch.Tensor', List[List[float]]],
        values: Union['torch.Tensor', List[List[float]]],
        *,
        action_masks: Union['torch.Tensor', List[List[bool]]],
        terminated: Union['torch.Tensor', List[bool]],
        truncated: Union['torch.Tensor', List[bool]],
        bootstrap_values: Optional[Union['torch.Tensor', List[Optional[float]]]] = None,
        effective_lengths: Optional[Union['torch.Tensor', List[int]]] = None,
        normalize: Optional[bool] = None,
        **kwargs,
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        import torch

        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        values = torch.as_tensor(values, dtype=torch.float32, device=rewards.device)
        action_masks = torch.as_tensor(action_masks, dtype=torch.bool, device=rewards.device)
        if rewards.dim() == 1:
            rewards, values, action_masks = rewards.unsqueeze(0), values.unsqueeze(0), action_masks.unsqueeze(0)
        if rewards.shape != values.shape or rewards.shape != action_masks.shape:
            raise ValueError('rewards, values, and action_masks must have identical shapes')

        batch_size = rewards.shape[0]
        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=rewards.device).flatten()
        truncated = torch.as_tensor(truncated, dtype=torch.bool, device=rewards.device).flatten()
        if terminated.numel() != batch_size or truncated.numel() != batch_size:
            raise ValueError('terminated and truncated must have one value per sequence')
        if bool((terminated & truncated).any()):
            raise ValueError('a trajectory cannot be both terminated and truncated')

        if bootstrap_values is None:
            bootstrap = [None] * batch_size
        else:
            bootstrap = list(bootstrap_values)
            if len(bootstrap) != batch_size:
                raise ValueError('bootstrap_values must have one value per sequence')

        if effective_lengths is None:
            lengths = action_masks.sum(dim=-1).tolist()
        else:
            lengths = torch.as_tensor(effective_lengths).flatten().tolist()
            if len(lengths) != batch_size:
                raise ValueError('effective_lengths must have one value per sequence')

        advantages = torch.zeros_like(rewards)
        for batch_idx in range(batch_size):
            positions = action_masks[batch_idx].nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                raise ValueError(f'trajectory {batch_idx} has no action tokens')
            if lengths[batch_idx] <= 0:
                raise ValueError('effective lengths must be positive')
            if bool(truncated[batch_idx]) and bootstrap[batch_idx] is None:
                raise ValueError(f'truncated trajectory {batch_idx} requires a bootstrap value')
            if not bool(terminated[batch_idx]) and not bool(truncated[batch_idx]):
                raise ValueError(f'trajectory {batch_idx} must be terminated or truncated')

            lambda_value = self.gae_lambda
            if lambda_value is None:
                lambda_value = 1.0 - 1.0 / (self.alpha * float(lengths[batch_idx]))
                lambda_value = min(1.0, max(0.0, lambda_value))

            next_advantage = rewards.new_zeros(())
            for index in range(positions.numel() - 1, -1, -1):
                position = positions[index]
                if index + 1 < positions.numel():
                    next_value = values[batch_idx, positions[index + 1]].detach()
                elif bool(truncated[batch_idx]):
                    next_value = torch.as_tensor(bootstrap[batch_idx], device=rewards.device, dtype=torch.float32)
                else:
                    next_value = rewards.new_zeros(())
                delta = rewards[batch_idx, position] + self.gamma * next_value - values[batch_idx, position]
                next_advantage = delta + self.gamma * lambda_value * next_advantage
                advantages[batch_idx, position] = next_advantage

        returns = (advantages + values.detach()).masked_fill(~action_masks, 0.0)
        should_normalize = self.normalize if normalize is None else normalize
        if should_normalize:
            valid = advantages[action_masks]
            if valid.numel() > 1:
                normalized = (advantages - valid.mean()) / valid.std(unbiased=False).clamp_min(1e-8)
                advantages = torch.where(action_masks, normalized, advantages)
        advantages = advantages.masked_fill(~action_masks, 0.0)
        return advantages, returns
