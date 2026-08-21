# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from .base import Advantage

if TYPE_CHECKING:
    import torch


class GAEAdvantage(Advantage):
    """Token-level generalized advantage estimation for terminal completions."""

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95, normalize: bool = True):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('gamma must be in [0, 1]')
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError('gae_lambda must be in [0, 1]')
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize = normalize

    @staticmethod
    def build_token_rewards(
        rewards: Union['torch.Tensor', List[float]],
        lengths: List[int],
        *,
        old_logps: Optional[List[List[float]]] = None,
        ref_logps: Optional[List[List[float]]] = None,
        kl_coef: float = 0.0,
    ) -> List[List[float]]:
        import torch

        rewards = torch.as_tensor(rewards, dtype=torch.float32).flatten().tolist()
        if len(rewards) != len(lengths):
            raise ValueError('rewards and lengths must have the same batch size')
        if (old_logps is None) != (ref_logps is None):
            raise ValueError('old_logps and ref_logps must be provided together')

        token_rewards = []
        for i, (reward, length) in enumerate(zip(rewards, lengths)):
            if length <= 0:
                raise ValueError('completion lengths must be positive')
            values = [0.0] * length
            if old_logps is not None:
                if len(old_logps[i]) != length or len(ref_logps[i]) != length:
                    raise ValueError(f'log-prob length mismatch at sample {i}')
                values = [-kl_coef * (float(old) - float(ref)) for old, ref in zip(old_logps[i], ref_logps[i])]
            values[-1] += float(reward)
            token_rewards.append(values)
        return token_rewards

    def __call__(
        self,
        rewards: Union['torch.Tensor', List[List[float]]],
        values: Union['torch.Tensor', List[List[float]]],
        *,
        masks: Optional[Union['torch.Tensor', List[List[bool]]]] = None,
        normalize: Optional[bool] = None,
        **kwargs,
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        import torch

        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        values = torch.as_tensor(values, dtype=torch.float32, device=rewards.device)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(0)
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if rewards.shape != values.shape:
            raise ValueError(f'rewards and values must have identical shapes, got {rewards.shape} and {values.shape}')

        if masks is None:
            masks = torch.ones_like(rewards, dtype=torch.bool)
        else:
            masks = torch.as_tensor(masks, dtype=torch.bool, device=rewards.device)
            if masks.shape != rewards.shape:
                raise ValueError('masks must have the same shape as rewards')

        advantages = torch.zeros_like(rewards)
        for batch_idx in range(rewards.shape[0]):
            valid = masks[batch_idx].nonzero(as_tuple=True)[0]
            last_gae = rewards.new_zeros(())
            for j in range(len(valid) - 1, -1, -1):
                pos = valid[j]
                if j + 1 < len(valid):
                    next_value = values[batch_idx, valid[j + 1]]
                else:
                    next_value = rewards.new_zeros(())
                delta = rewards[batch_idx, pos] + self.gamma * next_value - values[batch_idx, pos]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
                advantages[batch_idx, pos] = last_gae

        returns = advantages + values
        should_normalize = self.normalize if normalize is None else normalize
        if should_normalize:
            valid_advantages = advantages[masks]
            if valid_advantages.numel() > 1:
                mean = valid_advantages.mean()
                std = valid_advantages.std(unbiased=False)
                advantages = torch.where(masks, (advantages - mean) / (std + 1e-8), advantages)
        advantages = advantages.masked_fill(~masks, 0.0)
        returns = returns.masked_fill(~masks, 0.0)
        return advantages, returns
