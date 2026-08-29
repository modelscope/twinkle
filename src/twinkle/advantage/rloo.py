# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from .base import Advantage
from ._utils import apply_kl_in_reward, compute_gdpo_advantages, reduce_rewards

if TYPE_CHECKING:
    import torch


class RLOOAdvantage(Advantage):

    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none', 'gdpo'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """
        RLOO (Reinforce Leave-One-Out) advantages.

        For each sample, the baseline is the mean of OTHER samples in the group:
            baseline_i = (sum(rewards) - reward_i) / (K - 1)
            advantage_i = reward_i - baseline_i

        This reduces variance compared to using the full group mean.

        Args:
            rewards: Reward values, shape [batch_size] or list of floats.
            num_generations: Number of samples per prompt.
            scale: How to normalize advantages
                - 'group': Divide by group std
                - 'batch': Divide by batch std
                - 'none': No normalization

        Returns:
            advantages: Tensor of shape [batch_size]
        """
        import torch
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        if scale == 'gdpo':
            # GDPO recomputes advantages from the per-function matrix independently of the RLOO
            # baseline (identical to the GRPO+gdpo result), so it runs before the reduction below.
            if rewards.dim() <= 1:
                raise ValueError("scale='gdpo' requires a 2D [N, n_funcs] per-function reward matrix.")
            return compute_gdpo_advantages(rewards, kwargs.get('reward_weights'), num_generations)

        # reduce_rewards keeps the original plain-sum behavior when no reward_weights are given;
        # apply_kl_in_reward is a no-op unless kl_in_reward is requested.
        rewards = reduce_rewards(rewards, kwargs.get('reward_weights'))
        rewards = apply_kl_in_reward(
            rewards,
            kl_in_reward=kwargs.get('kl_in_reward', False),
            beta=kwargs.get('beta', 0.0),
            kl_values=kwargs.get('kl_values'),
        )

        # Guard against invalid num_generations
        if num_generations <= 1 or rewards.numel() % num_generations != 0:
            raise ValueError('Invalid')

        K = num_generations
        grouped = rewards.view(-1, K)

        # RLOO: baseline = (sum - self) / (K - 1)
        group_sum = grouped.sum(dim=1, keepdim=True)
        baselines = (group_sum - grouped) / (K - 1)
        advantages = grouped - baselines

        if scale == 'group':
            group_std = grouped.std(dim=1, keepdim=True)
            advantages = advantages / (group_std + 1e-8)
        elif scale == 'batch':
            batch_std = grouped.std()
            advantages = advantages / (batch_std + 1e-8)

        return advantages.view(-1)
