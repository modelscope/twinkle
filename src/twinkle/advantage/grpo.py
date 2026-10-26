# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from .base import Advantage
from ._utils import apply_kl_in_reward, compute_gdpo_advantages, reduce_rewards

if TYPE_CHECKING:
    import torch


class GRPOAdvantage(Advantage):

    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none', 'gdpo'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """
            GRPO-style advantages: subtract group mean.

            For each group of samples from the same prompt:
                advantage_i = reward_i - mean(rewards_in_group)

            Args:
                rewards: Reward values, shape [batch_size] or list of floats.
                num_generations: Number of samples per prompt.
                scale: How to normalize advantages
                    - 'group': Divide by group std
                    - 'batch': Divide by batch std
                    - 'none': No normalization

            Returns:
                advantages: Tensor of shape [batch_size]

            Example:
                >>> rewards = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
                >>> advantages = compute_advantages(rewards, num_generations=4)
        """
        import torch
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        if scale == 'gdpo':
            # Per-function group-normalize -> weighted-sum -> batch-normalize. Needs the 2D
            # [N, n_funcs] matrix + weights, so it runs before the reward reduction below.
            if rewards.dim() <= 1:
                raise ValueError("scale='gdpo' requires a 2D [N, n_funcs] per-function reward matrix.")
            return compute_gdpo_advantages(rewards, kwargs.get('reward_weights'), num_generations)

        # reduce_rewards keeps the original plain-sum behavior when no reward_weights are given;
        # apply_kl_in_reward is a no-op unless kl_in_reward is requested -- so a scalar single-reward
        # call runs exactly the original path.
        rewards = reduce_rewards(rewards, kwargs.get('reward_weights'))
        rewards = apply_kl_in_reward(
            rewards,
            kl_in_reward=kwargs.get('kl_in_reward', False),
            beta=kwargs.get('beta', 0.0),
            kl_values=kwargs.get('kl_values'),
        )

        if num_generations <= 0 or rewards.numel() % num_generations != 0:
            raise ValueError('Invalid')

        if num_generations == 1:
            # One sample per group: the group-relative advantage is identically zero (the group mean
            # is the sample itself; legacy's ``rewards - group_mean`` reduces to the same zeros).
            # Warn rather than refuse, and do NOT fall back to batch-centered or raw rewards: no
            # scale below re-centers, so that silently turned a degenerate GRPO into un-baselined
            # REINFORCE while 'none' paradoxically batch-centered (its own contract says no
            # normalization).
            import warnings

            warnings.warn('num_generations == 1: GRPO group-relative advantages are identically zero. '
                          'Raise num_generations for a meaningful group baseline.')
            return torch.zeros_like(rewards)

        grouped = rewards.view(-1, num_generations)
        group_mean = grouped.mean(dim=1, keepdim=True)
        advantages = grouped - group_mean

        if scale == 'group':
            group_std = grouped.std(dim=1, keepdim=True)
            advantages = advantages / (group_std + 1e-8)
        elif scale == 'batch':
            batch_std = grouped.std()
            advantages = advantages / (batch_std + 1e-8)

        return advantages.view(-1)
