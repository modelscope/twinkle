# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from .base import Advantage
from ._utils import apply_kl_in_reward, reduce_rewards

if TYPE_CHECKING:
    import torch


class GRPOPassKAdvantage(Advantage):
    """GRPO-Pass@k advantage estimator (https://arxiv.org/abs/2503.19595).

    Only the best response per group gets a non-zero advantage equal to
    ``r_max - r_second_max``, optionally normalized by the group std:
        advantage_best = (r_max - r_second_max) / (std + epsilon)   if norm_by_std
        advantage_best = r_max - r_second_max                        otherwise
        advantage_others = 0

    Requires at least 2 samples per group (num_generations >= 2).
    """

    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 2,
                 scale: Literal['group', 'batch', 'none'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """Compute GRPO-Pass@k advantages.

        Args:
            rewards: ``[N]`` scalar rewards (or ``[N, n_funcs]`` matrix).
            num_generations: Group size K (must be >= 2).
            scale: ``'group'`` normalizes by group std (default, matching verl's
                ``norm_adv_by_std_in_grpo=True``); ``'none'`` / ``'batch'`` do not
                normalize by group std.
            **kwargs:
                epsilon: Numerical stability constant (default 1e-6).
                reward_weights: Optional weights for multi-function reward reduction.
                kl_in_reward / beta / kl_values: Optional ref-model KL regularization.

        Returns:
            advantages: Tensor of shape ``[N]``.
        """
        import torch
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        epsilon = float(kwargs.get('epsilon', 1e-6))

        rewards = reduce_rewards(rewards, kwargs.get('reward_weights'))
        rewards = apply_kl_in_reward(
            rewards,
            kl_in_reward=kwargs.get('kl_in_reward', False),
            beta=kwargs.get('beta', 0.0),
            kl_values=kwargs.get('kl_values'),
        )

        if num_generations < 2:
            raise ValueError('GRPO-Pass@k requires num_generations >= 2')
        if rewards.numel() % num_generations != 0:
            raise ValueError(f'rewards numel ({rewards.numel()}) must be divisible by num_generations ({num_generations})')

        K = num_generations
        grouped = rewards.view(-1, K)
        n_groups = grouped.shape[0]

        advantages = torch.zeros_like(rewards.view(-1, K))
        norm_by_std = (scale == 'group')

        for g in range(n_groups):
            group_rewards = grouped[g]
            topk_vals, topk_idx = torch.topk(group_rewards, 2)
            r_max, r_second_max = topk_vals[0], topk_vals[1]
            adv_val = r_max - r_second_max
            if norm_by_std:
                std = group_rewards.std()
                adv_val = adv_val / (std + epsilon)
            advantages[g, topk_idx[0]] = adv_val

        return advantages.view(-1)
