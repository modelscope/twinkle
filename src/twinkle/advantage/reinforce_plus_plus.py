# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from .base import Advantage
from ._utils import apply_kl_in_reward, reduce_rewards

if TYPE_CHECKING:
    import torch


class ReinforcePlusPlusAdvantage(Advantage):

    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none', 'gdpo'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """REINFORCE++ advantages: subtract the group mean, then normalize by the ADVANTAGE std.

        The only difference from :class:`GRPOAdvantage` is what the normalizer is computed over:
        GRPO divides the group-centered advantage by the std of the raw *rewards*, REINFORCE++ by the
        std of the *advantages* themselves. Mirrors swift rl_core's ``reinforce_plus_plus`` estimator,
        including that ``scale='gdpo'`` is a no-op here (GDPO is only defined for the plain estimators),
        so REINFORCE++ + gdpo leaves the centered advantage unscaled.

        Args:
            rewards: ``[N]`` rewards or a ``[N, n_funcs]`` matrix (summed, weighted by
                ``reward_weights`` when given).
            num_generations: group size ``K``.
            scale: ``'group'`` / ``'batch'`` divide by the advantage std; ``'none'`` / ``'gdpo'`` do not.

        Returns:
            advantages: Tensor of shape ``[N]``.
        """
        import torch
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        rewards = reduce_rewards(rewards, kwargs.get('reward_weights'))
        rewards = apply_kl_in_reward(
            rewards,
            kl_in_reward=kwargs.get('kl_in_reward', False),
            beta=kwargs.get('beta', 0.0),
            kl_values=kwargs.get('kl_values'),
        )

        if num_generations <= 0 or rewards.numel() % num_generations != 0:
            raise ValueError('Invalid')

        K = num_generations
        grouped = rewards.view(-1, K)
        group_mean = grouped.mean(dim=1).repeat_interleave(K)
        advantages = rewards - group_mean

        if scale == 'batch':
            std = advantages.std().expand_as(advantages) if advantages.numel() > 1 else torch.zeros_like(advantages)
            advantages = advantages / (std + 1e-8)
        elif scale == 'group':
            std = (advantages.view(-1, K).std(dim=1).repeat_interleave(K) if K > 1 else torch.zeros_like(advantages))
            advantages = advantages / (std + 1e-8)

        return advantages.view(-1)
