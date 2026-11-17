# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from twinkle import remote_class, remote_function

from .base import Advantage
from ._utils import apply_kl_in_reward, nanstd, reduce_rewards

if TYPE_CHECKING:
    import torch


@remote_class(execute='first')
class ReinforcePlusPlusAdvantage(Advantage):

    @remote_function(execute='first', collect='first', stash=(0,))
    def __call__(self, trajs, **kwargs):
        """Trajectories and forward lanes in, training-target columns out -- see :class:`Advantage`."""
        return self._call(trajs, **kwargs)

    def _per_sequence(self,
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
        # A never-scored row stays NaN through the KL fold (an elementwise subtraction) and is kept
        # out of the group mean. Imputing 0 instead does two wrong things at once: it shifts every
        # scored row's advantage in that group, and it leaves the unscored row at -mean, a signal
        # built entirely from other rows' evidence. NaN - mean is NaN, so the row lands at exactly
        # zero below -- where grpo/rloo/passk land it too.
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
        group_mean = grouped.nanmean(dim=1).repeat_interleave(K)
        advantages = rewards - group_mean

        if scale == 'batch':
            std = nanstd(advantages).expand_as(advantages) if advantages.numel() > 1 else torch.zeros_like(advantages)
            advantages = advantages / (std + 1e-8)
        elif scale == 'group':
            std = (nanstd(advantages.view(-1, K), dim=1).repeat_interleave(K) if K > 1
                   else torch.zeros_like(advantages))
            advantages = advantages / (std + 1e-8)

        return torch.nan_to_num(advantages, nan=0.0).view(-1)
