# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from twinkle import remote_class, remote_function

from .base import Advantage
from ._utils import apply_kl_in_reward, compute_gdpo_advantages, nanstd, reduce_rewards

if TYPE_CHECKING:
    import torch


@remote_class(execute='first')
class RLOOAdvantage(Advantage):

    @remote_function(execute='first', collect='first', stash=(0,))
    def __call__(self, trajs, **kwargs):
        """Trajectories and forward lanes in, training-target columns out -- see :class:`Advantage`."""
        return self._call(trajs, **kwargs)

    def _per_sequence(self,
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

        # RLOO: baseline = mean of the OTHER K-1 samples. An unscored row rides as NaN out of
        # reduce_rewards: it stays out of every baseline (nansum + the valid mask) and lands at a
        # zero advantage. A group reduced to one valid sample has no "others" to average -- the
        # 0/0 baseline is NaN and nan_to_num zeroes the group below.
        valid = ~torch.isnan(grouped)
        count = valid.sum(dim=1, keepdim=True)
        group_sum = grouped.nansum(dim=1, keepdim=True)
        others_sum = group_sum - torch.nan_to_num(grouped, nan=0.0)
        others_count = (count - valid.to(count.dtype)).to(grouped.dtype)
        baselines = others_sum / others_count
        advantages = grouped - baselines

        if scale == 'group':
            group_std = nanstd(grouped, dim=1, keepdim=True)
            advantages = advantages / (group_std + 1e-8)
        elif scale == 'batch':
            batch_std = nanstd(grouped)
            advantages = advantages / (batch_std + 1e-8)

        return torch.nan_to_num(advantages, nan=0.0).view(-1)
