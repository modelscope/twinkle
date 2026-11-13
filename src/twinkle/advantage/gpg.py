# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from twinkle import remote_class, remote_function

from .base import Advantage
from ._utils import apply_kl_in_reward, reduce_rewards

if TYPE_CHECKING:
    import torch


@remote_class(execute='first')
class GPGAdvantage(Advantage):
    """GPG (Group Policy Gradient) advantage estimator.

    Adapted from https://github.com/AMAP-ML/GPG (VisualThinker-R1-Zero).

    Formula:
        alpha = batch_size / count_nonzero(scores)
        advantage_i = alpha * (score_i - group_mean_i) / f_norm

    The dynamic ``alpha`` reweights by the fraction of non-zero scores in the batch,
    effectively upweighting informative samples when many are zero-reward.
    """

    _reads_f_norm = True

    @remote_function(execute='first', collect='first', stash=(0,))
    def __call__(self, trajs, **kwargs):
        """Trajectories and forward lanes in, training-target columns out -- see :class:`Advantage`."""
        return self._call(trajs, **kwargs)

    def _per_sequence(self,
                      rewards: Union['torch.Tensor', List[float]],
                      num_generations: int = 1,
                      scale: Literal['group', 'batch', 'none'] = 'none',
                      **kwargs) -> 'torch.Tensor':
        """Compute GPG advantages.

        Args:
            rewards: ``[N]`` scalar rewards (or ``[N, n_funcs]`` matrix).
            num_generations: Group size K.
            scale: ``'none'`` (default); ``'batch'`` divides by batch std.
            **kwargs:
                f_norm: Float normalization constant (default 1.0).
                reward_weights: Optional weights for multi-function reward reduction.
                kl_in_reward / beta / kl_values: Optional ref-model KL regularization.

        Returns:
            advantages: Tensor of shape ``[N]``.
        """
        import torch
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        f_norm = float(kwargs.get('f_norm', 1.0))

        rewards = reduce_rewards(rewards, kwargs.get('reward_weights'))
        # An unscored row (nan out of reduce_rewards) records 0 here, exactly as the legacy
        # weighted-nansum fold did -- nan-aware group exclusion is grpo/rloo/passk's contract.
        rewards = torch.nan_to_num(rewards, nan=0.0)
        rewards = apply_kl_in_reward(
            rewards,
            kl_in_reward=kwargs.get('kl_in_reward', False),
            beta=kwargs.get('beta', 0.0),
            kl_values=kwargs.get('kl_values'),
        )

        if num_generations <= 0 or rewards.numel() % num_generations != 0:
            raise ValueError(f'rewards numel ({rewards.numel()}) must be divisible by num_generations ({num_generations})')

        K = num_generations
        bsz = rewards.numel()

        # Dynamic alpha: batch_size / count_nonzero(scores), clamped to avoid div-by-zero
        m = torch.count_nonzero(rewards)
        alpha = bsz / m.clamp(min=1).to(rewards.dtype)

        grouped = rewards.view(-1, K)
        group_mean = grouped.mean(dim=1, keepdim=True)
        advantages = alpha * (grouped - group_mean) / f_norm

        if scale == 'batch':
            std = advantages.std() if advantages.numel() > 1 else torch.ones(1, device=advantages.device)
            advantages = advantages / (std + 1e-8)

        return advantages.view(-1)
