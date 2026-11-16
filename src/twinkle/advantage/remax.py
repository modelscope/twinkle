# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from twinkle import remote_class, remote_function

from .base import Advantage
from ._utils import apply_kl_in_reward, reduce_rewards

if TYPE_CHECKING:
    import torch


@remote_class(execute='first')
class ReMaxAdvantage(Advantage):
    """ReMax advantage estimator (https://arxiv.org/abs/2310.10505).

    Unlike group-relative estimators, ReMax uses an externally provided baseline
    (typically the reward from greedy decoding) to reduce variance:
        advantage_i = reward_i - baseline_i

    The baseline must be supplied via ``kwargs['reward_baselines']`` — a tensor of
    shape ``[N]`` with one baseline value per sample.
    """

    @remote_function(execute='first', collect='first', stash=(0,))
    def __call__(self, trajs, **kwargs):
        """Trajectories and forward lanes in, training-target columns out -- see :class:`Advantage`."""
        return self._call(trajs, **kwargs)

    def _per_sequence(self,
                      rewards: Union['torch.Tensor', List[float]],
                      num_generations: int = 1,
                      scale: Literal['group', 'batch', 'none'] = 'none',
                      **kwargs) -> 'torch.Tensor':
        """Compute ReMax advantages.

        Args:
            rewards: ``[N]`` scalar rewards (or ``[N, n_funcs]`` matrix, reduced by weights).
            num_generations: Ignored for ReMax (no group-relative computation), kept for
                interface compatibility.
            scale: ``'none'`` (default, raw advantage), ``'batch'`` (divide by batch std).
            **kwargs:
                reward_baselines: ``[N]`` tensor of per-sample baselines (required).
                reward_weights: Optional weights for multi-function reward reduction.
                kl_in_reward / beta / kl_values: Optional ref-model KL regularization.

        Returns:
            advantages: Tensor of shape ``[N]``.
        """
        import torch
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        reward_baselines = kwargs.get('reward_baselines')
        if reward_baselines is None:
            raise ValueError("ReMaxAdvantage requires 'reward_baselines' in kwargs (per-sample greedy baselines).")
        if not isinstance(reward_baselines, torch.Tensor):
            reward_baselines = torch.tensor(reward_baselines, dtype=rewards.dtype, device=rewards.device)

        rewards = reduce_rewards(rewards, kwargs.get('reward_weights'))
        # A row whose every scorer failed is NaN out of the fold and must contribute no training
        # signal. Zeroing the reward alone does not achieve that: subtracting the baseline would
        # leave -baseline behind, a fabricated advantage on a row that was never scored. The same
        # holds on the baseline side -- a baseline whose own scorer failed folds to NaN, and
        # treating it as a real 0 would credit the row with its entire reward as excess over the
        # baseline. The mask is taken here because nan_to_num below destroys the only record of
        # which rows failed, on either side.
        unscored = torch.isnan(rewards) | torch.isnan(reward_baselines)
        rewards = torch.nan_to_num(rewards, nan=0.0)
        reward_baselines = torch.nan_to_num(reward_baselines, nan=0.0)
        rewards = apply_kl_in_reward(
            rewards,
            kl_in_reward=kwargs.get('kl_in_reward', False),
            beta=kwargs.get('beta', 0.0),
            kl_values=kwargs.get('kl_values'),
        )

        advantages = (rewards - reward_baselines).masked_fill(unscored, 0.0)

        if scale == 'batch':
            # Unscored rows stay out of the scale as well, so a batch's normalization is set by the
            # rows that actually carry signal -- the same exclusion grpo/rloo apply to their std.
            scored = advantages[~unscored]
            std = scored.std() if scored.numel() > 1 else torch.ones(1, device=advantages.device)
            advantages = advantages / (std + 1e-8)

        return advantages.view(-1)
