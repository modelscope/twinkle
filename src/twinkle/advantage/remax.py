# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from .base import Advantage
from ._utils import apply_kl_in_reward, reduce_rewards

if TYPE_CHECKING:
    import torch


class ReMaxAdvantage(Advantage):
    """ReMax advantage estimator (https://arxiv.org/abs/2310.10505).

    Unlike group-relative estimators, ReMax uses an externally provided baseline
    (typically the reward from greedy decoding) to reduce variance:
        advantage_i = reward_i - baseline_i

    The baseline must be supplied via ``kwargs['reward_baselines']`` — a tensor of
    shape ``[N]`` with one baseline value per sample.
    """

    def __call__(self,
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
        rewards = apply_kl_in_reward(
            rewards,
            kl_in_reward=kwargs.get('kl_in_reward', False),
            beta=kwargs.get('beta', 0.0),
            kl_values=kwargs.get('kl_values'),
        )

        advantages = rewards - reward_baselines

        if scale == 'batch':
            std = advantages.std() if advantages.numel() > 1 else torch.ones(1, device=advantages.device)
            advantages = advantages / (std + 1e-8)

        return advantages.view(-1)
