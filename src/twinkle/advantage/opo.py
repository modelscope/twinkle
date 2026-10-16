# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from .base import Advantage
from ._utils import apply_kl_in_reward, reduce_rewards

if TYPE_CHECKING:
    import torch


class OPOAdvantage(Advantage):
    """OPO (Online Policy Optimization) advantage estimator (https://arxiv.org/pdf/2505.23585).

    Uses a length-weighted group baseline instead of the plain group mean:
        baseline = sum(len_i * score_i) / sum(len_i)   (within each prompt group)
        advantage_i = score_i - baseline

    The response lengths must be supplied via ``kwargs['response_lengths']`` — a tensor of
    shape ``[N]`` with the number of valid tokens per sample.
    """

    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none'] = 'none',
                 **kwargs) -> 'torch.Tensor':
        """Compute OPO advantages.

        Args:
            rewards: ``[N]`` scalar rewards (or ``[N, n_funcs]`` matrix).
            num_generations: Group size K.
            scale: ``'none'`` (default, raw advantage), ``'batch'`` (divide by batch std).
            **kwargs:
                response_lengths: ``[N]`` tensor of valid response token counts (required).
                reward_weights: Optional weights for multi-function reward reduction.
                kl_in_reward / beta / kl_values: Optional ref-model KL regularization.

        Returns:
            advantages: Tensor of shape ``[N]``.
        """
        import torch
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)

        response_lengths = kwargs.get('response_lengths')
        if response_lengths is None:
            raise ValueError("OPOAdvantage requires 'response_lengths' in kwargs.")
        if not isinstance(response_lengths, torch.Tensor):
            response_lengths = torch.tensor(response_lengths, dtype=rewards.dtype, device=rewards.device)

        rewards = reduce_rewards(rewards, kwargs.get('reward_weights'))
        rewards = apply_kl_in_reward(
            rewards,
            kl_in_reward=kwargs.get('kl_in_reward', False),
            beta=kwargs.get('beta', 0.0),
            kl_values=kwargs.get('kl_values'),
        )

        if num_generations <= 0 or rewards.numel() % num_generations != 0:
            raise ValueError(f'rewards numel ({rewards.numel()}) must be divisible by num_generations ({num_generations})')

        K = num_generations
        grouped_rewards = rewards.view(-1, K)
        grouped_lengths = response_lengths.view(-1, K).to(rewards.dtype)

        # Length-weighted baseline per group: sum(len_i * score_i) / sum(len_i)
        len_sum = grouped_lengths.sum(dim=1, keepdim=True).clamp(min=1e-8)
        baseline = (grouped_lengths * grouped_rewards).sum(dim=1, keepdim=True) / len_sum

        advantages = grouped_rewards - baseline

        if scale == 'batch':
            std = advantages.std() if advantages.numel() > 1 else torch.ones(1, device=advantages.device)
            advantages = advantages / (std + 1e-8)

        return advantages.view(-1)
