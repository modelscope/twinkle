# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, List, Literal, Union

from .base import Advantage
from ._utils import apply_kl_in_reward, reduce_rewards

if TYPE_CHECKING:
    import torch


class ReinforcePPBaselineAdvantage(Advantage):
    """Reinforce++-baseline advantage estimator (https://arxiv.org/abs/2501.03262).

    Differs from :class:`ReinforcePlusPlusAdvantage` in the normalization step:
    after group-mean subtraction, applies a global whitening (subtract batch mean,
    divide by batch std) rather than dividing by the group advantage std.

    This matches verl's ``reinforce_plus_plus_baseline`` which calls
    ``masked_whiten(scores, response_mask)`` after mean subtraction. ``masked_whiten`` is
    ``(x - mean) * rsqrt(var_unbiased + 1e-8)``, so the epsilon lives INSIDE the square root with the
    Bessel-corrected (unbiased) variance -- replicating that form exactly (rather than ``x / (std + eps)``)
    is what keeps the two within fp32 atol=1e-6.

    Formula:
        centered_i = score_i - group_mean_i
        advantages = (centered - mean(centered)) * rsqrt(var_unbiased(centered) + 1e-8)
    """

    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none'] = 'batch',
                 **kwargs) -> 'torch.Tensor':
        """Compute Reinforce++-baseline advantages.

        Args:
            rewards: ``[N]`` scalar rewards (or ``[N, n_funcs]`` matrix).
            num_generations: Group size K.
            scale: ``'batch'`` (default) applies global whitening; ``'none'`` skips it.
            **kwargs:
                reward_weights: Optional weights for multi-function reward reduction.
                kl_in_reward / beta / kl_values: Optional ref-model KL regularization.

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
            raise ValueError(f'rewards numel ({rewards.numel()}) must be divisible by num_generations ({num_generations})')

        K = num_generations

        # Group-mean subtraction (verl uses a zero baseline for a size-1 group; K>=2 is the real GRPO case)
        if K > 1:
            grouped = rewards.view(-1, K)
            group_mean = grouped.mean(dim=1, keepdim=True)
            centered = (grouped - group_mean).view(-1)
        else:
            centered = rewards.clone()

        # Global whitening: verl's masked_whiten = (x - mean) * rsqrt(var_unbiased + 1e-8)
        if scale == 'batch' and centered.numel() > 1:
            mean = centered.mean()
            var = centered.var(unbiased=True)
            centered = (centered - mean) * torch.rsqrt(var + 1e-8)

        return centered.view(-1)
