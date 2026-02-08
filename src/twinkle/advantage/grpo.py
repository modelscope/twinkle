# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Union, List, Literal, TYPE_CHECKING
from .base import Advantage
if TYPE_CHECKING:
    import torch


class GRPOAdvantage(Advantage):

    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none'] = 'group',
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

        if rewards.dim() > 1:
            rewards = rewards.sum(dim=-1)

        if num_generations <= 0 or rewards.numel() % num_generations != 0:
            raise ValueError("Invalid")

        if num_generations == 1:
            if scale == 'batch':
                std = rewards.std() if rewards.numel() > 1 else torch.ones(1, device=rewards.device)
                return (rewards - rewards.mean()) / (std + 1e-8)
            elif scale == 'none':
                return rewards - rewards.mean()
            else:
                return rewards

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