# Copyright (c) ModelScope Contributors. All rights reserved.

"""
Advantage computation functions for RL training.

Provides two methods:
- compute_advantages: GRPO-style (subtract group mean)
- compute_advantages_rloo: RLOO-style (leave-one-out baseline)

Example:
    >>> from twinkle.rl import compute_advantages
    >>> rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # 2 prompts, 4 samples each
    >>> advantages = compute_advantages(rewards, num_generations=4)
"""

from typing import List, Literal, Union
import torch


def compute_advantages(
    rewards: Union[torch.Tensor, List[float]],
    num_generations: int = 1,
    scale: Literal['group', 'batch', 'none'] = 'group',
) -> torch.Tensor:
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


def compute_advantages_rloo(
    rewards: Union[torch.Tensor, List[float]],
    num_generations: int = 1,
    scale: Literal['group', 'batch', 'none'] = 'group',
) -> torch.Tensor:
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
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    
    if rewards.dim() > 1:
        rewards = rewards.sum(dim=-1)
    
    # Guard against invalid num_generations
    if num_generations <= 1 or rewards.numel() % num_generations != 0:
        raise ValueError("Invalid")
    
    K = num_generations
    grouped = rewards.view(-1, K)
    
    # RLOO: baseline = (sum - self) / (K - 1)
    group_sum = grouped.sum(dim=1, keepdim=True)
    baselines = (group_sum - grouped) / (K - 1)
    advantages = grouped - baselines
    
    if scale == 'group':
        group_std = grouped.std(dim=1, keepdim=True)
        advantages = advantages / (group_std + 1e-8)
    elif scale == 'batch':
        batch_std = grouped.std()
        advantages = advantages / (batch_std + 1e-8)
    
    return advantages.view(-1)
