# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared reward-shaping helpers for the group-relative advantage estimators.

These back the optional multi-reward / ref-KL / GDPO features that ``GRPOAdvantage`` and
``RLOOAdvantage`` expose through ``**kwargs``. Kept in one place so the two estimators cannot drift and
so a scalar single-reward call (no ``reward_weights`` / ``kl_*`` / ``scale='gdpo'``) still runs the
exact original code path.
"""
from typing import Optional, Sequence, Union

import torch


def nanstd(tensor: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    """Standard deviation ignoring NaNs, Bessel-corrected (NaN where fewer than 2 valid entries).

    Matches trl/swift semantics so GDPO normalization is identical to the legacy implementation.
    """
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    variance = torch.nanmean((tensor - mean)**2, dim=dim, keepdim=True)
    count = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=True)
    correction = count / (count - 1)
    correction = torch.where(count > 1, correction, torch.full_like(correction, float('nan')))
    variance = variance * correction  # Bessel's correction
    std = torch.sqrt(variance)
    if not keepdim and dim is not None:
        std = std.squeeze(dim)
    return std


def reduce_rewards(rewards: torch.Tensor, reward_weights: Optional[Sequence[float]]) -> torch.Tensor:
    """Collapse a ``[N, n_funcs]`` per-function reward matrix to ``[N]``.

    With no weights this is the plain sum the estimators used before (so existing single-reward and
    unweighted multi-reward behavior is unchanged). With weights it is the weighted ``nansum`` used by
    the legacy rl_core path, so a NaN from a skipped reward function drops out instead of poisoning the
    row.
    """
    if rewards.dim() <= 1:
        return rewards
    if reward_weights is None:
        return rewards.sum(dim=-1)
    weights = torch.as_tensor(reward_weights, dtype=rewards.dtype, device=rewards.device)
    return (rewards * weights.unsqueeze(0)).nansum(dim=1)


def apply_kl_in_reward(
    rewards: torch.Tensor,
    *,
    kl_in_reward: bool = False,
    beta: float = 0.0,
    kl_values: Optional[Union[torch.Tensor, Sequence[float]]] = None,
) -> torch.Tensor:
    """Subtract ``beta * kl_values`` from rewards BEFORE normalization (ref-model regularization)."""
    if kl_in_reward and beta != 0.0 and kl_values is not None:
        if not torch.is_tensor(kl_values):
            kl_values = torch.as_tensor(kl_values, dtype=rewards.dtype, device=rewards.device)
        rewards = rewards - beta * kl_values
    return rewards


def compute_gdpo_advantages(
    rewards_per_func: torch.Tensor,
    reward_weights: Optional[Sequence[float]],
    num_generations: int,
) -> torch.Tensor:
    """GDPO advantages: per-function group Z-score, weighted sum, then a batch Z-score.

    Unlike the plain estimators, GDPO normalizes each reward function within its prompt group first
    (so heterogeneous reward scales cannot dominate), then combines and batch-normalizes. It therefore
    needs the full ``[N, n_funcs]`` matrix rather than the summed reward.
    """
    K = num_generations
    n_funcs = rewards_per_func.shape[1]
    if reward_weights is None:
        weights = torch.ones(n_funcs, dtype=rewards_per_func.dtype, device=rewards_per_func.device)
    else:
        weights = torch.as_tensor(reward_weights, dtype=rewards_per_func.dtype, device=rewards_per_func.device)
    normalized = []
    for i in range(n_funcs):
        r_i = rewards_per_func[:, i].view(-1, K)
        g_mean = torch.nanmean(r_i, dim=1, keepdim=True)
        g_std = nanstd(r_i, dim=1, keepdim=True) + 1e-8
        norm_i = torch.nan_to_num((r_i - g_mean) / g_std, nan=0.0)
        normalized.append(weights[i] * norm_i.view(-1))
    summed = sum(normalized)
    return (summed - summed.mean()) / (summed.std() + 1e-8)
