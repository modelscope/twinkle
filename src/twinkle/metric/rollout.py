# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, Optional, Sequence

import numpy as np


def zero_variance_reward_group_indices(
    rewards: Sequence[float],
    num_generations: int,
) -> list[int]:
    """Return GRPO group indices that cannot produce a relative advantage."""
    if num_generations <= 0:
        raise ValueError('num_generations must be positive')
    if len(rewards) % num_generations != 0:
        raise ValueError('rewards must form complete num_generations groups')
    if len(rewards) == 0:
        return []

    grouped_rewards = np.asarray(rewards, dtype=np.float64).reshape(-1, num_generations)
    group_ranges = np.ptp(grouped_rewards, axis=1)
    return np.flatnonzero(np.isclose(group_ranges, 0.0)).astype(int).tolist()


def compute_grpo_rollout_metrics(
    *,
    completion_lengths: Sequence[int],
    stop_reasons: Sequence[str],
    rewards: Sequence[float],
    advantages: Sequence[float],
    num_generations: int,
    sampling_masks: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """Reduce one GRPO rollout batch into scalar diagnostics."""
    if len(stop_reasons) != len(completion_lengths):
        raise ValueError('stop_reasons must align with completion_lengths')
    if len(rewards) != len(completion_lengths):
        raise ValueError('rewards must align with completion_lengths')
    if num_generations <= 0 or len(rewards) % num_generations != 0:
        raise ValueError('rewards must form complete num_generations groups')
    if len(advantages) != len(rewards):
        raise ValueError('advantages must align with rewards')

    metrics: Dict[str, float] = {}
    if len(completion_lengths) > 0:
        lengths = np.asarray(completion_lengths, dtype=np.float64)
        metrics['rollout/completion_length_p95'] = float(np.percentile(lengths, 95))

    if len(stop_reasons) > 0:
        num_sequences = len(stop_reasons)
        metrics['rollout/stop_rate'] = sum(reason == 'stop' for reason in stop_reasons) / num_sequences
        metrics['rollout/length_stop_rate'] = (
            sum(reason == 'length' for reason in stop_reasons) / num_sequences)

    if len(rewards) > 0:
        grouped_rewards = np.asarray(rewards, dtype=np.float64).reshape(-1, num_generations)
        if num_generations > 1:
            group_stds = grouped_rewards.std(axis=1, ddof=1)
        else:
            group_stds = np.zeros(grouped_rewards.shape[0], dtype=np.float64)
        metrics['grpo/group_reward_std_mean'] = float(group_stds.mean())
        zero_variance_groups = zero_variance_reward_group_indices(rewards, num_generations)
        metrics['grpo/zero_variance_group_fraction'] = (
            len(zero_variance_groups) / grouped_rewards.shape[0])
        metrics['grpo/nonzero_advantage_fraction'] = float(
            (~np.isclose(np.asarray(advantages, dtype=np.float64), 0.0)).mean())

    if sampling_masks is not None:
        if len(sampling_masks) != len(completion_lengths):
            raise ValueError('sampling_masks must align with completion_lengths')
        support_sizes = []
        for sequence_idx, (sampling_mask, completion_length) in enumerate(
                zip(sampling_masks, completion_lengths)):
            offsets = sampling_mask.offsets
            if len(offsets) - 1 != completion_length:
                raise ValueError(
                    f'sampling mask {sequence_idx} has {len(offsets) - 1} rows, '
                    f'expected {completion_length}')
            support_sizes.extend(end - start for start, end in zip(offsets, offsets[1:]))

        if support_sizes:
            sizes = np.asarray(support_sizes, dtype=np.float64)
            metrics['replay/support_size_mean'] = float(sizes.mean())
            metrics['replay/support_size_p50'] = float(np.percentile(sizes, 50))
            metrics['replay/support_size_p95'] = float(np.percentile(sizes, 95))
            metrics['replay/support_size_max'] = float(sizes.max())
            metrics['replay/singleton_fraction'] = float((sizes == 1).mean())

    return metrics
