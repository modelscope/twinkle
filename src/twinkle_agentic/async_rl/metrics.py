# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stateless metrics specific to async RL policy and advantage semantics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def rollout_metrics(
        *,
        rewards: Mapping[str, Sequence[float]] | None = None,
        completion_lengths: Sequence[int] = (),
        stop_reasons: Sequence[str | None] = (),
        rollout_latency_s: float | None = None,
) -> dict[str, float | int]:
    """Summarize one RL rollout collection without retaining state."""
    metrics: dict[str, float | int] = {}
    reward_counts = [len(values) for values in (rewards or {}).values() if values]
    if len(set(reward_counts)) > 1:
        raise ValueError(f'reward metric lengths must match, got {reward_counts}')
    sample_count = len(completion_lengths) or (reward_counts[0] if reward_counts else 0)
    if completion_lengths and reward_counts and any(count != sample_count for count in reward_counts):
        raise ValueError(f'reward and completion metric lengths must match: {reward_counts} != {sample_count}')
    if stop_reasons and len(stop_reasons) != len(completion_lengths):
        raise ValueError('stop reason and completion metric lengths must match: '
                         f'{len(stop_reasons)} != {len(completion_lengths)}')
    if sample_count:
        metrics['sample_count'] = sample_count
    if completion_lengths:
        lengths = [int(value) for value in completion_lengths]
        output_tokens = sum(lengths)
        truncated_count = sum(reason == 'length' for reason in stop_reasons)
        metrics.update({
            'completion_length_mean': output_tokens / sample_count,
            'completion_length_p95': _p95(lengths),
            'completion_length_max': max(lengths),
            'completion_truncated_count': truncated_count,
            'completion_truncated_ratio': truncated_count / sample_count,
            'output_tokens': output_tokens,
        })
        if rollout_latency_s is not None:
            latency = float(rollout_latency_s)
            metrics['rollout_latency_s'] = latency
            metrics['output_tokens_per_s'] = output_tokens / latency if latency > 0 else 0.0
    elif rollout_latency_s is not None:
        metrics['rollout_latency_s'] = float(rollout_latency_s)

    for name, raw_values in (rewards or {}).items():
        values = [float(value) for value in raw_values]
        if not values:
            continue
        prefix = 'reward' if name == 'reward' else f'{name}_reward'
        metrics[prefix] = sum(values) / len(values)
        metrics[f'{prefix}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
    return metrics


def training_policy_metrics(
    sample_tags: tuple[dict[str, Any], ...],
    train_policy_version: int,
) -> dict[str, float | int]:
    if not sample_tags:
        raise ValueError('training batch must contain sample policy tags')
    final_versions = [int(tag['final_policy_version']) for tag in sample_tags]
    spans = [int(tag['policy_version_span']) for tag in sample_tags]
    gaps = [int(train_policy_version) - version for version in final_versions]
    if any(gap < 0 for gap in gaps):
        raise ValueError(
            f'training policy version {train_policy_version} is older than rollout versions {final_versions}')
    return {
        'policy_version_gap_mean': sum(gaps) / len(gaps),
        'policy_version_gap_p95': _p95(gaps),
        'policy_version_gap_max': max(gaps),
        'rollout_policy_span_mean': sum(spans) / len(spans),
        'rollout_policy_span_max': max(spans),
    }


def advantage_signal_metrics(
    rewards: Sequence[float],
    advantages: Sequence[float],
    *,
    num_generations: int,
    zero_tolerance: float = 1e-8,
) -> dict[str, float | int]:
    """Summarize whether GRPO groups provide a useful learning signal."""
    if num_generations <= 0:
        raise ValueError(f'num_generations must be positive, got {num_generations}')
    if len(rewards) != len(advantages):
        raise ValueError(f'rewards and advantages must have equal length: {len(rewards)} != {len(advantages)}')
    if len(rewards) == 0 or len(rewards) % num_generations:
        raise ValueError(f'advantage metrics require complete groups: sample_count={len(rewards)}, '
                         f'num_generations={num_generations}')

    reward_values = [float(value) for value in rewards]
    advantage_values = [float(value) for value in advantages]
    group_reward_stds: list[float] = []
    zero_advantage_groups = 0
    for start in range(0, len(reward_values), num_generations):
        group_rewards = reward_values[start:start + num_generations]
        group_advantages = advantage_values[start:start + num_generations]
        reward_mean = sum(group_rewards) / num_generations
        group_reward_stds.append(math.sqrt(sum((value - reward_mean)**2 for value in group_rewards) / num_generations))
        if max(abs(value) for value in group_advantages) <= zero_tolerance:
            zero_advantage_groups += 1

    advantage_mean = sum(advantage_values) / len(advantage_values)
    group_count = len(group_reward_stds)
    return {
        'group_count': group_count,
        'group_reward_std_mean': sum(group_reward_stds) / group_count,
        'zero_advantage_group_ratio': zero_advantage_groups / group_count,
        'positive_advantage_ratio': sum(value > zero_tolerance for value in advantage_values) / len(advantage_values),
        'advantage_mean': advantage_mean,
        'advantage_std':
        math.sqrt(sum((value - advantage_mean)**2 for value in advantage_values) / len(advantage_values)),
    }
