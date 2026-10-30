# Copyright (c) ModelScope Contributors. All rights reserved.
"""Data-parallel workload balancing for ``slice_dp`` dispatch.

``slice_dp`` hands every DP rank one contiguous ``DeviceMesh.get_slice`` chunk, and the step waits for the
slowest rank. ``plan_dp_balance`` permutes the batch so the chunks cost about the same, without changing how
many samples each rank receives.

Only batches that split evenly across the ranks are permuted. Losses such as ``GRPOLoss`` average over the
samples of one rank and DP then averages the ranks, so a sample weighs ``1 / (dp_size * rank_size)``; when
``get_slice`` gives some ranks one sample more, moving samples between ranks would change their weights.

A rank's cost is estimated by running the same micro-batch planner the model runs on its chunk, so the
estimate follows both the micro-batch layout and the input layout:

- packed (``padding_free=True``): every sequence costs ``workload_fn(length)``;
- padded: each micro-batch pads its rows to its longest sequence, so it costs ``rows * workload_fn(longest)``.

With dynamic batching the ranks are re-planned with a common micro-batch count, as the model does before
it starts the micro-batch loop.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .micro_batch import MicroBatchConfig, _batch_cost, _kk_partition, plan_micro_batches_for_lengths, sequence_length

WorkloadFn = Callable[[int], int]


def estimate_workload(length: int, hidden_size: int = 4096) -> int:
    """Relative forward cost of one dense-transformer sequence.

    Projections and MLP cost about ``12·h²·s`` FLOPs and attention ``2·h·s²``; dividing by ``2h`` keeps the
    ratio and gives ``6·h·s + s²``.
    """
    return 6 * hidden_size * length + length * length


def micro_batch_workload(lengths: Sequence[int], *, padding_free: bool, workload_fn: WorkloadFn = estimate_workload):
    """Estimated cost of one micro-batch with the given sequence lengths."""
    if not lengths:
        return 0
    if padding_free:
        return sum(workload_fn(length) for length in lengths)
    return len(lengths) * workload_fn(max(lengths))


@dataclass(frozen=True)
class _RankEstimate:
    workloads: list[int]
    tokens: list[int]
    computed_tokens: list[int]
    micro_batches: int


def _estimate_ranks(
    rank_lengths: list[list[int]],
    config: MicroBatchConfig | None,
    padding_free: bool,
    workload_fn: WorkloadFn,
) -> _RankEstimate:
    if config is None:
        plans = [[list(range(len(lengths)))] for lengths in rank_lengths]
    else:
        min_micro_batches = 1
        while True:
            plans = [
                plan_micro_batches_for_lengths(
                    lengths, config, padding_free=padding_free, min_micro_batches=min_micro_batches
                )
                for lengths in rank_lengths
            ]
            counts = [len(plan) for plan in plans]
            if not config.dynamic_batching or len(set(counts)) == 1:
                break
            min_micro_batches = max(counts)

    workloads, computed_tokens = [], []
    for lengths, plan in zip(rank_lengths, plans, strict=True):
        batches = [[lengths[index] for index in indices] for indices in plan]
        workloads.append(
            sum(micro_batch_workload(batch, padding_free=padding_free, workload_fn=workload_fn) for batch in batches)
        )
        computed_tokens.append(sum(_batch_cost(indices, lengths, padding_free) for indices in plan))
    return _RankEstimate(
        workloads=workloads,
        tokens=[sum(lengths) for lengths in rank_lengths],
        computed_tokens=computed_tokens,
        micro_batches=max(len(plan) for plan in plans),
    )


def _contiguous(group_count: int, dp_size: int) -> list[list[int]]:
    size = group_count // dp_size
    return [list(range(rank * size, (rank + 1) * size)) for rank in range(dp_size)]


def _karmarkar_karp(group_workloads: list[int], dp_size: int) -> list[list[int]]:
    return [sorted(part) for part in _kk_partition(group_workloads, dp_size, equal_size=True)]


def _snake(group_keys: list[tuple[int, int]], dp_size: int) -> list[list[int]]:
    """Deal groups from the heaviest down, reversing direction every round.

    Every rank then sees the same spread of lengths, so the micro-batch planner pads and splits all ranks
    alike, which is what padded micro-batches need.
    """
    ordered = sorted(range(len(group_keys)), key=lambda group: (group_keys[group], -group), reverse=True)
    ranks: list[list[int]] = [[] for _ in range(dp_size)]
    for start in range(0, len(ordered), dp_size):
        chunk = ordered[start : start + dp_size]
        targets = range(dp_size) if (start // dp_size) % 2 == 0 else range(dp_size - 1, -1, -1)
        for rank, group in zip(targets, chunk, strict=True):
            ranks[rank].append(group)
    return [sorted(groups) for groups in ranks]


def _imbalance(values: Sequence[int]) -> float:
    # max/mean estimates how much slower the step is than a perfectly even split.
    mean = sum(values) / len(values)
    return max(values) / mean if mean else 1.0


@dataclass(frozen=True)
class DPBalancePlan:
    """A batch permutation for ``slice_dp`` dispatch plus before/after balance metrics."""

    order: list[int]
    strategy: str = "identity"
    metrics: dict[str, float] = field(default_factory=dict)

    def apply(self, values: Any) -> Any:
        """Reorder one per-sample field (list, tuple, tensor or array) with the plan."""
        if len(values) != len(self.order):
            raise ValueError(f"cannot reorder {len(values)} values with a plan for {len(self.order)} samples")
        if isinstance(values, list):
            return [values[index] for index in self.order]
        if isinstance(values, tuple):
            return tuple(values[index] for index in self.order)
        if hasattr(values, "shape"):
            return values[self.order]
        raise TypeError(f"cannot reorder a field of type {type(values).__name__}")


def plan_dp_balance_for_lengths(
    lengths: Sequence[int],
    dp_size: int,
    *,
    padding_free: bool,
    micro_batch_config: MicroBatchConfig | None = None,
    group_size: int = 1,
    workload_fn: WorkloadFn = estimate_workload,
) -> DPBalancePlan:
    """Same as :func:`plan_dp_balance`, for samples given by their sequence lengths."""
    lengths = [int(length) for length in lengths]
    total = len(lengths)
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if group_size <= 0 or total % group_size:
        raise ValueError(f"batch size {total} must be a positive multiple of group_size={group_size}")
    group_count = total // group_size
    identity = list(range(total))
    if dp_size == 1 or group_count < dp_size:
        return DPBalancePlan(order=identity)
    if total % dp_size:
        # Uneven rank sizes give samples rank-dependent loss weights; see the module docstring.
        return DPBalancePlan(order=identity)
    if group_size > 1 and group_count % dp_size:
        raise ValueError(
            f"keeping groups of {group_size} together needs the group count ({group_count}) "
            f"to be divisible by dp_size={dp_size}"
        )
    sample_workloads = [workload_fn(length) for length in lengths]
    if min(sample_workloads) < 0:
        raise ValueError(f"workload_fn must return non-negative values, got {min(sample_workloads)}")

    def members(group: int) -> range:
        return range(group * group_size, (group + 1) * group_size)

    group_workloads = [sum(sample_workloads[index] for index in members(group)) for group in range(group_count)]
    group_longest = [max(lengths[index] for index in members(group)) for group in range(group_count)]
    candidates = {
        "identity": _contiguous(group_count, dp_size),
        "karmarkar_karp": _karmarkar_karp(group_workloads, dp_size),
        "snake": _snake(list(zip(group_longest, group_workloads, strict=True)), dp_size),
    }

    estimates: dict[str, tuple[list[int], _RankEstimate]] = {}
    for name, rank_groups in candidates.items():
        order = [index for groups in rank_groups for group in groups for index in members(group)]
        rank_size = total // dp_size
        rank_lengths = [
            [lengths[index] for index in order[rank * rank_size : (rank + 1) * rank_size]] for rank in range(dp_size)
        ]
        try:
            estimates[name] = order, _estimate_ranks(rank_lengths, micro_batch_config, padding_free, workload_fn)
        except ValueError:
            # The model rejects this layout too (for example a sequence over max_tokens_per_micro_batch).
            continue
    if not estimates:
        return DPBalancePlan(order=identity)

    # Ties keep the earlier candidate, so the batch is only permuted when the slowest rank gets lighter.
    strategy = min(estimates, key=lambda name: (max(estimates[name][1].workloads), estimates[name][1].micro_batches))
    order, after = estimates[strategy]
    metrics = {"dp_balance/rank_tokens_mean": sum(lengths) / dp_size}
    phases = [("after", after)]
    if "identity" in estimates:
        phases.insert(0, ("before", estimates["identity"][1]))
    for phase, estimate in phases:
        metrics[f"dp_balance/rank_tokens_max_{phase}"] = max(estimate.tokens)
        metrics[f"dp_balance/rank_tokens_min_{phase}"] = min(estimate.tokens)
        metrics[f"dp_balance/workload_imbalance_{phase}"] = _imbalance(estimate.workloads)
        metrics[f"dp_balance/micro_batches_{phase}"] = estimate.micro_batches
        if not padding_free:
            metrics[f"dp_balance/padding_ratio_{phase}"] = 1 - sum(estimate.tokens) / sum(estimate.computed_tokens)
    return DPBalancePlan(order=order, strategy=strategy, metrics=metrics)


def plan_dp_balance(
    inputs: Sequence[Any],
    dp_size: int,
    *,
    padding_free: bool,
    micro_batch_config: MicroBatchConfig | None = None,
    group_size: int = 1,
    workload_fn: WorkloadFn = estimate_workload,
) -> DPBalancePlan:
    """Plan a DP-balanced order for encoded ``inputs`` that will be dispatched with ``slice_dp``.

    Pass the processor's ``padding_free`` and the micro-batch options given to ``forward_backward``
    (``None`` when the model runs each rank's chunk as one batch). Apply the plan to ``inputs`` and every
    aligned per-sample field (old logps, advantages, ...) before calling the model; results returned by the
    model come back in the permuted order.

    Every rank keeps the sample count ``get_slice`` gives it, and ``group_size`` consecutive samples stay on
    one rank (for example chosen/rejected pairs). Within a rank, samples keep their original relative order.
    A batch whose size is not a multiple of ``dp_size`` keeps its order (see the module docstring).
    """
    return plan_dp_balance_for_lengths(
        [sequence_length(model_input) for model_input in inputs],
        dp_size,
        padding_free=padding_free,
        micro_batch_config=micro_batch_config,
        group_size=group_size,
        workload_fn=workload_fn,
    )
