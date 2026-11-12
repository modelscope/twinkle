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
import copy
import functools
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


def _reorder(value: Any, order: list[int], batch_size: int) -> Any:
    # Mirrors what slice_dp slices: lists and tensors along dim 0, and dicts of them, key by key.
    import torch

    if isinstance(value, dict):
        return {key: _reorder(item, order, batch_size) for key, item in value.items()}
    if isinstance(value, list) and len(value) == batch_size:
        return [value[index] for index in order]
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
        return value[order]
    return value


def _sliced_with_other_size(value: Any, batch_size: int) -> bool:
    import torch

    if isinstance(value, dict):
        return any(_sliced_with_other_size(item, batch_size) for item in value.values())
    if isinstance(value, list):
        return len(value) != batch_size
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return value.shape[0] != batch_size
    return False


def batch_aligned(args: tuple, kwargs: dict[str, Any], batch_size: int) -> bool:
    """Whether every argument ``slice_dp`` slices has one entry per sample, so a reorder keeps them aligned."""
    return not any(_sliced_with_other_size(value, batch_size) for value in (*args, *kwargs.values()))


def has_per_sample_values(value: Any) -> bool:
    """Conservatively detect fresh batch data on a loss call whose inputs were reordered earlier."""
    import numpy as np
    import torch

    if isinstance(value, dict):
        return any(has_per_sample_values(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return True
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.ndim > 0
    return False


def call_lengths(args: tuple, kwargs: dict[str, Any]) -> list[int] | None:
    """Sequence lengths of a ``slice_dp`` call's ``inputs``, or None when the call cannot be reordered."""
    inputs = kwargs.get("inputs")
    if kwargs.get("return_logits"):
        # Raw logits of packed inputs are laid out per rank, so they cannot be put back in order.
        return None
    if not isinstance(inputs, list) or not inputs:
        return None
    if not batch_aligned(args, kwargs, len(inputs)):
        return None
    # Lists of per-sample ObjectRefs can be permuted without fetching their payloads. A whole
    # tensor hidden behind one ref cannot: its shape/alignment are unknown, so skip that call.
    import sys
    ray = sys.modules.get('ray')

    def opaque(value):
        if ray is not None and isinstance(value, ray.ObjectRef):
            return True
        if isinstance(value, dict):
            return any(opaque(item) for item in value.values())
        if isinstance(value, list):
            return False  # the list itself is the batch axis; its elements travel together
        return callable(value) and getattr(value, '_is_lazy_collect', False)

    if any(opaque(value) for value in (*args, *kwargs.values())):
        return None
    supplied = kwargs.get('input_lengths')
    if supplied is not None:
        if not isinstance(supplied, list) or len(supplied) != len(inputs):
            raise ValueError('input_lengths must contain one sequence length per input')
        lengths = [int(length) for length in supplied]
        if any(length <= 0 for length in lengths):
            raise ValueError('input_lengths must be positive')
        return lengths
    if not all(isinstance(item, dict) and 'input_ids' in item for item in inputs):
        return None
    return [sequence_length(model_input) for model_input in inputs]


def _estimate_megatron_ranks(rank_lengths, *, micro_batch_size, padding_free, variable_seq_lengths,
                            sequence_multiple, workload_fn):
    """Mirror InputProcessor.collate_fn's fixed consecutive Megatron micro-batches and CP/SP padding."""
    workloads, tokens, computed_tokens = [], [], []
    for lengths in rank_lengths:
        rounded = [((length + sequence_multiple - 1) // sequence_multiple) * sequence_multiple for length in lengths]
        workload = computed = 0
        for start in range(0, len(lengths), micro_batch_size):
            batch = rounded[start:start + micro_batch_size]
            if padding_free:
                workload += sum(workload_fn(length) for length in batch)
                computed += sum(batch)
            else:
                width = max(batch) if variable_seq_lengths else max(rounded)
                workload += len(batch) * workload_fn(width)
                computed += len(batch) * width
        workloads.append(workload)
        tokens.append(sum(lengths))
        computed_tokens.append(computed)
    return _RankEstimate(workloads, tokens, computed_tokens, len(rank_lengths[0]) // micro_batch_size)


def plan_call(lengths: list[int], dp_size: int, spec: dict[str, Any], kwargs: dict[str, Any]) -> DPBalancePlan | None:
    """Plan a call from the model's ``_dp_balance_spec``, or None when the call keeps its order untouched."""
    if spec.get('backend', 'transformers') not in ('transformers', 'megatron'):
        return None
    workload_fn = estimate_workload
    if spec["hidden_size"] is not None:
        workload_fn = functools.partial(estimate_workload, hidden_size=spec["hidden_size"])
    try:
        config = MicroBatchConfig.from_kwargs(dict(kwargs))
        rank_estimator = None
        if spec.get('backend') == 'megatron':
            # Keep every DP replica's schedule identical. Short tails have different sample weights
            # in mcore's per-micro-batch mean path; dynamic batching is not this backend's contract.
            if len(lengths) % dp_size or (config is not None and config.dynamic_batching):
                return None
            rank_size = len(lengths) // dp_size
            size = config.micro_batch_size if config is not None else min(2, rank_size)
            if not size or rank_size % size or size % spec['group_size']:
                return None
            rank_estimator = functools.partial(
                _estimate_megatron_ranks, micro_batch_size=size, padding_free=spec['padding_free'],
                variable_seq_lengths=spec.get('variable_seq_lengths', True),
                sequence_multiple=spec.get('sequence_multiple', 1), workload_fn=workload_fn)
        elif spec['group_size'] > 1 and config is not None:
            if config.dynamic_batching or config.micro_batch_size % spec['group_size']:
                return None
        plan = plan_dp_balance_for_lengths(
            lengths, dp_size, padding_free=spec['padding_free'], micro_batch_config=config,
            group_size=spec['group_size'], workload_fn=workload_fn, rank_estimator=rank_estimator)
    except ValueError:
        # Options or batch shapes the model itself rejects; let it report them.
        return None
    return plan if plan.metrics else None


# Local mode: every rank holds only its own slice of the batch, so the ranks swap samples instead of a
# driver reordering the whole batch. A "leaf" is one value slice_dp would slice (a list, or a tensor along
# dim 0), addressed by its path through the call's args and kwargs.


def _leaves(value: Any, path: tuple) -> dict[tuple, Any]:
    import torch

    if isinstance(value, dict):
        leaves = {}
        for key, item in value.items():
            leaves.update(_leaves(item, (*path, key)))
        return leaves
    if isinstance(value, list) or (isinstance(value, torch.Tensor) and value.dim() > 0):
        return {path: value}
    return {}


def call_leaves(args: tuple, kwargs: dict[str, Any]) -> dict[tuple, Any]:
    """The values of a call that ``slice_dp`` would slice, keyed by their path."""
    leaves = {}
    for index, arg in enumerate(args):
        leaves.update(_leaves(arg, ("args", index)))
    for key, value in kwargs.items():
        leaves.update(_leaves(value, ("kwargs", key)))
    return leaves


def leaf_layout(leaves: dict[tuple, Any]) -> dict[tuple, Any]:
    """What the ranks must agree on before their leaves can be joined into one batch."""
    return {path: "list" if isinstance(value, list) else (value.dim(), str(value.dtype)) for path, value in leaves.items()}


def to_cpu(value: Any) -> Any:
    """``value`` with every tensor detached onto the CPU, for ``all_gather_object``."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        moved = copy.copy(value)
        for key, item in value.items():
            moved[key] = to_cpu(item)
        return moved
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value


def _device(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.device if value.device.type != "cpu" else None
    items = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
    for item in items:
        device = _device(item)
        if device is not None:
            return device
    return None


def _to(value: Any, device) -> Any:
    import torch

    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to(item, device) for item in value)
    return value


def _take(value: Any, indices: list[int]) -> Any:
    if isinstance(value, list):
        return [value[index] for index in indices]
    return value[indices]


def _with_leaves(value: Any, path: tuple, leaves: dict[tuple, Any]) -> Any:
    if path in leaves:
        return leaves[path]
    if isinstance(value, dict):
        return {key: _with_leaves(item, (*path, key), leaves) for key, item in value.items()}
    return value


def exchange_call(
    plan: DPBalancePlan, rank_leaves: list[dict[tuple, Any]], data_rank: int, args: tuple, kwargs: dict[str, Any]
) -> tuple[tuple, dict[str, Any]]:
    """This rank's share of the reordered batch, built from every data rank's leaves (in data-rank order).

    Tensors are joined the way ``collect_tensor_dict`` joins them for the Ray driver, padding the
    trailing dims to the longest rank, and land on the device the local value was on.
    """
    from twinkle.utils import pad_and_stack_tensors

    size = len(plan.order) // len(rank_leaves)
    mine = plan.order[data_rank * size : (data_rank + 1) * size]
    swapped = {}
    for path, local in call_leaves(args, kwargs).items():
        parts = [leaves[path] for leaves in rank_leaves]
        joined = [item for part in parts for item in part] if isinstance(local, list) else pad_and_stack_tensors(parts)
        swapped[path] = _to(_take(joined, mine), _device(local))
    return (
        tuple(_with_leaves(arg, ("args", index), swapped) for index, arg in enumerate(args)),
        {key: _with_leaves(value, ("kwargs", key), swapped) for key, value in kwargs.items()},
    )


def _rows(value: Any) -> int | None:
    import torch

    if isinstance(value, list):
        return len(value)
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return value.shape[0]
    return None


def _restore_rank(value: Any, rank_values: list[Any], sources: list[tuple[int, int]]) -> Any:
    from twinkle.utils import pad_and_stack_tensors

    if isinstance(value, dict):
        restored = copy.copy(value)
        for key, item in value.items():
            restored[key] = _restore_rank(
                item, [other.get(key) if isinstance(other, dict) else None for other in rank_values], sources
            )
        return restored
    size = len(sources)
    if _rows(value) != size or any(_rows(other) != size or type(other) is not type(value) for other in rank_values):
        # Not one entry per sample (a loss, a count): it stays this rank's own value.
        return value
    if isinstance(value, list):
        return _to([rank_values[rank][row] for rank, row in sources], _device(value))
    return pad_and_stack_tensors([rank_values[rank][row : row + 1] for rank, row in sources]).to(value.device)


def normalize_rank_output(output: Any, size: int) -> Any:
    """Merge Megatron's micro-batch logps before local per-sample output restoration.

    Ray's collector already performs this merge. Keep the worker's metric cache untouched by
    returning a shallow copy, and only merge tensors whose rows cover this complete rank shard.
    """
    import torch

    from twinkle.utils import pad_and_stack_tensors

    if isinstance(output, dict):
        logps = output.get('logps')
        if (isinstance(logps, list) and logps and all(isinstance(item, torch.Tensor) and item.ndim >= 2
                                                    for item in logps)
                and sum(item.shape[0] for item in logps) == size):
            output = copy.copy(output)
            output['logps'] = pad_and_stack_tensors(logps)
    return output


def restore_rank_outputs(output: Any, rank_outputs: list[Any], plan: DPBalancePlan, data_rank: int) -> Any:
    """Put this rank's own samples' per-sample outputs back, fetched from the ranks that computed them.

    ``rank_outputs`` holds every data rank's output, in data-rank order. Values that are not per-sample
    keep this rank's own value.
    """
    size = len(plan.order) // len(rank_outputs)
    position = [0] * len(plan.order)
    for index, sample in enumerate(plan.order):
        position[sample] = index
    sources = [divmod(position[sample], size) for sample in range(data_rank * size, (data_rank + 1) * size)]
    return _restore_rank(output, rank_outputs, sources)

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

    @property
    def reorders(self) -> bool:
        return self.order != list(range(len(self.order)))

    def reorder_call(self, args: tuple, kwargs: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
        """Reorder every batch-aligned argument of a ``slice_dp`` call, the way ``slice_dp`` would slice it."""
        size = len(self.order)
        return (
            tuple(_reorder(arg, self.order, size) for arg in args),
            {key: _reorder(value, self.order, size) for key, value in kwargs.items()},
        )

    def restore_outputs(self, outputs: Any) -> Any:
        """Put the batch-aligned fields of a collected result back in the caller's order."""
        inverse = [0] * len(self.order)
        for position, index in enumerate(self.order):
            inverse[index] = position
        return _reorder(outputs, inverse, len(self.order))


def plan_dp_balance_for_lengths(
    lengths: Sequence[int],
    dp_size: int,
    *,
    padding_free: bool,
    micro_batch_config: MicroBatchConfig | None = None,
    group_size: int = 1,
    workload_fn: WorkloadFn = estimate_workload,
    rank_estimator: Callable | None = None,
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
            estimates[name] = order, (rank_estimator(rank_lengths) if rank_estimator is not None else
                                      _estimate_ranks(rank_lengths, micro_batch_config, padding_free, workload_fn))
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
