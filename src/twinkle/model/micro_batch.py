# Copyright (c) ModelScope Contributors. All rights reserved.
# Packing algorithms are adapted from AReaL (Apache-2.0).
from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class MicroBatchConfig:
    micro_batch_size: int
    dynamic_batching: bool = False
    max_tokens_per_micro_batch: int | None = None
    packing_algorithm: Literal["ffd", "kk"] = "ffd"

    def __post_init__(self):
        if self.micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be positive, got {self.micro_batch_size}")
        if self.packing_algorithm not in ("ffd", "kk"):
            raise ValueError(f"packing_algorithm must be ffd or kk, got {self.packing_algorithm!r}")
        if self.dynamic_batching and (self.max_tokens_per_micro_batch is None or self.max_tokens_per_micro_batch <= 0):
            raise ValueError("max_tokens_per_micro_batch must be positive when dynamic_batching=true")

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> MicroBatchConfig | None:
        option_names = (
            "micro_batch_size",
            "dynamic_batching",
            "max_tokens_per_micro_batch",
            "packing_algorithm",
        )
        if not any(name in kwargs for name in option_names):
            return None
        if "micro_batch_size" not in kwargs:
            raise ValueError("micro_batch_size is required when configuring micro-batching")
        return cls(
            micro_batch_size=int(kwargs.pop("micro_batch_size")),
            dynamic_batching=bool(kwargs.pop("dynamic_batching", False)),
            max_tokens_per_micro_batch=kwargs.pop("max_tokens_per_micro_batch", None),
            packing_algorithm=kwargs.pop("packing_algorithm", "ffd"),
        )


def sequence_length(model_input: dict[str, Any]) -> int:
    input_ids = model_input["input_ids"]
    return int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(input_ids)


def _batch_cost(group: list[int], lengths: list[int], padding_free: bool) -> int:
    if not group:
        return 0
    values = [lengths[index] for index in group]
    return sum(values) if padding_free else max(values) * len(values)


def _fits(group: list[int], index: int, lengths: list[int], config: MicroBatchConfig, padding_free: bool) -> bool:
    if len(group) >= config.micro_batch_size:
        return False
    candidate = [*group, index]
    return _batch_cost(candidate, lengths, padding_free) <= config.max_tokens_per_micro_batch


def _ffd_allocate(
    lengths: list[int], config: MicroBatchConfig, padding_free: bool, min_micro_batches: int
) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(min_micro_batches)]
    for index in sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True):
        candidates = [
            group_index
            for group_index, group in enumerate(groups)
            if _fits(group, index, lengths, config, padding_free)
        ]
        if not candidates:
            groups.append([index])
            continue
        group_index = min(
            candidates,
            key=lambda candidate: (
                _batch_cost(groups[candidate], lengths, padding_free),
                len(groups[candidate]),
            ),
        )
        groups[group_index].append(index)
    return [group for group in groups if group]


class _KKSet:
    __slots__ = ("total", "items")

    def __init__(self):
        self.total = 0
        self.items: list[int] = []

    def add(self, index: int, value: int) -> None:
        self.items.append(index)
        self.total += value

    def merge(self, other: _KKSet) -> None:
        self.items.extend(other.items)
        self.total += other.total

    def __lt__(self, other: _KKSet) -> bool:
        return (self.total, len(self.items), self.items) < (other.total, len(other.items), other.items)


class _KKState:
    __slots__ = ("sets",)

    def __init__(self, items: list[tuple[int, int]], group_count: int):
        self.sets = [_KKSet() for _ in range(group_count)]
        for group, (index, value) in zip(self.sets, items, strict=False):
            group.add(index, value)
        self.sets.sort(reverse=True)

    @property
    def spread(self) -> int:
        return self.sets[0].total - self.sets[-1].total

    def merge(self, other: _KKState) -> None:
        for index in range(len(self.sets)):
            self.sets[index].merge(other.sets[-1 - index])
        self.sets.sort(reverse=True)

    def __lt__(self, other: _KKState) -> bool:
        return self.spread > other.spread


def _kk_partition(lengths: list[int], group_count: int) -> list[list[int]]:
    queue = []
    for value, index in sorted((value, index) for index, value in enumerate(lengths)):
        heapq.heappush(queue, _KKState([(index, value)], group_count))
    while len(queue) > 1:
        first = heapq.heappop(queue)
        second = heapq.heappop(queue)
        first.merge(second)
        heapq.heappush(queue, first)
    return [group.items for group in queue[0].sets if group.items]


def _kk_allocate(
    lengths: list[int], config: MicroBatchConfig, padding_free: bool, min_micro_batches: int
) -> list[list[int]]:
    capacity = config.max_tokens_per_micro_batch
    group_count = max(min_micro_batches, math.ceil(sum(lengths) / capacity))
    while group_count <= len(lengths):
        groups = _kk_partition(lengths, group_count)
        if all(
            len(group) <= config.micro_batch_size and _batch_cost(group, lengths, padding_free) <= capacity
            for group in groups
        ):
            return groups
        group_count += 1
    raise ValueError("unable to construct a valid KK micro-batch plan")


def plan_micro_batches(
    inputs: list[dict[str, Any]],
    config: MicroBatchConfig,
    *,
    padding_free: bool,
    min_micro_batches: int = 1,
) -> list[list[int]]:
    if not inputs:
        raise ValueError("cannot plan micro-batches for empty inputs")
    if min_micro_batches <= 0 or min_micro_batches > len(inputs):
        raise ValueError(f"invalid min_micro_batches={min_micro_batches} for {len(inputs)} inputs")
    if not config.dynamic_batching:
        group_count = max(min_micro_batches, math.ceil(len(inputs) / config.micro_batch_size))
        base_size, remainder = divmod(len(inputs), group_count)
        groups = []
        start = 0
        for group_index in range(group_count):
            size = base_size + int(group_index < remainder)
            groups.append(list(range(start, start + size)))
            start += size
        return groups
    lengths = [sequence_length(model_input) for model_input in inputs]
    capacity = config.max_tokens_per_micro_batch
    oversized = [length for length in lengths if length > capacity]
    if oversized:
        raise ValueError(f"sequence length {max(oversized)} exceeds max_tokens_per_micro_batch={capacity}")
    if config.packing_algorithm == "ffd":
        return _ffd_allocate(lengths, config, padding_free, min_micro_batches)
    return _kk_allocate(lengths, config, padding_free, min_micro_batches)


def select_batch(value: Any, indices: list[int], batch_size: int) -> Any:
    if isinstance(value, list):
        return [value[index] for index in indices] if len(value) == batch_size else value
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices) if len(value) == batch_size else value
    if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == batch_size:
        return value[indices]
    return value


def merge_micro_batch_outputs(
    outputs: list[dict[str, Any]],
    plan: list[list[int]],
    *,
    task: str = "causal_lm",
    micro_batch_scales: list[float] | None = None,
) -> dict[str, Any]:
    """Restore sample outputs and reduce auxiliary loss with explicit training scales."""
    merger = _MicroBatchOutputMerger(plan)
    return merger.merge(outputs, task=task, micro_batch_scales=micro_batch_scales)


class _MicroBatchOutputMerger:
    """Keep sample ordering shared across field-specific and recursive merge operations."""

    def __init__(self, plan):
        if not plan or any(not indices for indices in plan):
            raise ValueError("outputs must match a non-empty micro-batch plan")
        merged_sample_indices = [index for indices in plan for index in indices]
        if sorted(merged_sample_indices) != list(range(len(merged_sample_indices))):
            raise ValueError("micro-batch plan must cover every sample exactly once")
        self.plan = plan
        self.restore_order_indices = sorted(range(len(merged_sample_indices)), key=merged_sample_indices.__getitem__)

    @staticmethod
    def _invalid(path, reason):
        raise ValueError(f"{path}: {reason}")

    def merge(self, outputs, *, task, micro_batch_scales):
        if not outputs or len(outputs) != len(self.plan):
            raise ValueError("outputs must match a non-empty micro-batch plan")
        if any(set(output) != set(outputs[0]) for output in outputs):
            self._invalid("outputs", "inconsistent output keys across micro-batches")
        pooled_tasks = ("embedding", "generative_reranker", "seq_cls", "reranker")
        known_sequence_dims = {
            "logps": (1,),
            "values": (1,),
            "entropies": (1,),
            "logits": () if task in pooled_tasks else (1,),
            "embeddings": (),
            "routed_experts": (1,),
        }
        return {
            field: self._merge_field(
                field, [output[field] for output in outputs], known_sequence_dims, micro_batch_scales
            )
            for field in outputs[0]
        }

    def _merge_field(self, field, values, known_sequence_dims, micro_batch_scales):
        path = f"outputs.{field}"
        if field == "aux_loss":
            return self._reduce_aux_loss(values, path, micro_batch_scales)
        if field == "past_key_values":
            if any(value is not None for value in values):
                self._invalid(path, "forward_only expects None")
            return None
        if field in known_sequence_dims:
            return self._merge_values(values, path, known_sequence_dims[field])
        return self._merge_values(values, path)

    @staticmethod
    def _reduce_aux_loss(values, path, micro_batch_scales):
        import torch

        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"{path}: inconsistent optional outputs across micro-batches")
        if micro_batch_scales is None or len(micro_batch_scales) != len(values):
            raise ValueError(f"{path}: one training scale per micro-batch is required")
        if any(
            not isinstance(value, (int, float, torch.Tensor)) or (isinstance(value, torch.Tensor) and value.ndim != 0)
            for value in values
        ):
            raise ValueError(f"{path}: expected scalar auxiliary losses")
        return sum(value * scale for value, scale in zip(values, micro_batch_scales, strict=False))

    def _merge_values(self, values, path, sequence_dims=None):
        import torch

        first = values[0]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            self._invalid(path, "inconsistent optional outputs across micro-batches")
        if any(type(value) is not type(first) for value in values):
            self._invalid(path, "inconsistent output types across micro-batches")
        if isinstance(first, torch.Tensor):
            return self._merge_tensors(values, path, sequence_dims)
        if isinstance(first, dict):
            return self._merge_dicts(values, path, sequence_dims)
        if isinstance(first, tuple):
            return self._merge_tuples(values, path, sequence_dims)
        if isinstance(first, list):
            return self._merge_sample_lists(values, path)
        self._invalid(path, f"unsupported micro-batch output type: {type(first).__name__}")

    def _merge_tensors(self, values, path, sequence_dims):
        from twinkle.utils import pad_and_stack_tensors

        first = values[0]
        if first.ndim == 0:
            self._invalid(path, "scalar forward outputs require an explicit reduction rule")
        if any(
            value.ndim != first.ndim or value.dtype != first.dtype or value.device != first.device for value in values
        ):
            self._invalid(path, "tensor rank, dtype and device must agree across micro-batches")
        if any(value.shape[0] != len(indices) for value, indices in zip(values, self.plan, strict=False)):
            self._invalid(
                path, f"expected a sample dimension matching the plan; got {[tuple(v.shape) for v in values]}"
            )
        # Known fields specify their sequence axes; unknown tensors use generic padding.
        padding_dims = tuple(range(1, first.ndim)) if sequence_dims is None else sequence_dims
        for dim in range(1, first.ndim):
            if dim not in padding_dims and any(value.shape[dim] != first.shape[dim] for value in values):
                self._invalid(path, f"non-sequence dimension {dim} must agree across micro-batches")
        combined = pad_and_stack_tensors(values, pad_value=0)
        return combined[self.restore_order_indices]

    def _merge_dicts(self, values, path, sequence_dims):
        first = values[0]
        if any(set(value) != set(first) for value in values):
            self._invalid(path, "inconsistent output keys across micro-batches")
        return {
            key: self._merge_values([value[key] for value in values], f"{path}.{key}", sequence_dims) for key in first
        }

    def _merge_tuples(self, values, path, sequence_dims):
        if any(len(value) != len(values[0]) for value in values):
            self._invalid(path, "tuple lengths must agree across micro-batches")
        return tuple(
            self._merge_values([value[index] for value in values], f"{path}[{index}]", sequence_dims)
            for index in range(len(values[0]))
        )

    def _merge_sample_lists(self, values, path):
        if any(len(value) != len(indices) for value, indices in zip(values, self.plan, strict=False)):
            self._invalid(path, "output list does not match micro-batch sample counts")
        flattened = [item for value in values for item in value]
        return [flattened[index] for index in self.restore_order_indices]
