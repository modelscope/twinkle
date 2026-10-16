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
    packing_algorithm: Literal['ffd', 'kk'] = 'ffd'

    def __post_init__(self):
        if self.micro_batch_size <= 0:
            raise ValueError(f'micro_batch_size must be positive, got {self.micro_batch_size}')
        if self.packing_algorithm not in ('ffd', 'kk'):
            raise ValueError(f'packing_algorithm must be ffd or kk, got {self.packing_algorithm!r}')
        if self.dynamic_batching and (self.max_tokens_per_micro_batch is None or self.max_tokens_per_micro_batch <= 0):
            raise ValueError('max_tokens_per_micro_batch must be positive when dynamic_batching=true')

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> MicroBatchConfig | None:
        option_names = (
            'micro_batch_size',
            'dynamic_batching',
            'max_tokens_per_micro_batch',
            'packing_algorithm',
        )
        if not any(name in kwargs for name in option_names):
            return None
        if 'micro_batch_size' not in kwargs:
            raise ValueError('micro_batch_size is required when configuring micro-batching')
        return cls(
            micro_batch_size=int(kwargs.pop('micro_batch_size')),
            dynamic_batching=bool(kwargs.pop('dynamic_batching', False)),
            max_tokens_per_micro_batch=kwargs.pop('max_tokens_per_micro_batch', None),
            packing_algorithm=kwargs.pop('packing_algorithm', 'ffd'),
        )


def sequence_length(model_input: dict[str, Any]) -> int:
    input_ids = model_input['input_ids']
    return int(input_ids.shape[-1]) if hasattr(input_ids, 'shape') else len(input_ids)


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


def _ffd_allocate(lengths: list[int], config: MicroBatchConfig, padding_free: bool,
                  min_micro_batches: int) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(min_micro_batches)]
    for index in sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True):
        candidates = [
            group_index for group_index, group in enumerate(groups)
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
    __slots__ = ('total', 'items')

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
    __slots__ = ('sets', )

    def __init__(self, items: list[tuple[int, int]], group_count: int):
        self.sets = [_KKSet() for _ in range(group_count)]
        for group, (index, value) in zip(self.sets, items):
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


def _kk_allocate(lengths: list[int], config: MicroBatchConfig, padding_free: bool,
                 min_micro_batches: int) -> list[list[int]]:
    capacity = config.max_tokens_per_micro_batch
    group_count = max(min_micro_batches, math.ceil(sum(lengths) / capacity))
    while group_count <= len(lengths):
        groups = _kk_partition(lengths, group_count)
        if all(
                len(group) <= config.micro_batch_size and _batch_cost(group, lengths, padding_free) <= capacity
                for group in groups):
            return groups
        group_count += 1
    raise ValueError('unable to construct a valid KK micro-batch plan')


def plan_micro_batches(
    inputs: list[dict[str, Any]],
    config: MicroBatchConfig,
    *,
    padding_free: bool,
    min_micro_batches: int = 1,
) -> list[list[int]]:
    if not inputs:
        raise ValueError('cannot plan micro-batches for empty inputs')
    if min_micro_batches <= 0 or min_micro_batches > len(inputs):
        raise ValueError(f'invalid min_micro_batches={min_micro_batches} for {len(inputs)} inputs')
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
        raise ValueError(f'sequence length {max(oversized)} exceeds max_tokens_per_micro_batch={capacity}')
    if config.packing_algorithm == 'ffd':
        return _ffd_allocate(lengths, config, padding_free, min_micro_batches)
    return _kk_allocate(lengths, config, padding_free, min_micro_batches)


def select_batch(value: Any, indices: list[int], batch_size: int) -> Any:
    if isinstance(value, list):
        return [value[index] for index in indices] if len(value) == batch_size else value
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices) if len(value) == batch_size else value
    if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == batch_size:
        return value[indices]
    return value
