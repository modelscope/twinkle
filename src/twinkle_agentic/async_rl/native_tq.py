# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adapters and helpers for the native TransferQueue client API.

The async RL data path deliberately uses ``BatchMeta`` as its descriptor.  A
``kv_list`` result is a diagnostic snapshot, not a queue cursor, and therefore
must not be used to drive the hot path.
"""

from __future__ import annotations

from transfer_queue import GRPOGroupNSampler
from typing import Any, Protocol, Sequence


class AsyncTQClient(Protocol):

    async def async_get_meta(self,
                             *,
                             data_fields: list[str],
                             batch_size: int,
                             partition_id: str,
                             mode: str = 'fetch',
                             task_name: str | None = None,
                             sampling_config: dict[str, Any] | None = None) -> Any:
        ...

    async def async_get_data(self, metadata: Any) -> Any:
        ...

    async def async_put(self, data: Any, metadata: Any | None = None, partition_id: str | None = None) -> Any:
        ...

    async def async_clear_partition(self, partition_id: str) -> Any:
        ...

    async def async_check_consumption_status(self, task_name: str, partition_id: str) -> bool:
        ...

    async def async_set_custom_meta(self, metadata: Any) -> Any:
        ...


class ContextGRPOGroupNSampler(GRPOGroupNSampler):
    """Select complete prompt groups using the request's generation count."""

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = '',
        partition_id: str = '',
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        group_size = int(kwargs['n_samples_per_prompt'])
        if group_size <= 0:
            raise ValueError(f'n_samples_per_prompt must be positive, got {group_size}')
        if batch_size % group_size:
            raise ValueError(f'batch_size ({batch_size}) must be a multiple of n_samples_per_prompt ({group_size})')

        states = self._states.get(partition_id, {}).get(task_name, {})
        dp_rank = kwargs.get('dp_rank')
        batch_index = kwargs.get('batch_index')
        if dp_rank in states and batch_index in states[dp_rank]:
            return states[dp_rank][batch_index]

        ready = sorted(ready_indexes)
        selected: list[int] = []
        offset = 0
        while offset <= len(ready) - group_size and len(selected) < batch_size:
            group = ready[offset:offset + group_size]
            if all(right - left == 1 for left, right in zip(group, group[1:])):
                selected.extend(group)
                offset += group_size
            else:
                offset += 1

        if len(selected) != batch_size:
            return [], []

        result = (selected, selected.copy())
        if dp_rank is not None:
            states.setdefault(dp_rank, {})[batch_index] = result
            self._states.setdefault(partition_id, {})[task_name] = states
        return result


def batch_size_for_groups(groups: int, num_generations: int) -> int:
    if groups <= 0:
        raise ValueError(f'groups must be positive, got {groups}')
    if num_generations <= 0:
        raise ValueError(f'num_generations must be positive, got {num_generations}')
    return groups * num_generations


def validate_group_batch_size(batch_size: int, num_generations: int) -> None:
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    if num_generations <= 0:
        raise ValueError(f'num_generations must be positive, got {num_generations}')
    if batch_size % num_generations:
        raise ValueError(f'batch_size={batch_size} must be divisible by num_generations={num_generations}')


def metadata_size(metadata: Any) -> int:
    """Return the native BatchMeta size."""
    if metadata is None:
        return 0
    return int(metadata.size)


async def fetch_ready_batch(
    client: AsyncTQClient,
    *,
    data_fields: list[str],
    batch_size: int,
    partition_id: str,
    task_name: str,
    num_generations: int,
    sampling_config: dict[str, Any] | None = None,
) -> Any | None:
    """Fetch one complete group-aligned batch using TQ production status.

    The caller owns the outer service loop.  This helper performs one request
    only, which keeps shutdown and failure handling explicit and avoids hiding
    an unbounded wait in a data-plane utility.
    """

    validate_group_batch_size(batch_size, num_generations)
    config = dict(sampling_config or {})
    config['n_samples_per_prompt'] = num_generations
    metadata = await client.async_get_meta(
        data_fields=list(data_fields),
        batch_size=batch_size,
        partition_id=partition_id,
        mode='fetch',
        task_name=task_name,
        sampling_config=config,
    )
    return metadata if metadata_size(metadata) else None


async def append_fields(client: AsyncTQClient, data: Any, metadata: Any) -> Any:
    """Append fields to exactly the samples described by ``metadata``."""

    if metadata_size(metadata) == 0:
        raise ValueError('cannot append fields to an empty BatchMeta')
    return await client.async_put(data=data, metadata=metadata)


async def set_sample_tags(client: AsyncTQClient, metadata: Any, tags: Sequence[dict[str, Any]]) -> None:
    """Persist tags through the native metadata API in one controller request."""

    if metadata_size(metadata) != len(tags):
        raise ValueError(f'metadata size {metadata_size(metadata)} does not match tags {len(tags)}')
    metadata.update_custom_meta([dict(tag) for tag in tags])
    await client.async_set_custom_meta(metadata)


def split_batch_meta(metadata: Any, group_size: int) -> list[Any]:
    """Split a preallocated BatchMeta into contiguous prompt-group views."""

    size = metadata_size(metadata)
    if group_size <= 0:
        raise ValueError(f'group_size must be positive, got {group_size}')
    if size % group_size:
        raise ValueError(f'metadata size {size} is not divisible by group_size {group_size}')
    return [metadata.select_samples(list(range(start, start + group_size))) for start in range(0, size, group_size)]


async def preallocate_partition(
    client: AsyncTQClient,
    *,
    partition_id: str,
    prompt_fields: Any,
) -> Any:
    """Insert prompt rows once and return the native BatchMeta descriptor."""

    batch_size = getattr(prompt_fields, 'batch_size', None)
    if batch_size is None or len(batch_size) == 0 or int(batch_size[0]) <= 0:
        raise ValueError('prompt_fields must be a non-empty batched TensorDict')
    return await client.async_put(data=prompt_fields, partition_id=partition_id)


async def clear_partition(client: AsyncTQClient, partition_id: str) -> None:
    await client.async_clear_partition(partition_id)
