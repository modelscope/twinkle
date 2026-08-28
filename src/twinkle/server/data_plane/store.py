# Copyright (c) ModelScope Contributors. All rights reserved.
"""TransferQueue KV storage behind opaque client DataRef values."""
from __future__ import annotations

import uuid
from typing import Any

from twinkle.tq_utils import rows_to_tq_fields
from twinkle_client.common.json_utils import json_safe
from twinkle_client.types.component import DataRef


def _keys(ref: DataRef) -> list[str]:
    return [str(index) for index in range(ref.size)]


def _partition(ref: DataRef) -> str:
    """Resolve an opaque DataRef to its self-contained physical TQ partition."""
    return f'twinkle-client/{ref.ref_id}'


def _input_token_count(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        input_ids = row.get('input_ids')
        if input_ids is None and isinstance(row.get('train_input'), dict):
            input_ids = row['train_input'].get('input_ids')
        if isinstance(input_ids, (list, tuple)):
            total += len(input_ids)
    return total


def _rows_from_tensordict(data: Any, size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = list(data.keys())
    for index in range(size):
        rows.append({field: json_safe(data[field][index]) for field in fields})
    return rows


class TQDataRefStore:

    def __init__(self, config: dict[str, Any] | None = None):
        import transfer_queue as tq
        if config:
            from omegaconf import OmegaConf
            tq.init(OmegaConf.create(config))
        else:
            tq.init()

    async def put(
        self,
        rows: list[dict[str, Any]],
        *,
        kind: str = 'data',
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        if not rows:
            raise ValueError('rows must not be empty')
        if tags is not None and len(tags) != len(rows):
            raise ValueError(f'tag count {len(tags)} does not match row count {len(rows)}')
        import transfer_queue as tq
        ref = DataRef(
            ref_id=uuid.uuid4().hex,
            size=len(rows),
            fields=list(rows[0]),
            kind=kind,
            num_tokens=_input_token_count(rows),
        )
        await tq.async_kv_batch_put(
            keys=_keys(ref),
            partition_id=_partition(ref),
            fields=rows_to_tq_fields(rows),
            tags=tags,
        )
        return ref

    async def get(
        self,
        ref: DataRef,
        *,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        import transfer_queue as tq
        selected = fields if fields is not None else ref.fields
        data = await tq.async_kv_batch_get(
            keys=_keys(ref),
            partition_id=_partition(ref),
            select_fields=selected,
        )
        return _rows_from_tensordict(data, ref.size)

    async def get_tags(
        self,
        ref: DataRef,
    ) -> list[dict[str, Any]]:
        """Return TQ sample tags in the same order as the rows in ``ref``."""
        import transfer_queue as tq
        partition_id = _partition(ref)
        partitions = await tq.async_kv_list(partition_id=partition_id)
        partition = partitions.get(partition_id, {})
        return [dict(partition.get(key, {})) for key in _keys(ref)]

    async def append(
        self,
        ref: DataRef,
        rows: list[dict[str, Any]],
        *,
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        if len(rows) != ref.size:
            raise ValueError(f'append row count {len(rows)} does not match DataRef size {ref.size}')
        if not rows:
            raise ValueError('rows must not be empty')
        if tags is not None and len(tags) != len(rows):
            raise ValueError(f'tag count {len(tags)} does not match row count {len(rows)}')
        import transfer_queue as tq
        await tq.async_kv_batch_put(
            keys=_keys(ref),
            partition_id=_partition(ref),
            fields=rows_to_tq_fields(rows),
            tags=tags,
        )
        updates: dict[str, Any] = {
            'fields': list(dict.fromkeys([*ref.fields, *rows[0].keys()])),
        }
        if 'input_ids' in rows[0] or 'train_input' in rows[0]:
            updates['num_tokens'] = _input_token_count(rows)
        return ref.model_copy(update=updates)

    async def release(self, ref: DataRef) -> None:
        import transfer_queue as tq
        await tq.async_kv_clear(keys=_keys(ref), partition_id=_partition(ref))
