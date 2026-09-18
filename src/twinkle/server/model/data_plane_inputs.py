# Copyright (c) ModelScope Contributors. All rights reserved.
"""Utilities shared by DataPlane-backed model routes."""
from __future__ import annotations

import asyncio
from typing import Any

from twinkle_client.common.json_utils import json_safe


def model_result_rows(result: Any, batch_size: int) -> list[dict[str, Any]]:
    """Keep per-sample model outputs at DataPlane row granularity."""
    if isinstance(result, list) and len(result) == batch_size and all(isinstance(item, dict) for item in result):
        return result
    if isinstance(result, dict):
        batched = {name for name, value in result.items() if isinstance(value, list) and len(value) == batch_size}
        if batched:
            return [{
                name: value[index] if name in batched else value
                for name, value in result.items()
            } for index in range(batch_size)]
    return [{'result': result}]


def value_at_path(value: Any, path: str) -> Any:
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f'data field {path!r} does not exist')
        value = value[part]
    return value


def set_at_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split('.')
    current = target
    for part in parts[:-1]:
        nested = current.setdefault(part, {})
        if not isinstance(nested, dict):
            raise ValueError(f'cannot bind nested model argument {path!r}')
        current = nested
    current[parts[-1]] = value


async def resolve_data_plane_model_inputs(body: Any, data_plane: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve DataPlane references into model inputs and bound keyword arguments."""
    selected_fields = None
    if body.input_field is not None:
        selected_fields = list(
            dict.fromkeys([
                body.input_field,
                *(source.split('.', 1)[0] for source in body.kwarg_fields.values()),
            ]))
    batches = await asyncio.gather(*(data_plane.get(ref, fields=selected_fields) for ref in body.input_refs))
    rows = [row for batch in batches for row in batch]
    if body.input_field is None:
        kwarg_roots = {source.split('.', 1)[0] for source in body.kwarg_fields.values()}
        inputs = [{key: value for key, value in row.items() if key not in kwarg_roots} for row in rows]
    else:
        inputs = [value_at_path(row, body.input_field) for row in rows]

    field_kwargs: dict[str, Any] = {}
    for target_path, source_path in body.kwarg_fields.items():
        field_value = [value_at_path(row, source_path) for row in rows]
        set_at_path(field_kwargs, target_path, field_value)
    return inputs, field_kwargs


def merge_forward_kwargs(explicit: dict[str, Any], bound: dict[str, Any]) -> dict[str, Any]:
    collisions = set(explicit).intersection(bound)
    if collisions:
        names = ', '.join(sorted(collisions))
        raise ValueError(f'explicit model kwargs conflict with kwarg_fields: {names}')
    return {**explicit, **bound}


def data_plane_request_shape(body: Any) -> tuple[int, int]:
    return (
        sum(ref.num_tokens for ref in body.input_refs),
        sum(ref.size for ref in body.input_refs),
    )


def select_output_rows(
    result: Any,
    *,
    batch_size: int,
    output_fields: dict[str, str],
) -> list[dict[str, Any]]:
    rows = model_result_rows(json_safe(result), batch_size)
    if len(rows) != batch_size:
        raise ValueError(f'model returned {len(rows)} rows for an output_ref of size {batch_size}')
    return [{target: value_at_path(row, source) for source, target in output_fields.items()} for row in rows]
