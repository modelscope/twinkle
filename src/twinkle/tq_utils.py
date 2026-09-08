# Copyright (c) ModelScope Contributors. All rights reserved.
"""Small TransferQueue packing helpers shared by both async-RL modes."""
from __future__ import annotations

from numbers import Number
from typing import Any


def rows_to_tq_fields(rows: list[dict[str, Any]]):
    from tensordict import TensorDict

    if not rows:
        return TensorDict({}, batch_size=[0])
    field_names = tuple(rows[0].keys())
    expected = set(field_names)
    for row_index, row in enumerate(rows):
        actual = set(row)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f'TQ row {row_index} fields mismatch: missing={missing}, extra={extra}')
    columns = {field_name: [row[field_name] for row in rows] for field_name in field_names}
    return columns_to_tq_fields(columns, len(rows))


def columns_to_tq_fields(columns: dict[str, list[Any]], size: int):
    import torch
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorStack

    if size < 0:
        raise ValueError(f'TQ field size must be non-negative, got {size}')
    packed = {}
    for field_name, values in columns.items():
        if not isinstance(values, list):
            raise TypeError(f'TQ field {field_name!r} must be a list, got {type(values)!r}')
        if len(values) != size:
            raise ValueError(f'TQ field {field_name!r} must contain {size} values, got {len(values)}')
        if all(isinstance(item, Number) and not isinstance(item, bool) for item in values):
            packed[field_name] = torch.tensor(values)
        else:
            packed[field_name] = NonTensorStack(*values)
    return TensorDict(packed, batch_size=[size])
