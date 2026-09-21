# Copyright (c) ModelScope Contributors. All rights reserved.
"""TransferQueue field packing and the field-name schema both async-RL modes share.

Lives in ``data_format/`` because "rows/columns -> TensorDict" is a data-format
conversion, alongside ``input_feature`` / ``trajectory`` / ``encoding`` / ``message`` /
``output`` / ``sampling``. It used to sit at the package root as ``twinkle/tq_utils.py``
-- the only domain module there, unreachable via ``twinkle.<attr>``, under an unexplained
abbreviation and a ``_utils`` suffix that undersold what it does (it validates field
consistency and raises).

The field lists moved here from ``twinkle_agentic/async_rl/tq_utils.py``: the schema and
the packing logic belong in one file, and that shim existed only to re-export this
module. This deliberately means ``twinkle`` holds the RL training field names
(``logprobs`` / ``rewards`` / ``advantages`` / ``returns``) while ``twinkle_agentic``
only consumes them.

``torch`` / ``tensordict`` stay inside the functions: ``tensordict`` arrives with
``TransferQueue``, which is only in the ``async-rl`` extra, so this module must import
cleanly without it. Do NOT hoist them.
"""
from __future__ import annotations

from numbers import Number
from typing import Any

TRANSFORMERS_INPUT_FIELDS = (
    'input_ids',
    'labels',
    'attention_mask',
    'position_ids',
    'cu_seqlens',
    'completion_mask',
    'pixel_values',
    'image_grid_thw',
    'video_pixel_values',
    'video_grid_thw',
    'input_features',
    'feature_attention_mask',
)
REQUIRED_MODEL_INPUT_FIELDS = ('input_ids', 'labels', 'attention_mask', 'position_ids')
ROLLOUT_TRAIN_FIELDS = (*TRANSFORMERS_INPUT_FIELDS, 'logprobs', 'rewards', 'advantages', 'returns')


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
