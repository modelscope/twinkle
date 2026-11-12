# Copyright (c) ModelScope Contributors. All rights reserved.
"""Nan-skipping reductions over per-row values."""
import math
from typing import Any, List, Literal, Optional

from twinkle import remote_class, remote_function

OPS = ('sum', 'mean', 'variance', 'logsumexp')


def as_rows(values: Any) -> List[Any]:
    """A lane as a list of rows (a tensor lane is converted; ``None`` is an empty lane)."""
    if values is None:
        return []
    if hasattr(values, 'tolist'):
        values = values.tolist()
    return list(values) if isinstance(values, (list, tuple)) else [values]


def _reduce(values: Any, op: str) -> Optional[float]:
    """One reduction over a flat sequence's finite entries; ``None`` when none is finite."""
    flat = [float(value) for value in as_rows(values) if value is not None and not math.isnan(float(value))]
    if not flat:
        return None
    if op == 'sum':
        return sum(flat)
    mean = sum(flat) / len(flat)
    if op == 'mean':
        return mean
    if op == 'variance':
        return sum((value - mean)**2 for value in flat) / len(flat)
    top = max(flat)
    return top + math.log(sum(math.exp(value - top) for value in flat))


@remote_class(execute='first')
class Reduce:
    """A nan-skipping reduction over a lane.

    ``axis='token'`` reduces each row (a token sequence) to one value; ``'row'`` reduces the batch's
    scalar rows to one value; ``'group'`` reduces each consecutive block of ``group_size`` scalar
    rows to one value. A reduction with no finite entry is ``None``.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(self,
                 values: Any,
                 *,
                 op: Literal['sum', 'mean', 'variance', 'logsumexp'] = 'mean',
                 axis: Literal['token', 'row', 'group'] = 'token',
                 group_size: Optional[int] = None,
                 **kwargs) -> List[Optional[float]]:
        if op not in OPS:
            raise ValueError(f'reduce op {op!r} not in {OPS}.')
        rows = as_rows(values)
        if axis == 'token':
            return [_reduce(row, op) for row in rows]
        if axis == 'row':
            return [_reduce(rows, op)]
        if axis == 'group':
            if not group_size or len(rows) % group_size:
                raise ValueError(f"axis='group' needs a group_size dividing the {len(rows)} rows, got {group_size!r}.")
            return [_reduce(rows[start:start + group_size], op) for start in range(0, len(rows), group_size)]
        raise ValueError(f"reduce axis {axis!r} not in ('token', 'row', 'group').")
