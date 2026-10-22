# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tensor statistics helpers for the module profiler.

These are plain functions over ``torch.Tensor`` (and the nested containers a
forward may return) so both the built-in :func:`activation_stats` /
:func:`param_stats` collectors and user collectors can reuse them. Every
reduction ends in ``.item()``, which forces a device sync -- that is
intentional and affordable because the profiler only collects on gated steps,
never on every forward.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterator, Optional

import torch

__all__ = ['iter_tensors', 'primary_tensor', 'tensor_moments', 'tensor_numel', 'describe_shape']


def iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor reachable from a forward input/output.

    Walks tensors, mappings (transformers ``ModelOutput`` is an ordered mapping),
    sequences and dataclasses; anything else is skipped. Depth is bounded by the
    container nesting a forward actually returns.
    """
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            yield from iter_tensors(getattr(obj, field.name, None))


def primary_tensor(obj: Any) -> Optional[torch.Tensor]:
    """The first tensor in ``obj``, or None if it carries none."""
    for tensor in iter_tensors(obj):
        return tensor
    return None


def tensor_moments(tensor: torch.Tensor) -> Dict[str, float]:
    """mean / absmax / l2 / std of a tensor, computed in float32.

    Returns all-zero stats for an empty tensor rather than raising, so a module
    that legitimately emits a zero-element tensor (e.g. an all-padding slice) does
    not break profiling.
    """
    detached = tensor.detach()
    if detached.numel() == 0:
        return {'mean': 0.0, 'absmax': 0.0, 'l2': 0.0, 'std': 0.0}
    # Reduce in float32 for stability regardless of the stored dtype (bf16/fp16).
    values = detached.float()
    return {
        'mean': values.mean().item(),
        'absmax': values.abs().max().item(),
        'l2': values.norm().item(),
        'std': values.std().item() if values.numel() > 1 else 0.0,
    }


def tensor_numel(obj: Any) -> float:
    """Element count of the primary tensor (0.0 when there is none)."""
    tensor = primary_tensor(obj)
    return float(tensor.numel()) if tensor is not None else 0.0


def describe_shape(obj: Any) -> str:
    """A compact shape descriptor for display, e.g. ``'2x8x1024'`` or ``'(2x8, 4x16)'``.

    Non-numeric metadata (kept out of the float-only ``collect`` contract) used by
    ``print_summary`` to show what flowed through a module.
    """
    tensors = list(iter_tensors(obj))
    if not tensors:
        return '-'
    if len(tensors) == 1:
        return 'x'.join(str(dim) for dim in tensors[0].shape)
    return '(' + ', '.join('x'.join(str(dim) for dim in t.shape) for t in tensors) + ')'
