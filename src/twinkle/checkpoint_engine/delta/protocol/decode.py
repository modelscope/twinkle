# Copyright (c) ModelScope Contributors. All rights reserved.
"""Device-agnostic decoding of one received ``DeltaFlush`` (no vLLM dependency).

Shared by the vLLM weight-transfer adapter (which converts :class:`DeltaPatch` into vLLM's
``CheckpointWeightPatch``) and by tests.
"""
from __future__ import annotations

import json
import torch
from collections.abc import Iterable
from dataclasses import dataclass

from .wire import checksum


@dataclass
class DeltaPatch:
    """One parameter's patch. ``positions`` are flat int32 positions, ``None`` for dense."""

    name: str
    shape: tuple
    dtype: torch.dtype
    values: torch.Tensor
    positions: torch.Tensor | None = None


def decode_delta_wire(
    named_tensors: Iterable[tuple[str, torch.Tensor]], ) -> tuple[str, list[DeltaPatch]]:
    """Validate one DeltaFlush and slice it into per-parameter patches."""
    tensors = dict(named_tensors)
    try:
        spec_tensor = tensors['__delta_spec__']
        values = tensors['__values__']
        spec = json.loads(bytes(spec_tensor.detach().cpu().numpy().tobytes()).decode())
    except KeyError as exc:
        raise ValueError(f'DeltaFlush is missing {exc.args[0]}') from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('DeltaFlush contains an invalid JSON spec') from exc

    encoding = spec['encoding']
    if encoding not in {'dense', 'indices'}:
        raise ValueError(f'Unsupported DeltaFlush encoding: {encoding!r}')
    positions = tensors.get('__positions__')
    if positions is None:
        positions = torch.empty(0, dtype=torch.uint8, device=values.device)
    if encoding == 'dense' and positions.numel():
        raise ValueError('Dense DeltaFlush must not contain positions')
    if encoding == 'indices' and '__positions__' not in tensors:
        raise ValueError('Indices DeltaFlush requires __positions__')

    got_checksum = checksum(positions, values)
    expected_checksum = int(spec['checksum'])
    if got_checksum != expected_checksum:
        raise RuntimeError(
            f'Delta checksum mismatch in vLLM consumer: got {got_checksum}, expected {expected_checksum}')

    patches: list[DeltaPatch] = []
    for param_spec in spec['params']:
        name = param_spec['name']
        dtype = getattr(torch, param_spec['dtype'])
        patch_values = values[param_spec['val_start']:param_spec['val_end']]
        if encoding == 'dense':
            patch_positions = None
        else:
            if param_spec['pos_width'] != 4:
                raise ValueError(f'{name}: indices encoding requires pos_width=4')
            pos_bytes = positions[param_spec['pos_start']:param_spec['pos_end']]
            if pos_bytes.numel() != patch_values.numel() * 4:
                raise ValueError(f'{name}: position and value slices have inconsistent lengths')
            patch_positions = pos_bytes.view(torch.int32)
        patches.append(
            DeltaPatch(
                name=name,
                shape=tuple(param_spec['shape']),
                dtype=dtype,
                values=patch_values,
                positions=patch_positions,
            ))
    return encoding, patches
