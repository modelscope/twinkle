# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tensor normalization helpers for JSON-compatible RL inputs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def align_per_token_values(
    values: Any,
    target_shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str = 'values',
    padding_value: float = 0.0,
    valid_mask: Any | None = None,
) -> torch.Tensor:
    """Convert tensor-like per-token values and align them to a model batch.

    Component HTTP APIs intentionally use JSON-compatible values, so tensors
    sent by a client or read through DataPlane arrive as Python lists.  The
    computation layer calls this helper only for fields it knows are per-token
    tensors.  Rectangular values are converted directly; ragged rows are padded
    before validating and aligning them to ``target_shape``. Missing suffixes
    are accepted only when ``valid_mask`` marks those positions as padding.
    """
    import torch

    row_lengths: list[int] | None = None
    if torch.is_tensor(values):
        tensor = values
    else:
        try:
            tensor = torch.as_tensor(values)
        except (TypeError, ValueError):
            if not isinstance(values, (list, tuple)) or not values:
                raise TypeError(f'{name} must be a tensor or a non-empty sequence') from None

            rows = []
            for index, value in enumerate(values):
                try:
                    row = torch.as_tensor(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(f'{name}[{index}] cannot be converted to a tensor') from exc
                if row.ndim == 2 and row.shape[0] == 1:
                    row = row.squeeze(0)
                if row.ndim != 1:
                    raise ValueError(f'{name}[{index}] must be one-dimensional, got shape {tuple(row.shape)}')
                rows.append(row)
            row_lengths = [row.numel() for row in rows]
            tensor = torch.nn.utils.rnn.pad_sequence(
                rows,
                batch_first=True,
                padding_value=padding_value,
            )

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f'{name} must be two-dimensional, got shape {tuple(tensor.shape)}')

    target_batch_size, target_seq_len = target_shape
    batch_size, seq_len = tensor.shape
    if batch_size != target_batch_size:
        raise ValueError(f'{name} batch size ({batch_size}) does not match target batch size '
                         f'({target_batch_size})')
    mask = None
    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool)
        if tuple(mask.shape) != target_shape:
            raise ValueError(f'valid_mask shape {tuple(mask.shape)} does not match target shape '
                             f'{target_shape}')

    if row_lengths is not None:
        for index, row_len in enumerate(row_lengths):
            if row_len >= target_seq_len:
                continue
            if mask is None or bool(mask[index, row_len:].any().item()):
                raise ValueError(f'{name}[{index}] has {row_len} tokens but target sequence length '
                                 f'is {target_seq_len}')
    if seq_len < target_seq_len:
        if mask is None or bool(mask[:, seq_len:].any().item()):
            raise ValueError(f'{name} seq_len ({seq_len}) is smaller than target seq_len '
                             f'({target_seq_len})')
        tensor = torch.nn.functional.pad(
            tensor,
            (0, target_seq_len - seq_len),
            value=padding_value,
        )
    if seq_len > target_seq_len:
        tensor = tensor[:, :target_seq_len]

    return tensor.to(device=device, dtype=dtype)
