# Copyright (c) ModelScope Contributors. All rights reserved.
"""Validate the losses implemented by the Tinker compatibility bridge."""
from __future__ import annotations

from tinker import types

from twinkle.server.exceptions import BatchSizeError, RequestRejectedError


def validate_tinker_loss(loss_fn: str, inputs: list[types.Datum], *, data_world_size: int = 1) -> bool:
    """Reject unsupported losses and incomplete DPO pairs; return whether this is DPO.

    ``importance_sampling`` is also used by Twinkle's DPO cookbook, where every
    datum has ``ref_logps`` and chosen/rejected examples are interleaved. Ordinary
    RL examples do not need pairs. Call before enqueueing, and again at the backend
    boundary so direct backend callers cannot silently select another loss.
    """
    if loss_fn not in ('cross_entropy', 'importance_sampling'):
        raise RequestRejectedError(
            f'Unsupported Tinker loss_fn {loss_fn!r}; supported losses: cross_entropy, importance_sampling. '
            'importance_sampling uses Twinkle GRPO semantics, not the unclipped Tinker IS objective.')

    if loss_fn != 'importance_sampling':
        return False

    has_ref_logps = ['ref_logps' in datum.loss_fn_inputs for datum in inputs]
    if not any(has_ref_logps):
        return False
    if not all(has_ref_logps):
        raise RequestRejectedError('DPO requires ref_logps on every datum; cannot mix DPO and RL examples.')

    required_multiple = 2 * data_world_size
    if len(inputs) % required_multiple != 0:
        raise BatchSizeError(f'DPO batch size {len(inputs)} must be divisible by {required_multiple} '
                             'so each data-parallel shard receives complete chosen/rejected pairs.')
    return True
