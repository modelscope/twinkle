# Copyright (c) ModelScope Contributors. All rights reserved.
"""Parsing of the user-facing FSDP mixed-precision overrides.

``fsdp_config['mixed_precision']`` configures the precision of each FSDP field
independently, e.g. BF16 computation with FP32 gradient reduction::

    fsdp_config={'mixed_precision': {'param_dtype': 'bf16', 'reduce_dtype': 'fp32'}}

Every key is optional. Only the native FSDP2 path consumes this sub-config: it
resolves the dict through :func:`parse_mp_overrides` and maps the resolved
dtypes onto torch's ``MixedPrecisionPolicy`` (FSDP1's ``buffer_dtype`` has no
counterpart here because only FSDP2 is supported). The accelerate path is not
adapted and keeps its own plugin-level handling. Combinations whose effective
``output_dtype`` differs from ``param_dtype`` are rejected at strategy
construction (see ``native_fsdp._check_mp_dtypes``): such a unit-boundary dtype
cast breaks gradient-checkpointing recompute.
"""
from typing import Any

import torch

# The FSDP2 ``MixedPrecisionPolicy`` fields users may override. ``cast_forward_inputs``
# is intentionally not exposed: it must stay aligned with ``param_dtype``.
MP_OVERRIDE_FIELDS = ('param_dtype', 'reduce_dtype', 'output_dtype')

_STR_TO_DTYPE = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
_ALLOWED_DTYPES = tuple(_STR_TO_DTYPE.values())


def parse_mp_overrides(config: dict[str, Any] | None) -> dict[str, torch.dtype]:
    """Validate ``fsdp_config['mixed_precision']`` and resolve it to torch dtypes.

    Args:
        config: ``None``, or a dict mapping any of ``param_dtype`` / ``reduce_dtype`` /
            ``output_dtype`` to ``'bf16'`` / ``'fp16'`` / ``'fp32'`` (case-insensitive)
            or the equivalent ``torch.dtype``. A key whose value is ``None`` counts as
            unset and falls back to the strategy default.

    Returns:
        dict[str, torch.dtype]: the explicitly configured fields only; empty when
        nothing is configured, so callers can treat it as pure overrides on top of
        the ``mixed_precision`` string.

    Raises:
        ValueError: on a non-dict config, an unknown key, or an invalid value.
    """
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"fsdp_config['mixed_precision'] must be a dict, got {type(config).__name__}: {config!r}")
    unknown = sorted(set(config) - set(MP_OVERRIDE_FIELDS))
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in fsdp_config['mixed_precision']; "
                         f'valid keys are {list(MP_OVERRIDE_FIELDS)}.')
    resolved: dict[str, torch.dtype] = {}
    for field in MP_OVERRIDE_FIELDS:
        value = config.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            dtype = _STR_TO_DTYPE.get(value.lower())
        elif isinstance(value, torch.dtype) and value in _ALLOWED_DTYPES:
            dtype = value
        else:
            dtype = None
        if dtype is None:
            raise ValueError(f"Invalid value for fsdp_config['mixed_precision']['{field}']: {value!r}. "
                             f"Must be one of 'bf16', 'fp16', 'fp32' or the corresponding torch.dtype.")
        resolved[field] = dtype
    return resolved
