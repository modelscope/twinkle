# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run a fully frozen tower without autograd (legacy swift #10214).

A tower with no trainable parameter still keeps a graph when a trainable consumer sits downstream.
The decision is made per call, so unfreezing the tower restores autograd by itself.
"""
from __future__ import annotations

import contextlib
from collections.abc import Mapping
from functools import wraps
from types import MethodType

import torch
import torch.nn as nn

_ORIGINAL_FORWARD_ATTR = '_twinkle_frozen_module_forward'


def _needs_input_grad(value) -> bool:
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, Mapping):
        return any(_needs_input_grad(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_needs_input_grad(item) for item in value)
    # An unknown container may carry differentiable tensors: leave the graph alone.
    return not isinstance(value, (type(None), bool, int, float, str, torch.dtype, torch.device))


def patch_frozen_module(module: nn.Module, amp_dtype: torch.dtype | None = None) -> None:
    """Run ``module.forward`` without autograd while nothing in it or in its inputs needs a gradient.

    ``amp_dtype`` (the training mixed-precision dtype) runs the tower under autocast: a frozen tower
    kept in fp32 still drives flash-attention kernels that reject fp32, so the attention must see
    bf16/fp16. ``None`` (no mixed precision) leaves the dtype untouched.
    """
    if hasattr(module, _ORIGINAL_FORWARD_ATTR):
        return
    setattr(module, _ORIGINAL_FORWARD_ATTR, module.forward)
    use_amp = amp_dtype in (torch.bfloat16, torch.float16)

    @wraps(module.forward)
    def frozen_forward(self, *args, **kwargs):
        requires_grad = torch.is_grad_enabled()
        if requires_grad:
            requires_grad = any(parameter.requires_grad for parameter in self.parameters()) or _needs_input_grad(
                (args, kwargs))
        amp_ctx = (torch.autocast(device_type=next(self.parameters()).device.type, dtype=amp_dtype)
                   if use_amp else contextlib.nullcontext())
        with torch.set_grad_enabled(requires_grad), amp_ctx:
            return getattr(self, _ORIGINAL_FORWARD_ATTR)(*args, **kwargs)

    module.forward = MethodType(frozen_forward, module)
