# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run a fully frozen tower without autograd (legacy swift #10214).

A tower with no trainable parameter still keeps a graph when a trainable consumer sits downstream.
The decision is made per call, so unfreezing the tower restores autograd by itself.
"""
from __future__ import annotations

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


def patch_frozen_module(module: nn.Module) -> None:
    """Run ``module.forward`` without autograd while nothing in it or in its inputs needs a gradient."""
    if hasattr(module, _ORIGINAL_FORWARD_ATTR):
        return
    setattr(module, _ORIGINAL_FORWARD_ATTR, module.forward)

    @wraps(module.forward)
    def frozen_forward(self, *args, **kwargs):
        requires_grad = torch.is_grad_enabled()
        if requires_grad:
            requires_grad = any(parameter.requires_grad for parameter in self.parameters()) or _needs_input_grad(
                (args, kwargs))
        with torch.set_grad_enabled(requires_grad):
            return getattr(self, _ORIGINAL_FORWARD_ATTR)(*args, **kwargs)

    module.forward = MethodType(frozen_forward, module)
