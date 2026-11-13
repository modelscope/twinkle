# Copyright (c) ModelScope Contributors. All rights reserved.
"""Device dispatch for fused linear selective log-softmax."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _validate_inputs(x, weight, targets, temperature):
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError(f'x and weight must be 2D, got {tuple(x.shape)} and {tuple(weight.shape)}')
    if targets.dim() != 1 or targets.shape[0] != x.shape[0]:
        raise ValueError(f'targets must have shape [{x.shape[0]}], got {tuple(targets.shape)}')
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f'hidden dimensions do not match: {x.shape[1]} and {weight.shape[1]}')
    if x.device != weight.device or x.device != targets.device:
        raise ValueError('x, weight and targets must be on the same device')
    supported_dtypes = (torch.float32, torch.bfloat16)
    mixed_bf16 = x.dtype == torch.float32 and weight.dtype == torch.bfloat16
    if x.dtype not in supported_dtypes or weight.dtype not in supported_dtypes or (x.dtype != weight.dtype
                                                                                   and not mixed_bf16):
        raise ValueError(f'unsupported x/weight dtypes: {x.dtype} and {weight.dtype}')
    if x.shape[0] == 0 or weight.shape[0] == 0:
        raise ValueError('token and vocabulary dimensions must be non-empty')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f'temperature must be finite and positive, got {temperature!r}')


def fused_linear_selective_log_softmax_reference(x, weight, targets, temperature=1.0, return_entropy=False):
    """Native PyTorch semantics used on non-NPU devices and as the test oracle."""
    logits = F.linear(x, weight).div_(temperature)
    logits = logits.float()
    logsumexp = torch.logsumexp(logits, dim=-1)
    selected = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    logps = selected - logsumexp
    if not return_entropy:
        return logps
    probabilities = torch.softmax(logits, dim=-1)
    entropies = logsumexp - (probabilities * logits).sum(dim=-1)
    return logps, entropies


def fused_linear_selective_log_softmax(x, weight, targets, temperature=1.0, return_entropy=False):
    """Compute selected log-probabilities from hidden states without exposing logits.

    The Ascend implementation owns the logits workspace and reuses it for its
    gradient. It supports one first-order backward per forward. FP32 hidden
    states with a BF16 weight follow the usual mixed-precision cast. Other
    devices use the native PyTorch reference implementation.
    """
    temperature = float(temperature)
    _validate_inputs(x, weight, targets, temperature)
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    if x.device.type == 'npu':
        from .npu import npu_fused_linear_selective_log_softmax
        return npu_fused_linear_selective_log_softmax(x, weight, targets, temperature, return_entropy)
    return fused_linear_selective_log_softmax_reference(x, weight, targets, temperature, return_entropy)
