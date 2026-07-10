# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fused RMSNorm impls for Ascend NPU.

Designed for class-replacement: do not define ``__init__``; rely on the
attributes already present on the original instance.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from twinkle import get_logger

logger = get_logger()


class NpuRMSNorm(nn.Module):
    """Class-replacement impl for HF RMSNorm variants.

    Required instance attributes (provided by the original class):
      - ``weight``: ``nn.Parameter``
      - ``variance_epsilon`` *or* ``eps``: float
    """

    def _twinkle_residual_param(self) -> bool:
        cached = getattr(self, '_twinkle_residual_cached', None)
        if cached is None:
            cached = not hasattr(self, 'variance_epsilon')
            self._twinkle_residual_cached = cached
            if cached:
                logger.debug('[NPU] NpuRMSNorm using residual parameterization (1.0 + weight)')
        return cached

    def _twinkle_eps(self) -> float:
        return getattr(self, 'variance_epsilon', getattr(self, 'eps', 1e-6))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import torch_npu
        input_dtype = hidden_states.dtype
        if _FORCE_FP32:
            hidden_states = hidden_states.to(torch.float32)
            if self._twinkle_residual_param():
                scale = (1.0 + self.weight).float()
            else:
                scale = self.weight.float()
        else:
            if self._twinkle_residual_param():
                scale = (1.0 + self.weight).to(input_dtype)
            else:
                scale = self.weight.to(input_dtype)
        out = torch_npu.npu_rms_norm(hidden_states, scale, epsilon=self._twinkle_eps())[0]
        return out.to(input_dtype) if _FORCE_FP32 else out


# Resolved once at import: matches the legacy "patch-time, process-wide" invariant.
# Mid-process env mutation will not retroactively change behavior.
_FORCE_FP32 = os.environ.get('TWINKLE_NPU_GATED_RMSNorm_FP32', '0').lower() in ('1', 'true', 'on', 'yes')


def npu_gated_rms_norm_forward(self, hidden_states, gate=None):
    """Forward replacement for Gated RMSNorm variants (e.g. Qwen3.5-MoE)."""
    import torch_npu

    input_dtype = hidden_states.dtype
    _eps = getattr(self, 'variance_epsilon', getattr(self, 'eps', 1e-6))

    if _FORCE_FP32:
        hidden_states = hidden_states.to(torch.float32)
        weight = self.weight.float()
        gate = gate.to(torch.float32) if gate is not None else None
    else:
        weight = self.weight

    hidden_states = torch_npu.npu_rms_norm(hidden_states, weight, epsilon=_eps)[0]
    if gate is not None:
        hidden_states = hidden_states * F.silu(gate)
    return hidden_states.to(input_dtype)
