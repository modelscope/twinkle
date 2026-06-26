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
        """Lazily detect residual parameterization (e.g. Qwen3.5: scale = 1 + weight)."""
        cached = getattr(self, '_twinkle_residual_cached', None)
        if cached is None:
            cached = abs(self.weight.data.mean().item()) < 0.3
            self._twinkle_residual_cached = cached
            if cached:
                logger.debug('[NPU] NpuRMSNorm using residual parameterization (1.0 + weight)')
        return cached

    def _twinkle_eps(self) -> float:
        return getattr(self, 'variance_epsilon', getattr(self, 'eps', 1e-6))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import torch_npu
        target_dtype = hidden_states.dtype
        if self._twinkle_residual_param():
            scale = (1.0 + self.weight).to(target_dtype)
        else:
            scale = self.weight.to(target_dtype)
        return torch_npu.npu_rms_norm(hidden_states, scale, epsilon=self._twinkle_eps())[0]


def npu_gated_rms_norm_forward(self, hidden_states, gate=None):
    """Forward replacement for Gated RMSNorm variants (e.g. Qwen3.5-MoE).

    Reads FP32-mode preference from env ``TWINKLE_NPU_GATED_RMSNorm_FP32`` once
    and caches it on the instance.
    """
    import torch_npu

    input_dtype = hidden_states.dtype
    _eps = getattr(self, 'variance_epsilon', getattr(self, 'eps', 1e-6))

    force_fp32 = getattr(self, '_twinkle_force_fp32', None)
    if force_fp32 is None:
        force_fp32 = os.environ.get('TWINKLE_NPU_GATED_RMSNorm_FP32', '0').lower() in (
            '1', 'true', 'on', 'yes'
        )
        self._twinkle_force_fp32 = force_fp32

    if force_fp32:
        hidden_states = hidden_states.to(torch.float32)
        weight = self.weight.float()
        gate = gate.to(torch.float32) if gate is not None else None
    else:
        weight = self.weight

    hidden_states = torch_npu.npu_rms_norm(hidden_states, weight, epsilon=_eps)[0]
    if gate is not None:
        hidden_states = hidden_states * F.silu(gate)
    return hidden_states.to(input_dtype)