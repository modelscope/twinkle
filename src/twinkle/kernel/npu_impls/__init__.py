# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-layer NPU implementations consumed by ``npu_builtin()``.

Each impl is contracted to be applied via ``m.__class__ = ImplCls`` (class
replacement) or ``setattr(module, attr, fn)`` (function replacement). No impl
here is meant to be instantiated directly.
"""
from .rms_norm import NpuRMSNorm, npu_gated_rms_norm_forward

__all__ = ['NpuRMSNorm', 'npu_gated_rms_norm_forward']