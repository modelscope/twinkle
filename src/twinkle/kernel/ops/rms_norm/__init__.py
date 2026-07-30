# Copyright (c) ModelScope Contributors. All rights reserved.
"""rms_norm / gated_rms_norm op registration.

Liger's RMSNorm is parameterized per family (casting_mode / offset, see the
adapter classes in liger.py), so the liger load factory dispatches variants
by mapping target: gemma4 -> Gemma4Replacement; gemma* -> GemmaReplacement;
qwen3_5* -> Qwen35Replacement (residual parameterization — llama-cast would
produce NaN); everything else -> the default llama-cast Replacement.
"""
from __future__ import annotations

from typing import Any

from ...registry import KernelImpl, is_liger_available, is_npu_available, lazy_import, register_op


def _liger_rms_norm_load(target: Any):
    name = target if isinstance(target, str) else f'{target.__module__}.{target.__qualname__}'
    if 'gemma4' in name:
        from .liger import LigerRMSNormGemma4Replacement as cls
    elif 'gemma' in name:
        from .liger import LigerRMSNormGemmaReplacement as cls
    elif 'qwen3_5' in name:
        from .liger import LigerRMSNormQwen35Replacement as cls
    else:
        from .liger import LigerRMSNormReplacement as cls
    return cls


register_op(
    'rms_norm',
    implementations={
        'npu': KernelImpl(
            load=lazy_import('twinkle.kernel.ops.rms_norm.npu:NpuRMSNorm'),
            available=is_npu_available,
        ),
        'liger': KernelImpl(
            load=_liger_rms_norm_load,
            available=is_liger_available,
        ),
    },
)

# Gated RMSNorm (qwen3_5 family, forward-level replacement); no liger impl
register_op(
    'gated_rms_norm',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.rms_norm.npu:npu_gated_rms_norm_forward'),
            available=is_npu_available,
        ),
    },
)
