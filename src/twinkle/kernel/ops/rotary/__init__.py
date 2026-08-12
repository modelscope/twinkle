# Copyright (c) ModelScope Contributors. All rights reserved.
"""rotary / multimodal_rotary op registration.

The liger exclusion for the qwen3_5 family is NOT expressed at the registry
layer — it is decided in config.py, where the rotary targets of those two
families use a ('npu',)-only chain (liger's full-rotation implementation is
incompatible with partial_rotary).
"""
from __future__ import annotations

from ...registry import KernelImpl, is_liger_available, is_npu_available, lazy_import, register_op

register_op(
    'rotary',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.rotary.npu:npu_apply_rotary_pos_emb'),
            available=is_npu_available,
        ),
        'liger':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.rotary.liger:liger_rotary_pos_emb'),
            available=is_liger_available,
        ),
    },
)

# Qwen2.5-VL multimodal rope; no liger impl
register_op(
    'multimodal_rotary',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.rotary.npu:npu_apply_multimodal_rotary_pos_emb'),
            available=is_npu_available,
        ),
    },
)
