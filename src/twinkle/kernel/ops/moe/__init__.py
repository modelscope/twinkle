# Copyright (c) ModelScope Contributors. All rights reserved.
"""moe_experts / moe_block op registration.

moe_experts: the npu backend is a forward-level replacement (CANN
grouped-matmul fast path); the liger backend is a LigerExperts class
replacement that only takes effect when the user explicitly puts 'liger'
in the backend chain — the default chain is ('npu',), so the liger class
replacement never shadows the npu fast path (absorbs the drop semantics
of the old _prefer_cann_on_npu).
"""
from __future__ import annotations

from ...registry import KernelImpl, is_liger_available, is_npu_available, lazy_import, register_op

register_op(
    'moe_experts',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.moe.npu:npu_packed_moe_experts_forward'),
            available=is_npu_available,
        ),
        'liger':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.moe.liger:LigerExperts'),
            available=is_liger_available,
        ),
    },
)

register_op(
    'moe_block',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.moe.npu:npu_qwen3_5_moe_sparse_block_forward'),
            available=is_npu_available,
        ),
    },
)
