# Copyright (c) ModelScope Contributors. All rights reserved.
"""swiglu op registration: npu / liger implementations, class-forward replacement."""
from __future__ import annotations

from ...registry import KernelImpl, is_liger_available, is_npu_available, lazy_import, register_op

register_op(
    'swiglu',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.swiglu.npu:npu_swiglu_forward'),
            available=is_npu_available,
        ),
        'liger':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.swiglu.liger:liger_swiglu_forward'),
            available=is_liger_available,
        ),
    },
)
