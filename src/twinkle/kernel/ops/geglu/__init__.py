# Copyright (c) ModelScope Contributors. All rights reserved.
"""geglu op registration: forward-level replacement for gemma-family MLPs, liger only."""
from __future__ import annotations

from ...registry import KernelImpl, is_liger_available, lazy_import, register_op

register_op(
    'geglu',
    implementations={
        'liger': KernelImpl(
            load=lazy_import('twinkle.kernel.ops.geglu.liger:liger_geglu_forward'),
            available=is_liger_available,
        ),
    },
)
