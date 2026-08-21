# Copyright (c) ModelScope Contributors. All rights reserved.
"""fla op registration: Qwen3.5 Flash Linear Attention per-instance patching
(custom installer, needs the model passed to kernelize).

Availability = NPU platform + flash-linear-attention installed; when fla is
missing the whole chain fails and falls back per the warn/debug rules,
preserving the "missing fla only warns, other patches proceed" semantics.
"""
from __future__ import annotations

from ...registry import KernelImpl, exists, is_npu_available, lazy_import, register_op


def install_fla(model, target, impl) -> None:
    impl(model)  # impl = apply_qwen3_5_fla


def _fla_available() -> tuple[bool, str | None]:
    ok, reason = is_npu_available()
    if not ok:
        return ok, reason
    if not exists('flash-linear-attention'):
        return False, 'flash-linear-attention not installed'
    return True, None


register_op(
    'fla',
    implementations={
        'npu': KernelImpl(
            load=lazy_import('twinkle.kernel.ops.fla.npu:apply_qwen3_5_fla'),
            available=_fla_available,
        ),
    },
    installer=install_fla,
)
