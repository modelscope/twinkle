# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4 SAS/LI op registration: three forward-level replacements gated
by the ``TWINKLE_NPU_DSV4_SAS`` env var.

When enabled, the full patch set is applied:

  - ``DeepseekV4Attention.forward``     → NPU sparse attention (SAS)
  - ``DeepseekV4Indexer.forward``       → Lightning Indexer (LI)
  - ``DeepseekV4CSACompressor.forward`` → full replacement returning a
    3-tuple ``(compressed_kv, block_bias, top_k_indices)``

LI is always on under SAS — there is no use case for SAS without LI
(CSA would fall back to the slower stock indexer) or LI without SAS
(indices would go unused). The CSA compressor is a **full forward
replacement** rather than a wrapper (see ``npu.py`` docstring for details).
"""
from __future__ import annotations

import os

from ...registry import KernelImpl, is_npu_available, lazy_import, register_op


def _dsv4_sas_available() -> tuple[bool, str | None]:
    env = os.environ.get('TWINKLE_NPU_DSV4_SAS', '').lower().strip()
    if not env or env in ('0', 'false', 'off', 'no'):
        return False, 'TWINKLE_NPU_DSV4_SAS not enabled'
    ok, reason = is_npu_available()
    if not ok:
        return ok, reason
    return True, None


_DSV4_BASE = 'twinkle.kernel.ops.dsv4_sas_li.npu'

register_op(
    'dsv4_attention',
    implementations={
        'npu': KernelImpl(
            load=lazy_import(f'{_DSV4_BASE}:npu_dsv4_attention_forward'),
            available=_dsv4_sas_available,
        ),
    },
)

register_op(
    'dsv4_indexer',
    implementations={
        'npu': KernelImpl(
            load=lazy_import(f'{_DSV4_BASE}:npu_dsv4_indexer_forward'),
            available=_dsv4_sas_available,
        ),
    },
)

register_op(
    'dsv4_csa_compressor',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import(f'{_DSV4_BASE}:npu_dsv4_csa_compressor_forward'),
            available=_dsv4_sas_available,
        ),
    },
)
