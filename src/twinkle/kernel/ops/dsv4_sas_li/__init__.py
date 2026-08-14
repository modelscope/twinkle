# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4 SAS/LI op registration: three forward-level replacements.

No env var gate: the op registration is lazy (only loaded when
``kernelize`` resolves the ``KernelChoice``), the C++ extension is
JIT-compiled on first use, and the SAS/LI forward has ``try/except``
fallback to standard attention if the ACLNN kernel is unavailable.

The full patch set is applied:

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

from ...registry import KernelImpl, is_npu_available, lazy_import, register_op

_DSV4_BASE = 'twinkle.kernel.ops.dsv4_sas_li.npu'

register_op(
    'dsv4_attention',
    implementations={
        'npu': KernelImpl(
            load=lazy_import(f'{_DSV4_BASE}:npu_dsv4_attention_forward'),
            available=is_npu_available,
        ),
    },
)

register_op(
    'dsv4_indexer',
    implementations={
        'npu': KernelImpl(
            load=lazy_import(f'{_DSV4_BASE}:npu_dsv4_indexer_forward'),
            available=is_npu_available,
        ),
    },
)

register_op(
    'dsv4_csa_compressor',
    implementations={
        'npu': KernelImpl(
            load=lazy_import(f'{_DSV4_BASE}:npu_dsv4_csa_compressor_forward'),
            available=is_npu_available,
        ),
    },
)
