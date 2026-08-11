# Copyright (c) ModelScope Contributors. All rights reserved.
"""sdpa_attention op registration: installs the NPU SDPA forward into the
transformers global attention registry (not a plain class/attr replacement,
hence the custom installer).

The default config references this op via the logical target 'sdpa'; the
logical target is only a mapping label passed to install_sdpa and is never
resolved by the generic replacer.
"""
from __future__ import annotations

from twinkle import get_logger
from ...registry import KernelImpl, is_npu_available, lazy_import, register_op

logger = get_logger()


def install_sdpa(model, target, impl) -> None:
    """One-shot install of SDPA attention forward (global modeling_utils dict).

    ``AttentionInterface._global_mapping`` is a private transformers attribute;
    guard against its removal so an upstream change can't take down the rest
    of kernelize.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    except ImportError:
        return
    try:
        AttentionInterface._global_mapping['sdpa'] = impl
    except AttributeError:
        logger.warning('[SDPA] AttentionInterface._global_mapping unavailable; skipping')
    ALL_ATTENTION_FUNCTIONS['sdpa'] = impl


register_op(
    'sdpa_attention',
    implementations={
        'npu':
        KernelImpl(
            load=lazy_import('twinkle.kernel.ops.sdpa_attention.npu:npu_sdpa_attention_forward'),
            available=is_npu_available,
        ),
    },
    installer=install_sdpa,
)
