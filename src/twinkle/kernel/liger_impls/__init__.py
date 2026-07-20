# Copyright (c) ModelScope Contributors. All rights reserved.
"""Class-replacement adapters that bridge Liger Kernel modules onto
HuggingFace instances when applied via ``kernelize``.

``kernelize`` swaps ``m.__class__ = impl_cls`` *without* calling ``__init__``
(see ``Kernel.md`` caveats). Liger's own ``LigerRMSNorm.forward`` reads
instance attributes (``offset`` / ``casting_mode`` / ``in_place`` / ``row_mode``)
that HuggingFace RMSNorm variants do not define. Liger's monkey-patch path
sets those attributes eagerly via ``_patch_rms_norm_module``; the adapters here
do the same lazily inside ``forward`` so the class-replacement contract is
honoured and no global state is mutated.

SwiGLU / Experts / RoPE need no adapter — Liger's classes/functions read only
attributes (``gate_proj`` / ``up_proj`` / ``down_proj`` / ``weight``) that the
HuggingFace instances already provide, so they are re-exported verbatim.
"""
from .rms_norm import (LigerRMSNormGemma4Replacement, LigerRMSNormGemmaReplacement, LigerRMSNormQwen35Replacement,
                       LigerRMSNormReplacement)
from .swiglu import liger_geglu_forward, liger_swiglu_forward

__all__ = [
    'LigerRMSNormReplacement',
    'LigerRMSNormGemmaReplacement',
    'LigerRMSNormGemma4Replacement',
    'LigerRMSNormQwen35Replacement',
    'liger_swiglu_forward',
    'liger_geglu_forward',
]
