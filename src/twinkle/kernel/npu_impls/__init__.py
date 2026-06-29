# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-layer NPU implementations consumed by ``npu_builtin()``.

Each impl is contracted to be applied via ``m.__class__ = ImplCls`` (class
replacement) or ``setattr(module, attr, fn)`` (function replacement). No impl
here is meant to be instantiated directly.
"""
from .attention import npu_sdpa_attention_forward
from .fla import apply_qwen3_5_fla
from .moe import GmmFunction, npu_grouped_mm, npu_packed_moe_experts_forward, npu_qwen3_5_moe_sparse_block_forward
from .rms_norm import NpuRMSNorm, npu_gated_rms_norm_forward
from .rotary import npu_apply_multimodal_rotary_pos_emb, npu_apply_rotary_pos_emb
from .swiglu import npu_swiglu_forward

__all__ = [
    'NpuRMSNorm',
    'npu_gated_rms_norm_forward',
    'npu_apply_rotary_pos_emb',
    'npu_apply_multimodal_rotary_pos_emb',
    'npu_swiglu_forward',
    'npu_sdpa_attention_forward',
    'GmmFunction',
    'npu_grouped_mm',
    'npu_packed_moe_experts_forward',
    'npu_qwen3_5_moe_sparse_block_forward',
    'apply_qwen3_5_fla',
]
