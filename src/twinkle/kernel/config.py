# Copyright (c) ModelScope Contributors. All rights reserved.
"""Built-in default mapping: target -> replacement (KernelChoice).

Pure data file: never imports transformers / torch_npu / liger_kernel, no side effects;
targets are uniformly dotted-path strings (class targets are resolved by default_installer into __class__ replacement,
targets whose family is not installed are skipped at DEBUG level during install).

``kernelize(model)`` ≡ ``kernelize(model, DEFAULT_KERNEL_CONFIG)``。
User customization = copy + override: ``{**DEFAULT_KERNEL_CONFIG, <target>: <value>}`` --
the mapping fully replaces the default config, it is not incremental. This dict must not be mutated in place at runtime.
"""
from __future__ import annotations

from typing import Any

from .registry import KernelChoice

# ── Qwen families (npu first, liger fallback) ──────────────────────────
# (module, rms_cls, mlp_classes, extra_rms_classes)
_QWEN_DENSE = [
    ('transformers.models.qwen2.modeling_qwen2', 'Qwen2RMSNorm', ['Qwen2MLP'], []),
    ('transformers.models.qwen3.modeling_qwen3', 'Qwen3RMSNorm', ['Qwen3MLP'], []),
    ('transformers.models.qwen2_5_vl.modeling_qwen2_5_vl', 'Qwen2_5_VLRMSNorm', ['Qwen2MLP', 'Qwen2_5_VLMLP'], []),
    ('transformers.models.qwen3_5.modeling_qwen3_5', 'Qwen3_5RMSNorm', ['Qwen3_5MLP',
                                                                        'Qwen3_5VisionMLP'], ['Qwen3_5VisionRMSNorm']),
]
# (module, rms_cls, mlp_cls, experts_cls, block_cls)
_QWEN_MOE = [
    ('transformers.models.qwen3_moe.modeling_qwen3_moe', 'Qwen3MoeRMSNorm', 'Qwen3MoeMLP', 'Qwen3MoeExperts',
     'Qwen3MoeSparseMoeBlock'),
    ('transformers.models.qwen3_5_moe.modeling_qwen3_5_moe', 'Qwen3_5MoeRMSNorm', 'Qwen3_5MoeMLP', 'Qwen3_5MoeExperts',
     'Qwen3_5MoeSparseMoeBlock'),
]
# qwen3_5 Gated RMSNorm (forward-level replacement, no liger impl)
_QWEN_GATED_RMS = {
    'transformers.models.qwen3_5.modeling_qwen3_5': 'Qwen3_5GatedRMSNorm',
    'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe': 'Qwen3_5MoeGatedRMSNorm',
}
# Qwen3.5 uses partial_rotary_factor=0.25 + mrope_interleaved; liger's full-rotation
# rotate_half impl is incompatible -> rotary chains for these two families are ('npu',) only
_QWEN35_ROPE_NPU_ONLY = (
    'transformers.models.qwen3_5.modeling_qwen3_5',
    'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
)

# ── llama families x8 (liger only) ──────────────────────────────────────
_LLAMA_STYLE = [
    ('transformers.models.llama.modeling_llama', 'LlamaRMSNorm', 'LlamaMLP'),
    ('transformers.models.mistral.modeling_mistral', 'MistralRMSNorm', 'MistralMLP'),
    ('transformers.models.mixtral.modeling_mixtral', 'MixtralRMSNorm', 'MixtralBlockSparseTop2MLP'),
    ('transformers.models.phi3.modeling_phi3', 'Phi3RMSNorm', 'Phi3MLP'),
    ('transformers.models.glm4.modeling_glm4', 'Glm4RMSNorm', 'Glm4MLP'),
    ('transformers.models.olmo2.modeling_olmo2', 'Olmo2RMSNorm', 'Olmo2MLP'),
    ('transformers.models.granite.modeling_granite', 'GraniteRMSNorm', 'GraniteMLP'),
    ('transformers.models.internvl.modeling_internvl', 'InternVLRMSNorm', 'InternVLMLP'),
]

# ── gemma families x4 (liger only: rms_norm + geglu) ────────────────────
_GEMMA_STYLE = [
    ('transformers.models.gemma.modeling_gemma', 'GemmaRMSNorm', 'GemmaMLP'),
    ('transformers.models.gemma2.modeling_gemma2', 'Gemma2RMSNorm', 'Gemma2MLP'),
    ('transformers.models.gemma3.modeling_gemma3', 'Gemma3RMSNorm', 'Gemma3MLP'),
    ('transformers.models.gemma4.modeling_gemma4', 'Gemma4RMSNorm', 'Gemma4TextMLP'),
]


def _rope_backends(module: str) -> tuple[str, ...]:
    return ('npu', ) if module in _QWEN35_ROPE_NPU_ONLY else ('npu', 'liger')


def _build() -> dict[Any, Any]:
    cfg: dict[Any, Any] = {}

    # Qwen dense：rms_norm / rotary / swiglu
    for mod, rms, mlps, extra_rms in _QWEN_DENSE:
        cfg[f'{mod}.{rms}'] = KernelChoice(op='rms_norm', backends=('npu', 'liger'))
        for extra in extra_rms:
            cfg[f'{mod}.{extra}'] = KernelChoice(op='rms_norm', backends=('npu', 'liger'))
        cfg[f'{mod}.apply_rotary_pos_emb'] = KernelChoice(op='rotary', backends=_rope_backends(mod))
        for mlp in mlps:
            cfg[f'{mod}.{mlp}.forward'] = KernelChoice(op='swiglu', backends=('npu', 'liger'))

    # Qwen MoE: dense trio + moe_experts / moe_block
    # moe_experts default chain ('npu',): liger's LigerExperts class replacement only
    # takes effect when explicitly chosen by the user; it never blocks the npu CANN
    # grouped-matmul fast path (absorbs the drop semantics of the old _prefer_cann_on_npu)
    for mod, rms, mlp, experts, block in _QWEN_MOE:
        cfg[f'{mod}.{rms}'] = KernelChoice(op='rms_norm', backends=('npu', 'liger'))
        cfg[f'{mod}.apply_rotary_pos_emb'] = KernelChoice(op='rotary', backends=_rope_backends(mod))
        cfg[f'{mod}.{mlp}.forward'] = KernelChoice(op='swiglu', backends=('npu', 'liger'))
        cfg[f'{mod}.{experts}.forward'] = KernelChoice(op='moe_experts', backends=('npu', ))
        cfg[f'{mod}.{block}.forward'] = KernelChoice(op='moe_block', backends=('npu', ))

    # Qwen gated rms_norm (npu only)
    for mod, gated in _QWEN_GATED_RMS.items():
        cfg[f'{mod}.{gated}.forward'] = KernelChoice(op='gated_rms_norm', backends=('npu', ))

    # Qwen2.5-VL multimodal rope (npu only)
    cfg['transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb'] = KernelChoice(
        op='multimodal_rotary', backends=('npu', ))

    # llama families x3 ops, gemma families x2 ops (liger only)
    for mod, rms, mlp in _LLAMA_STYLE:
        cfg[f'{mod}.{rms}'] = KernelChoice(op='rms_norm', backends=('liger', ))
        cfg[f'{mod}.apply_rotary_pos_emb'] = KernelChoice(op='rotary', backends=('liger', ))
        cfg[f'{mod}.{mlp}.forward'] = KernelChoice(op='swiglu', backends=('liger', ))
    for mod, rms, mlp in _GEMMA_STYLE:
        cfg[f'{mod}.{rms}'] = KernelChoice(op='rms_norm', backends=('liger', ))
        cfg[f'{mod}.{mlp}.forward'] = KernelChoice(op='geglu', backends=('liger', ))

    # logical target: handled by a custom installer (never resolved by the generic replacer)
    cfg['sdpa'] = KernelChoice(op='sdpa_attention', backends=('npu', ))
    cfg['fla'] = KernelChoice(op='fla', backends=('npu', ))

    # DeepSeek-V4 SAS + LI (lazy: op registration + C++ JIT + try/except fallback)
    _dsv4 = 'transformers.models.deepseek_v4.modeling_deepseek_v4'
    cfg[f'{_dsv4}.DeepseekV4Attention.forward'] = KernelChoice(op='dsv4_attention', backends=('npu', ))
    cfg[f'{_dsv4}.DeepseekV4Indexer.forward'] = KernelChoice(op='dsv4_indexer', backends=('npu', ))
    cfg[f'{_dsv4}.DeepseekV4CSACompressor.forward'] = KernelChoice(op='dsv4_csa_compressor', backends=('npu', ))
    return cfg


DEFAULT_KERNEL_CONFIG = _build()
