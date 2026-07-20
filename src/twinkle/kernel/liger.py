# Copyright (c) ModelScope Contributors. All rights reserved.
"""``liger_builtin()`` returns the Liger Kernel replacement bundle.

This mirrors ``npu_builtin()`` but for Liger Kernel. Two key differences from
the NPU bundle, both consequences of Liger being cross-device:

  1. **No device gating.** Values are *bare* impls (not ``{'cuda': impl}``).
     Liger's modules self-dispatch across CUDA (Triton) and Ascend NPU (the
     auto-applied ``backends/_ascend`` backend) via ``infer_device`` /
     ``select_impl`` in ``liger_kernel``. Device-conditional wrapping would
     wrongly skip Liger on NPU, where it is fully supported.

  2. **No side effects.** Unlike ``npu_builtin`` (which installs global SDPA
     and per-instance FLA), this bundle contains only class/function
     replacements consumable by ``kernelize``. The process-level
     ``nn.functional.cross_entropy`` swap and fused-linear-CE ``forward``
     replacement are deliberately *excluded* — they belong to the loss layer,
     not the kernel layer (see ROADMAP / ``twinkle.loss``).

Composing with ``npu_builtin`` on NPU: both bundles are NPU implementations of
the same operators. Plain dict merge lets the user pick precedence (later keys
win) — see ``Kernel.md``:

    model = kernelize(model, {**npu_builtin(model), **liger_builtin(model)})  # Liger wins
    model = kernelize(model, {**liger_builtin(model), **npu_builtin(model)})  # Twinkle-NPU wins
    model = kernelize(model, liger_builtin(model))                            # Liger only

RMSNorm uses Liger-aware adapter classes (``liger_impls.rms_norm``) that set the
Liger-specific instance attributes lazily in ``forward``; SwiGLU/GeGLU use
function-level forward replacements; RoPE and the Qwen3-MoE expert/MLP classes
are wired as bare Liger classes (their forwards read only attributes the HF
instances already provide).
"""
from __future__ import annotations

import importlib
import torch.nn as nn
from typing import Any

from twinkle import get_logger
from twinkle.utils.device_mesh import Platform

logger = get_logger()


def _import_optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _has(maybe_none, attr_path: str) -> bool:
    """Return True if ``attr_path`` (may contain one dot, e.g. ``Cls.member``)
    exists on ``maybe_none`` (a module or None)."""
    if maybe_none is None:
        return False
    if '.' in attr_path:
        head, _, tail = attr_path.partition('.')
        owner = getattr(maybe_none, head, None)
        return owner is not None and hasattr(owner, tail)
    return hasattr(maybe_none, attr_path)


def _add_class_if_present(bundle, module_path, class_name, impl_cls):
    mod = _import_optional(module_path)
    if mod is None:
        return
    cls = getattr(mod, class_name, None)
    if isinstance(cls, type):
        bundle[cls] = impl_cls


def _add_fwd_if_present(bundle, module_path, class_name, fn):
    """Function-level forward replacement: ``bundle['<module>.<Cls>.forward'] = fn``.

    Only added when the HF module imports AND the target class exists, so the
    bundle stays safe when a model family is not installed.
    """
    mod = _import_optional(module_path)
    if mod is None:
        return
    cls = getattr(mod, class_name, None)
    if isinstance(cls, type):
        bundle[f'{module_path}.{class_name}.forward'] = fn


def _add_attr_if_present(bundle, module_path, attr_name, impl):
    mod = _import_optional(module_path)
    if mod is None:
        return
    if _has(mod, attr_name):
        bundle[f'{module_path}.{attr_name}'] = impl


# ── family registrations ──────────────────────────────────────────────────────
# Each helper mirrors the subset of liger_kernel's apply_liger_kernel_to_<family>
# that is expressible as a pure class/function replacement (RoPE, RMSNorm,
# SwiGLU/GeGLU). Fused-linear-CE and the global cross_entropy swap are excluded.


def _add_llama_style(bundle, module_path, rms_name, mlp_name, with_rope=True):
    """llama / qwen / mistral / mixtral / phi3 / olmo2 / glm4 / granite / internvl:
    llama-cast RMSNorm + SwiGLU MLP + (optional) RoPE."""
    from liger_kernel.transformers import liger_rotary_pos_emb

    from .liger_impls import LigerRMSNormReplacement, liger_swiglu_forward

    _add_class_if_present(bundle, module_path, rms_name, LigerRMSNormReplacement)
    _add_fwd_if_present(bundle, module_path, mlp_name, liger_swiglu_forward)
    if with_rope:
        _add_attr_if_present(bundle, module_path, 'apply_rotary_pos_emb', liger_rotary_pos_emb)


def _add_gemma_style(bundle, module_path, rms_name, mlp_name):
    """gemma / gemma2 / gemma3_text: gemma-cast RMSNorm (offset=1.0) + GeGLU MLP.
    RoPE differs per gemma variant and is left to the model-specific entries."""
    from .liger_impls import LigerRMSNormGemmaReplacement, liger_geglu_forward

    _add_class_if_present(bundle, module_path, rms_name, LigerRMSNormGemmaReplacement)
    _add_fwd_if_present(bundle, module_path, mlp_name, liger_geglu_forward)


def _add_qwen2(bundle):
    _add_llama_style(bundle, 'transformers.models.qwen2.modeling_qwen2', 'Qwen2RMSNorm', 'Qwen2MLP')


def _add_qwen3(bundle):
    _add_llama_style(bundle, 'transformers.models.qwen3.modeling_qwen3', 'Qwen3RMSNorm', 'Qwen3MLP')


def _add_qwen3_moe(bundle):
    base = 'transformers.models.qwen3_moe.modeling_qwen3_moe'
    from liger_kernel.transformers import LigerExperts, LigerQwen3MoeSwiGLUMLP, liger_rotary_pos_emb

    from .liger_impls import LigerRMSNormReplacement, liger_swiglu_forward

    _add_class_if_present(bundle, base, 'Qwen3MoeRMSNorm', LigerRMSNormReplacement)
    _add_attr_if_present(bundle, base, 'apply_rotary_pos_emb', liger_rotary_pos_emb)
    # transformers v5+: experts are a separate fused class; v4 uses Qwen3MoeMLP.
    _add_class_if_present(bundle, base, 'Qwen3MoeExperts', LigerExperts)
    _add_fwd_if_present(bundle, base, 'Qwen3MoeMLP', liger_swiglu_forward)
    # Qwen3MoeMLP (v4) is structurally identical to LigerQwen3MoeSwiGLUMLP; offer
    # class replacement too so v4 users get the fused SiLU kernel without a
    # forward-level swap shadowing it.
    mod = _import_optional(base)
    if mod is not None and isinstance(getattr(mod, 'Qwen3MoeMLP', None), type):
        if not any(k == getattr(mod, 'Qwen3MoeMLP') for k in bundle):
            bundle[getattr(mod, 'Qwen3MoeMLP')] = LigerQwen3MoeSwiGLUMLP


def _add_qwen3_5(bundle):
    from .liger_impls import LigerRMSNormQwen35Replacement, liger_swiglu_forward

    base = 'transformers.models.qwen3_5.modeling_qwen3_5'
    if _import_optional(base) is None:
        return
    _add_class_if_present(bundle, base, 'Qwen3_5RMSNorm', LigerRMSNormQwen35Replacement)
    _add_class_if_present(bundle, base, 'Qwen3_5VisionRMSNorm', LigerRMSNormQwen35Replacement)
    # RoPE intentionally NOT replaced: Qwen3.5 uses partial_rotary_factor=0.25 +
    # mrope_interleaved=True; Liger's liger_rotary_pos_emb assumes full-rotation
    # with the rotate_half convention, which is incompatible. Let npu_builtin's
    # npu_apply_rotary_pos_emb (which handles Partial-RoPE) take precedence.
    _add_fwd_if_present(bundle, base, 'Qwen3_5MLP', liger_swiglu_forward)
    _add_fwd_if_present(bundle, base, 'Qwen3_5VisionMLP', liger_swiglu_forward)


def _add_qwen3_5_moe(bundle):
    from liger_kernel.transformers import LigerExperts

    from .liger_impls import LigerRMSNormQwen35Replacement, liger_swiglu_forward

    base = 'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe'
    if _import_optional(base) is None:
        return
    _add_class_if_present(bundle, base, 'Qwen3_5MoeRMSNorm', LigerRMSNormQwen35Replacement)
    # RoPE intentionally NOT replaced (same reason as _add_qwen3_5).
    _add_class_if_present(bundle, base, 'Qwen3_5MoeExperts', LigerExperts)
    _add_fwd_if_present(bundle, base, 'Qwen3_5MoeMLP', liger_swiglu_forward)


def _add_qwen2_5_vl(bundle):
    from liger_kernel.transformers import liger_rotary_pos_emb

    from .liger_impls import LigerRMSNormReplacement, liger_swiglu_forward

    base = 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl'
    if _import_optional(base) is None:
        return
    _add_class_if_present(bundle, base, 'Qwen2_5_VLRMSNorm', LigerRMSNormReplacement)
    _add_attr_if_present(bundle, base, 'apply_rotary_pos_emb', liger_rotary_pos_emb)
    _add_fwd_if_present(bundle, base, 'Qwen2MLP', liger_swiglu_forward)
    _add_fwd_if_present(bundle, base, 'Qwen2_5_VLMLP', liger_swiglu_forward)


def _add_gemma4(bundle):
    base = 'transformers.models.gemma4.modeling_gemma4'
    if _import_optional(base) is None:
        return
    from .liger_impls import LigerRMSNormGemma4Replacement, liger_geglu_forward

    _add_class_if_present(bundle, base, 'Gemma4RMSNorm', LigerRMSNormGemma4Replacement)
    _add_fwd_if_present(bundle, base, 'Gemma4TextMLP', liger_geglu_forward)


def liger_builtin(model: nn.Module | None = None) -> dict[Any, Any]:
    """Return the Liger Kernel built-in mapping; composes with ``kernelize``.

    Args:
        model: Accepted for API symmetry with ``npu_builtin(model)``; Liger's
            pure class/function replacements have no per-instance side effects,
            so the model is not traversed here.

    Returns:
        A ``dict`` whose keys are HF ``nn.Module`` subclasses (class replacement)
        or dotted paths (function/forward replacement) and whose values are
        *bare* Liger impls (no device gating). Missing model families are
        silently skipped — the bundle only contains entries for installed
        transformers model modules.

    On NPU, Liger's Triton-on-Ascend kernels are slower than the CANN vendor
    ops in ``npu_impls`` for the bandwidth-bound per-layer ops (RMSNorm,
    SwiGLU, RoPE). ``_prefer_cann_on_npu`` post-processes the bundle to swap
    those Liger impls for their CANN equivalents and drops the LigerExperts
    class replacement (which shadows ``npu_builtin``'s faster forward-level
    MoE expert replacement). Non-Qwen families without a CANN equivalent keep
    their Liger impls. On CUDA, the bundle is unchanged.

    Raises:
        ImportError: if ``liger_kernel`` is not importable (caught at the first
            ``from liger_kernel ...`` statement inside the family helpers).
    """
    bundle: dict[Any, Any] = {}

    # ── Qwen family (primary on Twinkle) ──────────────────────────────────────
    _add_qwen2(bundle)
    _add_qwen3(bundle)
    _add_qwen3_moe(bundle)
    _add_qwen3_5(bundle)
    _add_qwen3_5_moe(bundle)
    _add_qwen2_5_vl(bundle)

    # ── Llama-style dense / MoE families ─────────────────────────────────────
    _add_llama_style(bundle, 'transformers.models.llama.modeling_llama', 'LlamaRMSNorm', 'LlamaMLP')
    _add_llama_style(bundle, 'transformers.models.mistral.modeling_mistral', 'MistralRMSNorm', 'MistralMLP')
    _add_llama_style(bundle, 'transformers.models.mixtral.modeling_mixtral', 'MixtralRMSNorm',
                     'MixtralBlockSparseTop2MLP')
    _add_llama_style(bundle, 'transformers.models.phi3.modeling_phi3', 'Phi3RMSNorm', 'Phi3MLP')
    _add_llama_style(bundle, 'transformers.models.glm4.modeling_glm4', 'Glm4RMSNorm', 'Glm4MLP')
    _add_llama_style(bundle, 'transformers.models.olmo2.modeling_olmo2', 'Olmo2RMSNorm', 'Olmo2MLP')
    _add_llama_style(bundle, 'transformers.models.granite.modeling_granite', 'GraniteRMSNorm', 'GraniteMLP')
    _add_llama_style(bundle, 'transformers.models.internvl.modeling_internvl', 'InternVLRMSNorm', 'InternVLMLP')

    # ── Gemma family (gemma-cast RMSNorm + GeGLU) ────────────────────────────
    _add_gemma_style(bundle, 'transformers.models.gemma.modeling_gemma', 'GemmaRMSNorm', 'GemmaMLP')
    _add_gemma_style(bundle, 'transformers.models.gemma2.modeling_gemma2', 'Gemma2RMSNorm', 'Gemma2MLP')
    _add_gemma_style(bundle, 'transformers.models.gemma3.modeling_gemma3', 'Gemma3RMSNorm', 'Gemma3MLP')
    _add_gemma4(bundle)

    if not bundle:
        logger.warning('[liger_builtin] No Liger entries were registered — '
                       'is liger_kernel installed and are transformers model modules importable?')

    # On NPU, prefer CANN vendor ops over Liger's Triton-on-Ascend kernels for
    # the per-layer ops where CANN is significantly faster (RMSNorm: single-pass
    # vs Liger's 2-pass tiled; SwiGLU/RoPE: one CANN op vs Triton launch + extra
    # allocations). Also drop LigerExperts class replacements that shadow
    # npu_builtin's faster forward-level MoE expert replacement.
    if Platform.device_prefix() == 'npu':
        _prefer_cann_on_npu(bundle)
    return bundle


def _prefer_cann_on_npu(bundle: dict[Any, Any]) -> None:
    """In-place: swap Liger Triton impls for CANN vendor ops on NPU.

    Replaces:
      - LigerRMSNorm* class values  -> NpuRMSNorm (single-pass CANN aclnn)
      - liger_swiglu_forward values -> npu_swiglu_forward (one CANN fused op)
      - liger_rotary_pos_emb values -> npu_apply_rotary_pos_emb (no Q/K copies)

    Removes:
      - LigerExperts / LigerQwen3MoeSwiGLUMLP class keys — they replace
        ``m.__class__`` which shadows npu_builtin's forward-level
        ``npu_packed_moe_experts_forward`` (a different dict key), leaving
        the fast CANN grouped-matmul MoE path unused.
    """
    from .npu_impls import NpuRMSNorm, npu_apply_rotary_pos_emb, npu_swiglu_forward

    # Stringify once for fast membership checks.
    def _val_name(v) -> str:
        return getattr(v, '__name__', getattr(v, '__qualname__', ''))

    keys_to_drop: list[Any] = []
    for key, val in list(bundle.items()):
        name = _val_name(val)

        # Drop LigerExperts / LigerQwen3MoeSwiGLUMLP class replacements.
        if isinstance(val, type) and any(
                s in name for s in ('LigerExperts', 'LigerQwen3MoeSwiGLUMLP', 'LigerQwen3_5MoeSwiGLUMLP')):
            keys_to_drop.append(key)
            continue

        # Swap LigerRMSNorm* class replacements -> NpuRMSNorm.
        if isinstance(val, type) and name.startswith('LigerRMSNorm'):
            bundle[key] = NpuRMSNorm
            continue

        # Swap liger_swiglu_forward -> npu_swiglu_forward.
        if name == 'liger_swiglu_forward':
            bundle[key] = npu_swiglu_forward
            continue

        # Swap liger_rotary_pos_emb -> npu_apply_rotary_pos_emb.
        if name == 'liger_rotary_pos_emb':
            bundle[key] = npu_apply_rotary_pos_emb
            continue

    for key in keys_to_drop:
        del bundle[key]

    if keys_to_drop:
        logger.info(
            '[liger_builtin] NPU: dropped %d LigerExperts/LigerMoeMLP class replacement(s) '
            'so npu_builtin\'s CANN grouped-matmul MoE forward takes effect', len(keys_to_drop))
