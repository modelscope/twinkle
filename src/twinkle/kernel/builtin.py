# Copyright (c) ModelScope Contributors. All rights reserved.
"""``npu_builtin()`` returns the bundle of Ascend NPU replacements.

All values are wrapped in ``{'npu': impl}`` so the bundle composes safely on
CUDA/CPU systems — non-NPU devices silently skip every entry.

GMM is **not** included by default (without EP it causes ~8x slowdown). Opt
in by merging:

    {**npu_builtin(model), 'transformers.integrations.moe._grouped_mm':
                            {'npu': npu_grouped_mm}}
"""
from __future__ import annotations

import importlib
from typing import Any

import torch.nn as nn

from twinkle import get_logger

logger = get_logger()


def _import_optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def npu_builtin(model: nn.Module | None = None) -> dict[Any, dict[str, Any]]:
    """Return the NPU builtin mapping; optionally apply per-instance FLA."""
    from .npu_impls.attention import npu_sdpa_attention_forward
    from .npu_impls.fla import apply_qwen3_5_fla
    from .npu_impls.moe import (
        npu_packed_moe_experts_forward,
        npu_qwen3_5_moe_sparse_block_forward,
    )
    from .npu_impls.rms_norm import NpuRMSNorm, npu_gated_rms_norm_forward
    from .npu_impls.rotary import (
        npu_apply_multimodal_rotary_pos_emb,
        npu_apply_rotary_pos_emb,
    )
    from .npu_impls.swiglu import npu_swiglu_forward

    bundle: dict[Any, dict[str, Any]] = {}

    # SDPA attention (global)
    bundle['transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS'] = {'npu': _SdpaPatchSentinel()}
    # NOTE: ALL_ATTENTION_FUNCTIONS is a dict, not a function. We can't setattr
    # it. We instead install the sdpa entry by a small bootstrap below.
    # Remove the sentinel approach in favor of explicit module-level entries:
    bundle.pop('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', None)

    # Apply SDPA install eagerly (one-shot module-level mutation).
    _install_sdpa(npu_sdpa_attention_forward)

    # === per-family class + function entries ===
    _add_qwen2_entries(bundle, NpuRMSNorm, npu_apply_rotary_pos_emb, npu_swiglu_forward)
    _add_qwen3_entries(bundle, NpuRMSNorm, npu_apply_rotary_pos_emb, npu_swiglu_forward)
    _add_qwen3_moe_entries(
        bundle, NpuRMSNorm, npu_apply_rotary_pos_emb, npu_swiglu_forward,
        npu_packed_moe_experts_forward, npu_qwen3_5_moe_sparse_block_forward,
    )
    _add_qwen2_5_vl_entries(
        bundle, NpuRMSNorm, npu_apply_rotary_pos_emb, npu_swiglu_forward,
        npu_apply_multimodal_rotary_pos_emb,
    )
    _add_qwen3_5_entries(
        bundle, NpuRMSNorm, npu_gated_rms_norm_forward, npu_apply_rotary_pos_emb,
        npu_swiglu_forward,
    )
    _add_qwen3_5_moe_entries(
        bundle, NpuRMSNorm, npu_gated_rms_norm_forward, npu_apply_rotary_pos_emb,
        npu_swiglu_forward, npu_packed_moe_experts_forward,
        npu_qwen3_5_moe_sparse_block_forward,
    )

    # === FLA (side-effect; mapping-incompatible) ===
    apply_qwen3_5_fla(model)

    return bundle


class _SdpaPatchSentinel:
    pass  # unused; placeholder retained for clarity in diffs


def _install_sdpa(impl) -> None:
    """One-shot install of SDPA attention forward (global modeling_utils dict)."""
    try:
        from transformers.modeling_utils import (
            ALL_ATTENTION_FUNCTIONS,
            AttentionInterface,
        )
    except ImportError:
        return
    AttentionInterface._global_mapping['sdpa'] = impl
    ALL_ATTENTION_FUNCTIONS['sdpa'] = impl


# ---- helpers that conditionally add entries based on module availability ----

def _add_class_if_present(bundle, module_path, class_name, impl_cls):
    mod = _import_optional(module_path)
    if mod is None:
        return
    cls = getattr(mod, class_name, None)
    if isinstance(cls, type):
        bundle[cls] = {'npu': impl_cls}


def _add_swiglu_if_present(bundle, module_path, class_name, fn):
    mod = _import_optional(module_path)
    if mod is None:
        return
    cls = getattr(mod, class_name, None)
    if isinstance(cls, type):
        # Function-level: wrap as string-keyed forward replacement.
        # We override on the *class object*, not the module attribute, by
        # using a class-key with a synthetic impl wrapping the forward.
        # The simplest way is to subclass and reassign __class__, but here
        # we follow the legacy approach of overwriting the class's forward:
        bundle[f'{module_path}.{class_name}.forward'] = {'npu': fn}


def _add_attr_if_present(bundle, module_path, attr_name, impl):
    mod = _import_optional(module_path)
    if mod is None:
        return
    if '.' in attr_name:
        # Dotted attr like 'Qwen3MoeExperts.forward': resolve the class on
        # the module, then check the trailing member on the class.
        head, _, tail = attr_name.partition('.')
        owner = getattr(mod, head, None)
        if owner is None or not hasattr(owner, tail):
            return
    else:
        if not hasattr(mod, attr_name):
            return
    bundle[f'{module_path}.{attr_name}'] = {'npu': impl}


def _add_qwen2_entries(bundle, rms_cls, rope_fn, swiglu_fn):
    # Qwen2 (used by Qwen2.5-VL etc. via inheritance)
    _add_class_if_present(bundle, 'transformers.models.qwen2.modeling_qwen2', 'Qwen2RMSNorm', rms_cls)
    _add_attr_if_present(bundle, 'transformers.models.qwen2.modeling_qwen2', 'apply_rotary_pos_emb', rope_fn)
    _add_swiglu_if_present(bundle, 'transformers.models.qwen2.modeling_qwen2', 'Qwen2MLP', swiglu_fn)


def _add_qwen3_entries(bundle, rms_cls, rope_fn, swiglu_fn):
    base = 'transformers.models.qwen3.modeling_qwen3'
    _add_class_if_present(bundle, base, 'Qwen3RMSNorm', rms_cls)
    _add_attr_if_present(bundle, base, 'apply_rotary_pos_emb', rope_fn)
    _add_swiglu_if_present(bundle, base, 'Qwen3MLP', swiglu_fn)


def _add_qwen3_moe_entries(bundle, rms_cls, rope_fn, swiglu_fn, experts_fn, sparse_fn):
    base = 'transformers.models.qwen3_moe.modeling_qwen3_moe'
    _add_class_if_present(bundle, base, 'Qwen3MoeRMSNorm', rms_cls)
    _add_attr_if_present(bundle, base, 'apply_rotary_pos_emb', rope_fn)
    _add_swiglu_if_present(bundle, base, 'Qwen3MoeMLP', swiglu_fn)
    _add_attr_if_present(bundle, base, 'Qwen3MoeExperts.forward', experts_fn)
    _add_attr_if_present(bundle, base, 'Qwen3MoeSparseMoeBlock.forward', sparse_fn)


def _add_qwen2_5_vl_entries(bundle, rms_cls, rope_fn, swiglu_fn, multimodal_rope_fn):
    base = 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl'
    _add_class_if_present(bundle, base, 'Qwen2_5_VLRMSNorm', rms_cls)
    _add_attr_if_present(bundle, base, 'apply_rotary_pos_emb', rope_fn)
    _add_attr_if_present(bundle, base, 'apply_multimodal_rotary_pos_emb', multimodal_rope_fn)
    _add_swiglu_if_present(bundle, base, 'Qwen2MLP', swiglu_fn)
    _add_swiglu_if_present(bundle, base, 'Qwen2_5_VLMLP', swiglu_fn)


def _add_qwen3_5_entries(bundle, rms_cls, gated_rms_fn, rope_fn, swiglu_fn):
    base = 'transformers.models.qwen3_5.modeling_qwen3_5'
    if _import_optional(base) is None:
        return
    _add_class_if_present(bundle, base, 'Qwen3_5RMSNorm', rms_cls)
    _add_class_if_present(bundle, base, 'Qwen3_5VisionRMSNorm', rms_cls)
    _add_attr_if_present(bundle, base, 'apply_rotary_pos_emb', rope_fn)
    _add_swiglu_if_present(bundle, base, 'Qwen3_5MLP', swiglu_fn)
    _add_swiglu_if_present(bundle, base, 'Qwen3_5VisionMLP', swiglu_fn)
    # Qwen3_5GatedRMSNorm: forward-level replacement
    _add_attr_if_present(bundle, base, 'Qwen3_5GatedRMSNorm.forward', gated_rms_fn)


def _add_qwen3_5_moe_entries(bundle, rms_cls, gated_rms_fn, rope_fn, swiglu_fn,
                             experts_fn, sparse_fn):
    base = 'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe'
    if _import_optional(base) is None:
        return
    _add_class_if_present(bundle, base, 'Qwen3_5MoeRMSNorm', rms_cls)
    _add_attr_if_present(bundle, base, 'apply_rotary_pos_emb', rope_fn)
    _add_swiglu_if_present(bundle, base, 'Qwen3_5MoeMLP', swiglu_fn)
    _add_attr_if_present(bundle, base, 'Qwen3_5MoeExperts.forward', experts_fn)
    _add_attr_if_present(bundle, base, 'Qwen3_5MoeSparseMoeBlock.forward', sparse_fn)
    _add_attr_if_present(bundle, base, 'Qwen3_5MoeGatedRMSNorm.forward', gated_rms_fn)