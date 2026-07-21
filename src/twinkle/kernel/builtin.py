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


def npu_builtin(model: nn.Module | None = None) -> dict[Any, dict[str, Any]]:
    """Return the NPU builtin mapping; optionally apply per-instance FLA."""
    from .npu_impls.attention import (npu_dsv4_attention_forward, npu_dsv4_csa_compressor_forward,
                                      npu_dsv4_indexer_forward, npu_sdpa_attention_forward)
    from .npu_impls.fla import apply_qwen3_5_fla
    from .npu_impls.moe import npu_packed_moe_experts_forward, npu_qwen3_5_moe_sparse_block_forward
    from .npu_impls.rms_norm import NpuRMSNorm, npu_gated_rms_norm_forward
    from .npu_impls.rotary import npu_apply_multimodal_rotary_pos_emb, npu_apply_rotary_pos_emb
    from .npu_impls.swiglu import npu_swiglu_forward

    bundle: dict[Any, dict[str, Any]] = {}

    is_npu_platform = Platform.device_prefix() == 'npu'

    # Apply SDPA install eagerly (one-shot module-level mutation) on NPU
    # platforms. The NPU impl inverts boolean masks, which is wrong for
    # CUDA/CPU execution, so non-NPU platforms must not mutate the global HF
    # registry even if ``torch_npu`` is importable in the environment.
    if is_npu_platform:
        _install_sdpa(npu_sdpa_attention_forward)

    # === per-family class + function entries ===
    _add_qwen2_entries(bundle, NpuRMSNorm, npu_apply_rotary_pos_emb, npu_swiglu_forward)
    _add_qwen3_entries(bundle, NpuRMSNorm, npu_apply_rotary_pos_emb, npu_swiglu_forward)
    _add_qwen3_moe_entries(
        bundle,
        NpuRMSNorm,
        npu_apply_rotary_pos_emb,
        npu_swiglu_forward,
        npu_packed_moe_experts_forward,
        npu_qwen3_5_moe_sparse_block_forward,
    )
    _add_qwen2_5_vl_entries(
        bundle,
        NpuRMSNorm,
        npu_apply_rotary_pos_emb,
        npu_swiglu_forward,
        npu_apply_multimodal_rotary_pos_emb,
    )
    _add_qwen3_5_entries(
        bundle,
        NpuRMSNorm,
        npu_gated_rms_norm_forward,
        npu_apply_rotary_pos_emb,
        npu_swiglu_forward,
    )
    _add_qwen3_5_moe_entries(
        bundle,
        NpuRMSNorm,
        npu_gated_rms_norm_forward,
        npu_apply_rotary_pos_emb,
        npu_swiglu_forward,
        npu_packed_moe_experts_forward,
        npu_qwen3_5_moe_sparse_block_forward,
    )
    _add_deepseek_v4_entries(
        bundle,
        npu_dsv4_attention_forward,
        npu_dsv4_indexer_forward,
        npu_dsv4_csa_compressor_forward,
    )

    # === FLA (side-effect; mapping-incompatible) ===
    if is_npu_platform:
        from twinkle.utils.import_utils import exists
        if exists('mindspeed>=0.12.1') and exists('flash-linear-attention'):
            apply_qwen3_5_fla(model)
        else:
            logger.warning('[NPU] [FLA] mindspeed or flash-linear-attention is not installed, '
                           'or mindspeed version smaller than 0.12.1; '
                           'FLA patch for Qwen3.5 requires mindspeed and flash-linear-attention. '
                           'Install them with: pip install flash-linear-attention, '
                           'and install mindspeed from source code via '
                           'https://gitcode.com/Ascend/MindSpeed. '
                           'Other NPU patches will still apply.')
    return bundle


def _install_sdpa(impl) -> None:
    """One-shot install of SDPA attention forward (global modeling_utils dict).

    ``AttentionInterface._global_mapping`` is a private transformers attribute;
    guard against its removal so an upstream change can't take down the rest
    of ``npu_builtin()``.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    except ImportError:
        return
    try:
        AttentionInterface._global_mapping['sdpa'] = impl
    except AttributeError:
        logger.warning('[NPU] [SDPA] AttentionInterface._global_mapping unavailable; skipping')
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


def _add_qwen3_5_moe_entries(bundle, rms_cls, gated_rms_fn, rope_fn, swiglu_fn, experts_fn, sparse_fn):
    base = 'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe'
    if _import_optional(base) is None:
        return
    _add_class_if_present(bundle, base, 'Qwen3_5MoeRMSNorm', rms_cls)
    _add_attr_if_present(bundle, base, 'apply_rotary_pos_emb', rope_fn)
    _add_swiglu_if_present(bundle, base, 'Qwen3_5MoeMLP', swiglu_fn)
    _add_attr_if_present(bundle, base, 'Qwen3_5MoeExperts.forward', experts_fn)
    _add_attr_if_present(bundle, base, 'Qwen3_5MoeSparseMoeBlock.forward', sparse_fn)
    _add_attr_if_present(bundle, base, 'Qwen3_5MoeGatedRMSNorm.forward', gated_rms_fn)


def _add_deepseek_v4_entries(bundle, attention_fn, indexer_fn, csa_compressor_fn):
    """Register DeepSeek-V4 NPU attention / indexer / compressor forwards.

    Opt-in via the single env var ``TWINKLE_NPU_DSV4_SAS`` (Sparse Attention).
    When enabled, the full patch set is applied as one unit:

      - ``DeepseekV4Attention.forward``     → NPU sparse attention (SAS)
      - ``DeepseekV4Indexer.forward``       → Lightning Indexer (LI) — selects
        top-512 compressed blocks per query for CSA layers via mindspeed
      - ``DeepseekV4CSACompressor.forward`` → full replacement returning a
        3-tuple ``(compressed_kv, block_bias, top_k_indices)``

    HCA compressor is **not** patched: its stock forward returns a 2-tuple,
    which the SAS attention forward handles via ``len(compressor_out)`` —
    HCA layers don't use top-k (``cmp_sparse_indices = None``), so
    ``top_k_indices`` staying ``None`` is the correct behavior.

    LI is always on under SAS — there is no use case for SAS without LI
    (CSA would fall back to the slower stock indexer) or LI without SAS
    (indices would go unused). The CSA compressor is a **full forward
    replacement** rather than a wrapper: the stock forward already calls
    ``self.indexer(...)`` to build ``block_bias``, so a wrapper that fetched
    ``top_k_indices`` by re-calling the indexer would mutate
    ``DeepseekV4CSACache`` twice (``store_compression_weights`` appends on
    every call). Under gradient checkpointing the recomputed forward would
    see a cache already mutated by the first forward, producing a different
    compressed length and triggering ``CheckpointError``. The replacement
    calls the indexer **once** and returns ``top_k_indices`` alongside the
    other outputs.
    """
    base = 'transformers.models.deepseek_v4.modeling_deepseek_v4'
    if _import_optional(base) is None:
        return

    _add_attr_if_present(bundle, base, 'DeepseekV4Attention.forward', attention_fn)
    _add_attr_if_present(bundle, base, 'DeepseekV4Indexer.forward', indexer_fn)
    _add_attr_if_present(bundle, base, 'DeepseekV4CSACompressor.forward', csa_compressor_fn)
    logger.info('[NPU] [DSV4] SAS + LI patch registered (CSA uses Lightning Indexer top-k)')
