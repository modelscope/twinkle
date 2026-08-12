# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen3.5 Flash Linear Attention enablement for Ascend NPU.

Delegates to fla's native operators (``fla.modules.convolution.causal_conv1d``
and ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``) so that no MindSpeed
or twinkle-own Triton kernels are referenced. The fla package ships its own
Ascend-backend dispatch (see ``fla/backends``), so the same call path works on
NPU without a vendor-specific reimplementation.

This mirrors the ms-swift approach (``swift.model.npu_patch.mindspeed.
patch_mindspeed_fla_gdn_implementation``) which prefers upstream fla over
MindSpeed's GDN implementation.
"""
from __future__ import annotations

import importlib
import os
import torch

from twinkle import get_logger

logger = get_logger()


def _is_env_enabled(var: str, default: bool = True) -> bool:
    env = os.environ.get(var, '').lower().strip()
    if not env:
        return default
    if env in ('1', 'true', 'on', 'yes'):
        return True
    if env in ('0', 'false', 'off', 'no'):
        return False
    return default


def _import_optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def npu_causal_conv1d_fn(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
    backend: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    """Adapter for twinkle's ``causal_conv1d_fn`` call signature.

    Delegates to fla's native ``causal_conv1d``. Handles two input layouts:
    standard Qwen3.5 path passes ``x=[B, D, T]`` (needs transpose), SP path
    passes ``x=[B, T, D]`` (no transpose needed). Detected via
    ``x.shape[-1] == D and x.shape[1] != D``; when ``T == D`` (ambiguous)
    defaults to transposing (standard path).
    """
    del seq_idx, backend
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

    D, W = weight.shape[0], weight.shape[1]
    is_bt_d_layout = (
        x.dim() == 3 and weight.dim() == 2 and x.shape[-1] == D  # last dim is the channel dim
        and x.shape[1] != D  # second dim is NOT D -> genuinely [B, T, D]
        and D != W  # sanity: D != kernel_size
    )
    if is_bt_d_layout:
        y, _ = fla_causal_conv1d(x=x, weight=weight, bias=bias, activation=activation, cu_seqlens=cu_seqlens)
        return y
    else:
        x_t = x.transpose(1, 2).contiguous()
        y_t, _ = fla_causal_conv1d(x=x_t, weight=weight, bias=bias, activation=activation, cu_seqlens=cu_seqlens)
        return y_t.transpose(1, 2).contiguous()


def apply_qwen3_5_fla(model=None) -> int:
    """Enable Flash Linear Attention fast path for Qwen3.5 on NPU.

    Returns the count of patched per-layer instances (0 when disabled or when
    prerequisites are missing). Safe to call multiple times.
    """
    if not _is_env_enabled('TWINKLE_NPU_FLA', default=True):
        logger.info('[NPU] [FLA] Disabled by TWINKLE_NPU_FLA')
        return 0

    if _import_optional('torch_npu') is None:
        logger.info('[NPU] [FLA] Skip: torch_npu unavailable')
        return 0

    # 1. Confirm the fla native operators are actually importable BEFORE
    #    flipping any global availability flags. If we flip the flag and then
    #    fail to install the kernel, HF transformers would route Qwen3.5 onto
    #    a FLA fast path whose kernel is missing -> runtime failure on NPU.
    try:
        from fla.modules.convolution import causal_conv1d as _fla_causal_conv1d  # noqa: F401
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
    except ImportError as exc:
        logger.warning('[NPU] [FLA] fla native operators unavailable: %s', exc)
        return 0

    # 2. Only now can we safely claim FLA is available: flip the global flags
    #    and install the kernel path on Qwen3.5 modeling modules.
    def _is_fla_available() -> bool:
        return True

    for utils_mod_name in ('transformers.utils', 'transformers.utils.import_utils'):
        utils_mod = _import_optional(utils_mod_name)
        if utils_mod is not None:
            setattr(utils_mod, 'is_flash_linear_attention_available', _is_fla_available)

    # 3. Patch Qwen3.5 modeling modules
    fla_target_modules = [
        'transformers.models.qwen3_5.modeling_qwen3_5',
        'transformers.models.qwen3_5_moe.modeling_qwen3_5_moe',
    ]
    for module_name in fla_target_modules:
        module = _import_optional(module_name)
        if module is None:
            continue
        setattr(module, 'is_flash_linear_attention_available', _is_fla_available)
        setattr(module, 'is_fast_path_available', True)
        if hasattr(module, 'FusedRMSNormGated'):
            setattr(module, 'FusedRMSNormGated', None)
        setattr(module, 'chunk_gated_delta_rule', fla_chunk_gated_delta_rule)

    # 4. Traverse model and patch per-layer attributes
    if model is None:
        return 0

    root = getattr(model, 'model', getattr(model, 'module', model))
    if not hasattr(root, 'named_modules'):
        return 0

    patched_instances = 0
    for _name, _module in root.named_modules():
        if hasattr(_module, 'chunk_gated_delta_rule') and callable(getattr(_module, 'chunk_gated_delta_rule')):
            if _module.chunk_gated_delta_rule is not fla_chunk_gated_delta_rule:
                _module.chunk_gated_delta_rule = fla_chunk_gated_delta_rule
                _module._twinkle_npu_patched = True
                patched_instances += 1
        if hasattr(_module, 'causal_conv1d_fn'):
            if getattr(_module, 'causal_conv1d_fn') is not npu_causal_conv1d_fn:
                _module.causal_conv1d_fn = npu_causal_conv1d_fn

    if patched_instances:
        logger.info('[NPU] [FLA] Patched %d linear attention instance(s)', patched_instances)
    return patched_instances
