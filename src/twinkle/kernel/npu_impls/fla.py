# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen3.5 Flash Linear Attention enablement for Ascend NPU."""
from __future__ import annotations

import importlib
import os

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

    # 1. Confirm the MindSpeed Triton kernel is actually importable BEFORE
    #    flipping any global availability flags. If we flip the flag and then
    #    fail to install the kernel, HF transformers would route Qwen3.5 onto
    #    a FLA fast path whose kernel is missing -> runtime failure on NPU.
    try:
        from twinkle.kernel.chunk_gated_delta_rule import chunk_gated_delta_rule as mindspeed_fla
        from twinkle.kernel.causal_conv1d import npu_causal_conv1d_fn
    except ImportError as exc:
        logger.warning('[NPU] [FLA] MindSpeed unavailable: %s', exc)
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
        setattr(module, 'chunk_gated_delta_rule', mindspeed_fla)

    # 4. Traverse model and patch per-layer attributes
    if model is None:
        return 0

    root = getattr(model, 'model', getattr(model, 'module', model))
    if not hasattr(root, 'named_modules'):
        return 0

    patched_instances = 0
    for _name, _module in root.named_modules():
        if hasattr(_module, 'chunk_gated_delta_rule') and callable(
                getattr(_module, 'chunk_gated_delta_rule')):
            if _module.chunk_gated_delta_rule is not mindspeed_fla:
                _module.chunk_gated_delta_rule = mindspeed_fla
                _module._twinkle_npu_patched = True
                patched_instances += 1
        if hasattr(_module, 'causal_conv1d_fn'):
            if getattr(_module, 'causal_conv1d_fn') is not npu_causal_conv1d_fn:
                _module.causal_conv1d_fn = npu_causal_conv1d_fn

    if patched_instances:
        logger.info('[NPU] [FLA] Patched %d linear attention instance(s)', patched_instances)
    return patched_instances