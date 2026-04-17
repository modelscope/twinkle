"""MindSpeed runtime bootstrap for Twinkle's Megatron NPU path.

This module deliberately keeps two phases separate:
1. Early import-time patching via ``mindspeed.megatron_adaptor`` before
   ``mcore_bridge`` is imported.
2. Runtime args synthesis and ``repatch()`` once ``ModelConfig`` exists.
"""

import argparse
import json
import torch
from typing import Any, Dict

from twinkle import Platform
from twinkle.utils import get_logger

logger = get_logger()

_MINDSPEED_IMPORTED = False
_LAST_RUNTIME_SIGNATURE = None


def _is_npu() -> bool:
    return Platform.device_prefix() == 'npu'


def ensure_mindspeed_adaptor_patched() -> None:
    """Import MindSpeed's official adaptor before any mcore/TE import on NPU.

    ``mcore_bridge.__init__`` immediately imports its patcher, and that patcher
    pulls in ``megatron.core`` and TE symbols at module import time. MindSpeed's
    patch stack must land before that import chain, otherwise TE symbols and
    ``torch.compile``-related hooks are bound too early.
    """
    global _MINDSPEED_IMPORTED
    if not _is_npu() or _MINDSPEED_IMPORTED:
        return
    import mindspeed.megatron_adaptor  # noqa: F401
    _MINDSPEED_IMPORTED = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _is_runtime_value(value: Any) -> bool:
    return isinstance(value, (type(None), bool, int, float, str, list, tuple, dict, torch.dtype))


def _compute_optimization_level(config: Any) -> int:
    num_moe_experts = getattr(config, 'num_moe_experts', None)
    has_moe = num_moe_experts not in (None, 0, 1)
    # MindSpeed's context-parallel feature stack is gated behind optimization
    # level 2. If Twinkle launches a CP run with the default level 0, the CP
    # patch set never gets registered and ring state stays uninitialized.
    if int(getattr(config, 'context_parallel_size', 1) or 1) > 1:
        return 2
    if getattr(config, 'multi_latent_attention', False):
        return 2
    if has_moe and getattr(config, 'moe_grouped_gemm', False):
        return 2
    if getattr(config, 'schedules_method', None) == 'dualpipev':
        return 2
    return 0


def _force_megatron_cp_te_patch(runtime_args: argparse.Namespace) -> None:
    """Twinkle-side override for MindSpeed TE CP class selection on NPU.

    MindSpeed 0.15.3 routes TE context parallel through a factory that only
    accepts `kvallgather_cp_algo`. Twinkle still wants the default
    `megatron_cp_algo` ring path for the Megatron smoke, so we override the TE
    class back to the older `MindSpeedCPDotProductAttention` from the Twinkle
    runtime layer instead of changing MindSpeed sources.
    """
    if not _is_npu():
        return
    if int(getattr(runtime_args, 'context_parallel_size', 1)) <= 1:
        return
    if getattr(runtime_args, 'context_parallel_algo', 'megatron_cp_algo') != 'megatron_cp_algo':
        return

    from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
    from mindspeed.patch_utils import MindSpeedPatchesManager

    MindSpeedPatchesManager.register_patch(
        'megatron.core.extensions.transformer_engine.TEDotProductAttention',
        MindSpeedCPDotProductAttention,
        force_patch=True,
    )
    MindSpeedPatchesManager.apply_patches()
    logger.info('Forced TEDotProductAttention to MindSpeedCPDotProductAttention for megatron_cp_algo.')


def _ensure_megatron_cp_ring_state(runtime_args: argparse.Namespace) -> None:
    """Initialize MindSpeed's ring CP globals when the default path is selected.

    MindSpeed 0.15.x already owns the real ring-attention logic, but Twinkle can
    still end up with the TE class patched back to the legacy CP path while the
    ring globals remain unset. If that happens, the first forward dies in
    ``get_ring_ranks_for_intra_window()`` even though the model parallel groups
    are already up. We repair the MindSpeed module state here, from Twinkle, so
    the shared runtime behavior stays intact without editing MindSpeed sources.
    """
    if not _is_npu():
        return
    if int(getattr(runtime_args, 'context_parallel_size', 1)) <= 1:
        return
    if getattr(runtime_args, 'context_parallel_algo', 'megatron_cp_algo') != 'megatron_cp_algo':
        return
    if not torch.distributed.is_initialized():
        return

    from mindspeed.core.context_parallel import model_parallel_utils as cp_utils

    try:
        cp_utils.get_ring_ranks_for_intra_window()
        return
    except AssertionError:
        pass

    from megatron.core import mpu

    cp_utils.initialize_context_parallel_group_for_double_ring(
        mpu.get_tensor_model_parallel_world_size(),
        mpu.get_pipeline_model_parallel_world_size(),
        mpu.get_context_parallel_world_size(),
        {},
    )
    logger.info('Initialized MindSpeed ring CP state for megatron_cp_algo from Twinkle bootstrap.')


def build_mindspeed_runtime_args(config: Any) -> argparse.Namespace:
    """Build the runtime namespace MindSpeed 0.15.3 consumes on NPU.

    We start from MindSpeed feature defaults and overlay the current
    ``ModelConfig`` values. The config object is already the single source of
    truth in the new Twinkle + mcore-bridge architecture, so we do not keep a
    second Twinkle-side args protocol here.
    """
    from mindspeed.args_utils import get_mindspeed_args

    defaults = get_mindspeed_args(get_defaults=True)
    values: Dict[str, Any] = vars(defaults).copy()

    for key, value in vars(config).items():
        if key.startswith('_') or key in {'bridge', 'model_meta', 'hf_config'}:
            continue
        if not _is_runtime_value(value):
            continue
        values[key] = value

    num_moe_experts = getattr(config, 'num_moe_experts', None)
    if num_moe_experts not in (None, 0):
        values['num_experts'] = num_moe_experts
        values['num_moe_experts'] = num_moe_experts

    if getattr(config, 'multi_latent_attention', False):
        values['multi_head_latent_attention'] = True
    if getattr(config, 'qk_head_dim', None) is not None:
        values['qk_nope_head_dim'] = config.qk_head_dim
    if getattr(config, 'qk_pos_emb_head_dim', None) is not None:
        values['qk_rope_head_dim'] = config.qk_pos_emb_head_dim
    # MindSpeed's CP rotary-pos helper reads this flag directly even when the
    # base Twinkle/MCore config path does not define it.
    values.setdefault('reset_position_ids', False)

    params_dtype = getattr(config, 'params_dtype', None)
    if params_dtype == torch.bfloat16:
        values['bf16'] = True
        values['fp16'] = False
    elif params_dtype == torch.float16:
        values['fp16'] = True
        values['bf16'] = False
    elif params_dtype is not None:
        values['fp16'] = False
        values['bf16'] = False

    values['optimization_level'] = _compute_optimization_level(config)
    return argparse.Namespace(**values)


def configure_mindspeed_runtime_args(config: Any) -> argparse.Namespace:
    """Install current runtime args and repatch MindSpeed on signature changes."""
    global _LAST_RUNTIME_SIGNATURE

    if not _is_npu():
        return argparse.Namespace()

    ensure_mindspeed_adaptor_patched()

    from mindspeed import args_utils
    from mindspeed.megatron_adaptor import repatch

    runtime_args = build_mindspeed_runtime_args(config)
    args_utils._MINDSPEED_ARGS = runtime_args

    runtime_signature = json.dumps(
        {
            k: _jsonable(v)
            for k, v in sorted(vars(runtime_args).items())
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    if runtime_signature != _LAST_RUNTIME_SIGNATURE:
        repatch(vars(runtime_args))
        _LAST_RUNTIME_SIGNATURE = runtime_signature
        logger.info(
            'Configured MindSpeed runtime args for NPU, optimization_level=%s',
            getattr(runtime_args, 'optimization_level', None),
        )
    _force_megatron_cp_te_patch(runtime_args)
    _ensure_megatron_cp_ring_state(runtime_args)
    return runtime_args
