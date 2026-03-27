# Copyright (c) ModelScope Contributors. All rights reserved.
"""Private MindSpeed runtime args helpers for NPU Megatron runs."""

import argparse
import json
import torch
from typing import Any, Dict

from .utils import convert_hf_config


def sanitize_mindspeed_values(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if isinstance(key, str) and key.isidentifier()}


def _resolve_optimization_level(values: Dict[str, Any]) -> int:
    if any((
            bool(values.get('multi_latent_attention')),
            bool(values.get('multi_head_latent_attention')),
            values.get('q_lora_rank') is not None,
            values.get('num_experts', 0) > 0,
            bool(values.get('moe_grouped_gemm')),
            bool(values.get('moe_fb_overlap')),
            bool(values.get('moe_alltoall_overlap_comm')),
            bool(values.get('moe_allgather_overlap_comm')),
            bool(values.get('balanced_moe_experts')),
            values.get('schedules_method') == 'dualpipev',
    )):
        return 2
    return 0


def _update_sanitized(values: Dict[str, Any], section: Dict[str, Any]) -> None:
    values.update(sanitize_mindspeed_values(section))


def _build_fixed_runtime_defaults() -> Dict[str, Any]:
    # Fixed MindSpeed / TE runtime defaults.
    return {
        'transformer_impl': 'transformer_engine',
        'fp8': None,
        'optimizer_selection': 'fused_adamw',
        'shape_order': 'SBH',
        'use_ascend_mc2': False,
        'enable_gloo_process_groups': True,
        'disable_gloo_group': False,
    }


def _build_topology_and_shape_defaults(args: Any, values: Dict[str, Any], rope_scaling: Dict[str,
                                                                                             Any]) -> Dict[str, Any]:
    # Core topology and transformer shape.
    return {
        'tensor_model_parallel_size': args.tp_size,
        'pipeline_model_parallel_size': args.pp_size,
        'context_parallel_size': args.cp_size,
        'expert_model_parallel_size': args.ep_size,
        'expert_tensor_parallel_size': args.etp_size,
        'virtual_pipeline_model_parallel_size': args.vpp_size,
        'sequence_parallel': bool(args.sequence_parallel),
        'num_layers': int(args.num_layers),
        'hidden_size': int(args.hidden_size),
        'num_attention_heads': int(args.num_attention_heads),
        'num_query_groups': int(args.num_query_groups or args.num_attention_heads),
        'ffn_hidden_size': int(args.ffn_hidden_size),
        'mtp_num_layers': int(args.mtp_num_layers or 0),
        'bf16': args.params_dtype == torch.bfloat16,
        'fp16': args.params_dtype == torch.float16,
        'position_embedding_type': values.get('position_embedding_type', 'rope'),
        'rope_scaling_type': rope_scaling.get('rope_type') or rope_scaling.get('type'),
        'yarn_scaling_factor': rope_scaling.get('factor'),
        'rope_scaling_mscale': rope_scaling.get('mscale'),
        'rope_scaling_mscale_all_dim': rope_scaling.get('mscale_all_dim'),
    }


def _build_moe_runtime_defaults(values: Dict[str, Any], args: Any, num_experts: int) -> Dict[str, Any]:
    # MoE runtime knobs.
    return {
        'num_experts': num_experts,
        'num_moe_experts': num_experts or None,
        'moe_grouped_gemm': bool(values.get('moe_grouped_gemm', False) or num_experts > 0),
        'moe_token_dispatcher_type': values.get('moe_token_dispatcher_type')
        or ('alltoall' if num_experts > 0 else None),
        'moe_router_topk': int(values.get('moe_router_topk', args.num_experts_per_tok) or 2),
    }


def _build_mla_runtime_defaults(values: Dict[str, Any], q_lora_rank: Any, multi_latent_attention: bool,
                                qk_layernorm: bool, args: Any) -> Dict[str, Any]:
    # MLA / DeepSeek-style attention knobs.
    return {
        'multi_latent_attention': multi_latent_attention,
        'multi_head_latent_attention': multi_latent_attention,
        'q_lora_rank': q_lora_rank,
        'kv_lora_rank': values.get('kv_lora_rank'),
        'qk_layernorm': qk_layernorm,
        'use_qk_norm': qk_layernorm,
        'qk_nope_head_dim': values.get('qk_head_dim', values.get('qk_nope_head_dim')),
        'qk_rope_head_dim': values.get('qk_pos_emb_head_dim', values.get('qk_rope_head_dim')),
        'v_head_dim': values.get('v_head_dim', args.kv_channels),
    }


def build_mindspeed_namespace(args: Any, defaults: Dict[str, Any]) -> argparse.Namespace:
    """Build MindSpeed runtime args namespace from Twinkle args.

    If there are fields with the same name, the one at the lowest level will be overwritten.

    Merges three layers in order of precedence (later layers override earlier ones):
    1. MindSpeed defaults (~100+ fields from register_args) - lowest priority
    2. HF config (layers, heads, MoE params via convert_hf_config()) - medium priority
    3. Twinkle args (tp/pp/cp/ep, dtype) - highest priority, overrides all

    Args:
        args: TwinkleMegatronArgs instance.
        defaults: MindSpeed default values from args_utils.get_mindspeed_args().

    Returns:
        Merged MindSpeed runtime arguments as Namespace.
    """
    if getattr(args, 'fp8', None):
        raise RuntimeError('MindSpeed NPU TE bootstrap does not support FP8.')

    values = sanitize_mindspeed_values(defaults.copy())
    hf_config = getattr(args, 'hf_config', None)
    if hf_config is not None:
        values.update(sanitize_mindspeed_values(convert_hf_config(hf_config)))

    rope_scaling = args.rope_scaling if isinstance(args.rope_scaling, dict) else {}
    num_experts = int(getattr(args, 'num_experts', 0) or values.get('num_experts', 0) or 0)
    q_lora_rank = values.get('q_lora_rank', getattr(args, 'q_lora_rank', None))
    multi_latent_attention = bool(
        getattr(args, 'multi_latent_attention', False) or values.get('multi_latent_attention', False)
        or values.get('multi_head_latent_attention', False) or q_lora_rank is not None)
    qk_layernorm = bool(getattr(args, 'qk_layernorm', False) or values.get('qk_layernorm', False))

    _update_sanitized(values, _build_fixed_runtime_defaults())
    _update_sanitized(values, _build_topology_and_shape_defaults(args, values, rope_scaling))
    _update_sanitized(values, _build_moe_runtime_defaults(values, args, num_experts))
    _update_sanitized(values,
                      _build_mla_runtime_defaults(values, q_lora_rank, multi_latent_attention, qk_layernorm, args))
    values['optimization_level'] = _resolve_optimization_level(values)
    return argparse.Namespace(**sanitize_mindspeed_values(values))


def get_mindspeed_signature(namespace: argparse.Namespace) -> str:
    return json.dumps(sanitize_mindspeed_values(vars(namespace).copy()), sort_keys=True, default=str)
