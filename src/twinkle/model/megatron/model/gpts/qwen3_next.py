# Copyright (c) ModelScope Contributors. All rights reserved.
# Reference: swift/swift/megatron/model/gpts/qwen3_next.py
# Qwen3-Next / Qwen3.5 series model support for Megatron

import megatron.core
import torch
from copy import deepcopy
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, _get_extra_te_kwargs
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_mtp_block_spec
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import (gather_from_sequence_parallel_region,
                                           reduce_scatter_to_sequence_parallel_region)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params, is_fa_min_version
from packaging import version
from typing import List, Optional, Tuple, Union

from twinkle import get_logger
from twinkle.model.megatron.args import get_args
from twinkle.model.megatron.model.register import MegatronModelLoader

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
mcore_015 = version.parse(megatron.core.__version__) >= version.parse('0.15.0rc0')
try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward
    from flashattn_hopper.flash_attn_interface import flash_attn_with_kvcache as flash_attn3_with_kvcache
    HAVE_FA3 = True
except Exception:
    HAVE_FA3 = False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    import transformer_engine  # pylint: disable=unused-import
    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None

logger = get_logger()


class Qwen3NextRMSNorm(torch.nn.Module):
    """
    Zero-Centered RMSNorm for Qwen3-Next/Qwen3.5.
    Uses (1 + weight) scaling to match HuggingFace implementation exactly.
    This eliminates the need for +1/-1 offset during weight conversion.
    """

    def __init__(self, config, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.config = config
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, hidden_states):
        output = self._norm(hidden_states.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(hidden_states)


class Qwen3NextSelfAttention(SelfAttention):
    """Full attention with output gate for Qwen3-Next/Qwen3.5 models.

    QKV projection produces [Q_heads, gate_heads, K_heads, V_heads] where
    Q and gate are interleaved: Q0, gate0, Q1, gate1, ...
    """

    def __init__(self, config, submodules: SelfAttentionSubmodules, *args, **kwargs):
        super(SelfAttention, self).__init__(config, submodules, *args, attention_type='self', **kwargs)
        kwargs_pg = {}
        if mcore_015:
            kwargs_pg['tp_group'] = self.pg_collection.tp
        elif mcore_013:
            kwargs_pg['tp_group'] = self.model_comm_pgs.tp
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            2 * self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            **kwargs_pg,
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from megatron.core.utils import nvtx_range_pop, nvtx_range_push
        except ImportError:

            def nvtx_range_pop(*args, **kwargs):
                return

            def nvtx_range_push(*args, **kwargs):
                return

        if hasattr(self.config, 'no_rope_freq'):
            no_rope = (self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False)
            if no_rope:
                rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version(
                '2.7.3'), 'flash attn verion v2.7.3 and above is required for dynamic batching.'

        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, ) * 2

        nvtx_range_push(suffix='qkv')
        query, key, value, gate = self.get_query_key_value_tensors(hidden_states, key_value_states)
        nvtx_range_pop(suffix='qkv')

        in_decode_mode = (inference_context is not None and inference_context.is_decode_only() and not self.training)

        nvtx_range_push(suffix='adjust_key_value')
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[self.layer_number]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        if (in_decode_mode and self.config.enable_cuda_graph and inference_context.is_static_batching()):
            raise ValueError('CUDA graphs must use flash decode with static batching!')

        result = self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )
        if mcore_013:
            query, key, value, rotary_pos_emb, attn_mask_type, block_table = result
        else:
            query, key, value, rotary_pos_emb, attn_mask_type = result

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix='adjust_key_value')

        kwargs_cp = {}
        if mcore_015:
            kwargs_cp['cp_group'] = self.pg_collection.cp
        elif mcore_013:
            kwargs_cp['cp_group'] = self.model_comm_pgs.cp
        nvtx_range_push(suffix='rotary_pos_emb')
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = (
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params.cu_seqlens_q_padded is not None else packed_seq_params.cu_seqlens_q)
                cu_seqlens_kv = (
                    packed_seq_params.cu_seqlens_kv_padded
                    if packed_seq_params.cu_seqlens_kv_padded is not None else packed_seq_params.cu_seqlens_kv)
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q, **kwargs_cp)
                else:
                    query = inference_context.apply_rotary_emb_query(query, q_pos_emb, self.config, cu_seqlens_q,
                                                                     **kwargs_cp)
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv, **kwargs_cp)
        nvtx_range_pop(suffix='rotary_pos_emb')

        nvtx_range_push(suffix='core_attention')
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, kv_lengths_decode_only, max_seqlen_k = (inference_context.cu_kv_lengths())
                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    kv_lengths_decode_only,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix='core_attention')

        core_attn_out = core_attn_out * torch.sigmoid(gate.reshape_as(core_attn_out))
        nvtx_range_push(suffix='linear_proj')
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix='linear_proj')

        return output, bias

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition // self.num_query_groups_per_partition * 2 + 2)
             * self.hidden_size_per_attention_head),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition
             * self.hidden_size_per_attention_head * 2),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        query, gate = query[:, :, ::2], query[:, :, 1::2]
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, gate


def _gated_delta_net_forward(self, hidden_states: torch.Tensor, **kwargs):
    """Shared forward logic for all GatedDeltaNet variants."""
    args = get_args()
    if args.sequence_parallel and args.tensor_model_parallel_size > 1:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)
    seq_len = hidden_states.shape[0]
    packed_seq_params = kwargs.get('packed_seq_params')
    thd_format = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
    if thd_format and not getattr(args, 'packing', False):
        new_hidden_states = hidden_states.new_zeros(
            (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item(), hidden_states.shape[-1]))
        attention_mask = hidden_states.new_zeros((packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item()),
                                                 dtype=torch.bool)
        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        for i in range(packed_seq_params.num_samples):
            start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            attention_mask[i, :end - start] = True
            new_hidden_states[i, :end - start] = hidden_states[start:end, 0]
        hidden_states = new_hidden_states
    else:
        hidden_states = hidden_states.transpose(0, 1)
        attention_mask = kwargs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = (~attention_mask).sum(dim=(1, 2)) > 0
    res = super(type(self), self).forward(hidden_states=hidden_states, attention_mask=attention_mask)
    if thd_format and not getattr(args, 'packing', False):
        res = res[attention_mask][:, None]
        res = torch.concat([res, res.new_zeros(seq_len - res.shape[0], 1, res.shape[2])])
    else:
        res = res.transpose(0, 1)
    if args.sequence_parallel and args.tensor_model_parallel_size > 1:
        res = reduce_scatter_to_sequence_parallel_region(res) / args.tensor_model_parallel_size
    return res, None


def _gated_delta_net_init(self, hf_cls, config, submodules, layer_number, **kwargs):
    """Shared __init__ logic for all GatedDeltaNet variants."""
    assert config.context_parallel_size == 1, 'Qwen3-Next/Qwen3.5 currently does not support context parallel.'
    hf_cls.__init__(self, config, layer_number)
    self.config = config
    extra_kwargs = _get_extra_te_kwargs(config)
    self.to(dtype=extra_kwargs['params_dtype'], device=extra_kwargs['device'])


try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet as _Qwen3_5MoeGatedDeltaNet
except ImportError:
    _Qwen3_5MoeGatedDeltaNet = object

try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet as _Qwen3NextGatedDeltaNet
except ImportError:
    _Qwen3NextGatedDeltaNet = object


class Qwen3NextGatedDeltaNet(_HuggingFaceModule, _Qwen3NextGatedDeltaNet):
    """GatedDeltaNet for linear attention layers in Qwen3-Next models."""

    def __init__(self, config, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
        assert _Qwen3NextGatedDeltaNet is not object, 'please update the `transformers` version.'
        _gated_delta_net_init(self, _Qwen3NextGatedDeltaNet, config, submodules, layer_number, **kwargs)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return _gated_delta_net_forward(self, hidden_states, **kwargs)


class Qwen3_5MoeGatedDeltaNet(_HuggingFaceModule, _Qwen3_5MoeGatedDeltaNet):
    """GatedDeltaNet for Qwen3.5-MoE linear attention layers."""

    def __init__(self, config, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
        assert _Qwen3_5MoeGatedDeltaNet is not object, 'please update the `transformers` version.'
        _gated_delta_net_init(self, _Qwen3_5MoeGatedDeltaNet, config, submodules, layer_number, **kwargs)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return _gated_delta_net_forward(self, hidden_states, **kwargs)


def get_local_layer_specs(config, layer_specs, vp_stage=None):
    """Get the layer specs for layers assigned to this pipeline stage.

    Mirrors swift.megatron.utils.get_local_layer_specs for distributing
    heterogeneous layer specs across pipeline stages.
    """
    from megatron.core import mpu
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    if pp_size <= 1:
        return layer_specs
    num_layers = len(layer_specs)
    layers_per_stage = num_layers // pp_size
    remainder = num_layers % pp_size
    start = pp_rank * layers_per_stage + min(pp_rank, remainder)
    if pp_rank < remainder:
        layers_per_stage += 1
    return layer_specs[start:start + layers_per_stage]


def get_qwen3_next_layer_spec(config, args, gated_delta_net_cls):
    """Build the heterogeneous transformer layer specs for Qwen3-Next/Qwen3.5.

    Returns a TransformerBlockSubmodules with per-layer specs matching
    the model's layer_types (linear_attention / full_attention).
    """
    config.hetereogenous_dist_checkpoint = True
    config.hidden_act = 'silu'
    config.rms_norm_eps = config.layernorm_epsilon
    config.dtype = args.params_dtype

    layer_norm_impl = Qwen3NextRMSNorm
    kwargs = {'use_kitchen': config.use_kitchen} if hasattr(config, 'use_kitchen') and mcore_013 else {}
    moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=getattr(config, 'moe_grouped_gemm', True),
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=getattr(config, 'moe_use_legacy_grouped_gemm', False),
        **kwargs,
    )
    layer_specs = []
    for layer_type in config.layer_types:
        layer_spec = deepcopy(moe_layer_spec)
        if layer_type == 'linear_attention':
            layer_spec.submodules.self_attention.module = gated_delta_net_cls
        elif layer_type == 'full_attention':
            layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
            layer_spec.submodules.self_attention.module = Qwen3NextSelfAttention
        layer_spec.submodules.input_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules,
                   'pre_mlp_layernorm') and layer_spec.submodules.pre_mlp_layernorm is not IdentityOp:
            layer_spec.submodules.pre_mlp_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, 'q_layernorm'):
            layer_spec.submodules.self_attention.submodules.q_layernorm = layer_norm_impl
        if hasattr(layer_spec.submodules.self_attention.submodules, 'k_layernorm'):
            layer_spec.submodules.self_attention.submodules.k_layernorm = layer_norm_impl
        if (getattr(config, 'moe_use_shared_expert_gate', False) and hasattr(layer_spec.submodules, 'mlp')
                and hasattr(layer_spec.submodules.mlp.submodules, 'shared_experts')):
            layer_spec.submodules.mlp.submodules.shared_experts.params = {'gate': True}
        layer_specs.append(layer_spec)

    local_layer_specs = get_local_layer_specs(config, layer_specs)
    block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)

    return block_spec


def get_qwen3_next_mtp_block_spec(config, transformer_layer_spec, **kwargs):
    """Build MTP block spec with Qwen3NextRMSNorm."""
    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=True, **kwargs)
    for layer_spec in mtp_block_spec.layer_specs:
        layer_spec.submodules.enorm = Qwen3NextRMSNorm
        layer_spec.submodules.hnorm = Qwen3NextRMSNorm
        layer_spec.submodules.layer_norm = Qwen3NextRMSNorm
    return mtp_block_spec


class Qwen3NextLoader(MegatronModelLoader):
    """Loader for Qwen3-Next models with heterogeneous linear/full attention layers."""
    gated_delta_net = Qwen3NextGatedDeltaNet

    def post_config(self, config, args, mg_config_dict):
        layer_types = mg_config_dict.get('layer_types')
        if layer_types is not None:
            config.layer_types = layer_types
            for attr in ('linear_num_value_heads', 'linear_num_key_heads', 'linear_key_head_dim',
                         'linear_value_head_dim', 'linear_conv_kernel_dim'):
                val = mg_config_dict.get(attr)
                if val is not None:
                    setattr(config, attr, val)

    def get_layer_spec(self, config, args, mg_config_dict):
        return get_qwen3_next_layer_spec(config, args, self.gated_delta_net)

    def get_mtp_block_spec(self, config, layer_spec, **kwargs):
        return get_qwen3_next_mtp_block_spec(config, layer_spec, **kwargs)
