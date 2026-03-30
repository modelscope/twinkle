import importlib
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional

from twinkle.patch import Patch
from twinkle.model.transformers.strategy.sequence_parallel.utils import head_to_seq_shard, seq_to_head_shard


def _sp_rank(sequence_parallel) -> int:
    if getattr(sequence_parallel, 'sp_world_size', 1) <= 1:
        return 0
    if getattr(sequence_parallel, '_sp_group', None) is None:
        return 0
    return dist.get_rank(group=sequence_parallel._sp_group)


def _slice_conv_channel_params(
    params: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    head_indices: torch.Tensor,
) -> torch.Tensor:
    if params.dim() == 1:
        reshaped = params.reshape(num_heads, head_dim)
        return reshaped.index_select(0, head_indices).reshape(-1)
    if params.dim() == 2:
        reshaped = params.reshape(num_heads, head_dim, params.shape[-1])
        return reshaped.index_select(0, head_indices).reshape(-1, params.shape[-1])
    raise ValueError(f'Unsupported channel param dims: {params.dim()}')


def _run_depthwise_causal_conv(
    mixed_qkv_hp: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    module: torch.nn.Module,
    origin_causal_conv1d_fn,
) -> torch.Tensor:
    batch_size, seq_len, local_heads, channel_dim = mixed_qkv_hp.shape
    x = mixed_qkv_hp.reshape(batch_size, seq_len, local_heads * channel_dim).transpose(1, 2)
    if origin_causal_conv1d_fn is not None:
        x = origin_causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=bias,
            activation=module.activation,
            seq_idx=None,
        )
    else:
        conv = F.conv1d(x, weight.unsqueeze(1), bias, padding=module.conv_kernel_size - 1, groups=x.shape[1])
        x = F.silu(conv[:, :, :seq_len])
    x = x.transpose(1, 2).reshape(batch_size, seq_len, local_heads, channel_dim)
    return x


class Qwen35LinearAttentionSPPatch(Patch):

    @staticmethod
    def _select_rule(module: torch.nn.Module, origin_rule, full_seq_len: int):
        recurrent_rule = getattr(module, 'recurrent_gated_delta_rule', None)
        module_impl = importlib.import_module(module.__class__.__module__)
        torch_recurrent_rule = getattr(module_impl, 'torch_recurrent_gated_delta_rule', None)
        effective_chunk_size = 64
        if recurrent_rule is not None and full_seq_len % effective_chunk_size != 0:
            rule = torch_recurrent_rule or recurrent_rule

            def recurrent_rule_wrapper(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                chunk_size: Optional[int] = None,
                initial_state: Optional[torch.Tensor] = None,
                output_final_state: bool = False,
                use_qk_l2norm_in_kernel: bool = False,
            ):
                del chunk_size
                return rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )

            return recurrent_rule_wrapper, None
        return origin_rule, None

    @staticmethod
    def _run_head_parallel(
        sequence_parallel,
        mod: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sp_world_size = int(sequence_parallel.sp_world_size or 1)
        if sp_world_size > 1 and mod.num_v_heads % sp_world_size != 0:
            raise RuntimeError(
                'SequenceParallel: Qwen3.5 head-parallel linear attention requires '
                f'sp_world_size ({sp_world_size}) to divide linear_num_value_heads ({mod.num_v_heads}).'
            )
        module_impl = importlib.import_module(mod.__class__.__module__)
        apply_mask_to_padding_states = getattr(module_impl, 'apply_mask_to_padding_states')
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, local_seq_len, _ = hidden_states.shape
        if local_seq_len <= 0:
            raise RuntimeError('SequenceParallel: Qwen3.5 head-parallel linear attention requires non-empty local sequences.')

        mixed_qkv = mod.in_proj_qkv(hidden_states)
        z = mod.in_proj_z(hidden_states).reshape(batch_size, local_seq_len, mod.num_v_heads, mod.head_v_dim)
        b = mod.in_proj_b(hidden_states).reshape(batch_size, local_seq_len, mod.num_v_heads, 1)
        a = mod.in_proj_a(hidden_states).reshape(batch_size, local_seq_len, mod.num_v_heads, 1)
        query_proj, key_proj, value_proj = torch.split(mixed_qkv, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        query_proj = query_proj.reshape(batch_size, local_seq_len, mod.num_k_heads, mod.head_k_dim)
        key_proj = key_proj.reshape(batch_size, local_seq_len, mod.num_k_heads, mod.head_k_dim)
        value_proj = value_proj.reshape(batch_size, local_seq_len, mod.num_v_heads, mod.head_v_dim)

        qk_expand = mod.num_v_heads // mod.num_k_heads
        if qk_expand * mod.num_k_heads != mod.num_v_heads:
            raise RuntimeError(
                'SequenceParallel: Qwen3.5 head-parallel linear attention requires '
                'linear_num_value_heads to be an integer multiple of linear_num_key_heads.'
            )
        query_proj = query_proj.repeat_interleave(qk_expand, dim=2)
        key_proj = key_proj.repeat_interleave(qk_expand, dim=2)
        mixed_qkv_expanded = torch.cat([query_proj, key_proj, value_proj], dim=-1)
        mixed_qkv_hp = seq_to_head_shard(mixed_qkv_expanded, sequence_parallel)
        z_hp = seq_to_head_shard(z, sequence_parallel)
        b_hp = seq_to_head_shard(b, sequence_parallel)
        a_hp = seq_to_head_shard(a, sequence_parallel)

        local_v_heads = mod.num_v_heads // sp_world_size
        sp_rank = _sp_rank(sequence_parallel)
        start_v = sp_rank * local_v_heads
        end_v = start_v + local_v_heads
        local_value_head_indices = torch.arange(start_v, end_v, device=hidden_states.device, dtype=torch.long)
        local_qk_head_indices = torch.div(local_value_head_indices, qk_expand, rounding_mode='floor')

        conv_weight = mod.conv1d.weight.squeeze(1)
        q_weight = conv_weight[:mod.key_dim]
        k_weight = conv_weight[mod.key_dim:2 * mod.key_dim]
        v_weight = conv_weight[2 * mod.key_dim:]
        local_q_weight = _slice_conv_channel_params(
            q_weight, num_heads=mod.num_k_heads, head_dim=mod.head_k_dim, head_indices=local_qk_head_indices)
        local_k_weight = _slice_conv_channel_params(
            k_weight, num_heads=mod.num_k_heads, head_dim=mod.head_k_dim, head_indices=local_qk_head_indices)
        local_v_weight = _slice_conv_channel_params(
            v_weight, num_heads=mod.num_v_heads, head_dim=mod.head_v_dim, head_indices=local_value_head_indices)
        local_conv_weight = torch.cat([local_q_weight, local_k_weight, local_v_weight], dim=0)

        conv_bias = mod.conv1d.bias
        local_conv_bias = None
        if conv_bias is not None:
            q_bias = conv_bias[:mod.key_dim]
            k_bias = conv_bias[mod.key_dim:2 * mod.key_dim]
            v_bias = conv_bias[2 * mod.key_dim:]
            local_q_bias = _slice_conv_channel_params(
                q_bias, num_heads=mod.num_k_heads, head_dim=mod.head_k_dim, head_indices=local_qk_head_indices)
            local_k_bias = _slice_conv_channel_params(
                k_bias, num_heads=mod.num_k_heads, head_dim=mod.head_k_dim, head_indices=local_qk_head_indices)
            local_v_bias = _slice_conv_channel_params(
                v_bias, num_heads=mod.num_v_heads, head_dim=mod.head_v_dim, head_indices=local_value_head_indices)
            local_conv_bias = torch.cat([local_q_bias, local_k_bias, local_v_bias], dim=0)

        mixed_qkv_hp = _run_depthwise_causal_conv(
            mixed_qkv_hp,
            weight=local_conv_weight,
            bias=local_conv_bias,
            module=mod,
            origin_causal_conv1d_fn=mod.causal_conv1d_fn,
        )
        query_hp, key_hp, value_hp = torch.split(mixed_qkv_hp, [mod.head_k_dim, mod.head_k_dim, mod.head_v_dim], dim=-1)
        local_dt_bias = mod.dt_bias[start_v:end_v]
        local_a_log = mod.A_log[start_v:end_v]
        beta = b_hp.squeeze(-1).sigmoid()
        g = -local_a_log.float().exp() * F.softplus(a_hp.squeeze(-1).float() + local_dt_bias)
        rule, chunk_size = Qwen35LinearAttentionSPPatch._select_rule(mod, mod.chunk_gated_delta_rule, query_hp.shape[1])
        chunk_kwargs = {
            'g': g,
            'beta': beta,
            'initial_state': None,
            'output_final_state': False,
            'use_qk_l2norm_in_kernel': True,
        }
        if chunk_size is not None:
            chunk_kwargs['chunk_size'] = chunk_size
        core_attn_out, _ = rule(query_hp, key_hp, value_hp, **chunk_kwargs)
        core_attn_out = core_attn_out.reshape(-1, mod.head_v_dim)
        z_hp = z_hp.reshape(-1, mod.head_v_dim)
        core_attn_out = mod.norm(core_attn_out, z_hp)
        core_attn_out = core_attn_out.reshape(batch_size, -1, local_v_heads, mod.head_v_dim)
        core_attn_out = head_to_seq_shard(core_attn_out, sequence_parallel)
        core_attn_out = core_attn_out.reshape(batch_size, local_seq_len, mod.value_dim)
        return mod.out_proj(core_attn_out)

    def __call__(self, module, *args, **kwargs):
        del module, args
        sequence_parallel = kwargs.get('sequence_parallel', None)
        if sequence_parallel is None:
            return
        if int(getattr(sequence_parallel, 'rp_world_size', 1) or 1) > 1:
            raise NotImplementedError(
                'Qwen35LinearAttentionSPPatch does not support rp_world_size > 1 (derived ring attention).')
        if os.environ.get('QWEN35_SP_LINEAR_HEAD_PARALLEL', '0') != '1':
            return

        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        except Exception:
            return

        if getattr(Qwen3_5GatedDeltaNet, '_twinkle_sp_linear_patched', False):
            return

        origin_forward = Qwen3_5GatedDeltaNet.forward

        def sp_linear_forward(
            mod,
            hidden_states: torch.Tensor,
            cache_params=None,
            cache_position=None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            if sequence_parallel.world_size <= 1:
                return origin_forward(
                    mod,
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
            if cache_params is not None or cache_position is not None:
                raise NotImplementedError(
                    'Qwen35LinearAttentionSPPatch does not support cached/incremental path when SP is enabled.')
            return Qwen35LinearAttentionSPPatch._run_head_parallel(sequence_parallel, mod, hidden_states, attention_mask)

        Qwen3_5GatedDeltaNet.forward = sp_linear_forward
        Qwen3_5GatedDeltaNet._twinkle_sp_linear_patched = True
