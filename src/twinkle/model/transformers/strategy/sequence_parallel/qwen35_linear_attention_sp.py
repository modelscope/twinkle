import importlib
import os
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from twinkle.patch import Patch
from twinkle.model.transformers.strategy.sequence_parallel.utils import get_cu_seqlens_from_position_ids
from twinkle.model.transformers.strategy.sequence_parallel.utils import head_to_seq_shard, seq_to_head_shard


def _sp_is_enabled(sequence_parallel_context) -> bool:
    return bool(sequence_parallel_context is not None and getattr(sequence_parallel_context, 'world_size', 1) > 1)


def _get_sp_rank(sequence_parallel_context) -> int:
    if not _sp_is_enabled(sequence_parallel_context):
        return 0
    if getattr(sequence_parallel_context, '_sp_group', None) is None:
        return 0
    return dist.get_rank(group=sequence_parallel_context._sp_group)


def _maybe_slice_tensor_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    raise TypeError(f'Unexpected tensor output type: {type(output)}')


def _is_packed_position_ids(position_ids: Optional[torch.Tensor]) -> bool:
    if position_ids is None or not torch.is_tensor(position_ids):
        return False
    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.dim() != 2:
        return False
    for row in position_ids:
        valid = row[row >= 0]
        if valid.numel() == 0:
            continue
        if int((valid == 0).sum().item()) > 1:
            return True
    return False


def _resolve_local_padding_mask(
    attention_mask: Optional[torch.Tensor],
    local_seq_len: int,
    sequence_parallel_context,
) -> Optional[torch.Tensor]:
    if attention_mask is None or not torch.is_tensor(attention_mask):
        return attention_mask
    if attention_mask.dim() != 2:
        return attention_mask
    if attention_mask.shape[-1] == local_seq_len:
        return attention_mask
    if not _sp_is_enabled(sequence_parallel_context):
        return attention_mask
    real_position_ids = getattr(sequence_parallel_context, 'real_position_ids', None)
    if real_position_ids is None:
        return attention_mask
    return sequence_parallel_context.split(attention_mask, dim=1, position_ids=real_position_ids)


def _build_varlen_metadata(
    position_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    full_seq_len: int,
    cu_seq_lens_q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    packed_valid_mask = None
    if attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
        packed_valid_mask = attention_mask.to(dtype=torch.bool)
    elif position_ids is not None and torch.is_tensor(position_ids):
        pos = position_ids[0] if position_ids.dim() == 3 else position_ids
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        packed_valid_mask = (pos != -1)

    if packed_valid_mask is None:
        raise ValueError('Packed linear attention requires a 2D attention_mask or valid position_ids.')
    if packed_valid_mask.shape[-1] != full_seq_len:
        raise ValueError(f'Packed mask length mismatch: expected {full_seq_len}, got {packed_valid_mask.shape[-1]}.')

    if cu_seq_lens_q is not None:
        return packed_valid_mask, cu_seq_lens_q.to(dtype=torch.int32, device=packed_valid_mask.device)

    if position_ids is None:
        raise ValueError('Packed linear attention requires position_ids to derive cu_seqlens when not provided.')
    pos = position_ids[0] if position_ids.dim() == 3 else position_ids
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    if pos.shape[0] != 1:
        raise ValueError('Packed linear attention without explicit cu_seq_lens_q only supports batch_size == 1.')
    safe_pos = pos.clone()
    safe_pos[safe_pos < 0] = 0
    return packed_valid_mask, get_cu_seqlens_from_position_ids(safe_pos).to(dtype=torch.int32, device=safe_pos.device)


def _pack_varlen_tensor(tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    tail_shape = tensor.shape[2:]
    packed = tensor[valid_mask]
    return packed.reshape(1, packed.shape[0], *tail_shape).contiguous()


def _unpack_varlen_tensor(
    tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    full_seq_len: int,
) -> torch.Tensor:
    tail_shape = tensor.shape[2:]
    output = tensor.new_zeros((batch_size, full_seq_len, *tail_shape))
    output[valid_mask] = tensor.reshape(-1, *tail_shape)
    return output


def _ensure_linear_attention_fast_path(mod: torch.nn.Module):
    module_impl = importlib.import_module(mod.__class__.__module__)
    mod.causal_conv1d_fn = getattr(mod, 'causal_conv1d_fn', None) or getattr(module_impl, 'causal_conv1d_fn', None)
    mod.causal_conv1d_update = getattr(mod, 'causal_conv1d_update', None) or getattr(module_impl,
                                                                                       'causal_conv1d_update', None)
    mod.chunk_gated_delta_rule = getattr(mod, 'chunk_gated_delta_rule', None) or getattr(module_impl,
                                                                                           'chunk_gated_delta_rule',
                                                                                           None) or getattr(
                                                                                               module_impl,
                                                                                               'torch_chunk_gated_delta_rule',
                                                                                               None)
    mod.recurrent_gated_delta_rule = getattr(mod, 'recurrent_gated_delta_rule', None) or getattr(
        module_impl, 'fused_recurrent_gated_delta_rule', None) or getattr(module_impl, 'torch_recurrent_gated_delta_rule',
                                                                          None)
    if mod.causal_conv1d_update is None:
        mod.causal_conv1d_update = getattr(module_impl, 'torch_causal_conv1d_update', None)
    if mod.chunk_gated_delta_rule is None or mod.recurrent_gated_delta_rule is None:
        raise ImportError('Qwen35LinearAttentionSPPatch requires gated delta rule implementations to be available.')
    return module_impl


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


def _get_local_conv_params(
    mod: torch.nn.Module,
    *,
    sp_rank: int,
    local_num_k_heads: int,
    local_num_v_heads: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    local_qk_head_indices = torch.arange(
        sp_rank * local_num_k_heads,
        (sp_rank + 1) * local_num_k_heads,
        device=mod.conv1d.weight.device,
        dtype=torch.long,
    )
    local_value_head_indices = torch.arange(
        sp_rank * local_num_v_heads,
        (sp_rank + 1) * local_num_v_heads,
        device=mod.conv1d.weight.device,
        dtype=torch.long,
    )
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

    conv_bias = getattr(mod.conv1d, 'bias', None)
    if conv_bias is None:
        return local_conv_weight, None
    q_bias = conv_bias[:mod.key_dim]
    k_bias = conv_bias[mod.key_dim:2 * mod.key_dim]
    v_bias = conv_bias[2 * mod.key_dim:]
    local_q_bias = _slice_conv_channel_params(
        q_bias, num_heads=mod.num_k_heads, head_dim=mod.head_k_dim, head_indices=local_qk_head_indices)
    local_k_bias = _slice_conv_channel_params(
        k_bias, num_heads=mod.num_k_heads, head_dim=mod.head_k_dim, head_indices=local_qk_head_indices)
    local_v_bias = _slice_conv_channel_params(
        v_bias, num_heads=mod.num_v_heads, head_dim=mod.head_v_dim, head_indices=local_value_head_indices)
    return local_conv_weight, torch.cat([local_q_bias, local_k_bias, local_v_bias], dim=0)


def _apply_varlen_conv(
    mod: torch.nn.Module,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    cu_seq_lens_q: Optional[torch.Tensor],
) -> torch.Tensor:
    if mod.causal_conv1d_fn is None:
        raise ImportError(
            'Qwen35LinearAttentionSPPatch requires fla.modules.convolution.causal_conv1d for prefill/train.')
    x = mixed_qkv.transpose(1, 2).contiguous()
    try:
        output = mod.causal_conv1d_fn(
            x=x,
            weight=conv_weight,
            bias=conv_bias,
            activation=mod.activation,
            seq_idx=None,
            backend='triton',
            cu_seqlens=cu_seq_lens_q,
        )
    except TypeError:
        output = mod.causal_conv1d_fn(
            x=x,
            weight=conv_weight,
            bias=conv_bias,
            activation=mod.activation,
            seq_idx=None,
        )
    output = _maybe_slice_tensor_output(output)
    if output.dim() == 2:
        output = output.unsqueeze(0)
    if output.dim() == 3 and output.shape[1] == conv_weight.shape[0]:
        output = output.transpose(1, 2).contiguous()
    return output


def _apply_decode_conv(
    mod: torch.nn.Module,
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if mod.causal_conv1d_update is None:
        raise ImportError(
            'Qwen35LinearAttentionSPPatch decode requires a causal_conv1d_update implementation from flash-linear-attention '
            'or causal-conv1d.')
    mixed_qkv_t = mixed_qkv.transpose(1, 2).contiguous()
    output = mod.causal_conv1d_update(
        mixed_qkv_t,
        conv_state,
        conv_weight,
        conv_bias,
        mod.activation,
    )
    output = _maybe_slice_tensor_output(output)
    if output.dim() == 2:
        output = output.unsqueeze(1)
    elif output.dim() == 3 and output.shape[1] == conv_weight.shape[0]:
        output = output.transpose(1, 2).contiguous()
    return output


class Qwen35LinearAttentionSPPatch(Patch):

    @staticmethod
    def _run_forward(
        mod: torch.nn.Module,
        hidden_states: torch.Tensor,
        *,
        cache_params=None,
        cache_position=None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seq_lens_q: Optional[torch.Tensor] = None,
        sequence_parallel_context=None,
    ) -> torch.Tensor:
        module_impl = _ensure_linear_attention_fast_path(mod)
        apply_mask_to_padding_states = getattr(module_impl, 'apply_mask_to_padding_states')

        local_attention_mask = _resolve_local_padding_mask(attention_mask, hidden_states.shape[1], sequence_parallel_context)
        hidden_states = apply_mask_to_padding_states(hidden_states, local_attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        has_previous_state = bool(cache_params is not None and getattr(cache_params, 'has_previous_state', False))
        use_precomputed_states = has_previous_state and seq_len == 1 and cache_position is not None

        if cache_params is not None:
            conv_state = cache_params.conv_states[mod.layer_idx]
            recurrent_state = cache_params.recurrent_states[mod.layer_idx]
        else:
            conv_state = None
            recurrent_state = None

        mixed_qkv = mod.in_proj_qkv(hidden_states)
        z = mod.in_proj_z(hidden_states).reshape(batch_size, seq_len, mod.num_v_heads, mod.head_v_dim)
        b = mod.in_proj_b(hidden_states)
        a = mod.in_proj_a(hidden_states)

        sp_enabled = _sp_is_enabled(sequence_parallel_context)
        if sp_enabled:
            sp_world_size = int(sequence_parallel_context.sp_world_size)
            if mod.num_k_heads % sp_world_size != 0 or mod.num_v_heads % sp_world_size != 0:
                raise RuntimeError(
                    'Qwen35LinearAttentionSPPatch requires sp_world_size to divide both '
                    f'linear_num_key_heads ({mod.num_k_heads}) and linear_num_value_heads ({mod.num_v_heads}).')
            local_num_k_heads = mod.num_k_heads // sp_world_size
            local_num_v_heads = mod.num_v_heads // sp_world_size
            q_proj, k_proj, v_proj = torch.split(mixed_qkv, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
            q_proj = q_proj.reshape(batch_size, seq_len, mod.num_k_heads, mod.head_k_dim)
            k_proj = k_proj.reshape(batch_size, seq_len, mod.num_k_heads, mod.head_k_dim)
            v_proj = v_proj.reshape(batch_size, seq_len, mod.num_v_heads, mod.head_v_dim)
            q_proj = seq_to_head_shard(q_proj, sequence_parallel_context)
            k_proj = seq_to_head_shard(k_proj, sequence_parallel_context)
            v_proj = seq_to_head_shard(v_proj, sequence_parallel_context)
            z = seq_to_head_shard(z, sequence_parallel_context)
            b = seq_to_head_shard(b.reshape(batch_size, seq_len, mod.num_v_heads, 1), sequence_parallel_context).squeeze(-1)
            a = seq_to_head_shard(a.reshape(batch_size, seq_len, mod.num_v_heads, 1), sequence_parallel_context).squeeze(-1)
            seq_after_shard = q_proj.shape[1]
            mixed_qkv = torch.cat(
                (
                    q_proj.reshape(batch_size, seq_after_shard, local_num_k_heads * mod.head_k_dim),
                    k_proj.reshape(batch_size, seq_after_shard, local_num_k_heads * mod.head_k_dim),
                    v_proj.reshape(batch_size, seq_after_shard, local_num_v_heads * mod.head_v_dim),
                ),
                dim=-1,
            )
            sp_rank = _get_sp_rank(sequence_parallel_context)
            conv_weight, conv_bias = _get_local_conv_params(
                mod, sp_rank=sp_rank, local_num_k_heads=local_num_k_heads, local_num_v_heads=local_num_v_heads)
        else:
            local_num_k_heads = mod.num_k_heads
            local_num_v_heads = mod.num_v_heads
            sp_rank = 0
            b = b.reshape(batch_size, seq_len, mod.num_v_heads)
            a = a.reshape(batch_size, seq_len, mod.num_v_heads)
            conv_weight = mod.conv1d.weight.squeeze(1)
            conv_bias = getattr(mod.conv1d, 'bias', None)

        packed_valid_mask = None
        packed_cu_seqlens = cu_seq_lens_q
        packed_seq_len = mixed_qkv.shape[1]
        full_position_ids = getattr(sequence_parallel_context, 'real_position_ids', None) if sequence_parallel_context is not None else None
        use_varlen_pack = not use_precomputed_states and (
            cu_seq_lens_q is not None or _is_packed_position_ids(full_position_ids))
        if use_varlen_pack:
            packed_valid_mask, packed_cu_seqlens = _build_varlen_metadata(
                position_ids=full_position_ids,
                attention_mask=attention_mask,
                full_seq_len=packed_seq_len,
                cu_seq_lens_q=cu_seq_lens_q,
            )
            mixed_qkv = _pack_varlen_tensor(mixed_qkv, packed_valid_mask)
            b = _pack_varlen_tensor(b, packed_valid_mask)
            a = _pack_varlen_tensor(a, packed_valid_mask)

        if use_precomputed_states:
            if conv_state is None:
                raise RuntimeError('Qwen35LinearAttentionSPPatch decode requires initialized convolution state.')
            mixed_qkv = _apply_decode_conv(mod, mixed_qkv, conv_state, conv_weight, conv_bias)
        else:
            if cache_params is not None:
                cache_params.conv_states[mod.layer_idx] = F.pad(
                    mixed_qkv.transpose(1, 2).contiguous(), (mod.conv_kernel_size - mixed_qkv.shape[1], 0))
            mixed_qkv = _apply_varlen_conv(mod, mixed_qkv, conv_weight, conv_bias, packed_cu_seqlens)

        local_key_dim = local_num_k_heads * mod.head_k_dim
        local_value_dim = local_num_v_heads * mod.head_v_dim
        query, key, value = torch.split(mixed_qkv, [local_key_dim, local_key_dim, local_value_dim], dim=-1)
        qkv_batch_size = 1 if use_varlen_pack else batch_size
        query = query.reshape(qkv_batch_size, query.shape[1], local_num_k_heads, mod.head_k_dim)
        key = key.reshape(qkv_batch_size, key.shape[1], local_num_k_heads, mod.head_k_dim)
        value = value.reshape(qkv_batch_size, value.shape[1], local_num_v_heads, mod.head_v_dim)

        beta = b.sigmoid()
        head_slice = slice(sp_rank * local_num_v_heads, (sp_rank + 1) * local_num_v_heads) if sp_enabled else slice(None)
        g = -mod.A_log[head_slice].float().exp() * F.softplus(a.float() + mod.dt_bias[head_slice])

        if local_num_v_heads // local_num_k_heads > 1:
            repeat = local_num_v_heads // local_num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        if use_precomputed_states:
            core_attn_out, last_recurrent_state = mod.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            chunk_kwargs = {
                'g': g,
                'beta': beta,
                'initial_state': None,
                'output_final_state': cache_params is not None,
                'use_qk_l2norm_in_kernel': True,
            }
            if packed_cu_seqlens is not None:
                chunk_kwargs['cu_seqlens'] = packed_cu_seqlens
            try:
                core_attn_out, last_recurrent_state = mod.chunk_gated_delta_rule(query, key, value, **chunk_kwargs)
            except TypeError:
                chunk_kwargs.pop('cu_seqlens', None)
                core_attn_out, last_recurrent_state = mod.chunk_gated_delta_rule(query, key, value, **chunk_kwargs)

        if cache_params is not None:
            cache_params.recurrent_states[mod.layer_idx] = last_recurrent_state

        if use_varlen_pack:
            core_attn_out = _unpack_varlen_tensor(core_attn_out, packed_valid_mask, batch_size, packed_seq_len)
        core_attn_out = head_to_seq_shard(core_attn_out, sequence_parallel_context)
        core_attn_out = mod.norm(core_attn_out.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim))
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, local_value_dim if not sp_enabled else mod.value_dim)
        return mod.out_proj(core_attn_out)

    def __call__(self, module, *args, **kwargs):
        del module, args
        sequence_parallel = kwargs.get('sequence_parallel', None)
        if sequence_parallel is None:
            return
        if int(getattr(sequence_parallel, 'rp_world_size', 1) or 1) > 1:
            raise NotImplementedError(
                'Qwen35LinearAttentionSPPatch does not support rp_world_size > 1 (derived ring attention).')
        if os.environ.get('QWEN35_SP_LINEAR_HEAD_PARALLEL', '1') != '1':
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
            **extra_kwargs,
        ):
            sequence_parallel_context = extra_kwargs.pop('sequence_parallel_context', sequence_parallel)
            cu_seq_lens_q = extra_kwargs.pop('cu_seq_lens_q', None)
            if not _sp_is_enabled(sequence_parallel_context):
                return origin_forward(
                    mod,
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
            return Qwen35LinearAttentionSPPatch._run_forward(
                mod,
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                sequence_parallel_context=sequence_parallel_context,
            )

        Qwen3_5GatedDeltaNet.forward = sp_linear_forward
        Qwen3_5GatedDeltaNet._twinkle_sp_linear_patched = True
