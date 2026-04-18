import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.utils.import_utils import is_flash_linear_attention_available
from typing import Any, Optional, Tuple

from twinkle.model.transformers.strategy.sequence_parallel.utils import head_to_seq_shard, seq_to_head_shard
from twinkle.patch import Patch

if is_flash_linear_attention_available():
    from fla.modules.convolution import causal_conv1d as _FLA_CAUSAL_CONV1D_FN
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _FLA_CHUNK_GATED_DELTA_RULE
else:
    _FLA_CAUSAL_CONV1D_FN = None
    _FLA_CHUNK_GATED_DELTA_RULE = None

_SP_LINEAR_KERNEL_IMPORT_ERROR = ('Qwen3.5 linear attention sequence parallel requires flash-linear-attention. '
                                  'Install: https://github.com/fla-org/flash-linear-attention#installation')


def _sp_is_enabled(sequence_parallel_context) -> bool:
    return bool(sequence_parallel_context is not None and getattr(sequence_parallel_context, 'world_size', 1) > 1)


def _get_sp_rank(sequence_parallel_context) -> int:
    if not _sp_is_enabled(sequence_parallel_context):
        return 0
    if getattr(sequence_parallel_context, '_sp_group', None) is None:
        return 0
    return dist.get_rank(group=sequence_parallel_context._sp_group)


def _get_local_padding_mask(
    attention_mask: torch.Tensor,
    local_seq_len: int,
    sequence_parallel_context,
) -> torch.Tensor:
    if attention_mask.shape[-1] == local_seq_len or not _sp_is_enabled(sequence_parallel_context):
        return attention_mask
    return sequence_parallel_context.split(
        attention_mask,
        dim=1,
        position_ids=sequence_parallel_context.real_position_ids,
    )


def _ensure_linear_attention_kernels(mod: torch.nn.Module):
    mod.causal_conv1d_fn = _FLA_CAUSAL_CONV1D_FN
    mod.chunk_gated_delta_rule = _FLA_CHUNK_GATED_DELTA_RULE
    if mod.chunk_gated_delta_rule is None or mod.causal_conv1d_fn is None:
        raise ImportError(_SP_LINEAR_KERNEL_IMPORT_ERROR)


def _get_local_conv_weights(
    mod: torch.nn.Module,
    *,
    sp_rank: int,
    local_num_k_heads: int,
    local_num_v_heads: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    local_key_dim = local_num_k_heads * mod.head_k_dim
    local_value_dim = local_num_v_heads * mod.head_v_dim
    conv_weight = mod.conv1d.weight.squeeze(1)
    if conv_weight.shape[0] != (2 * mod.key_dim + mod.value_dim):
        raise ValueError(
            f'Unexpected conv weight dim {conv_weight.shape[0]}, expected {2 * mod.key_dim + mod.value_dim}.')
    key_offset = sp_rank * local_key_dim
    value_offset = sp_rank * local_value_dim
    local_q_weight = conv_weight[key_offset:key_offset + local_key_dim]
    local_k_weight = conv_weight[mod.key_dim + key_offset:mod.key_dim + key_offset + local_key_dim]
    local_v_weight = conv_weight[2 * mod.key_dim + value_offset:2 * mod.key_dim + value_offset + local_value_dim]
    local_conv_weight = torch.cat([local_q_weight, local_k_weight, local_v_weight], dim=0)

    conv_bias = getattr(mod.conv1d, 'bias', None)
    if conv_bias is None:
        return local_conv_weight, None
    local_q_bias = conv_bias[key_offset:key_offset + local_key_dim]
    local_k_bias = conv_bias[mod.key_dim + key_offset:mod.key_dim + key_offset + local_key_dim]
    local_v_bias = conv_bias[2 * mod.key_dim + value_offset:2 * mod.key_dim + value_offset + local_value_dim]
    return local_conv_weight, torch.cat([local_q_bias, local_k_bias, local_v_bias], dim=0)


class Qwen3_5GatedDeltaNetUlyssesPatch(Patch):

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
        _ensure_linear_attention_kernels(mod)
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

        local_attention_mask = attention_mask
        if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
            local_attention_mask = _get_local_padding_mask(
                attention_mask,
                hidden_states.shape[1],
                sequence_parallel_context,
            )
        hidden_states = apply_mask_to_padding_states(hidden_states, local_attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        has_previous_state = bool(cache_params is not None and getattr(cache_params, 'has_previous_state', False))
        use_precomputed_states = has_previous_state and seq_len == 1 and cache_position is not None
        if use_precomputed_states:
            raise NotImplementedError(
                'Qwen3.5 linear attention sequence parallel only supports training/prefill paths; decode with '
                'cached states is not supported.')

        mixed_qkv = mod.in_proj_qkv(hidden_states)
        z = mod.in_proj_z(hidden_states).reshape(batch_size, seq_len, mod.num_v_heads, mod.head_v_dim)
        b = mod.in_proj_b(hidden_states)
        a = mod.in_proj_a(hidden_states)

        sp_enabled = _sp_is_enabled(sequence_parallel_context)
        if sp_enabled:
            sp_world_size = int(sequence_parallel_context.sp_world_size)
            if mod.num_k_heads % sp_world_size != 0 or mod.num_v_heads % sp_world_size != 0:
                raise RuntimeError(
                    'Qwen3.5 linear attention sequence parallel requires sp_world_size to divide both '
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
            b = seq_to_head_shard(b.reshape(batch_size, seq_len, mod.num_v_heads, 1),
                                  sequence_parallel_context).squeeze(-1)
            a = seq_to_head_shard(a.reshape(batch_size, seq_len, mod.num_v_heads, 1),
                                  sequence_parallel_context).squeeze(-1)
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
            conv_weight, conv_bias = _get_local_conv_weights(
                mod, sp_rank=sp_rank, local_num_k_heads=local_num_k_heads, local_num_v_heads=local_num_v_heads)
        else:
            local_num_k_heads = mod.num_k_heads
            local_num_v_heads = mod.num_v_heads
            sp_rank = 0
            b = b.reshape(batch_size, seq_len, mod.num_v_heads)
            a = a.reshape(batch_size, seq_len, mod.num_v_heads)
            conv_weight = mod.conv1d.weight.squeeze(1)
            conv_bias = getattr(mod.conv1d, 'bias', None)

        packed_cu_seqlens = None
        if cu_seq_lens_q is not None:
            packed_cu_seqlens = cu_seq_lens_q.to(dtype=torch.int32, device=mixed_qkv.device)
        elif sequence_parallel_context is not None:
            packed_cu_seqlens = getattr(sequence_parallel_context, 'extra_kwargs', {}).get('cu_seq_lens_q')
            if packed_cu_seqlens is not None:
                packed_cu_seqlens = packed_cu_seqlens.to(dtype=torch.int32, device=mixed_qkv.device)
        if bool(getattr(sequence_parallel_context, 'extra_kwargs', {}).get('is_packed',
                                                                           False)) and packed_cu_seqlens is None:
            raise ValueError(
                'Packed Qwen3.5 linear attention sequence parallel requires cu_seq_lens_q to be populated by '
                'sequence parallel input preparation.')
        if cache_params is not None:
            cache_params.conv_states[mod.layer_idx] = F.pad(
                mixed_qkv.transpose(1, 2).contiguous(), (mod.conv_kernel_size - mixed_qkv.shape[1], 0))
        mixed_qkv, _ = mod.causal_conv1d_fn(
            x=mixed_qkv,
            weight=conv_weight,
            bias=conv_bias,
            activation=mod.activation,
            seq_idx=None,
            backend='triton',
            cu_seqlens=packed_cu_seqlens,
        )
        if mixed_qkv.dim() == 2:
            mixed_qkv = mixed_qkv.unsqueeze(0)
        if mixed_qkv.dim() != 3:
            raise ValueError(f'Unexpected conv output dims: {tuple(mixed_qkv.shape)}')

        local_key_dim = local_num_k_heads * mod.head_k_dim
        local_value_dim = local_num_v_heads * mod.head_v_dim
        query, key, value = torch.split(mixed_qkv, [local_key_dim, local_key_dim, local_value_dim], dim=-1)
        query = query.reshape(batch_size, query.shape[1], local_num_k_heads, mod.head_k_dim)
        key = key.reshape(batch_size, key.shape[1], local_num_k_heads, mod.head_k_dim)
        value = value.reshape(batch_size, value.shape[1], local_num_v_heads, mod.head_v_dim)

        beta = b.sigmoid()
        head_slice = slice(sp_rank * local_num_v_heads,
                           (sp_rank + 1) * local_num_v_heads) if sp_enabled else slice(None)
        g = -mod.A_log[head_slice].float().exp() * F.softplus(a.float() + mod.dt_bias[head_slice])

        if local_num_v_heads // local_num_k_heads > 1:
            repeat = local_num_v_heads // local_num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        chunk_kwargs = {
            'g': g,
            'beta': beta,
            'initial_state': None,
            'output_final_state': cache_params is not None,
            'use_qk_l2norm_in_kernel': True,
        }
        if packed_cu_seqlens is not None:
            chunk_kwargs['cu_seqlens'] = packed_cu_seqlens
        core_attn_out, last_recurrent_state = mod.chunk_gated_delta_rule(query, key, value, **chunk_kwargs)

        if cache_params is not None:
            cache_params.recurrent_states[mod.layer_idx] = last_recurrent_state

        if sp_enabled:
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
            raise NotImplementedError('Qwen3.5 linear attention sequence parallel does not support rp_world_size > 1 '
                                      '(derived ring attention).')

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
            if cu_seq_lens_q is None and sequence_parallel_context is not None:
                cu_seq_lens_q = getattr(sequence_parallel_context, 'extra_kwargs', {}).get('cu_seq_lens_q')
            if not _sp_is_enabled(sequence_parallel_context):
                return origin_forward(
                    mod,
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )
            return Qwen3_5GatedDeltaNetUlyssesPatch._run_forward(
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
