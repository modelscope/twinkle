import functools
import torch
import torch_npu


class GmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_list: torch.Tensor, weight_ekn: torch.Tensor):
        assert x.dim() == 2, f'x must be [M, K], got {tuple(x.shape)}'
        assert group_list.dim() == 1, f'group_list must be [E], got {tuple(group_list.shape)}'
        assert weight_ekn.dim() == 3, f'weight_ekn must be [E, K, N], got {tuple(weight_ekn.shape)}'
        assert group_list.numel() == weight_ekn.size(0), (
            f'group_list len {group_list.numel()} != num_experts {weight_ekn.size(0)}')
        assert x.size(1) == weight_ekn.size(1), (
            f'input dim mismatch: x.shape={tuple(x.shape)}, weight_ekn.shape={tuple(weight_ekn.shape)}')

        group_list = group_list.to(torch.int64)

        ctx.save_for_backward(x, group_list, weight_ekn)

        outputs = torch_npu.npu_grouped_matmul(
            [x],
            [weight_ekn],
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, group_list, weight_ekn = ctx.saved_tensors

        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight_ekn.transpose(-2, -1).contiguous()],
            bias=None,
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )[0]

        grad_weight = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=group_list,
            group_type=2,
            split_item=3,
            group_list_type=1,
        )[0]

        return grad_input, None, grad_weight.contiguous()


def _grouped_mm_npu(input: torch.Tensor, weight_ekn: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    assert input.dim() == 2, f'input must be [M, K], got {tuple(input.shape)}'
    assert weight_ekn.dim() == 3, f'weight_ekn must be [E, K, N], got {tuple(weight_ekn.shape)}'
    assert offs.dim() == 1, f'offs must be [E], got {tuple(offs.shape)}'
    assert weight_ekn.size(0) == offs.numel(), (
        f'weight_ekn.size(0)={weight_ekn.size(0)} != offs.numel()={offs.numel()}')

    counts = torch.empty_like(offs)
    counts[0] = offs[0]
    if offs.numel() > 1:
        counts[1:] = offs[1:] - offs[:-1]
    counts = counts.to(torch.int64)

    return GmmFunction.apply(input, counts, weight_ekn)


def apply_hf_moe_grouped_mm_patch():
    import transformers.integrations.moe as hf_moe

    hf_moe._grouped_mm = _grouped_mm_npu
    print('[PATCH] transformers.integrations.moe._grouped_mm -> _grouped_mm_npu')


def _deepseek_v4_rms_norm_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dtype != self.weight.dtype:
        hidden_states = hidden_states.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


def _deepseek_v4_unweighted_rms_norm_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    weight = getattr(self, '_twinkle_npu_rms_weight', None)
    if (
        weight is None
        or weight.device != hidden_states.device
        or weight.dtype != hidden_states.dtype
        or weight.shape[-1] != hidden_states.shape[-1]
    ):
        weight = torch.ones(hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
        self._twinkle_npu_rms_weight = weight
    return torch_npu.npu_rms_norm(hidden_states, weight, epsilon=self.eps)[0]


def apply_deepseek_v4_rms_norm_patch():
    try:
        from transformers.models.deepseek_v4 import modeling_deepseek_v4
    except Exception as exc:
        print(f'[PATCH] skip DeepSeek V4 RMSNorm patch: {exc}')
        return

    if not getattr(modeling_deepseek_v4.DeepseekV4RMSNorm.forward, '_twinkle_npu_patched', False):
        _deepseek_v4_rms_norm_forward_npu._twinkle_npu_patched = True
        _deepseek_v4_rms_norm_forward_npu._twinkle_old_forward = modeling_deepseek_v4.DeepseekV4RMSNorm.forward
        modeling_deepseek_v4.DeepseekV4RMSNorm.forward = _deepseek_v4_rms_norm_forward_npu

    if not getattr(modeling_deepseek_v4.DeepseekV4UnweightedRMSNorm.forward, '_twinkle_npu_patched', False):
        _deepseek_v4_unweighted_rms_norm_forward_npu._twinkle_npu_patched = True
        _deepseek_v4_unweighted_rms_norm_forward_npu._twinkle_old_forward = (
            modeling_deepseek_v4.DeepseekV4UnweightedRMSNorm.forward
        )
        modeling_deepseek_v4.DeepseekV4UnweightedRMSNorm.forward = _deepseek_v4_unweighted_rms_norm_forward_npu

    print('[PATCH] DeepseekV4RMSNorm/DeepseekV4UnweightedRMSNorm.forward -> torch_npu.npu_rms_norm')



def _npu_sparse_attn_shared_kv(query: torch.Tensor, ori_kv: torch.Tensor, cmp_kv: torch.Tensor,
                              cmp_indices: torch.Tensor, sinks: torch.Tensor, scale: float,
                              cmp_ratio: int, sliding_window: int) -> torch.Tensor:
    """Adapter from HF DeepSeek V4 attention tensors to MindSpeed SFA.

    HF uses [B, H, S, D] for q/kv. MindSpeed SparseAttnSharedKV expects BSND query
    and split original/compressed shared-KV tensors.
    """
    from mindspeed.ops.npu_sparse_attn_shared_kv import SparseAttnSharedKV

    query = query.transpose(1, 2).contiguous()  # [B, S, H, D]
    ori_kv = ori_kv.squeeze(1).contiguous()  # [B, S_ori, D]

    if cmp_kv is not None:
        cmp_kv = cmp_kv.squeeze(1).contiguous()  # [B, S_cmp, D]

    if cmp_indices is not None:
        cmp_indices = cmp_indices.to(torch.int32).contiguous().unsqueeze(2)  # [B, S, 1, K]

    batch_size, q_len, num_heads, head_dim = query.shape
    kv_len = ori_kv.shape[1]
    topk = 0 if cmp_indices is None or cmp_ratio != 4 else cmp_indices.shape[-1]

    output = SparseAttnSharedKV.apply(
        query,
        ori_kv.unsqueeze(2).contiguous(),
        None if cmp_kv is None else cmp_kv.unsqueeze(2).contiguous(),
        None,  # cu_seq_lens_q; TND is not supported by this adapter.
        None,  # cu_seq_lens_ori_kv
        None,  # cu_seq_lens_cmp_kv
        None,  # ori_sparse_indices; original KV uses band mask mode.
        cmp_indices if cmp_ratio == 4 else None,
        sinks.float(),
        scale,
        cmp_ratio,
        4,  # ori_mask_mode: band/sliding original KV
        3,  # cmp_mask_mode: sparse compressed KV
        sliding_window - 1,
        0,
        num_heads,
        1,  # DeepSeek V4 has shared KV / single KV head.
        head_dim,
        batch_size,
        q_len,
        kv_len,
        topk,
        'BSND',
        'BSND',
    )
    return output.transpose(1, 2).contiguous()  # [B, H, S, D]


def apply_deepseek_v4_indexer_capture_patch():
    try:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
    except Exception as exc:
        print(f'[PATCH] skip DeepseekV4Indexer capture patch: {exc}')
        return

    if getattr(DeepseekV4Indexer.forward, '_twinkle_npu_patched', False):
        return

    old_forward = DeepseekV4Indexer.forward

    @functools.wraps(old_forward)
    def new_forward(self, *args, **kwargs):
        indices = old_forward(self, *args, **kwargs)
        self._twinkle_last_top_k_indices = indices
        return indices

    new_forward._twinkle_npu_patched = True
    new_forward._twinkle_old_forward = old_forward
    DeepseekV4Indexer.forward = new_forward
    print('[PATCH] DeepseekV4Indexer.forward captures top_k_indices')


def apply_deepseek_v4_sfa_attention_patch():
    try:
        import torch.nn.functional as F
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4Attention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
    except Exception as exc:
        print(f'[PATCH] skip DeepseekV4 SFA attention patch: {exc}')
        return

    if getattr(DeepseekV4Attention.forward, '_twinkle_npu_patched', False):
        return

    def _get_position_cos_sin(self, position_embeddings):
        if isinstance(position_embeddings, dict):
            rope_layer_type = getattr(self, 'rope_layer_type', 'main')
            return position_embeddings[rope_layer_type]
        return position_embeddings

    def _get_compress_ratio(self):
        if self.layer_type == 'compressed_sparse_attention':
            return self.config.compress_rates['compressed_sparse_attention']
        if self.layer_type == 'heavily_compressed_attention':
            return self.config.compress_rates['heavily_compressed_attention']
        return 0

    def new_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = _get_position_cos_sin(self, position_embeddings)

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        ori_kv = kv
        compressed_kv = None
        block_bias = None
        top_k_indices = None

        if self.compressor is not None:
            compressed_kv, block_bias = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            indexer = getattr(self.compressor, 'indexer', None)
            if indexer is not None:
                top_k_indices = getattr(indexer, '_twinkle_last_top_k_indices', None)

        use_npu_sfa = (
            q.device.type == 'npu'
            and self.compressor is not None
            and self.layer_type == 'compressed_sparse_attention'
            and compressed_kv is not None
            and top_k_indices is not None
            and not getattr(self, '_twinkle_npu_sfa_disabled', False)
        )

        if use_npu_sfa:
            try:
                attn_output = _npu_sparse_attn_shared_kv(
                    query=q,
                    ori_kv=ori_kv,
                    cmp_kv=compressed_kv,
                    cmp_indices=top_k_indices,
                    sinks=self.sinks,
                    scale=self.scaling,
                    cmp_ratio=_get_compress_ratio(self),
                    sliding_window=self.sliding_window,
                )
                attn_weights = None
            except Exception as exc:
                self._twinkle_npu_sfa_disabled = True
                print(f'[PATCH] DeepSeek V4 NPU SFA fallback to HF attention: {exc}')
                use_npu_sfa = False

        if not use_npu_sfa:
            kv = ori_kv if compressed_kv is None else torch.cat([ori_kv, compressed_kv], dim=2)

            if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
                if block_bias is not None:
                    attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
                else:
                    attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                q,
                kv,
                kv,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,
                **kwargs,
            )

        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights

    new_forward._twinkle_npu_patched = True
    new_forward._twinkle_old_forward = DeepseekV4Attention.forward
    DeepseekV4Attention.forward = new_forward
    print('[PATCH] DeepseekV4Attention.forward -> NPU SFA adapter with HF fallback')


def apply_deepseek_v4_sfa_patch():
    apply_deepseek_v4_indexer_capture_patch()
    apply_deepseek_v4_sfa_attention_patch()


def apply_npu_patch():
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    apply_hf_moe_grouped_mm_patch()
    apply_deepseek_v4_rms_norm_patch()
    apply_deepseek_v4_sfa_patch()
