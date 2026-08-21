# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4 NPU attention (SAS) / Lightning Indexer (LI) forwards.

Used as class-attribute replacements on the corresponding HF classes via
the ``DEFAULT_KERNEL_CONFIG`` mapping (see ``config`` and ``registry``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from twinkle import get_logger

logger = get_logger()

_sas_logged = False
_li_logged = False


def npu_dsv4_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    position_ids,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    """Drop-in ``DeepseekV4Attention.forward`` using NPU sparse attention (SAS).

    Falls back to the standard HF attention interface when the ACLNN
    extension is unavailable. Expects ``self.compressor(...)`` to return a
    3-tuple ``(compressed_kv, block_bias, top_k_indices)`` (see
    :func:`npu_dsv4_make_compressor_wrapper`).
    """
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb,
                                                                      eager_attention_forward)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings[self.rope_layer_type]

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
        compressor_out = self.compressor(hidden_states, q_residual, position_ids, past_key_values, self.layer_idx)
        if len(compressor_out) == 3:
            compressed_kv, block_bias, top_k_indices = compressor_out
        else:
            compressed_kv, block_bias = compressor_out

    use_sas = True
    if self.layer_type == 'sliding_attention':
        cmp_ratio = 1
        cmp_kv_arg = None
        cmp_sparse_indices = None
    elif self.layer_type == 'compressed_sparse_attention':
        cmp_ratio = self.config.compress_rates['compressed_sparse_attention']
        if compressed_kv is not None and compressed_kv.shape[2] > 0:
            cmp_kv_arg = compressed_kv.squeeze(1).contiguous()
            if top_k_indices is not None:
                cmp_sparse_indices = top_k_indices.clamp(min=0).to(torch.int32)
            else:
                cmp_sparse_indices = None
        else:
            use_sas = False
            cmp_kv_arg = None
            cmp_sparse_indices = None
    else:
        cmp_ratio = self.config.compress_rates['heavily_compressed_attention']
        if compressed_kv is not None and compressed_kv.shape[2] > 0:
            cmp_kv_arg = compressed_kv.squeeze(1).contiguous()
        else:
            use_sas = False
            cmp_kv_arg = None
        cmp_sparse_indices = None

    try:
        from .aclnn_ops import npu_sparse_attn_shared_kv
        attn_output = npu_sparse_attn_shared_kv(
            query=q.transpose(1, 2).contiguous(),
            ori_kv=ori_kv.squeeze(1).contiguous(),
            cmp_kv=cmp_kv_arg,
            cmp_sparse_indices=cmp_sparse_indices,
            sinks=self.sinks.float(),
            softmax_scale=self.scaling,
            cmp_ratio=cmp_ratio,
            ori_win_left=self.sliding_window - 1,
        )
        global _sas_logged
        if not _sas_logged:
            logger.info(
                '[NPU] [DSV4-SAS] Twinkle sparse attention active '
                '(layer_type=%s, cmp_ratio=%s, topk=%s)',
                self.layer_type,
                cmp_ratio,
                0 if cmp_sparse_indices is None else cmp_sparse_indices.shape[-1],
            )
            _sas_logged = True
        attn_weights = None
    except (ImportError, RuntimeError):
        use_sas = False

    if not use_sas:
        if compressed_kv is not None:
            kv = torch.cat([kv, compressed_kv], dim=2)
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            if block_bias is not None:
                attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                                    eager_attention_forward)
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


def npu_dsv4_indexer_forward(
    self,
    hidden_states,
    q_residual,
    position_ids,
    past_key_values,
    layer_idx,
):
    """Drop-in ``DeepseekV4Indexer.forward`` using NPU Lightning Indexer (LI).

    Falls back to a pure-torch top-k implementation when the ACLNN
    extension is unavailable.
    """
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    batch, seq_len, _ = hidden_states.shape
    cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights('indexer', kv, gate)

    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        ratio = self.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

        new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
        new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float('-inf'))
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
        if cache_layer is not None:
            prior_kv, prior_gate = cache_layer.update_overlap_state('indexer', chunk_kv, chunk_gate, self.head_dim)
            if prior_kv is not None:
                new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

        compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
        positions = torch.arange(n_windows, device=compressed.device)
        positions = positions * self.compress_rate + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    compressed_kv = (compressed if cache_layer is None else cache_layer.update_compressor_states('indexer', compressed))

    cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
    q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
    q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

    def torch_indexer_top_k_indices():
        index_scores = self.scorer(q, compressed_kv, hidden_states)
        compressed_len = compressed_kv.shape[1]
        top_k = min(self.index_topk, compressed_len)
        if compressed_len > 0:
            causal_threshold = (position_ids + 1) // self.compress_rate
            entry_indices = torch.arange(compressed_len, device=index_scores.device)
            future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
            index_scores = index_scores.masked_fill(future_mask, float('-inf'))
            top_k_indices = index_scores.topk(top_k, dim=-1).indices
            invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
            top_k_indices = torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)
            if top_k < self.index_topk:
                padding = top_k_indices.new_full((batch, seq_len, self.index_topk - top_k), -1)
                top_k_indices = torch.cat([top_k_indices, padding], dim=-1)
            return top_k_indices
        return index_scores.new_full((batch, seq_len, self.index_topk), -1, dtype=torch.long)

    if compressed_kv.shape[1] > 0:
        try:
            from .aclnn_ops import npu_lightning_indexer

            scorer = self.scorer
            weights = scorer.weights_proj(hidden_states).to(torch.bfloat16) * scorer.weights_scaling
            q_indexer = q.to(torch.bfloat16)
            k_indexer = compressed_kv.to(torch.bfloat16).unsqueeze(2)
            top_k_indices, _ = npu_lightning_indexer(
                q_indexer,
                k_indexer,
                weights,
                sparse_count=self.index_topk,
                sparse_mode=3,
                cmp_ratio=self.compress_rate,
            )
            top_k_indices = top_k_indices.squeeze(2)
            global _li_logged
            if not _li_logged:
                logger.info(
                    '[NPU] [DSV4-LI] Twinkle lightning indexer active '
                    '(sparse_count=%s, cmp_ratio=%s)',
                    self.index_topk,
                    self.compress_rate,
                )
                _li_logged = True
            return top_k_indices
        except (ImportError, RuntimeError, NameError):
            pass

    return torch_indexer_top_k_indices()


def npu_dsv4_csa_compressor_forward(
    self,
    hidden_states,
    q_residual,
    position_ids,
    past_key_values,
    layer_idx,
):
    """Drop-in ``DeepseekV4CSACompressor.forward`` that returns a 3-tuple.

    Identical to the stock CSA compressor forward, but returns
    ``(compressed_kv, block_bias, top_k_indices)`` so the SAS attention forward
    can use ``top_k_indices`` directly — **without re-invoking the indexer**.

    The stock forward already calls ``self.indexer(...)`` internally to build
    ``block_bias``; a wrapper that called the indexer a second time to fetch
    ``top_k_indices`` would mutate ``DeepseekV4CSACache`` twice
    (``store_compression_weights`` appends on every call). Under
    gradient checkpointing the recomputed forward sees a cache already mutated
    by the first forward, producing a different compressed length
    (e.g. 714 vs 712 tokens) and triggering ``CheckpointError``.
    """
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    batch, seq_len, _ = hidden_states.shape
    cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None:
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights('compressor', kv, gate)

    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        ratio = self.compress_rate
        chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
        chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias

        new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
        new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float('-inf'))
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
        if cache_layer is not None:
            prior_kv, prior_gate = cache_layer.update_overlap_state('compressor', chunk_kv, chunk_gate, self.head_dim)
            if prior_kv is not None:
                new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

        compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
        positions = torch.arange(n_windows, device=compressed.device)
        positions = positions * self.compress_rate + first_window_position
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
        compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
        compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states('compressor', compressed)
    compressed_kv = compressed.unsqueeze(1)

    # Lightning Indexer — called ONCE here; the result is returned alongside
    # block_bias so the SAS attention forward need not re-invoke the indexer.
    top_k_indices = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
    compressed_len = compressed_kv.shape[2]
    valid = top_k_indices >= 0
    safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
    block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float('-inf'))
    block_bias.scatter_(-1, safe_indices.unsqueeze(1).to(torch.int64), 0.0)
    return compressed_kv, block_bias[..., :compressed_len], top_k_indices
