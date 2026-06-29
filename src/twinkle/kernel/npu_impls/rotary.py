# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fused RoPE impls for Ascend NPU (lazy ``torch_npu`` import)."""
from __future__ import annotations

import torch


def _resolve_unsqueeze_dim(position_ids=None, unsqueeze_dim=1):
    if isinstance(position_ids, int) and unsqueeze_dim == 1:
        return position_ids
    return unsqueeze_dim


def _make_apply_npu_rotary_emb():
    """Closure with per-shape Partial-RoPE detection cache."""
    _cached_partial: dict[tuple[int, int], bool] = {}

    def _apply(q, k, cos, sin):
        import torch_npu
        rotary_dim = cos.shape[-1]
        query_dim = q.shape[-1]
        shape_key = (rotary_dim, query_dim)

        use_partial = _cached_partial.get(shape_key)
        if use_partial is None:
            use_partial = rotary_dim < query_dim
            _cached_partial[shape_key] = use_partial

        if use_partial:
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin).to(q.dtype)
            k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin).to(k.dtype)
            q_embed = torch.cat([q_embed, q_pass], dim=-1)
            k_embed = torch.cat([k_embed, k_pass], dim=-1)
        else:
            q_embed = torch_npu.npu_rotary_mul(q, cos, sin).to(q.dtype)
            k_embed = torch_npu.npu_rotary_mul(k, cos, sin).to(k.dtype)
        return q_embed, k_embed

    return _apply


_apply_npu_rotary_emb = _make_apply_npu_rotary_emb()


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Fused RoPE via ``torch_npu.npu_rotary_mul`` with Partial-RoPE support."""
    unsqueeze_dim = _resolve_unsqueeze_dim(position_ids, unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _apply_npu_rotary_emb(q, k, cos, sin)


def npu_apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Multimodal RoPE for Qwen2.5-VL with Partial-RoPE support."""
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    return _apply_npu_rotary_emb(q, k, cos, sin)