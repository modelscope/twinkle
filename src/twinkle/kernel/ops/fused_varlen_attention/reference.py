# Copyright (c) ModelScope Contributors. All rights reserved.
"""分段（packed）序列的精确 PyTorch 参考实现：逐文档调用 SDPA，不依赖任何融合内核。"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _segments(cu_seqlens, total):
    values = [int(v) for v in cu_seqlens]
    if len(values) < 2:
        raise ValueError('cu_seqlens 至少需要两个元素')
    if values[0] != 0 or values[-1] != total:
        raise ValueError('cu_seqlens 必须从 0 开始并以序列总长结束')
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError('cu_seqlens 必须严格递增（不支持空文档）')
    return list(zip(values, values[1:]))


def _repeat_kv(tensor, groups):
    if groups == 1:
        return tensor
    heads, length, dim = tensor.shape
    return tensor[:, None].expand(heads, groups, length, dim).reshape(heads * groups, length, dim)


def fused_varlen_attention_reference(query, key, value, cu_seqlens, *, scale=None, causal=True):
    """query/key/value 为 [T, H, D] 的拼接序列；按 cu_seqlens 逐文档独立计算注意力。"""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError('参考实现要求 [T, H, D] 三维输入')
    heads, head_dim = query.shape[1], query.shape[-1]
    kv_heads = key.shape[1]
    if key.shape != value.shape or heads % kv_heads:
        raise ValueError('key/value 形状需一致，且 query 头数需为 key 头数的整数倍')
    scale = head_dim**-0.5 if scale is None else scale
    groups = heads // kv_heads
    outputs = []
    for start, end in _segments(cu_seqlens, query.shape[0]):
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = _repeat_kv(key[start:end].transpose(0, 1), groups).unsqueeze(0)
        v = _repeat_kv(value[start:end].transpose(0, 1), groups).unsqueeze(0)
        length = end - start
        mask = None
        if causal:
            mask = torch.full((length, length), 0.0, dtype=q.dtype, device=q.device)
            mask = mask.masked_fill(torch.triu(torch.ones_like(mask, dtype=torch.bool), 1), float('-inf'))
        outputs.append(F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)[0].transpose(0, 1))
    return torch.cat(outputs, 0)
