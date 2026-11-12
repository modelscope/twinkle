# Copyright (c) ModelScope Contributors. All rights reserved.
"""分段（packed）序列的变长因果融合注意力；按设备选择后端，NPU 走单次融合调用。"""
from __future__ import annotations

import torch

from .reference import fused_varlen_attention_reference

_SUPPORTED = (torch.float32, torch.bfloat16)


def _normalize(cu_seqlens, total):
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.ndim != 1:
            raise ValueError('cu_seqlens 必须是一维张量')
        values = [int(v) for v in cu_seqlens.tolist()]
    else:
        values = [int(v) for v in cu_seqlens]
    if len(values) < 2:
        raise ValueError('cu_seqlens 至少需要两个元素')
    if values[0] != 0 or values[-1] != total:
        raise ValueError('cu_seqlens 必须从 0 开始并以序列总长结束')
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError('cu_seqlens 必须严格递增（不支持空文档）')
    return values


def _validate(query, key, value):
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError('query/key/value 必须为 [T, H, D] 拼接布局')
    if key.shape != value.shape:
        raise ValueError('key 与 value 形状必须一致')
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError('query/key 的序列长度与 head_dim 必须一致')
    if query.shape[1] % key.shape[1]:
        raise ValueError('query 头数必须是 key/value 头数的整数倍（GQA/MQA）')
    if any(t.dtype != query.dtype for t in (key, value)) or query.dtype not in _SUPPORTED:
        raise ValueError('融合变长注意力仅支持 FP32/BF16，且三者精度一致')
    if any(t.device != query.device for t in (key, value)):
        raise ValueError('query/key/value 必须在同一设备')


def fused_varlen_attention(query, key, value, cu_seqlens, *, scale=None, causal=True, backend=None):
    """拼接序列上的变长因果注意力，等价于逐文档独立计算，但只启动一次融合内核。

    query/key/value: ``[T, H, D]`` / ``[T, H_kv, D]``，T 为所有文档长度之和。
    cu_seqlens: 长度为文档数+1 的累积长度（从 0 开始，末项等于 T）。
    返回 ``[T, H, D]``；反向由同一算子给出，不依赖调用方的循环。
    """
    _validate(query, key, value)
    boundaries = _normalize(cu_seqlens, query.shape[0])
    if backend is None:
        backend = 'npu' if query.device.type in ('npu', 'privateuseone') else 'torch'
    if backend == 'npu':
        from .npu import npu_fused_varlen_attention
        return npu_fused_varlen_attention(query, key, value, boundaries, scale=scale, causal=causal)
    if backend == 'torch':
        return fused_varlen_attention_reference(query, key, value, boundaries, scale=scale, causal=causal)
    raise ValueError(f'未知后端 {backend!r}，可选 torch/npu')
