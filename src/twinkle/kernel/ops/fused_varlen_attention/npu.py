# Copyright (c) ModelScope Contributors. All rights reserved.
"""昇腾 NPU 后端：整段拼接序列交给一次融合注意力调用，因果掩码由块掩码显式给定。

本文件依赖两条 CANN 行为，二者都由 ``_verify_vendor_semantics`` 在进程内自检：

1. ``npu_fusion_attention`` 在此版本上不会依据 ``sparse_mode``/``next_tockens`` 自动生成
   因果掩码，只传 ``sparse_mode=3`` 会得到**非因果**注意力（不报错，静默算错）。
2. 变长 TND 布局下，只有 ``2048x2048`` 上三角布尔块掩码配 ``sparse_mode=3`` 的前向与反向
   同时成立；``(T, T)`` 掩码配 ``sparse_mode=0`` 前向正确，反向报
   ``aclnnFlashAttentionUnpaddingScoreGrad failed, error code 16``。
"""
from __future__ import annotations

import torch

from .adapter import additive_mask as _additive_mask
from .adapter import boundary_list
from .adapter import causal_sdpa as _sdpa_causal
from .adapter import sdpa_attention as _sdpa_fallback
from .adapter import varlen_attention_forward

#: 块掩码边长。CANN 的 compressed sparse mask 只接受固定的 2048x2048 块，
#: 与 ``sparse_mode=3``（rightDownCausal）配合时按分段边界对齐，不会跨文档。
BLOCK_MASK = 2048
#: 自检用的最大误差；FP32 下实测 ~1e-7，留两个数量级余量。
SELF_CHECK_TOL = 1e-4

_MASK_CACHE: dict = {}
_VERIFIED: set = set()


def _block_mask(device):
    key = (device.type, device.index)
    mask = _MASK_CACHE.get(key)
    if mask is None:
        mask = torch.triu(torch.ones(BLOCK_MASK, BLOCK_MASK, dtype=torch.bool, device=device), 1)
        _MASK_CACHE[key] = mask
    return mask


def _vendor_attention(query, key, value, boundaries, scale):
    """直接调用 CANN 变长注意力；不做任何校验，供自检与主路径共用。"""
    import torch_npu

    output = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        query.shape[1],
        'TND',
        atten_mask=_block_mask(query.device),
        scale=scale,
        keep_prob=1.0,
        pre_tockens=2147483647,
        next_tockens=0,
        actual_seq_qlen=boundaries,
        actual_seq_kvlen=boundaries,
        sparse_mode=3,
    )
    return output[0] if isinstance(output, (tuple, list)) else output


def _verify_vendor_semantics(device):
    """首次在设备上调用时，用一个 8 token / 2 文档的小例子核对分段因果语义。

    CANN 一旦改掉掩码语义，这里会直接抛错，而不是让训练静默产出错误结果。
    """
    from .reference import fused_varlen_attention_reference

    key = (device.type, device.index)
    if key in _VERIFIED:
        return
    generator = torch.Generator(device=device).manual_seed(0)
    total, heads, dim, boundaries = 8, 2, 8, [0, 5, 8]
    query = torch.randn(total, heads, dim, device=device, dtype=torch.float32, generator=generator)
    keys = torch.randn(total, heads, dim, device=device, dtype=torch.float32, generator=generator)
    value = torch.randn(total, heads, dim, device=device, dtype=torch.float32, generator=generator)
    got = _vendor_attention(query, keys, value, boundaries, dim**-0.5)
    want = fused_varlen_attention_reference(query, keys, value, boundaries)
    difference = (got - want).abs().max().item()
    if not difference <= SELF_CHECK_TOL:
        raise RuntimeError(f'融合变长注意力自检失败（{device}）：最大误差 {difference:.3e}，'
                           f'说明当前 CANN/torch_npu 的分段因果语义与实现不符，拒绝继续训练')
    _VERIFIED.add(key)


def _as_list(cu_seqlens):
    """``adapter.boundary_list`` 的别名：保留既有调用点与测试的入口名。"""
    return boundary_list(cu_seqlens)


def _expand_bf16_gqa(query, key, value):
    """与 Twinkle 原生 SDPA 的 ``repeat_kv`` BF16 反向归约路径对齐。"""
    if query.dtype != torch.bfloat16 or query.shape[1] == key.shape[1]:
        return key, value
    groups = query.shape[1] // key.shape[1]

    def expand(tensor):
        total, heads, dim = tensor.shape
        return tensor[:, :, None, :].expand(total, heads, groups, dim).reshape(total, heads * groups, dim)

    return expand(key), expand(value)


def npu_fused_varlen_attention(query, key, value, cu_seqlens, *, scale=None, causal=True):
    """``cu_seqlens`` 可为累积长度列表或整型张量；张量在此转成列表（CANN 只接受 ``list[int]``）。"""
    if not causal:
        raise NotImplementedError('NPU 融合路径只支持因果注意力；非因果请用 backend="torch"')
    key, value = _expand_bf16_gqa(query, key, value)
    actual = _as_list(cu_seqlens)
    _verify_vendor_semantics(query.device)
    return _vendor_attention(query, key, value, actual, query.shape[-1]**-0.5 if scale is None else scale)


def npu_varlen_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    is_causal=None,
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_length_q=None,
    max_length_k=None,
    **kwargs,
):
    """``transformers`` flash-attention 接口的昇腾变长实现。

    只接受 padding-free 拼接输入（batch=1 且提供 ``cu_seq_lens_q``）：拿不到分段边界时
    回退到原生 SDPA（padded 前向、生成、正确性对照用），绝不把无边界输入当成分段输入。
    单文档直接使用原生 causal SDPA；多文档使用 TND 融合 varlen。
    """
    return varlen_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout,
        scaling,
        sliding_window,
        is_causal,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
        packed_attention=npu_fused_varlen_attention,
    )
