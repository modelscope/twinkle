# Copyright (c) ModelScope Contributors. All rights reserved.
"""``transformers`` 注意力接口的适配层：不关心底层用哪种分段注意力实现。

``npu`` 后端把它接到 CANN 融合内核，``torch`` 后端接到逐文档精确参考实现，两条路径
共用同一套掩码合成、分段边界校验和单文档快路径，避免适配层语义随后端漂移。
"""
from __future__ import annotations

import torch

from .reference import _segments

_LIST_CACHE: dict = {}


def boundary_list(cu_seqlens):
    """把累积长度转成 ``list[int]``，供 CANN 与分段校验共用。

    一次前向里每层拿到的都是同一个张量对象，逐层 ``.tolist()`` 会逐层触发一次主机同步，
    把设备流水线打空（28 层就是 56 次）。这里按「对象 identity + 版本号」缓存上一次结果：
    张量被就地改写会推进 ``_version``，缓存自动失效，不会读到过期边界。
    ``inference_mode`` 下创建的张量没有版本计数，因此不参与缓存（否则既读不到版本号，
    也无法判断是否被改写）。
    """
    if isinstance(cu_seqlens, (list, tuple)):
        return [int(value) for value in cu_seqlens]
    if cu_seqlens.is_inference():
        return [int(value) for value in cu_seqlens.tolist()]
    key = (cu_seqlens.device.type, cu_seqlens.device.index)
    cached = _LIST_CACHE.get(key)
    if cached is not None and cached[0] is cu_seqlens and cached[1] == cu_seqlens._version:
        return cached[2]
    values = [int(value) for value in cu_seqlens.tolist()]
    _LIST_CACHE[key] = (cu_seqlens, cu_seqlens._version, values)
    return values


def _masked_min(dtype):
    return torch.finfo(dtype).min if dtype.is_floating_point else torch.finfo(torch.float32).min


def additive_mask(attention_mask, query, key, is_causal=True):
    """把 HF 的 None/2D padding 掩码/4D 掩码统一成 SDPA 可用的加性掩码。

    因果掩码按 ``query`` 在 ``key`` 中的**右下角**对齐：``query`` 长度小于 ``key`` 时
    （KV-cache 解码、chunked 解码）放行全部历史 key，只有真正在未来的位置被屏蔽。
    左上角对齐会把解码时长度 1 的 query 限定在第 0 个 key 上，静默给出错误结果。
    """
    seq_q, seq_kv = query.shape[2], key.shape[2]
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask[..., :seq_kv]
    additive = torch.zeros(seq_q, seq_kv, dtype=query.dtype, device=query.device)
    if is_causal:
        offset = max(seq_kv - seq_q, 0)
        blocked = torch.triu(torch.ones(seq_q, seq_kv, dtype=torch.bool, device=query.device), diagonal=offset + 1)
        additive = additive.masked_fill(blocked, _masked_min(query.dtype))
    if attention_mask is not None and attention_mask.ndim == 2:
        padding = attention_mask[:, :seq_kv] == 0
        blocked_value = torch.tensor(_masked_min(query.dtype), dtype=query.dtype, device=query.device)
        additive = additive[None, None] + torch.where(padding[:, None, None, :], blocked_value,
                                                      torch.zeros((), dtype=query.dtype, device=query.device))
    return additive


def sdpa_attention(module, query, key, value, attention_mask, dropout, scaling, is_causal):
    """非 packed 调用（padded 前向、生成、正确性对照）走 HF 语义的 SDPA，掩码自行补全。"""
    import torch.nn.functional as F
    from transformers.integrations.sdpa_attention import repeat_kv

    groups = getattr(module, 'num_key_value_groups', 1)
    if groups > 1:
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)
    mask = additive_mask(attention_mask, query, key, is_causal)
    output = F.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous(), None


def causal_sdpa(module, query, key, value, dropout, scaling):
    """单文档快路径：方阵输入用原生 ``is_causal``，带 cache 时退回显式偏移掩码。"""
    import torch.nn.functional as F

    if query.shape[2] == key.shape[2]:
        groups = getattr(module, 'num_key_value_groups', 1)
        if groups > 1:
            from transformers.integrations.sdpa_attention import repeat_kv
            key = repeat_kv(key, groups)
            value = repeat_kv(value, groups)
        output = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=None,
            dropout_p=dropout,
            scale=scaling,
            is_causal=True,
        )
        return output.transpose(1, 2).contiguous(), None
    return sdpa_attention(module, query, key, value, None, dropout, scaling, True)


def _boundaries(cu_seqlens, total, name):
    values = boundary_list(cu_seqlens)
    try:
        _segments(values, total)
    except ValueError as error:
        raise ValueError(f'{name}: {error}') from error
    return values


def varlen_attention_forward(
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
    *,
    packed_attention,
    **kwargs,
):
    """``transformers`` flash-attention 接口的变长实现。

    只接受 padding-free 拼接输入（batch=1 且提供 ``cu_seq_lens_q``）：拿不到分段边界时
    回退到原生 SDPA（padded 前向、生成、正确性对照用），绝不把无边界输入当成分段输入。
    单文档直接使用原生 causal SDPA；多文档使用 ``packed_attention``。
    """
    if dropout:
        raise ValueError('训练态融合变长注意力不支持 dropout')
    if sliding_window is not None:
        raise NotImplementedError('融合变长注意力暂不支持滑动窗口')
    causal = True if is_causal is None else bool(is_causal)
    if cu_seq_lens_q is None:
        return sdpa_attention(module, query, key, value, attention_mask, dropout, scaling, causal)
    if not causal:
        raise NotImplementedError('融合变长注意力只支持因果注意力；非因果请用原生 SDPA')
    if attention_mask is not None:
        raise NotImplementedError('padding-free 拼接输入不接受 attention_mask；拼接布局请传 None')
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError('融合变长注意力要求 [1, H, S, D] 的 padding-free 拼接输入')
    _, heads, seq_len, head_dim = query.shape
    if key.shape[2] != seq_len:
        raise NotImplementedError('融合变长注意力只支持拼接自注意力：query/key 序列长度必须一致')
    boundaries = _boundaries(cu_seq_lens_q, seq_len, 'cu_seq_lens_q')
    if cu_seq_lens_k is not None and _boundaries(cu_seq_lens_k, key.shape[2], 'cu_seq_lens_k') != boundaries:
        raise NotImplementedError('融合变长注意力只支持 query/key 共享分段边界')
    if len(boundaries) == 2:
        return causal_sdpa(module, query, key, value, dropout, scaling)
    flat_q = query.transpose(1, 2).reshape(seq_len, heads, head_dim)
    flat_k = key.transpose(1, 2).reshape(seq_len, key.shape[1], head_dim)
    flat_v = value.transpose(1, 2).reshape(seq_len, value.shape[1], head_dim)
    output = packed_attention(flat_q, flat_k, flat_v, boundaries, scale=scaling)
    return output.reshape(1, seq_len, heads, head_dim), None
