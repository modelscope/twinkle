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

#: 块掩码边长。CANN 的 compressed sparse mask 只接受固定的 2048x2048 块，
#: 与 ``sparse_mode=3``（rightDownCausal）配合时按分段边界对齐，不会跨文档。
BLOCK_MASK = 2048
#: 自检用的最大误差；FP32 下实测 ~1e-7，留两个数量级余量。
SELF_CHECK_TOL = 1e-4

_MASK_CACHE: dict = {}
_VERIFIED: set = set()
#: 上一次「整型张量 -> list[int]」转换的缓存，见 :func:`_as_list`。
_LIST_CACHE: dict = {}


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
    """把累积长度转成 CANN 要求的 ``list[int]``。

    一次前向里每层拿到的都是同一个张量对象，逐层 ``.tolist()`` 会逐层触发一次主机同步，
    把设备流水线打空（28 层就是 56 次）。这里按「对象 identity + 版本号」缓存上一次结果：
    张量被就地改写会推进 ``_version``，缓存自动失效，不会读到过期边界。
    """
    if isinstance(cu_seqlens, (list, tuple)):
        return [int(value) for value in cu_seqlens]
    key = (cu_seqlens.device.type, cu_seqlens.device.index)
    cached = _LIST_CACHE.get(key)
    if cached is not None and cached[0] is cu_seqlens and cached[1] == cu_seqlens._version:
        return cached[2]
    values = [int(value) for value in cu_seqlens.tolist()]
    _LIST_CACHE[key] = (cu_seqlens, cu_seqlens._version, values)
    return values


def npu_fused_varlen_attention(query, key, value, cu_seqlens, *, scale=None, causal=True):
    """``cu_seqlens`` 可为累积长度列表或整型张量；张量在此转成列表（CANN 只接受 ``list[int]``）。"""
    if not causal:
        raise NotImplementedError('NPU 融合路径只支持因果注意力；非因果请用 backend="torch"')
    actual = _as_list(cu_seqlens)
    _verify_vendor_semantics(query.device)
    return _vendor_attention(query, key, value, actual, query.shape[-1]**-0.5 if scale is None else scale)


def _masked_min(dtype):
    return torch.finfo(dtype).min if dtype.is_floating_point else torch.finfo(torch.float32).min


def _additive_mask(attention_mask, query, key):
    """把 HF 的 None/2D padding 掩码/4D 因果掩码统一成 SDPA 可用的加性掩码。"""
    seq_q, seq_kv = query.shape[2], key.shape[2]
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask[..., :seq_kv]
    additive = torch.zeros(seq_q, seq_kv, dtype=query.dtype, device=query.device)
    causal = torch.triu(torch.ones(seq_q, seq_kv, dtype=torch.bool, device=query.device), 1)
    additive = additive.masked_fill(causal, _masked_min(query.dtype))
    if attention_mask is not None and attention_mask.ndim == 2:
        padding = attention_mask[:, :seq_kv] == 0
        blocked = torch.tensor(_masked_min(query.dtype), dtype=query.dtype, device=query.device)
        additive = additive[None, None] + torch.where(padding[:, None, None, :], blocked,
                                                      torch.zeros((), dtype=query.dtype, device=query.device))
    return additive


def _sdpa_fallback(module, query, key, value, attention_mask, dropout, scaling, is_causal):
    """非 packed 调用（padded 前向、正确性对照）走 HF 语义的 SDPA，掩码自行补全。

    ``_attn_implementation`` 为 flash 时上游不会构造 4D 因果掩码，只可能给 2D padding 掩码，
    直接下传会被 FA 拒绝（``get unsupported atten_mask shape``），因此这里显式合成。
    """
    import torch.nn.functional as F
    from transformers.integrations.sdpa_attention import repeat_kv

    if hasattr(module, 'num_key_value_groups'):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    mask = _additive_mask(attention_mask, query, key)
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


def _sdpa_causal(module, query, key, value, dropout, scaling):
    import torch.nn.functional as F
    from transformers.integrations.sdpa_attention import repeat_kv

    if hasattr(module, 'num_key_value_groups'):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
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
    回退到原生 SDPA（padded 前向、正确性对照用），绝不把无边界输入当成分段输入。
    单文档直接使用原生 causal SDPA；多文档使用 TND 融合 varlen。
    """
    if dropout:
        raise ValueError('训练态融合变长注意力不支持 dropout')
    if sliding_window is not None:
        raise NotImplementedError('融合变长注意力暂不支持滑动窗口')
    if cu_seq_lens_q is None:
        return _sdpa_fallback(module, query, key, value, attention_mask, dropout, scaling, is_causal)
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError('融合变长注意力要求 [1, H, S, D] 的 padding-free 拼接输入')
    if len(cu_seq_lens_q) == 2:
        return _sdpa_causal(module, query, key, value, dropout, scaling)
    _, heads, seq_len, head_dim = query.shape
    flat_q = query.transpose(1, 2).reshape(seq_len, heads, head_dim)
    flat_k = key.transpose(1, 2).reshape(seq_len, key.shape[1], head_dim)
    flat_v = value.transpose(1, 2).reshape(seq_len, value.shape[1], head_dim)
    output = npu_fused_varlen_attention(flat_q, flat_k, flat_v, cu_seq_lens_q, scale=scaling, causal=True)
    return output.reshape(1, seq_len, heads, head_dim), None
