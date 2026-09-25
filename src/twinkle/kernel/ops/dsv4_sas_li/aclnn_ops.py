# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4 NPU fusion operators (SAS + LI) backed by self-compiled ACLNN.

JIT-compiles thin C++ bindings that invoke the ACLNN kernels directly
(``aclnnSparseAttnSharedkv``, ``aclnnLightningIndexer``, etc.), no mindspeed.
"""
from __future__ import annotations

import torch

from twinkle import get_logger
from .aclnn.builder import build_op

logger = get_logger()

TORCH_MAX_INT = 9223372036854775807

_sas_op = _li_op = _li_grad_op = None


def _ensure_ops():
    global _sas_op, _li_op, _li_grad_op
    if _sas_op is None:
        _sas_op = build_op('sparse_attn_sharedkv', ['sparse_attn_sharedkv/binding.cpp'])
        _li_op = build_op('lightning_indexer', ['lightning_indexer/binding.cpp'])
        _li_grad_op = build_op('lightning_indexer_grad', ['lightning_indexer_grad/binding.cpp'])
        logger.info('[NPU] [ACLNN] SAS + LI ops compiled successfully')


class SparseAttnSharedKV(torch.autograd.Function):
    """SAS: Shared-KV sparse attention (forward + backward via ACLNN).

    Metadata is computed OUTSIDE this Function (in the convenience wrapper)
    so it is NOT recomputed by gradient checkpointing's backward recompute.
    """

    @staticmethod
    def forward(ctx, query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, metadata, softmax_scale, cmp_ratio,
                ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout):
        _ensure_ops()
        ori_stride = ori_kv.stride(0)
        cmp_stride = cmp_kv.stride(0) if cmp_kv is not None else 0

        result, lse = _sas_op.npu_sparse_attn_sharedkv(query, ori_kv, cmp_kv, None, cmp_sparse_indices, None, None,
                                                       None, None, None, None, None, sinks, metadata, softmax_scale,
                                                       cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_stride, cmp_stride,
                                                       ori_win_left, ori_win_right, layout, layout, True)

        ctx.save_for_backward(query, ori_kv, cmp_kv, result, lse, cmp_sparse_indices, sinks)
        ctx.scale, ctx.cmp_ratio = softmax_scale, cmp_ratio
        ctx.ori_mm, ctx.cmp_mm = ori_mask_mode, cmp_mask_mode
        ctx.ori_wl, ctx.ori_wr = ori_win_left, ori_win_right
        ctx.layout = layout
        return result

    @staticmethod
    def backward(ctx, grad_output):
        q, ori_kv, cmp_kv, result, lse, cmp_si, sinks = ctx.saved_tensors
        q_g, kv_g, cmp_g, sinks_g = _sas_op.npu_sparse_attn_sharedkv_grad(q, ori_kv, cmp_kv, grad_output, result, lse,
                                                                          None, cmp_si, None, None, None, sinks,
                                                                          ctx.scale, ctx.cmp_ratio, ctx.ori_mm,
                                                                          ctx.cmp_mm, ctx.ori_wl, ctx.ori_wr,
                                                                          ctx.layout)
        return (q_g, kv_g, cmp_g, None, sinks_g, None) + (None, ) * 8


def npu_sparse_attn_shared_kv(query,
                              ori_kv,
                              cmp_kv,
                              cmp_sparse_indices,
                              sinks,
                              softmax_scale,
                              cmp_ratio,
                              ori_mask_mode=4,
                              cmp_mask_mode=3,
                              ori_win_left=127,
                              ori_win_right=0,
                              layout='BSND'):
    """Convenience wrapper: ``[B,S,N,D]`` query, ``[B,S,D]`` shared KV.

    Metadata is pre-computed here (outside the autograd Function) so gradient
    checkpointing's backward recomputation does NOT re-invoke the metadata kernel.
    """
    _ensure_ops()
    b, s_q, n_h, h_d = query.shape
    s_kv = ori_kv.size(1)
    topk = 0 if cmp_sparse_indices is None else cmp_sparse_indices.size(-1)
    has_cmp_kv = cmp_kv is not None

    e = torch.tensor([]).npu()
    metadata = _sas_op.npu_sparse_attn_sharedkv_metadata(e, e, e, e, e, n_h, 1, h_d, b, s_q, s_kv, 0, topk, cmp_ratio,
                                                         ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                                                         layout, layout, True, has_cmp_kv)

    query = query.contiguous()
    ori_kv = ori_kv.unsqueeze(2).contiguous()
    cmp_kv = cmp_kv if cmp_kv is None else cmp_kv.unsqueeze(2).contiguous()
    if cmp_sparse_indices is not None:
        cmp_sparse_indices = cmp_sparse_indices.unsqueeze(2).contiguous()
    return SparseAttnSharedKV.apply(query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, metadata, softmax_scale,
                                    cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                                    layout).contiguous()


class LightningIndexer(torch.autograd.Function):
    """LI: Lightning Indexer (forward + backward via ACLNN)."""

    @staticmethod
    def forward(ctx, query, key, weights, sparse_count, sparse_mode, cmp_ratio):
        _ensure_ops()
        indices, values = _li_op.npu_lightning_indexer(query, key, weights, None, None, None, 'BSND', 'BSND',
                                                       sparse_count, sparse_mode, TORCH_MAX_INT, TORCH_MAX_INT,
                                                       cmp_ratio, True)
        ctx.save_for_backward(query, key, weights, indices)
        ctx.sparse_mode, ctx.cmp_ratio = sparse_mode, cmp_ratio
        return indices, values

    @staticmethod
    def backward(ctx, grad_indices, grad_values):
        q, key, weights, indices = ctx.saved_tensors
        q_g, k_g, w_g = _li_grad_op.npu_lightning_indexer_grad(q, key, grad_values, indices, weights, None, None,
                                                               'BSND', ctx.sparse_mode, TORCH_MAX_INT, TORCH_MAX_INT,
                                                               ctx.cmp_ratio, None)
        return q_g, k_g, w_g, None, None, None


def npu_lightning_indexer(query, key, weights, sparse_count=2048, sparse_mode=3, cmp_ratio=1):
    """Convenience wrapper for the Lightning Indexer."""
    return LightningIndexer.apply(query, key, weights, sparse_count, sparse_mode, cmp_ratio)
