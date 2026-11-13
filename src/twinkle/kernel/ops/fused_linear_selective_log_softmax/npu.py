# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ascend Triton fused lm_head projection and selective log-softmax (BF16)."""
from __future__ import annotations

import torch

from .kernels import backward_launch, stats_launch

# Entropy's E_p[z] reduction runs in FP32 over row chunks; chunking is over rows,
# so per-row math is chunk-size independent.
_ENTROPY_CHUNK_BYTES = 128 << 20


class _FusedLinearSelectiveLogSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, targets, temperature, return_entropy):
        logits = torch.mm(x, weight.t())
        logits.div_(temperature)
        rows = x.shape[0]
        lse = torch.empty(rows, dtype=torch.float32, device=x.device)
        logps = torch.empty(rows, dtype=torch.float32, device=x.device)
        expected = torch.empty_like(logps) if return_entropy else lse
        stats_launch(logits, targets, lse, logps)
        if return_entropy:
            chunk_rows = max(1, _ENTROPY_CHUNK_BYTES // (logits.shape[-1] * 4))
            for start in range(0, rows, chunk_rows):
                stop = min(start + chunk_rows, rows)
                wide = logits[start:stop].float()
                probabilities = torch.softmax(wide, dim=-1)
                expected[start:stop] = (probabilities * wide).sum(dim=-1)
            entropies = lse - expected
        else:
            entropies = None

        ctx.save_for_backward(logits, targets, lse, expected, x, weight)
        ctx.return_entropy = return_entropy
        ctx.temperature = temperature
        ctx.consumed = False
        if return_entropy:
            return logps, entropies
        return logps

    @staticmethod
    def backward(ctx, grad_logps, grad_entropies=None):
        if ctx.consumed:
            raise RuntimeError('fused linear selective log-softmax does not support repeated backward')
        ctx.consumed = True
        logits, targets, lse, expected, x, weight = ctx.saved_tensors
        if grad_logps is None:
            grad_logps = torch.zeros_like(lse)
        if ctx.return_entropy and grad_entropies is None:
            grad_entropies = torch.zeros_like(lse)
        if not ctx.return_entropy:
            grad_entropies = None

        if grad_entropies is not None:
            grad_entropies = grad_entropies.contiguous()
        grad_logps = grad_logps.contiguous()
        backward_launch(logits, targets, lse, expected, grad_logps, grad_entropies)
        logits.div_(ctx.temperature)
        grad_x = torch.mm(logits, weight) if ctx.needs_input_grad[0] else None
        grad_weight = torch.mm(logits.t(), x) if ctx.needs_input_grad[1] else None
        return grad_x, grad_weight, None, None, None


def npu_fused_linear_selective_log_softmax(x, weight, targets, temperature=1.0, return_entropy=False):
    # The functional entry has already validated the integer dtype and normalized to
    # int64; this explicit range check mirrors native gather, which rejects out-of-range
    # indices instead of silently producing wrong probabilities in the Triton kernels.
    if not torch.logical_and(targets >= 0, targets < weight.shape[0]).all():
        raise IndexError(f'targets must be within the vocabulary range [0, {weight.shape[0]})')
    return _FusedLinearSelectiveLogSoftmax.apply(x.contiguous(), weight.contiguous(),
                                                 targets.contiguous(), temperature, return_entropy)
