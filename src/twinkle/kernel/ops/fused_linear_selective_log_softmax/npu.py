# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ascend fused lm_head projection and selective log-softmax."""
from __future__ import annotations

import torch

from .kernels import backward_launch, stats_launch


class _FusedLinearSelectiveLogSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, targets, temperature, return_entropy):
        logits = torch.mm(x, weight.t())
        logits.div_(temperature)
        rows = x.shape[0]
        lse = torch.empty(rows, dtype=torch.float32, device=x.device)
        logps = torch.empty(rows, dtype=torch.float32, device=x.device)
        entropies = None
        expected = torch.empty_like(logps) if return_entropy else lse
        if x.dtype == torch.float32:
            temporary_bytes = 32 << 20 if return_entropy else 64 << 20
            chunk_rows = max(1, temporary_bytes // (logits.shape[-1] * 4))
            for start in range(0, rows, chunk_rows):
                stop = min(start + chunk_rows, rows)
                chunk = logits[start:stop]
                lse[start:stop] = torch.logsumexp(chunk, dim=-1)
                selected = torch.gather(chunk, dim=-1, index=targets[start:stop].unsqueeze(-1)).squeeze(-1)
                logps[start:stop] = selected - lse[start:stop]
                if return_entropy:
                    probabilities = torch.softmax(chunk, dim=-1)
                    expected[start:stop] = (probabilities * chunk).sum(dim=-1)
        else:
            stats_launch(logits, targets, lse, logps)
            if return_entropy:
                chunk_rows = max(1, (32 << 20) // (logits.shape[-1] * 4))
                for start in range(0, rows, chunk_rows):
                    stop = min(start + chunk_rows, rows)
                    wide = logits[start:stop].float()
                    probabilities = torch.softmax(wide, dim=-1)
                    expected[start:stop] = (probabilities * wide).sum(dim=-1)
        if return_entropy:
            entropies = lse - expected

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
        if logits.dtype == torch.float32:
            temporary_bytes = 32 << 20 if ctx.return_entropy else 64 << 20
            chunk_rows = max(1, temporary_bytes // (logits.shape[-1] * 4))
            for start in range(0, logits.shape[0], chunk_rows):
                stop = min(start + chunk_rows, logits.shape[0])
                chunk = logits[start:stop]
                probabilities = torch.softmax(chunk, dim=-1)
                grad_chunk = torch.zeros_like(chunk)
                grad_logp = grad_logps[start:stop].unsqueeze(-1)
                grad_chunk.scatter_add_(dim=-1, index=targets[start:stop].unsqueeze(-1), src=grad_logp)
                grad_chunk.sub_(probabilities * grad_logp)
                if ctx.return_entropy:
                    grad_entropy = grad_entropies[start:stop].unsqueeze(-1)
                    grad_chunk.add_(probabilities * (grad_entropy * (expected[start:stop].unsqueeze(-1) - chunk)))
                chunk.copy_(grad_chunk)
        else:
            backward_launch(logits, targets, lse, expected, grad_logps, grad_entropies)
        logits.div_(ctx.temperature)
        grad_x = torch.mm(logits, weight) if ctx.needs_input_grad[0] else None
        grad_weight = torch.mm(logits.t(), x) if ctx.needs_input_grad[1] else None
        return grad_x, grad_weight, None, None, None


def npu_fused_linear_selective_log_softmax(x, weight, targets, temperature=1.0, return_entropy=False):
    return _FusedLinearSelectiveLogSoftmax.apply(x.contiguous(), weight.contiguous(),
                                                 targets.contiguous().long(), temperature, return_entropy)
