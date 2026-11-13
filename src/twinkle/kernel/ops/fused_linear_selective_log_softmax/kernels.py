# Copyright (c) ModelScope Contributors. All rights reserved.
"""Triton-Ascend kernels for selected-token log-probability and entropy."""
from __future__ import annotations

import triton
import triton.language as tl

BLOCK = 4096
NUM_WARPS = 8


@triton.jit
def stats_row_kernel(LOGITS, TARGETS, LSE, LOGPS, V, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * V
    target = tl.load(TARGETS + row)
    running_max = -float('inf')
    running_sum = 0.0
    target_logit = 0.0
    for i in range(0, tl.cdiv(V, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(LOGITS + base + offsets, mask=offsets < V, other=-float('inf')).to(tl.float32)
        new_max = tl.maximum(running_max, tl.max(values, 0))
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(tl.exp(values - new_max), 0)
        running_max = new_max
        local_target = target - i * BLOCK_SIZE
        target_logit += tl.sum(
            tl.where((tl.arange(0, BLOCK_SIZE) == local_target) & (local_target >= 0) & (local_target < BLOCK_SIZE),
                     values, 0.0), 0)
    lse = running_max + tl.log(running_sum)
    tl.store(LSE + row, lse)
    tl.store(LOGPS + row, target_logit - lse)


@triton.jit
def backward_row_kernel(LOGITS, TARGETS, LSE, EXPECTED, GRAD_LOGPS, GRAD_ENTROPIES, V, HAS_ENTROPY: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * V
    target = tl.load(TARGETS + row)
    lse = tl.load(LSE + row)
    grad_logp = tl.load(GRAD_LOGPS + row).to(tl.float32)
    if HAS_ENTROPY:
        grad_entropy = tl.load(GRAD_ENTROPIES + row).to(tl.float32)
        expected = tl.load(EXPECTED + row)
    else:
        grad_entropy = 0.0
        expected = 0.0
    for i in range(0, tl.cdiv(V, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < V
        logits = tl.load(LOGITS + base + offsets, mask=mask, other=-float('inf')).to(tl.float32)
        probabilities = tl.exp(logits - lse)
        grad = -(probabilities * grad_logp)
        if HAS_ENTROPY:
            grad += probabilities * (grad_entropy * (expected - logits))
        grad = tl.where(offsets == target, grad + grad_logp, grad)
        tl.store(LOGITS + base + offsets, grad.to(LOGITS.dtype.element_ty), mask=mask)


def stats_launch(logits, targets, lse, logps):
    rows, vocab = logits.shape
    stats_row_kernel[(rows, )](logits, targets, lse, logps, vocab, BLOCK, num_warps=NUM_WARPS)


def backward_launch(logits, targets, lse, expected, grad_logps, grad_entropies):
    rows, vocab = logits.shape
    has_entropy = grad_entropies is not None
    if not has_entropy:
        grad_entropies = grad_logps
    backward_row_kernel[(rows, )](
        logits, targets, lse, expected, grad_logps, grad_entropies, vocab, has_entropy, BLOCK, num_warps=NUM_WARPS)
