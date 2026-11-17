# Copyright (c) ModelScope Contributors. All rights reserved.
"""Triton-Ascend kernels for selected-token log-probability and entropy."""
from __future__ import annotations

import triton
import triton.language as tl

BLOCK = 4096
NUM_WARPS = 8


@triton.jit
def stats_row_kernel(LOGITS, TARGETS, ROW_MAX, ROW_SUM, LSE, LOGPS, V, BLOCK_SIZE: tl.constexpr):
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
    tl.store(ROW_MAX + row, running_max)
    tl.store(ROW_SUM + row, running_sum)
    tl.store(LSE + row, lse)
    tl.store(LOGPS + row, target_logit - lse)


@triton.jit
def backward_row_kernel(LOGITS, TARGETS, ROW_MAX, ROW_SUM, GRAD_LOGPS, V, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * V
    target = tl.load(TARGETS + row)
    row_max = tl.load(ROW_MAX + row)
    row_sum = tl.load(ROW_SUM + row)
    grad_logp = tl.load(GRAD_LOGPS + row).to(tl.float32)
    for i in range(0, tl.cdiv(V, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < V
        logits = tl.load(LOGITS + base + offsets, mask=mask, other=-float('inf')).to(tl.float32)
        probabilities = tl.exp(logits - row_max) / row_sum
        grad = -(probabilities * grad_logp)
        grad = tl.where(offsets == target, grad + grad_logp, grad)
        tl.store(LOGITS + base + offsets, grad.to(LOGITS.dtype.element_ty), mask=mask)


@triton.jit
def expected_row_kernel(LOGITS, ROW_MAX, ROW_SUM, EXPECTED, V, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = row * V
    row_max = tl.load(ROW_MAX + row)
    row_sum = tl.load(ROW_SUM + row)
    running = 0.0
    for i in range(0, tl.cdiv(V, BLOCK_SIZE)):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < V
        logits = tl.load(LOGITS + base + offsets, mask=valid, other=0.0).to(tl.float32)
        # Masked lanes contribute 0 to both factors, avoiding exp(-inf) * (-inf) = NaN.
        probabilities = tl.where(valid, tl.exp(logits - row_max) / row_sum, 0.0)
        running += tl.sum(probabilities * tl.where(valid, logits, 0.0), 0)
    tl.store(EXPECTED + row, running)


def expected_launch(logits, row_max, row_sum, expected):
    rows, vocab = logits.shape
    expected_row_kernel[(rows, )](logits, row_max, row_sum, expected, vocab, BLOCK, num_warps=NUM_WARPS)


def stats_launch(logits, targets, row_max, row_sum, lse, logps):
    rows, vocab = logits.shape
    stats_row_kernel[(rows, )](logits, targets, row_max, row_sum, lse, logps, vocab, BLOCK, num_warps=NUM_WARPS)


def backward_launch(logits, targets, row_max, row_sum, grad_logps):
    rows, vocab = logits.shape
    backward_row_kernel[(rows, )](logits, targets, row_max, row_sum, grad_logps, vocab, BLOCK, num_warps=NUM_WARPS)
