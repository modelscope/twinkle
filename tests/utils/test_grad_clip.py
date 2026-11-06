# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for gradient normalization + clipping (single-process standard path)."""
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from twinkle.utils.grad_clip import normalize_and_clip_grad_norm


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]



def _parameter(grad_value):
    parameter = nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([float(grad_value)])
    return parameter


class TestNormalizeAndClipGradNorm:

    def test_grads_are_divided_by_num_tokens(self):
        parameters = [_parameter(3.0), _parameter(4.0)]
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=2, max_grad_norm=10.0, norm_type=2.0)
        # 3/2 and 4/2 -> L2 norm 2.5, below the clip bound.
        assert norm == pytest.approx(2.5)
        assert parameters[0].grad.item() == pytest.approx(1.5)
        assert parameters[1].grad.item() == pytest.approx(2.0)

    def test_norm_is_reported_before_clipping(self):
        parameters = [_parameter(3.0), _parameter(4.0)]
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=1, max_grad_norm=0.5, norm_type=2.0)
        assert norm == pytest.approx(5.0)

    def test_clipping_scales_grads_to_the_bound(self):
        parameters = [_parameter(3.0), _parameter(4.0)]
        normalize_and_clip_grad_norm(parameters, num_tokens=2, max_grad_norm=1.25, norm_type=2.0)
        # coefficient = 1.25 / 2.5 = 0.5
        assert parameters[0].grad.item() == pytest.approx(0.75)
        assert parameters[1].grad.item() == pytest.approx(1.0)

    def test_inf_norm_clips_by_max_abs(self):
        parameters = [_parameter(3.0), _parameter(4.0)]
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=2, max_grad_norm=10.0, norm_type=float('inf'))
        assert norm == pytest.approx(2.0)

    def test_non_positive_num_tokens_is_treated_as_one(self):
        parameters = [_parameter(4.0)]
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=0, max_grad_norm=10.0, norm_type=2.0)
        assert norm == pytest.approx(4.0)

    def test_parameters_without_grads_are_skipped(self):
        plain = nn.Parameter(torch.tensor([1.0]))
        with_grad = _parameter(4.0)
        norm = normalize_and_clip_grad_norm(
            [plain, with_grad], num_tokens=2, max_grad_norm=10.0, norm_type=2.0)
        assert norm == pytest.approx(2.0)
        assert plain.grad is None

    def test_no_grads_at_all_returns_zero(self):
        plain = nn.Parameter(torch.tensor([1.0]))
        assert normalize_and_clip_grad_norm(
            [plain], num_tokens=2, max_grad_norm=1.0, norm_type=2.0) == 0.0


# ---------------------------------------------------------------------------
# DDP replicated grads must not be SUM-reduced across ranks (regression #225).
# The transformers backend clips with an explicit dp group; under DDP the backward
# allreduce already replicated the gradients, so every rank holds the SAME grads and
# the local norm IS the global norm. A SUM over identical per-rank square sums inflates
# the reported norm (and over-clips) by sqrt(world_size) -- silent at world_size=1,
# which is why the single-process tests above and the single-card parity grid missed it.
# FSDP2 (fully_shard) keeps shards as DTensors and takes the all_dtensor branch, so only
# replicated-local grads reach the code under test here.
# ---------------------------------------------------------------------------
def _ddp_worker(rank: int, world: int, port: int, out) -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('gloo', rank=rank, world_size=world)
    try:
        # Identical grads on every rank == the state after a DDP backward allreduce.
        grads = [torch.tensor([3.0, 4.0]), torch.tensor([12.0])]
        parameters = [nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(1))]
        for param, grad in zip(parameters, grads):
            param.grad = grad.clone()
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=2, max_grad_norm=1e9, norm_type=2.0,
            group=dist.group.WORLD)
        out[rank] = float(norm)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world', [2, 4])
def test_ddp_replicated_grads_not_sum_reduced(world):
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_ddp_worker, args=(world, _free_port(), out), nprocs=world, join=True)
    # (9+16)/4 + 144/4 = 42.25 -> sqrt = 6.5; the pre-fix SUM path returned sqrt(world * 42.25).
    expected = 42.25 ** 0.5
    for rank in range(world):
        assert out[rank] == pytest.approx(expected), (
            f'rank {rank}: norm {out[rank]} != {expected} (SUM-reduced by sqrt({world})?)')
