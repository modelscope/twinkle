# Copyright (c) ModelScope Contributors. All rights reserved.
"""NCCL checkpoint engines across real GPUs: full sync (reused and rebuilt groups) and delta sync."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ._hccl_loopback import build_rounds, make_state
from ._nccl_harness import make_full_state, run

BUCKET = 4096


def _assert_full_rounds(out, rounds, n_recv):
    for rank in range(1, n_recv + 1):
        for r, expected in enumerate(rounds):
            got = out['recv'][(rank, r)][0]
            assert sorted(got) == sorted(expected), (rank, r)
            for name, tensor in expected.items():
                assert got[name].dtype == tensor.dtype and torch.equal(got[name], tensor), (rank, r, name)


@pytest.mark.accel(3)
def test_full_sync_reuses_one_group_across_rounds():
    rounds = [make_full_state(0), make_full_state(1)]
    out = run('full', dict(n_recv=2, bucket_size=BUCKET, rebuild_group=False, rounds=rounds))
    _assert_full_rounds(out, rounds, 2)
    groups = {out['trainer'][('trainer_round', r)]['group'] for r in range(len(rounds))}
    assert len(groups) == 1 and None not in groups
    for rank in range(1, 3):
        assert out['recv'][(rank, 0)][1]['group'] == out['recv'][(rank, 1)][1]['group']
    for r in range(len(rounds)):
        assert out['trainer'][('trainer_finalized', r)]['held_destroyed'] is False
    assert out['closed'][0]['held_destroyed'] is True
    for state in out['closed'].values():
        assert state['group'] is None


@pytest.mark.accel(3)
def test_full_sync_rebuild_group_builds_a_fresh_group_each_round():
    rounds = [make_full_state(2), make_full_state(3), make_full_state(4)]
    out = run('full', dict(n_recv=2, bucket_size=BUCKET, rebuild_group=True, rounds=rounds))
    _assert_full_rounds(out, rounds, 2)
    for r in range(len(rounds)):
        finalized = out['trainer'][('trainer_finalized', r)]
        assert finalized['group'] is None and finalized['held_destroyed'] is True


@pytest.mark.accel(4)
def test_full_sync_three_receivers_single_bucket():
    rounds = [make_full_state(5)]
    out = run('full', dict(n_recv=3, bucket_size=64 << 20, rebuild_group=False, rounds=rounds))
    _assert_full_rounds(out, rounds, 3)


@pytest.mark.accel(3)
def test_delta_seed_then_sparse_rounds():
    init = make_state(7)
    rounds, expected = build_rounds(init, [{'layer.w1': 0, 'layer.w2': 500, 'layer.bias': 3},
                                           {'layer.w1': 5, 'layer.w2': 0, 'layer.bias': 64}, {}], seed=8)
    out = run('delta', dict(n_recv=2, bucket_size=1024, init_state=init, rounds=rounds))
    for rank in (1, 2):
        for r, want in enumerate(expected):
            got, flushes = out['recv'][(rank, r)]
            assert sorted(got) == sorted(want), (rank, r)
            for name, tensor in want.items():
                assert torch.equal(got[name].view(torch.int16), tensor.view(torch.int16)), (rank, r, name)
        assert out['recv'][(rank, 0)][1] > 1  # the seed spans several 1 KiB flushes
        assert out['recv'][(rank, 3)][1] == 0  # zero-change round: terminal manifest, no broadcast
    for state in out['closed'].values():
        assert state['group'] is None


def test_xccl_engine_broadcasts_through_its_own_group():
    from twinkle.checkpoint_engine.xpu_checkpoint_engine import XCCLCheckpointEngine

    class RecordingPG:

        def __init__(self):
            self.calls = []

        def broadcast(self, tensors, opts):
            self.calls.append((tensors[0], opts.rootRank))
            return SimpleNamespace(wait=lambda: None)

    engine = XCCLCheckpointEngine(bucket_size=16, rebuild_group=True)
    assert engine._comm is None
    engine._pg = pg = RecordingPG()
    tensor = torch.zeros(4, dtype=torch.uint8)
    engine._group_broadcast(tensor)
    assert pg.calls == [(tensor, 0)]
    engine.finalize()
    assert engine._pg is None and engine._store is None and engine._comm is None
