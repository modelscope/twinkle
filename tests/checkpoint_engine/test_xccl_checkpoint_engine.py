# Copyright (c) ModelScope Contributors. All rights reserved.
"""The XPU full-sync checkpoint engine across real Kunlunxin cards."""
from __future__ import annotations

import pytest
import torch

from ._xccl_harness import make_state, run

BUCKET = 4096


def _assert_rounds(out, rounds, n_recv):
    for rank in range(1, n_recv + 1):
        for r, expected in enumerate(rounds):
            got = out['recv'][(rank, r)][0]
            assert sorted(got) == sorted(expected), (rank, r)
            for name, tensor in expected.items():
                assert got[name].dtype == tensor.dtype and torch.equal(got[name], tensor), (rank, r, name)


@pytest.mark.accel(3, backend='xccl')
def test_full_sync_reuses_one_group_across_rounds():
    rounds = [make_state(0), make_state(1)]
    out = run(dict(n_recv=2, bucket_size=BUCKET, rebuild_group=False, rounds=rounds))
    _assert_rounds(out, rounds, 2)
    groups = {out['trainer'][('trainer_round', r)] for r in range(len(rounds))}
    assert len(groups) == 1 and None not in groups, 'the group must be reused, not rebuilt'
    for r in range(len(rounds)):
        assert out['trainer'][('trainer_finalized', r)]['held_destroyed'] is False
    for state in out['closed'].values():
        assert state['group'] is None, 'close() must drop the group even when it is reused'


@pytest.mark.accel(3, backend='xccl')
def test_full_sync_rebuild_group_builds_a_fresh_group_each_round():
    rounds = [make_state(2), make_state(3)]
    out = run(dict(n_recv=2, bucket_size=BUCKET, rebuild_group=True, rounds=rounds))
    _assert_rounds(out, rounds, 2)
    for r in range(len(rounds)):
        finalized = out['trainer'][('trainer_finalized', r)]
        assert finalized['group'] is None and finalized['held_destroyed'] is True


@pytest.mark.accel(4, backend='xccl')
def test_full_sync_three_receivers_single_bucket():
    # One bucket large enough for the whole state: no chunking, every weight arrives as a view.
    rounds = [make_state(5)]
    out = run(dict(n_recv=3, bucket_size=1 << 20, rebuild_group=False, rounds=rounds))
    _assert_rounds(out, rounds, 3)
