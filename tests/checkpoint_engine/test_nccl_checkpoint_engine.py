# Copyright (c) ModelScope Contributors. All rights reserved.
"""NCCL checkpoint engines across real GPUs: full sync (reused and rebuilt groups) and delta sync."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from twinkle.checkpoint_engine.zmq_auth import send_auth_pyobj
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

class _FakeComm:
    """Single-process stand-in: all_reduce adds the peers' contribution to this rank's state."""

    def __init__(self, peers=None, barrier_error=None):
        self.device = torch.device('cpu')
        self.peers = peers
        self.barrier_error = barrier_error
        self.reduces = 0
        self.destroyed = False

    def barrier(self):
        if self.barrier_error is not None:
            raise self.barrier_error

    def all_reduce(self, tensor):
        self.reduces += 1
        tensor += torch.tensor(self.peers, dtype=tensor.dtype)

    def destroy(self):
        self.destroyed = True


def _rendezvous_engine(rank, world_size, comm, socket):
    from twinkle.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
    engine = NCCLCheckpointEngine(bucket_size=16)
    engine.rank, engine.world_size, engine._comm, engine.socket = rank, world_size, comm, socket
    return engine


def test_rendezvous_receiver_failure_still_joins_the_collective():
    def poll(timeout):
        raise RuntimeError('boom')

    comm = _FakeComm(peers=[1, 0, 0])
    engine = _rendezvous_engine(1, 3, comm, SimpleNamespace(poll=poll))
    with pytest.raises(RuntimeError, match='boom'):
        engine._rendezvous_subscribers()
    assert comm.reduces == 1


def test_rendezvous_master_aborts_when_a_peer_failed():
    socket = SimpleNamespace(send_string=lambda *a, **k: None, send=lambda *a, **k: None)
    comm = _FakeComm(peers=[1, 0, 1])
    engine = _rendezvous_engine(0, 3, comm, socket)
    with pytest.raises(RuntimeError, match='1 peer rank'):
        engine._rendezvous_subscribers()
    assert comm.reduces == 1


def test_failed_init_destroys_the_half_built_group(monkeypatch):
    from twinkle.checkpoint_engine import nccl_checkpoint_engine as nccl
    comm = _FakeComm(barrier_error=TimeoutError('peer missing'))
    monkeypatch.setattr(nccl, 'create_communicator', lambda *a, **k: comm)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    engine = nccl.NCCLCheckpointEngine(bucket_size=16)
    with pytest.raises(TimeoutError):
        engine.init_process_group(0, 2, nccl.MasterMetadata('127.0.0.1', 1, '127.0.0.1', 2))
    assert comm.destroyed and engine._comm is None and not engine._group_initialized


def test_delta_receiver_skips_stale_wire_probes():
    from twinkle.checkpoint_engine.delta.transport.nccl_engine import DeltaShardedCheckpointEngine
    engine = DeltaShardedCheckpointEngine(bucket_size=16)
    engine.rank = 1
    engine.auth_key = 'k'
    wire = SimpleNamespace(sent=[], send=lambda msg: wire.sent.append(msg))
    send_auth_pyobj(wire, {'wire_probe': True}, engine.auth_key)
    topics = [engine.topic + '__wire_probe__', engine.topic + '__wire_probe__', engine.topic]
    dropped = []
    engine.socket = SimpleNamespace(recv_string=lambda: topics.pop(0),
                                    recv=lambda copy=True: dropped.append(1) or wire.sent[0],
                                    recv_pyobj=lambda: {'terminal_empty': True})
    assert list(engine.receive_weights()) == []
    assert not topics and len(dropped) == 2


def test_delta_receiver_rejects_unsigned_wire_probe():
    from twinkle.checkpoint_engine.delta.transport.nccl_engine import DeltaShardedCheckpointEngine
    engine = DeltaShardedCheckpointEngine(bucket_size=16)
    engine.rank = 1
    engine.auth_key = 'k'
    engine.socket = SimpleNamespace(recv_string=lambda: engine.topic + '__wire_probe__',
                                    recv=lambda copy=True: b'\x00' * 40,
                                    recv_pyobj=lambda: {'terminal_empty': True})
    with pytest.raises(RuntimeError, match='HMAC verification failed'):
        list(engine.receive_weights())
