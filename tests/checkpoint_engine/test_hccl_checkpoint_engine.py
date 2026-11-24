# Copyright (c) ModelScope Contributors. All rights reserved.
"""HCCL full-sync checkpoint engine: group lifecycle with fakes, plus a CPU multi-process loopback."""
from __future__ import annotations

import asyncio
import os

import torch

import pytest

from twinkle.checkpoint_engine import hccl_checkpoint_engine as hccl
from twinkle.checkpoint_engine.hccl_checkpoint_engine import MasterMetadata
from ._hccl_cpu_harness import make_state, run

META = MasterMetadata(zmq_ip='1.2.3.4', zmq_port=111, dist_ip='1.2.3.4', dist_port=222, auth_key='k',
                      hccl_env={'HCCL_IF_BASE_PORT': '20222'})


class FakeComm:
    """Communicator stand-in: records the calls the engine makes on its group."""

    def __init__(self, barrier_error=None, fill=None):
        self.device = torch.device('cpu')
        self.barrier_error = barrier_error
        self.fill = fill
        self.barriers = 0
        self.broadcasts: list[int] = []
        self.syncs = 0
        self.destroyed = False

    def barrier(self):
        self.barriers += 1
        if self.barrier_error is not None:
            raise self.barrier_error

    def broadcast(self, tensor, src=0):
        assert src == 0
        self.broadcasts.append(tensor.numel())
        if self.fill is not None:
            self.fill(tensor)

    def synchronize(self):
        self.syncs += 1

    def destroy(self):
        self.destroyed = True


class CpuEngine(hccl.HCCLCheckpointEngine):
    """The engine with CPU bucket buffers (no NPU on a test box)."""

    _buffer_device = 'cpu'


@pytest.fixture
def comms(monkeypatch):
    """Hand out prepared FakeComms and record what the engine asked ``create_communicator`` for."""
    made: list[FakeComm] = []
    calls: list[dict] = []
    queued: list[FakeComm] = []

    def fake_create(rendezvous, rank, world_size, device, backend=None, **options):
        calls.append(dict(rendezvous=rendezvous, rank=rank, world_size=world_size, device=device,
                          backend=backend, **options))
        comm = queued.pop(0) if queued else FakeComm()
        made.append(comm)
        return comm

    # prepare() really binds its ZMQ server, so the node IP and ports must be bindable here.
    free = [hccl.find_free_port() for _ in range(4)]
    ports = iter(free)
    monkeypatch.setattr(hccl, 'create_communicator', fake_create)
    monkeypatch.setattr(hccl, 'find_node_ip', lambda: '127.0.0.1')
    monkeypatch.setattr(hccl, 'find_free_port', lambda: next(ports))
    return type('Comms', (), {'made': made, 'calls': calls, 'queued': queued, 'ports': free})


def test_prepare_publishes_the_rendezvous_and_the_hccl_socket_env(comms, monkeypatch):
    from twinkle.utils.platforms.npu import _derive_hccl_socket_env_defaults

    for key in _derive_hccl_socket_env_defaults(0):
        monkeypatch.delenv(key, raising=False)
    # Already pinned by this trainer's default process group: the rollout ranks must match it.
    monkeypatch.setenv('HCCL_IF_BASE_PORT', '12345')
    engine = CpuEngine(bucket_size=4096)
    engine.is_master = True

    md = engine.prepare()
    zmq_port, dist_port = comms.ports[:2]
    assert (md.zmq_ip, md.zmq_port, md.dist_ip, md.dist_port) == ('127.0.0.1', zmq_port, '127.0.0.1', dist_port)
    assert md.hccl_env == dict(_derive_hccl_socket_env_defaults(dist_port), HCCL_IF_BASE_PORT='12345')
    # Pin-then-publish: the layout is applied here, so what the rollout ranks are told is exactly
    # what this process will use. A bare snapshot could be a layout nothing ever applies.
    assert {key: os.environ[key] for key in md.hccl_env} == md.hccl_env
    assert len(md.auth_key) >= 32
    assert engine.prepare() == md  # idempotent
    assert engine.send_buf.device.type == 'cpu' and engine.send_buf.numel() == 4096


def test_prepare_non_master_publishes_nothing(comms):
    engine = CpuEngine(bucket_size=16)
    assert engine.prepare() is None and engine.hccl_env == {}


def test_init_process_group_builds_one_group_and_rendezvouses(comms):
    engine = CpuEngine(bucket_size=16, rebuild_group=False)
    engine.init_process_group(0, 3, META)

    assert comms.calls == [dict(rendezvous=hccl.Rendezvous('1.2.3.4', 222, group_name='twinkle_ckpt'), rank=0,
                                world_size=3, device=torch.device('cpu'), backend='hccl',
                                socket_env={'HCCL_IF_BASE_PORT': '20222'})]
    assert comms.made[0].barriers == 1
    assert (engine.rank, engine.world_size, engine._group_initialized) == (0, 3, True)

    engine.init_process_group(0, 3, META)  # reuse: no second group, no second barrier
    assert len(comms.calls) == 1 and comms.made[0].barriers == 1


def test_init_process_group_without_a_published_env_derives_it_in_the_backend(comms):
    engine = CpuEngine(bucket_size=16)
    engine.init_process_group(1, 2, MasterMetadata(zmq_ip='h', zmq_port=1, dist_ip='h', dist_port=2))
    assert comms.calls[0]['socket_env'] is None


def test_non_participating_rank_joins_no_group(comms):
    engine = CpuEngine(bucket_size=16)
    engine.init_process_group(-1, 3, META)
    assert comms.calls == [] and engine._comm is None and engine._group_initialized


def test_failed_barrier_destroys_the_half_built_group(comms):
    comms.queued.append(FakeComm(barrier_error=TimeoutError('peer missing')))
    engine = CpuEngine(bucket_size=16)
    with pytest.raises(TimeoutError):
        engine.init_process_group(0, 2, META)
    assert comms.made[0].destroyed and engine._comm is None and not engine._group_initialized


def test_broadcast_bucket_waits_for_the_device(comms):
    engine = CpuEngine(bucket_size=16)
    engine._comm = comm = FakeComm()
    engine._broadcast_bucket(torch.zeros(8, dtype=torch.uint8))
    assert comm.broadcasts == [8] and comm.syncs == 1


def test_finalize_keeps_the_group_unless_rebuild_group(comms):
    engine = CpuEngine(bucket_size=16, rebuild_group=False)
    engine.init_process_group(0, 2, META)
    comm = engine._comm
    engine.finalize()
    assert engine._comm is comm and not comm.destroyed
    engine.close()  # close() tears down regardless of the reuse setting
    assert comm.destroyed and engine._comm is None and engine.rank is None
    assert engine.rebuild_group is False


def test_send_weights_without_receivers_broadcasts_every_bucket(comms):
    engine = CpuEngine(bucket_size=64)
    engine.is_master = True
    engine.prepare()
    engine.init_process_group(0, 1, META)
    comm = engine._comm
    weights = {'a': torch.ones(40, dtype=torch.uint8), 'b': torch.ones(40, dtype=torch.uint8)}

    asyncio.run(engine.send_weights(iter(weights.items())))
    engine.close()
    # 80 bytes through a 64-byte bucket: two broadcasts, each followed by a host-side wait.
    assert comm.broadcasts == [64, 64] and comm.syncs == 2


def test_non_participating_rank_drains_the_weights_without_sending(comms):
    engine = CpuEngine(bucket_size=16)
    engine.init_process_group(-1, 2, META)
    consumed = []

    def weights():
        for name in ('a', 'b'):
            consumed.append(name)
            yield name, torch.ones(4, dtype=torch.uint8)

    asyncio.run(engine.send_weights(weights()))
    assert consumed == ['a', 'b']


# --------------------------------------------------------------------------- CPU loopback
def _assert_rounds(out, rounds, n_recv):
    for rank in range(1, n_recv + 1):
        for r, expected in enumerate(rounds):
            got = out['recv'][(rank, r)][0]
            assert sorted(got) == sorted(expected), (rank, r)
            for name, tensor in expected.items():
                assert got[name].dtype == tensor.dtype and got[name].shape == tensor.shape, (rank, r, name)
                assert torch.equal(got[name], tensor), (rank, r, name)


def test_cpu_loopback_reuses_one_group_across_rounds():
    rounds = [make_state(0), make_state(1)]
    out = run(dict(n_recv=2, bucket_size=4096, rebuild_group=False, rounds=rounds))
    _assert_rounds(out, rounds, 2)
    groups = {out['trainer'][('trainer_round', r)]['group'] for r in range(len(rounds))}
    assert len(groups) == 1 and None not in groups
    for rank in (1, 2):
        assert out['recv'][(rank, 0)][1]['group'] == out['recv'][(rank, 1)][1]['group']
    for r in range(len(rounds)):
        assert out['trainer'][('trainer_finalized', r)]['held_destroyed'] is False
    assert out['closed'][0]['held_destroyed'] is True
    for state in out['closed'].values():
        assert state['group'] is None


def test_cpu_loopback_rebuild_group_builds_a_fresh_group_each_round():
    rounds = [make_state(2), make_state(3)]
    out = run(dict(n_recv=1, bucket_size=4096, rebuild_group=True, rounds=rounds))
    _assert_rounds(out, rounds, 1)
    envs = [out['trainer'][('trainer_group', r)]['hccl_env'] for r in range(len(rounds))]
    assert all(env.get('HCCL_IF_BASE_PORT') for env in envs)
    for r in range(len(rounds)):
        finalized = out['trainer'][('trainer_finalized', r)]
        assert finalized['group'] is None and finalized['held_destroyed'] is True
        assert finalized['prepared'] is False


def test_cpu_loopback_single_bucket_three_receivers():
    rounds = [make_state(4)]
    out = run(dict(n_recv=3, bucket_size=1 << 20, rebuild_group=False, rounds=rounds))
    _assert_rounds(out, rounds, 3)


# --------------------------------------------------------------------------- real NPUs (HCCL)
def _npus_available(count: int) -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available() and torch.npu.device_count() >= count


@pytest.mark.accel(2)
@pytest.mark.parametrize('rebuild_group', [False, True])
def test_npu_full_sync_over_hccl(rebuild_group):
    if not _npus_available(2):
        pytest.skip('needs 2 npu devices')
    rounds = [make_state(5), make_state(6)]
    out = run(dict(device='npu', n_recv=1, bucket_size=4096, rebuild_group=rebuild_group, rounds=rounds),
              timeout=300.0)
    _assert_rounds(out, rounds, 1)
    for r in range(len(rounds)):
        assert out['trainer'][('trainer_round', r)]['group'] is not None
        assert out['trainer'][('trainer_finalized', r)]['held_destroyed'] is rebuild_group
    # With rebuild_group the last finalize already tore the group down; otherwise close() does.
    assert out['closed'][0]['held_destroyed'] is (None if rebuild_group else True)
    assert all(state['group'] is None for state in out['closed'].values())


@pytest.mark.accel(4)
def test_npu_full_sync_three_receivers():
    """One broadcaster fans out to three NPU receivers over a single HCCL group."""
    if not _npus_available(4):
        pytest.skip('needs 4 npu devices')
    rounds = [make_state(7), make_state(8)]
    out = run(dict(device='npu', n_recv=3, bucket_size=4096, rebuild_group=False, rounds=rounds),
              timeout=420.0)
    _assert_rounds(out, rounds, 3)
    for rank in (1, 2, 3):
        assert out['recv'][(rank, 0)][1]['group'] is not None
    assert out['closed'][0]['held_destroyed'] is True
    assert all(state['group'] is None for state in out['closed'].values())
