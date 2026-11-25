# Copyright (c) ModelScope Contributors. All rights reserved.
"""Device-free behaviour of the XPU checkpoint engine, including its decoupling from the GPU one."""
import pytest
import torch

from twinkle.checkpoint_engine.base import CheckpointEngine
from twinkle.checkpoint_engine.xpu_checkpoint_engine import MasterMetadata, XCCLCheckpointEngine


def _metadata():
    return MasterMetadata(zmq_ip='127.0.0.1', zmq_port=12345, dist_ip='127.0.0.1', dist_port=23456)


def test_engine_is_independent_of_the_nccl_engine():
    # The XPU group is built differently (a directly built ProcessGroupNCCL deadlocks when two
    # ranks share a local device index), so this engine owns its implementation instead of
    # pushing XPU seams into the GPU one.
    from twinkle.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

    assert XCCLCheckpointEngine.__bases__ == (CheckpointEngine, )
    assert not issubclass(XCCLCheckpointEngine, NCCLCheckpointEngine)
    assert XCCLCheckpointEngine.backend_name == 'xccl'


def test_engine_asks_for_the_xccl_backend_on_an_indexed_device(monkeypatch):
    calls = []
    monkeypatch.setattr('twinkle.checkpoint_engine.xpu_checkpoint_engine.create_communicator',
                        lambda *args, **kwargs: calls.append((args, kwargs)) or _FakeComm())
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)

    engine = XCCLCheckpointEngine()
    engine.init_process_group(0, 2, _metadata())

    (args, kwargs), = calls
    # An indexed device: a worker spanning several cards has no unambiguous ambient one.
    assert args[3] == torch.device('cuda', 3)
    assert kwargs['backend'] == 'xccl'
    assert engine._buffer_device == 'cuda:3'


class _FakeComm:
    """Enough of the Communicator contract for the group lifecycle."""

    def __init__(self, barrier_error=None):
        self.device = torch.device('cpu')
        self.barrier_error = barrier_error
        self.barriers = 0
        self.destroyed = False

    def barrier(self):
        self.barriers += 1
        if self.barrier_error is not None:
            raise self.barrier_error

    def destroy(self):
        self.destroyed = True


def test_group_is_reused_across_rounds_then_destroyed_by_close(monkeypatch):
    comm = _FakeComm()
    monkeypatch.setattr('twinkle.checkpoint_engine.xpu_checkpoint_engine.create_communicator',
                        lambda *args, **kwargs: comm)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)

    engine = XCCLCheckpointEngine(rebuild_group=False)
    engine.init_process_group(0, 2, _metadata())
    engine.init_process_group(0, 2, _metadata())
    assert comm.barriers == 1, 'a reused group must not re-barrier'

    engine.finalize()
    assert comm.destroyed is False, 'rebuild_group=False keeps the channel across rounds'
    engine.close()
    assert comm.destroyed is True and engine._comm is None


def test_failed_barrier_leaves_no_half_initialized_group(monkeypatch):
    comm = _FakeComm(barrier_error=RuntimeError('peer never joined'))
    monkeypatch.setattr('twinkle.checkpoint_engine.xpu_checkpoint_engine.create_communicator',
                        lambda *args, **kwargs: comm)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)

    engine = XCCLCheckpointEngine()
    with pytest.raises(RuntimeError, match='peer never joined'):
        engine.init_process_group(0, 2, _metadata())
    # The next call may arrive with fresh rendezvous metadata, so nothing may be cached.
    assert comm.destroyed is True
    assert engine._comm is None and engine._group_initialized is False


def test_non_participating_rank_builds_no_communicator(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('a non-participating trainer rank must not create a communicator')

    monkeypatch.setattr('twinkle.checkpoint_engine.xpu_checkpoint_engine.create_communicator', fail)
    engine = XCCLCheckpointEngine()
    engine.init_process_group(-1, 3, _metadata())
    assert (engine.rank, engine.world_size) == (-1, 3)
    assert engine._group_initialized is True and engine._comm is None


def test_build_topology_makes_trainer_rank0_the_only_sender():
    trainer, rollout = XCCLCheckpointEngine.build_topology(2, 3, [None, {'zmq_port': 1}])
    assert trainer['rank'] == [0, -1]
    assert rollout['rank'] == [1, 2, 3]
    assert trainer['world_size'] == [4, 4] and rollout['world_size'] == [4, 4, 4]
    # The master metadata is the first non-None entry, broadcast to every worker.
    assert trainer['master_metadata'] == [{'zmq_port': 1}] * 2
