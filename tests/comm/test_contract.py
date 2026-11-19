# Copyright (c) ModelScope Contributors. All rights reserved.
"""The contract ``Communicator`` enforces for every backend, plus single-process Gloo checks."""
import logging
import time
from datetime import timedelta

import pytest
import torch

from twinkle.comm import Communicator, ReduceOp, Rendezvous, create_communicator
from twinkle.comm.backends.gloo import GlooCommunicator
from twinkle.comm.base import normalize_device
from twinkle.utils.network import find_free_port


class _RecordingCommunicator(Communicator):
    """Implements only the mandatory hooks; records every call."""

    backend_name = 'recording'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []
        self.fail_destroy = False

    @classmethod
    def _create(cls, rendezvous, rank, world_size, device, *, timeout, **options):
        return cls(rank, world_size, device)

    def _broadcast(self, tensor, src, async_op):
        self.calls.append(('broadcast', tuple(tensor.shape), tensor.dtype, src))

    def _destroy(self):
        self.synchronize()
        self.calls.append(('destroy', ))
        if self.fail_destroy:
            raise OSError('teardown boom')


class _ReducingCommunicator(_RecordingCommunicator):

    def _all_reduce(self, tensor, op, async_op):
        self.calls.append(('all_reduce', tuple(tensor.shape), tensor.dtype, op))


class _GlooSubclass(GlooCommunicator):
    pass


def _gloo(world_size=1, **kwargs):
    return create_communicator(Rendezvous('127.0.0.1', find_free_port()), 0, world_size, 'cpu', backend='gloo',
                               **kwargs)


# ---- construction ----------------------------------------------------------------


@pytest.mark.parametrize('rank, world_size', [(-1, 2), (2, 2), (0, 0)])
def test_create_validates_group_before_connecting(rank, world_size):
    with pytest.raises(ValueError):
        _RecordingCommunicator.create(Rendezvous('h', 1), rank, world_size, 'cpu')


def test_create_rejects_unknown_options_and_bad_timeout():
    with pytest.raises(ValueError, match='does not accept options'):
        _RecordingCommunicator.create(Rendezvous('h', 1), 0, 1, 'cpu', typo=1)
    with pytest.raises(ValueError, match='timeout'):
        _RecordingCommunicator.create(Rendezvous('h', 1), 0, 1, 'cpu', timeout=timedelta(0))


def test_normalize_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    assert normalize_device('cuda') == torch.device('cuda', 3)
    assert normalize_device('cuda:1') == torch.device('cuda', 1)
    assert normalize_device(torch.device('cpu')) == torch.device('cpu')
    with pytest.raises(TypeError):
        normalize_device(0)
    comm = _RecordingCommunicator.create(Rendezvous('h', 1), 0, 1, 'cuda')
    assert comm.device_index == 3


# ---- teardown --------------------------------------------------------------------


def test_destroy_is_idempotent_and_can_synchronize_inside():
    comm = _RecordingCommunicator(0, 2, 'cpu')
    comm.destroy()
    comm.destroy()
    assert comm.calls == [('destroy', )]
    assert comm.destroyed


def test_destroy_logs_teardown_errors(caplog):
    comm = _RecordingCommunicator(0, 2, 'cpu')
    comm.fail_destroy = True
    with caplog.at_level(logging.WARNING, logger='twinkle.comm.base'):
        comm.destroy()
    assert comm.destroyed
    assert 'teardown boom' in caplog.text


def test_operations_after_destroy_raise():
    comm = _ReducingCommunicator(0, 2, 'cpu')
    comm.destroy()
    for call in (comm.barrier, comm.synchronize, lambda: comm.broadcast(torch.zeros(1)),
                 lambda: comm.all_reduce(torch.zeros(1))):
        with pytest.raises(RuntimeError, match='destroyed'):
            call()


# ---- tensor and argument checks ------------------------------------------------


def test_tensor_preconditions():
    comm = _ReducingCommunicator(0, 2, 'cpu')
    with pytest.raises(ValueError, match='contiguous'):
        comm.broadcast(torch.zeros(4, 4).t())
    with pytest.raises(ValueError, match='bound to cpu'):
        comm.broadcast(torch.empty(2, device='meta'))
    with pytest.raises(ValueError, match='contiguous'):
        comm.all_reduce(torch.zeros(4, 4).t())
    with pytest.raises(TypeError):
        comm.broadcast([1, 2])
    with pytest.raises(ValueError, match='src'):
        comm.broadcast(torch.zeros(1), src=2)
    assert comm.calls == []


def test_zero_element_tensors_never_reach_the_backend():
    comm = _ReducingCommunicator(0, 2, 'cpu')
    assert comm.broadcast(torch.empty(0)) is None
    assert comm.broadcast(torch.empty(0, 3), async_op=True).wait()
    assert comm.all_reduce(torch.empty(0)) is None
    assert comm.calls == []


def test_reduce_op_parsing():
    assert ReduceOp('SUM') is ReduceOp.SUM
    assert ReduceOp('Avg') is ReduceOp.AVG
    with pytest.raises(ValueError):
        ReduceOp('bogus')
    comm = _ReducingCommunicator(0, 2, 'cpu')
    with pytest.raises(ValueError):
        comm.all_reduce(torch.zeros(1), op='bogus')
    comm.all_reduce(torch.zeros(1), op='MAX')
    assert comm.calls == [('all_reduce', (1, ), torch.float32, ReduceOp.MAX)]


# ---- capabilities and defaults ---------------------------------------------------


def test_supports():
    assert _RecordingCommunicator.supports('broadcast')
    assert not _RecordingCommunicator.supports('all_reduce')
    assert not _RecordingCommunicator.supports('barrier')
    assert _ReducingCommunicator.supports('all_reduce')
    assert GlooCommunicator.supports('all_reduce') and GlooCommunicator.supports('barrier')
    assert _GlooSubclass.supports('all_reduce')
    for name in ('typo', 'destroy', 'create', 'synchronize'):
        with pytest.raises(ValueError, match='unknown communicator op'):
            GlooCommunicator.supports(name)


def test_missing_optional_op_raises_not_implemented():
    with pytest.raises(NotImplementedError, match='recording.*all_reduce'):
        _RecordingCommunicator(0, 2, 'cpu').all_reduce(torch.zeros(1))


def test_default_barrier_prefers_all_reduce():
    plain = _RecordingCommunicator(1, 2, 'cpu')
    plain.barrier()
    plain.barrier()
    assert plain.calls == [('broadcast', (1, ), torch.int32, 0)] * 2
    reducing = _ReducingCommunicator(1, 2, 'cpu')
    reducing.barrier()
    assert reducing.calls == [('all_reduce', (1, ), torch.int32, ReduceOp.SUM)]


# ---- gloo, single process --------------------------------------------------------


def test_gloo_single_rank_roundtrip():
    comm = _gloo()
    try:
        assert comm.native is not None
        t = torch.arange(4, dtype=torch.float32)
        comm.broadcast(t, src=0)
        comm.all_reduce(t, op=ReduceOp.AVG)
        assert comm.all_reduce(t, op='max', async_op=True).wait()
        assert torch.equal(t, torch.arange(4, dtype=torch.float32))
        comm.barrier()
    finally:
        comm.destroy()
    assert comm.native is None


def test_gloo_groups_coexist_in_one_process():
    a, b = _gloo(), _gloo()
    try:
        ta, tb = torch.ones(2), torch.zeros(2)
        a.broadcast(ta)
        b.broadcast(tb)
        a.barrier()
        b.barrier()
        assert torch.equal(ta, torch.ones(2)) and torch.equal(tb, torch.zeros(2))
    finally:
        a.destroy()
        b.destroy()


def test_gloo_create_times_out_when_a_rank_never_joins():
    start = time.monotonic()
    with pytest.raises(Exception, match='(?i)timeout|timed out'):
        _gloo(world_size=2, timeout=timedelta(seconds=2))
    assert time.monotonic() - start < 30
