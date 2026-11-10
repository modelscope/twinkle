# Copyright (c) ModelScope Contributors. All rights reserved.
"""Contract of the communicator interface, registry and platform mapping (no process spawning)."""
import pytest
import torch

from twinkle.comm import (Communicator, ReduceOp, Rendezvous, available_backends, create_communicator,
                          get_communicator_cls, register_backend)
from twinkle.comm import registry
from twinkle.comm.backends.gloo import GlooCommunicator
from twinkle.utils.network import find_free_port
from twinkle.utils.platforms import GPU, MPS, NPU, XPU, Metax, Platform


class _RecordingCommunicator(Communicator):
    """Implements only the mandatory surface; records every call."""

    backend_name = 'recording'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    @classmethod
    def create(cls, rendezvous, rank, world_size, device, **options):
        comm = cls(rank, world_size, device)
        comm.calls.append(('create', rendezvous, options))
        return comm

    def broadcast(self, tensor, src=0, async_op=False):
        self._check_alive()
        self.calls.append(('broadcast', tuple(tensor.shape), tensor.dtype, src))

    def _destroy(self):
        self.calls.append(('destroy', ))


@pytest.fixture
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, '_BACKENDS', dict(registry._BACKENDS))


@pytest.mark.parametrize('platform_cls, backend', [
    (GPU, 'nccl'),
    (Metax, 'nccl'),
    (NPU, 'hccl'),
    (XPU, 'xccl'),
    (MPS, 'gloo'),
])
def test_platform_communicator_backend(platform_cls, backend):
    assert platform_cls.communicator_backend() == backend


def test_gloo_is_registered_and_lazily_resolved(isolated_registry):
    assert 'gloo' in available_backends()
    assert isinstance(registry._BACKENDS['gloo'], str)
    assert get_communicator_cls('gloo') is GlooCommunicator
    assert registry._BACKENDS['gloo'] is GlooCommunicator


def test_unknown_backend_lists_available():
    with pytest.raises(ValueError, match=r"'nope'.*gloo"):
        get_communicator_cls('nope')


def test_default_backend_follows_platform(monkeypatch):
    monkeypatch.setattr(Platform, 'get_platform', staticmethod(lambda platform=None: MPS))
    assert get_communicator_cls() is GlooCommunicator


def test_register_backend_rejects_duplicates_unless_override(isolated_registry):
    register_backend('recording', _RecordingCommunicator)
    with pytest.raises(ValueError, match='already registered'):
        register_backend('recording', GlooCommunicator)
    register_backend('recording', GlooCommunicator, override=True)
    assert get_communicator_cls('recording') is GlooCommunicator


def test_register_backend_accepts_lazy_path(isolated_registry):
    register_backend('lazy', f'{__name__}:_RecordingCommunicator')
    assert get_communicator_cls('lazy') is _RecordingCommunicator


def test_create_communicator_forwards_everything(isolated_registry):
    register_backend('recording', _RecordingCommunicator)
    rdzv = Rendezvous('10.0.0.1', 29600)
    comm = create_communicator(rdzv, 2, 3, 'cpu', backend='recording', socket_env={'A': '1'})
    assert isinstance(comm, _RecordingCommunicator)
    assert (comm.rank, comm.world_size, comm.device) == (2, 3, torch.device('cpu'))
    assert comm.calls == [('create', rdzv, {'socket_env': {'A': '1'}})]


@pytest.mark.parametrize('rank, world_size', [(-1, 2), (2, 2), (0, 0)])
def test_constructor_validates_rank(rank, world_size):
    with pytest.raises(ValueError):
        _RecordingCommunicator(rank, world_size, 'cpu')


def test_optional_ops_default_and_supports():
    comm = _RecordingCommunicator(0, 2, 'cpu')
    with pytest.raises(NotImplementedError, match='recording.*all_reduce'):
        comm.all_reduce(torch.zeros(1))
    assert _RecordingCommunicator.supports('broadcast')
    assert not _RecordingCommunicator.supports('all_reduce')
    assert GlooCommunicator.supports('all_reduce')
    assert not GlooCommunicator.supports('not_an_op')


def test_default_barrier_is_one_element_broadcast_from_rank0():
    comm = _RecordingCommunicator(1, 2, 'cpu')
    comm.barrier()
    assert comm.calls == [('broadcast', (1, ), torch.uint8, 0)]


def test_destroy_is_idempotent_and_final():
    comm = _RecordingCommunicator(0, 2, 'cpu')
    comm.destroy()
    comm.destroy()
    assert comm.calls == [('destroy', )]
    assert comm.destroyed
    for call in (comm.barrier, comm.synchronize, lambda: comm.broadcast(torch.zeros(1))):
        with pytest.raises(RuntimeError, match='destroyed'):
            call()


def test_gloo_single_rank_roundtrip():
    """world_size=1 needs no peers, so the real Gloo path runs in-process."""
    comm = create_communicator(Rendezvous('127.0.0.1', find_free_port()), 0, 1, 'cpu', backend='gloo')
    try:
        t = torch.arange(4, dtype=torch.float32)
        comm.broadcast(t, src=0)
        comm.all_reduce(t, op=ReduceOp.MAX)
        assert torch.equal(t, torch.arange(4, dtype=torch.float32))
        comm.barrier()
    finally:
        comm.destroy()


def test_gloo_rejects_non_cpu_device():
    with pytest.raises(ValueError, match='CPU'):
        GlooCommunicator(0, 1, 'meta', store=None, pg=None)


def test_gloo_validates_src_before_communicating():
    comm = create_communicator(Rendezvous('127.0.0.1', find_free_port()), 0, 1, 'cpu', backend='gloo')
    try:
        with pytest.raises(ValueError, match='src'):
            comm.broadcast(torch.zeros(1), src=1)
    finally:
        comm.destroy()
