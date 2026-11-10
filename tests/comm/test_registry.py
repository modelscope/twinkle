# Copyright (c) ModelScope Contributors. All rights reserved.
"""Registry and platform mapping of ``twinkle.comm`` (no process spawning)."""
import subprocess
import sys
from datetime import timedelta

import pytest
import torch

from twinkle.comm import (Communicator, available_backends, create_communicator, get_communicator_cls,
                          register_backend, unregister_backend)
from twinkle.comm import registry
from twinkle.comm.backends.gloo import GlooCommunicator
from twinkle.utils.platforms import GPU, MPS, NPU, XPU, Metax, Platform


class _RecordingCommunicator(Communicator):

    backend_name = 'recording'
    supported_options = frozenset({'socket_env'})

    @classmethod
    def _create(cls, rendezvous, rank, world_size, device, *, timeout, **options):
        comm = cls(rank, world_size, device)
        comm.created_with = (rendezvous, timeout, options)
        return comm

    def _broadcast(self, tensor, src, async_op):
        return None

    def _destroy(self):
        pass


class _NotACommunicator:
    pass


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


def test_base_platform_declares_no_backend():
    with pytest.raises(NotImplementedError):
        Platform.communicator_backend()


def test_gloo_is_registered_and_lazily_resolved(isolated_registry):
    assert 'gloo' in available_backends()
    register_backend('gloo', 'twinkle.comm.backends.gloo:GlooCommunicator', override=True)
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


def test_register_backend_validates_target(isolated_registry):
    with pytest.raises(TypeError, match='Communicator subclass'):
        register_backend('bad', _NotACommunicator)
    with pytest.raises(ValueError, match='module:ClassName'):
        register_backend('bad', 'no_colon_here')


def test_lazy_path_resolution_errors_name_the_backend(isolated_registry):
    register_backend('missing', 'twinkle_no_such_module:Thing')
    with pytest.raises(ImportError, match="'missing'"):
        get_communicator_cls('missing')
    register_backend('wrong_type', f'{__name__}:_NotACommunicator')
    with pytest.raises(TypeError, match="'wrong_type'"):
        get_communicator_cls('wrong_type')


def test_register_backend_accepts_lazy_path(isolated_registry):
    register_backend('lazy', f'{__name__}:_RecordingCommunicator')
    assert get_communicator_cls('lazy') is _RecordingCommunicator


def test_unregister_backend(isolated_registry):
    register_backend('recording', _RecordingCommunicator)
    unregister_backend('recording')
    assert 'recording' not in available_backends()
    with pytest.raises(ValueError):
        unregister_backend('recording')


def test_create_communicator_forwards_everything(isolated_registry):
    from twinkle.comm import Rendezvous

    register_backend('recording', _RecordingCommunicator)
    rdzv = Rendezvous('10.0.0.1', 29600, group_name='ckpt')
    comm = create_communicator(rdzv, 2, 3, 'cpu', backend='recording', timeout=timedelta(seconds=7),
                               socket_env={'A': '1'})
    assert isinstance(comm, _RecordingCommunicator)
    assert (comm.rank, comm.world_size, comm.device) == (2, 3, torch.device('cpu'))
    assert comm.created_with == (rdzv, timedelta(seconds=7), {'socket_env': {'A': '1'}})


def test_create_communicator_rejects_unknown_options(isolated_registry):
    from twinkle.comm import Rendezvous

    register_backend('recording', _RecordingCommunicator)
    with pytest.raises(ValueError, match=r"\['socket_evn'\]"):
        create_communicator(Rendezvous('h', 1), 0, 1, 'cpu', backend='recording', socket_evn={})


def test_import_needs_no_accelerator_stack():
    code = ('import sys, twinkle.comm, twinkle.comm.backends.gloo\n'
            "leaked = [m for m in ('torch_npu', 'vllm', 'vllm_ascend') if m in sys.modules]\n"
            'assert not leaked, leaked\n')
    subprocess.run([sys.executable, '-c', code], check=True, timeout=120)
