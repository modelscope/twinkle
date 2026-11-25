# Copyright (c) ModelScope Contributors. All rights reserved.
"""Non-device tests for the XCCL backend's registration and contract."""
import importlib
import pytest
from twinkle.comm import Rendezvous, create_communicator, get_communicator_cls


def test_xccl_backend_is_lazily_registered():
    cls = get_communicator_cls('xccl')
    assert cls.__name__ == 'XCCLCommunicator'
    assert cls.backend_name == 'xccl'
    assert cls.supports('broadcast') is True
    # The readiness barrier is a store round-trip rather than the base collective fallback, so
    # it is valid before the group has run anything on the device.
    assert cls.supports('barrier') is True
    # Only the checkpoint engine's broadcast is needed; nothing votes through a collective.
    assert cls.supports('all_reduce') is False


def test_xccl_backend_module_imports_without_constructing_runtime():
    module = importlib.import_module('twinkle.comm.backends.xccl')
    assert hasattr(module, 'XCCLCommunicator')
    assert module.XCCLCommunicator.supports('broadcast')


def test_xccl_backend_takes_no_options():
    cls = get_communicator_cls('xccl')
    assert cls.supported_options == frozenset()
    with pytest.raises(ValueError, match='does not accept options'):
        cls.create(None, 0, 1, 'cpu', unsupported=True)


def test_xccl_rejects_non_cuda_device():
    # Kunlunxin mounts BKCL under torch's cuda namespace, so an XPU rank reports 'cuda';
    # anything else cannot host an XCCL group and must fail before the rendezvous.
    with pytest.raises(ValueError, match='cuda-alike'):
        create_communicator(Rendezvous('127.0.0.1', 1), 0, 1, 'cpu', backend='xccl')
