# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeltaHCCLCheckpointEngine with fake torch.npu, communicator and sockets (CPU only)."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from twinkle.checkpoint_engine.delta.transport import hccl_engine as mod
from twinkle.checkpoint_engine.delta.transport.hccl_engine import DeltaHCCLCheckpointEngine
from twinkle.checkpoint_engine.hccl_checkpoint_engine import MasterMetadata

from .test_delta_pipeline import FakeTrainingEngine

META = MasterMetadata(zmq_ip='1.2.3.4', zmq_port=111, dist_ip='1.2.3.4', dist_port=222)


class Log(list):
    pass


class FakeSocket:

    def __init__(self, log):
        self.log = log
        self.metas = []
        self.closed = False
        self.context = SimpleNamespace(term=lambda: log.append('ctx_term'))

    def send_string(self, s, flags=0):
        pass

    def send_pyobj(self, meta):
        self.log.append('manifest')
        self.metas.append(meta)

    def close(self, linger=None):
        self.closed = True


class FakeComm:

    def __init__(self, log):
        self.log = log
        self.comm = 'handle'
        self.destroyed = []

    def broadcast(self, tensor, src):
        assert src == 0 and tensor.dtype == torch.uint8
        self.log.append(('bcast', tensor.numel()))

    def destroyComm(self, handle):
        self.destroyed.append(handle)


@pytest.fixture
def env(monkeypatch):
    log = Log()
    calls = []
    comms = []
    counters = {'empty_cache': 0, 'sync': 0}

    def fake_init(**kwargs):
        calls.append(kwargs)
        log.append('build_comm')
        comm = FakeComm(log)
        comms.append(comm)
        return comm

    def sync():
        counters['sync'] += 1
        log.append('sync')

    def empty_cache():
        counters['empty_cache'] += 1

    monkeypatch.setattr(torch, 'npu',
                        SimpleNamespace(current_device=lambda: 0, synchronize=sync, empty_cache=empty_cache),
                        raising=False)
    monkeypatch.setattr(mod, 'stateless_init_process_group', fake_init)
    monkeypatch.setattr(mod, 'find_node_ip', lambda: '1.2.3.4')
    ports = iter([111, 222])
    monkeypatch.setattr(mod, 'find_free_port', lambda: next(ports))
    monkeypatch.setattr(mod, 'start_publisher', lambda ip, port: FakeSocket(log))
    # the pipeline broadcasts from `_wire_device`; keep tensors on CPU for the fake
    monkeypatch.setattr(DeltaHCCLCheckpointEngine, '_wire_device', property(lambda self: 'cpu'))
    return SimpleNamespace(log=log, calls=calls, comms=comms, counters=counters)


def make(master=True, rank=0, **kw):
    e = DeltaHCCLCheckpointEngine(bucket_size=1 << 20, batch_gather=4, **kw)
    e.is_master = master
    return e


def run(e, eng):
    return asyncio.run(e.send_weights(eng))


def test_import_without_npu_modules():
    code = ("import sys, twinkle.checkpoint_engine.delta.transport.hccl_engine as m;"
            "print('torch_npu' in sys.modules, 'vllm_ascend' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ['False', 'False']


def test_rebuild_group_rejected():
    with pytest.raises(NotImplementedError, match='rebuild_group=True is not supported'):
        DeltaHCCLCheckpointEngine(rebuild_group=True)
    e = DeltaHCCLCheckpointEngine()
    assert e.rebuild_group is False and e.wire_format == 'delta_flush' and e.topic == 'bucket_metadata'


def test_wire_device_is_current_npu(monkeypatch):
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_device=lambda: 3), raising=False)
    # `torch.device('npu', n)` only exists when the backend is registered; check the call shape
    captured = []
    monkeypatch.setattr(mod.torch, 'device', lambda *a: captured.append(a) or a)
    assert DeltaHCCLCheckpointEngine().__class__._wire_device.fget(None) == ('npu', 3)
    assert DeltaHCCLCheckpointEngine.__mro__[1].__name__ == 'DeltaSyncMixin'


def test_prepare_master_idempotent(env):
    e = make()
    first = e.prepare()
    sock = e.socket
    assert first == META
    assert e.prepare() == META and e.socket is sock


def test_prepare_non_master(env):
    e = make(master=False, rank=-1)
    assert e.prepare() is None and e.socket is None
    assert e.hccl_env is None


def test_prepare_publishes_effective_hccl_env(env, monkeypatch):
    from twinkle.utils.platforms.npu import _derive_hccl_socket_env_defaults
    for key in _derive_hccl_socket_env_defaults(0):
        monkeypatch.delenv(key, raising=False)
    # Already pinned by the FSDP process group (from MASTER_PORT): must be published as-is.
    monkeypatch.setenv('HCCL_IF_BASE_PORT', '12345')
    e = make()
    e.prepare()
    derived = _derive_hccl_socket_env_defaults(222)
    assert e.hccl_env == dict(derived, HCCL_IF_BASE_PORT='12345')
    assert 'HCCL_HOST_SOCKET_PORT_RANGE' not in os.environ  # read-only: prepare() must not mutate env
    e.close()
    assert e.hccl_env is None


def test_init_process_group_builds_no_communicator(env):
    e = make()
    e.prepare()
    e.init_process_group(0, 3, META)
    assert (e.rank, e._group_world_size, e._rendezvous) == (0, 3, META)
    other = make(master=False)
    other.init_process_group(-1, 3, META)
    assert other.rank == -1 and other._rendezvous is None
    assert env.calls == [] and e._comm is None


def test_end_to_end_rounds(env):
    e = make()
    e.prepare()
    e.init_process_group(0, 3, META)
    eng = FakeTrainingEngine([('a', (8,), [1, 3]), ('b', (4, 2), [0, 7, 5])])

    run(e, eng)  # seed: values-only flush, no positions
    assert env.calls == [dict(master_address='1.2.3.4', master_port=222, rank=0, world_size=3, device=0,
                              backend='hccl')]
    # comm built before the first manifest, then manifest -> values broadcast only -> synchronize
    assert [x[0] if isinstance(x, tuple) else x for x in env.log] == ['build_comm', 'manifest', 'bcast', 'sync']
    assert env.log[2][1] == (8 + 8) * 2  # a(8) + b(4x2) bf16 values
    assert env.counters['empty_cache'] == 1
    assert [m['encoding'] for m in e.socket.metas] == ['dense']

    env.log.clear()
    run(e, eng)  # steady: positions then values
    assert len(env.calls) == 1  # communicator reused
    assert env.log[0] == 'manifest'
    assert [x[0] if isinstance(x, tuple) else x for x in env.log] == ['manifest', 'bcast', 'bcast', 'sync']
    assert env.log[1][1] == 5 * 4 and env.log[2][1] == 5 * 2  # positions (uint8 bytes) then values
    assert env.counters['empty_cache'] == 2
    assert env.counters['sync'] == 2


def test_non_master_never_builds_or_publishes(env):
    e = make(master=False, rank=-1)
    e.prepare()
    e.init_process_group(-1, 3, META)
    eng = FakeTrainingEngine([('a', (8,), [1])])
    assert run(e, eng) is None and run(e, eng) is None
    assert env.calls == [] and env.log == [] and env.counters['empty_cache'] == 0


def test_terminal_only_round_builds_comm_but_broadcasts_nothing(env):
    e = make()
    e.prepare()
    e.init_process_group(0, 2, META)
    eng = FakeTrainingEngine([('a', (8,), [])])
    # skip the seed publish so the comm is still unbuilt at the terminal publish
    e._shard_seeded = True
    run(e, eng)
    assert env.log == ['build_comm', 'manifest']
    assert e.socket.metas[0]['terminal_empty'] is True
    assert len(env.calls) == 1


def test_close(env):
    e = make()
    e.close()  # safe before init
    e.prepare()
    e.init_process_group(0, 2, META)
    run(e, FakeTrainingEngine([('a', (8,), [1])]))
    comm, sock = env.comms[0], e.socket
    e.close()
    assert comm.destroyed == ['handle'] and sock.closed and 'ctx_term' in env.log
    assert e._comm is None and e.socket is None and e.rank is None
    e.close()  # idempotent
    assert comm.destroyed == ['handle']
    assert e._shard_seeded is False
