# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeltaSyncMixin driven by a fake CPU transport (no GPU, ray, zmq or NCCL)."""
from __future__ import annotations

import asyncio

import pytest
import torch

from twinkle.checkpoint_engine.delta.transport.nccl_engine import DeltaShardedCheckpointEngine
from twinkle.checkpoint_engine.delta.transport.pipeline import DeltaSyncMixin
from twinkle.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

METRIC_KEYS = {
    'checkpoint_engine/changed_ratio',
    'checkpoint_engine/changed_elems',
    'checkpoint_engine/payload_mbytes',
    'checkpoint_engine/flushes',
}


class _Socket:

    def __init__(self, log):
        self.log = log
        self.metas = []

    def send_string(self, s, flags=0):
        self.log.append('manifest_topic')

    def send_pyobj(self, meta):
        self.log.append('manifest')
        self.metas.append(meta)


class _Base:

    def __init__(self, *args, bucket_size=1 << 20, is_master=True, rank=0, **kwargs):
        self.bucket_size = bucket_size
        self.rollout_dtype = torch.bfloat16
        self.is_master = is_master
        self.rank = rank
        self.topic = 'topic'
        self.log = []
        self.socket = _Socket(self.log)


class FakeTransport(DeltaSyncMixin, _Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.broadcasts = []
        self.released = []

    @property
    def _wire_device(self):
        return 'cpu'

    def _ensure_wire(self):
        self.log.append('ensure_wire')

    def _broadcast_flush(self, pos_u8, val_u8):
        self.log.append('broadcast')
        self.broadcasts.append((pos_u8, val_u8))

    def _release_after_round(self, phase):
        self.released.append(phase)


class FakeTrainingEngine:

    def __init__(self, deltas):
        self.deltas = deltas  # list of (name, numel, changed idx)
        self.primed = 0

    def get_per_tensor_param(self):
        def gen():
            yield 'a', torch.arange(8, dtype=torch.bfloat16)
            yield 'b', torch.ones(4, 2, dtype=torch.bfloat16)
        return gen(), None

    def prime_delta_snapshots(self):
        self.primed += 1

    def get_per_tensor_param_delta_shard(self):
        def gen():
            for name, shape, idx in self.deltas:
                idx_t = torch.tensor(idx, dtype=torch.int64)
                val = torch.full((len(idx),), 2.0, dtype=torch.bfloat16)
                yield [(name, shape)], 'bfloat16', torch.tensor([len(idx)]), idx_t, val, None
        return gen(), None


def _run(t, eng):
    return asyncio.run(t.send_weights(eng))


def test_seed_then_steady():
    t = FakeTransport()
    eng = FakeTrainingEngine([('a', (8,), [1, 3]), ('b', (4, 2), [0, 7, 5])])
    m = _run(t, eng)
    assert set(m) == METRIC_KEYS
    assert eng.primed == 1 and t.released == ['seed']
    assert t.socket.metas and all(x['encoding'] == 'dense' and x['is_full'] for x in t.socket.metas)
    assert [x['is_last'] for x in t.socket.metas] == [False] * (len(t.socket.metas) - 1) + [True]
    assert all(p is None for p, _ in t.broadcasts)

    t.log.clear()
    t.socket.metas.clear()
    t.broadcasts.clear()
    m = _run(t, eng)
    assert eng.primed == 1 and t.released == ['seed', 'steady']
    assert set(m) == METRIC_KEYS
    assert m['checkpoint_engine/changed_elems'] == 5.0
    assert m['checkpoint_engine/changed_ratio'] == pytest.approx(5 / 16)
    metas = t.socket.metas
    assert len(metas) == 1
    assert metas[0]['encoding'] == 'indices' and metas[0]['is_full'] is False and metas[0]['is_last'] is True
    pos, val = t.broadcasts[0]
    assert pos.dtype == torch.uint8 and pos.numel() == 5 * 4
    assert pos.view(torch.int32).tolist() == [1, 3, 0, 7, 5]
    assert val.numel() == 5 * 2
    assert t.log == ['ensure_wire', 'manifest_topic', 'manifest', 'broadcast']


def test_small_bucket_multiple_flushes():
    t = FakeTransport(bucket_size=8)
    eng = FakeTrainingEngine([('a', (8,), [1, 3]), ('b', (4, 2), [0, 7, 5])])
    _run(t, eng)
    t.socket.metas.clear()
    m = _run(t, eng)
    flags = [x['is_last'] for x in t.socket.metas]
    assert len(flags) > 1 and flags == [False] * (len(flags) - 1) + [True]
    assert m['checkpoint_engine/flushes'] == float(len(flags))


def test_seed_small_bucket_single_last():
    t = FakeTransport(bucket_size=8)
    _run(t, FakeTrainingEngine([]))
    flags = [x['is_last'] for x in t.socket.metas]
    assert len(flags) > 1 and flags.count(True) == 1 and flags[-1] is True


def test_steady_no_change_terminal():
    t = FakeTransport()
    eng = FakeTrainingEngine([('a', (8,), [])])
    _run(t, eng)
    t.socket.metas.clear()
    t.broadcasts.clear()
    t.log.clear()
    m = _run(t, eng)
    assert len(t.socket.metas) == 1
    meta = t.socket.metas[0]
    assert meta['terminal_empty'] is True and meta['is_last'] is True and meta['is_full'] is False
    assert t.broadcasts == []
    assert t.log == ['ensure_wire', 'manifest_topic', 'manifest']
    assert m['checkpoint_engine/changed_elems'] == 0.0


def test_non_master_publishes_nothing():
    t = FakeTransport(is_master=False, rank=-1)
    eng = FakeTrainingEngine([('a', (8,), [1])])
    assert _run(t, eng) is None
    assert _run(t, eng) is None
    assert t.socket.metas == [] and t.broadcasts == [] and t.released == []
    assert eng.primed == 1


def test_unimplemented_hooks():
    class Bare(DeltaSyncMixin, _Base):
        pass

    b = Bare()
    with pytest.raises(NotImplementedError):
        b._wire_device
    with pytest.raises(NotImplementedError):
        b._broadcast_flush(None, torch.zeros(1, dtype=torch.uint8))
    with pytest.raises(NotImplementedError):
        b._release_after_round('seed')
    assert b._ensure_wire() is None


def test_encoding_must_be_indices():
    with pytest.raises(AssertionError):
        FakeTransport(encoding='dense')


def test_nccl_engine_wiring():
    assert issubclass(DeltaShardedCheckpointEngine, DeltaSyncMixin)
    assert issubclass(DeltaShardedCheckpointEngine, NCCLCheckpointEngine)
    mro = DeltaShardedCheckpointEngine.__mro__
    assert mro.index(DeltaSyncMixin) < mro.index(NCCLCheckpointEngine)
    assert DeltaShardedCheckpointEngine.MAX_ENTRY_ELEMS == DeltaSyncMixin.MAX_ENTRY_ELEMS

    e = object.__new__(DeltaShardedCheckpointEngine)
    sent = []
    e._broadcast = sent.append
    pos = torch.zeros(4, dtype=torch.uint8)
    val = torch.zeros(2, dtype=torch.uint8)
    e._broadcast_flush(None, val)
    e._broadcast_flush(pos, val)
    assert sent == [val, pos, val]
    assert e._release_after_round('seed') is None
    assert e._release_after_round('steady') is None
    assert e._ensure_wire() is None
