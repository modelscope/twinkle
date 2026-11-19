# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the full-sync bucket protocol with CPU buffers."""

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from twinkle.checkpoint_engine import nccl_checkpoint_engine as nccl


@pytest.mark.parametrize('sizes, broadcast_sizes', [([], []), ([0], []), ([4], [16]), ([3, 2, 3], [12, 8, 12])])
def test_nccl_broadcast_uses_payload_length(monkeypatch, sizes, broadcast_sizes):
    manifests, payloads = deque(), deque()
    sent_sizes, received_sizes = [], []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(nccl, 'send_auth_pyobj', lambda socket, metadata, *args, **kwargs: manifests.append(metadata))
    monkeypatch.setattr(nccl, 'recv_auth_pyobj', lambda *args, **kwargs: manifests.popleft())

    def broadcast(rank, bucket, src):
        if rank == 0:
            sent_sizes.append(bucket.numel())
            payloads.append(bucket.clone())
        else:
            received_sizes.append(bucket.numel())
            payload = payloads.popleft()
            assert payload.shape == bucket.shape
            bucket.copy_(payload)

    def engine(rank):
        engine = nccl.NCCLCheckpointEngine(bucket_size=16)
        engine.rank = rank
        engine._group_broadcast = lambda bucket: broadcast(rank, bucket, src=0)
        engine.send_buf = torch.empty(16, dtype=torch.uint8)
        engine.recv_buf = torch.empty_like(engine.send_buf)
        engine.socket = SimpleNamespace(send_string=lambda *args, **kwargs: None, recv_string=lambda: engine.topic)
        return engine

    weights = {str(i): torch.arange(n, dtype=torch.float32) + i for i, n in enumerate(sizes)}

    async def roundtrip():
        await engine(0).send_weights(iter(weights.items()))
        return {k: v.clone() async for k, v in engine(1).receive_weights()}

    received = asyncio.run(roundtrip())
    assert sent_sizes == received_sizes == broadcast_sizes
    assert received.keys() == weights.keys()
    for name in weights:
        torch.testing.assert_close(received[name], weights[name], rtol=0, atol=0)
    assert not manifests and not payloads


def _protocol_engine(monkeypatch, sizes):
    """Wire the real bucket protocol over CPU buffers; return (weights, roundtrip)."""
    manifests, payloads = deque(), deque()
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(nccl, 'send_auth_pyobj', lambda socket, metadata, *args, **kwargs: manifests.append(metadata))
    monkeypatch.setattr(nccl, 'recv_auth_pyobj', lambda *args, **kwargs: manifests.popleft())

    def broadcast(rank, bucket, src):
        if rank == 0:
            payloads.append(bucket.clone())
        else:
            bucket.copy_(payloads.popleft())

    def engine(rank):
        e = nccl.NCCLCheckpointEngine(bucket_size=16)
        e.rank = rank
        e._group_broadcast = lambda bucket: broadcast(rank, bucket, src=0)
        e.send_buf = torch.empty(16, dtype=torch.uint8)
        e.recv_buf = torch.empty_like(e.send_buf)
        e.socket = SimpleNamespace(send_string=lambda *a, **k: None, recv_string=lambda: e.topic)
        return e

    weights = {str(i): torch.arange(n, dtype=torch.float32) + i for i, n in enumerate(sizes)}

    async def roundtrip():
        await engine(0).send_weights(iter(weights.items()))
        return {k: v.clone() async for k, v in engine(1).receive_weights()}

    return weights, roundtrip


def test_receive_fingerprint_records_every_tensor(monkeypatch, tmp_path):
    # Multi-bucket ([3,2,3] -> buckets [12,8,12]): the recv probe must cover BOTH yield loops
    # (mid-bucket and final-bucket), with side='recv' (the field the reconciler reads), via='engine',
    # and bits equal to the published weight -- not a fabricated or partial record.
    import json

    from twinkle.utils.torch_utils import bits_fingerprint
    monkeypatch.setenv('TWINKLE_WEIGHT_SYNC_FINGERPRINT', str(tmp_path))
    weights, roundtrip = _protocol_engine(monkeypatch, [3, 2, 3])
    asyncio.run(roundtrip())
    recv_file = tmp_path / 'weightsync.engine.recv.rank1.jsonl'
    assert recv_file.exists(), 'receive-side fingerprint file missing'
    rows = [json.loads(l) for l in recv_file.read_text().splitlines()]
    assert len(rows) == len(weights)
    assert {r['side'] for r in rows} == {'recv'}
    assert {r['via'] for r in rows} == {'engine'}
    assert [r['seq'] for r in rows] == [0, 1, 2]
    for r in rows:
        assert r['bits'] == bits_fingerprint(weights[r['name']])
        assert r['shape'] == list(weights[r['name']].shape)


def test_receive_fingerprint_off_writes_nothing(monkeypatch, tmp_path):
    # Probe disabled (no env): the receive path must not create any fingerprint file.
    monkeypatch.delenv('TWINKLE_WEIGHT_SYNC_FINGERPRINT', raising=False)
    _, roundtrip = _protocol_engine(monkeypatch, [3, 2, 3])
    asyncio.run(roundtrip())
    assert not list(tmp_path.glob('weightsync.*'))


def test_receive_fingerprint_empty_batch(monkeypatch, tmp_path):
    # No tensors published -> no recv rows (a fabricated row would pair against nothing).
    monkeypatch.setenv('TWINKLE_WEIGHT_SYNC_FINGERPRINT', str(tmp_path))
    _, roundtrip = _protocol_engine(monkeypatch, [])
    asyncio.run(roundtrip())
    recv_file = tmp_path / 'weightsync.engine.recv.rank1.jsonl'
    assert not recv_file.exists() or recv_file.read_text() == ''
