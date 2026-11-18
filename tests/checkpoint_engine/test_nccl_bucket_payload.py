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

    monkeypatch.setattr(nccl, '_pg_broadcast', broadcast)

    def engine(rank):
        engine = nccl.NCCLCheckpointEngine(bucket_size=16)
        engine.rank = engine._pg = rank
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
