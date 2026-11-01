# Copyright (c) ModelScope Contributors. All rights reserved.
"""Read retry is bounded, request-local, and never replays writes."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError

from twinkle.server.state.backend.redis_backend import RedisBackend


@pytest.mark.asyncio
async def test_stale_read_retries_without_blocking_another_reader(monkeypatch):
    backend = RedisBackend('redis://127.0.0.1:1')
    entered, resume = asyncio.Event(), asyncio.Event()
    attempts = []

    async def read(key):
        attempts.append(key)
        if key == 'stale' and attempts.count(key) == 1:
            entered.set()
            await resume.wait()
            raise ConnectionError('stale connection')
        return '"' + key + '"'

    backend._client.get = read
    monkeypatch.setattr('twinkle.server.state.backend.redis_backend.asyncio.sleep', AsyncMock())
    try:
        first = asyncio.create_task(backend.get('stale'))
        await entered.wait()
        assert await backend.get('healthy') == 'healthy'
        resume.set()
        assert await first == 'stale'
        assert attempts == ['stale', 'healthy', 'stale']
    finally:
        resume.set()
        await backend.close()


@pytest.mark.asyncio
async def test_cancellation_is_not_retried():
    backend = RedisBackend('redis://127.0.0.1:1')
    backend._client.get = AsyncMock(side_effect=asyncio.CancelledError())
    try:
        with pytest.raises(asyncio.CancelledError):
            await backend.get('key')
        backend._client.get.assert_awaited_once()
    finally:
        await backend.close()
