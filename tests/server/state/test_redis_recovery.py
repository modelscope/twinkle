# Copyright (c) ModelScope Contributors. All rights reserved.
"""Read-only Redis reconnects are bounded; uncertain writes are never replayed."""
import shutil
from unittest.mock import AsyncMock, Mock

import pytest

redis = pytest.importorskip('redis')
ConnectionError = redis.exceptions.ConnectionError

from twinkle.server.state.backend.redis_backend import RedisBackend
from tests.server.integration.on_demand_training.managed_redis import ManagedRedis


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    yield  # These backend tests need no Ray cluster.


@pytest.mark.asyncio
async def test_stale_read_is_retried_but_write_is_not():
    backend = RedisBackend('redis://127.0.0.1:1/0')
    try:
        backend._client.get = AsyncMock(side_effect=[ConnectionError('stale socket'), '{"value": 1}'])
        assert await backend.get('owned') == {'value': 1}
        assert backend._client.get.await_count == 2
        backend._client.set = AsyncMock(side_effect=ConnectionError('lost acknowledgement'))
        with pytest.raises(ConnectionError):
            await backend.set('owned', {'value': 2})
        assert backend._client.set.await_count == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_persistent_outage_does_not_look_like_missing_state():
    backend = RedisBackend('redis://127.0.0.1:1/0')
    try:
        backend._client.get = AsyncMock(side_effect=ConnectionError('offline'))
        with pytest.raises(ConnectionError):
            await backend.get('controller-active')
        assert backend._client.get.await_count == 3
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which('redis-server'), reason='redis-server executable required')
async def test_real_restart_recovers_old_pooled_readers(tmp_path):
    managed = ManagedRedis(tmp_path)
    managed.start()
    first, stale, atomic = (RedisBackend(managed.url) for _ in range(3))
    try:
        await first.set('owner', {'epoch': 'original'})
        assert await stale.get('owner') == {'epoch': 'original'}
        assert await atomic.get('owner') == {'epoch': 'original'}
        managed.stop()
        with pytest.raises(ConnectionError):
            await first.get('owner')
        managed.start()
        assert await stale.get('owner') == {'epoch': 'original'}
        assert await stale.exists('owner')
        assert await first.mget(['owner', 'missing']) == [{'epoch': 'original'}, None]
        assert await stale.health_check()
        assert await atomic.update_atomic('counter', lambda value: int(value or 0) + 1) == 1
        assert await atomic.get('counter') == 1
    finally:
        await first.close()
        await stale.close()
        await atomic.close()
        managed.stop()


class PipelineStub:
    def __init__(self):
        self.watch = AsyncMock()
        self.get = AsyncMock(return_value='0')
        self.multi = Mock()
        self.set = Mock()
        self.execute = AsyncMock(return_value=[])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_transport_watch_error_after_exec_is_never_replayed():
    backend = RedisBackend('redis://127.0.0.1:1/0')
    pipe = PipelineStub()
    pipe.execute.side_effect = redis.exceptions.WatchError(
        'A ConnectionError occurred while watching one or more keys')
    backend._client.pipeline = lambda: pipe
    try:
        with pytest.raises(ConnectionError, match='outcome could not be confirmed'):
            await backend.update_atomic('counter', lambda value: value + 1)
        assert pipe.execute.await_count == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_connection_loss_before_exec_is_safe_to_retry():
    backend = RedisBackend('redis://127.0.0.1:1/0')
    pipe = PipelineStub()
    pipe.watch.side_effect = [ConnectionError('stale'), None]
    backend._client.pipeline = lambda: pipe
    try:
        assert await backend.update_atomic('counter', lambda value: value + 1) == 1
        assert pipe.watch.await_count == 2 and pipe.execute.await_count == 1
    finally:
        await backend.close()
