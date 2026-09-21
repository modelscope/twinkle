"""Cross-backend tests for the Ray-actor and Redis state backends.

Exercises atomic updates, leases, close semantics, and logical key prefixes:
- read-transform-write returns the new value
- ``transform`` returning ``None`` is a no-op and returns the existing value
- ``ttl`` shapes the new value's expiry
- concurrent atomic updates serialize correctly (no lost write)
- Redis: WATCH-style retries succeed under contention

Redis-backed cases are skipped automatically when the test Redis at
``TWINKLE_TEST_REDIS_URL`` is unreachable.
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import pytest
import pytest_asyncio
import uuid
from typing import Any

from twinkle.server.deployment import twinkle_server_error_handler
from twinkle.server.state.backend.base import ConcurrencyError
from twinkle.server.state.backend.memory_backend import RayActorBackend

REDIS_URL = os.environ.get('TWINKLE_TEST_REDIS_URL', 'redis://localhost:6379/0')


def _can_reach_redis() -> bool:
    try:
        from twinkle.server.state.backend.redis_backend import RedisBackend
    except ImportError:
        return False

    async def _check() -> bool:
        backend = RedisBackend(REDIS_URL)
        try:
            return await backend.health_check()
        except Exception:
            return False
        finally:
            try:
                await backend.close()
            except Exception:
                pass

    try:
        return asyncio.run(_check())
    except Exception:
        return False


_REDIS_AVAILABLE_AT_COLLECTION = _can_reach_redis()
_redis_skip = pytest.mark.skipif(
    not _REDIS_AVAILABLE_AT_COLLECTION,
    reason=f'Redis at {REDIS_URL} unreachable',
)


def _increment_or_init(current: Any | None, *, delta: int) -> int:
    return delta if current is None else int(current) + delta


def _no_op(current: Any | None) -> None:
    return None


def _replace_with(current: Any | None, *, value: Any) -> Any:
    return value


def _redis_backend():
    from twinkle.server.state.backend.redis_backend import RedisBackend

    return RedisBackend(REDIS_URL, key_prefix=f'twinkle-test-{uuid.uuid4().hex[:8]}::')


@pytest.fixture
def memory_backend() -> RayActorBackend:
    return RayActorBackend()


@pytest_asyncio.fixture
async def redis_backend():
    backend = _redis_backend()
    yield backend
    keys = await backend.keys('*')
    for k in keys:
        await backend.delete(k)
    await backend.close()


@pytest_asyncio.fixture(params=['memory', 'redis'])
async def backend_pair(request):
    prefix = f'twinkle-contract-{uuid.uuid4().hex[:8]}::'
    if request.param == 'redis':
        if not _REDIS_AVAILABLE_AT_COLLECTION:
            pytest.skip(f'Redis at {REDIS_URL} unreachable')
        from twinkle.server.state.backend.redis_backend import RedisBackend
        first = RedisBackend(REDIS_URL, key_prefix=prefix)
        second = RedisBackend(REDIS_URL, key_prefix=prefix)
    else:
        first = RayActorBackend(key_prefix=prefix)
        second = RayActorBackend(key_prefix=prefix)
    yield first, second
    for key in await second.keys('*'):
        await second.delete(key)
    await first.close()
    await second.close()


# ---------- Shared backend contract -------------------------------------- #


@pytest.mark.asyncio
async def test_close_releases_handle_without_dropping_shared_state(backend_pair) -> None:
    first, second = backend_pair
    await first.set('shared', {'value': 1})
    await first.close()
    assert await second.get('shared') == {'value': 1}


@pytest.mark.asyncio
async def test_key_prefix_is_hidden_from_logical_keys(backend_pair) -> None:
    first, _ = backend_pair
    await first.set('session::one', 1)
    assert await first.keys('session::*') == ['session::one']


# ---------- Memory backend ----------------------------------------------- #


@pytest.mark.asyncio
async def test_memory_update_atomic_read_transform_write(memory_backend) -> None:
    await memory_backend.set('k', 5)
    result = await memory_backend.update_atomic('k', functools.partial(_increment_or_init, delta=3))
    assert result == 8
    assert await memory_backend.get('k') == 8


@pytest.mark.asyncio
async def test_memory_update_atomic_none_is_noop(memory_backend) -> None:
    await memory_backend.set('k', 42)
    result = await memory_backend.update_atomic('k', _no_op)
    assert result == 42
    assert await memory_backend.get('k') == 42


@pytest.mark.asyncio
async def test_memory_update_atomic_initializes_when_missing(memory_backend) -> None:
    result = await memory_backend.update_atomic('fresh', functools.partial(_increment_or_init, delta=7))
    assert result == 7


@pytest.mark.asyncio
async def test_memory_update_atomic_respects_ttl(memory_backend) -> None:
    await memory_backend.update_atomic('leased', functools.partial(_replace_with, value='holder'), ttl=1)
    assert await memory_backend.get('leased') == 'holder'
    await asyncio.sleep(1.1)
    assert await memory_backend.get('leased') is None


@pytest.mark.asyncio
async def test_memory_update_atomic_concurrent_no_lost_writes(memory_backend) -> None:
    await memory_backend.set('counter', 0)

    async def hammer() -> None:
        for _ in range(50):
            await memory_backend.update_atomic('counter', functools.partial(_increment_or_init, delta=1))

    await asyncio.gather(*(hammer() for _ in range(8)))
    assert await memory_backend.get('counter') == 50 * 8


@pytest.mark.asyncio
async def test_memory_set_nx_with_ttl(memory_backend) -> None:
    assert await memory_backend.set_nx('lease', 'owner', ttl=1) is True
    assert await memory_backend.set_nx('lease', 'other', ttl=1) is False
    await asyncio.sleep(1.1)
    assert await memory_backend.set_nx('lease', 'next', ttl=1) is True


# ---------- Redis backend ------------------------------------------------ #


@_redis_skip
@pytest.mark.asyncio
async def test_redis_update_atomic_read_transform_write(redis_backend) -> None:
    await redis_backend.set('k', 5)
    result = await redis_backend.update_atomic('k', functools.partial(_increment_or_init, delta=3))
    assert result == 8
    assert await redis_backend.get('k') == 8


@_redis_skip
@pytest.mark.asyncio
async def test_redis_update_atomic_none_is_noop(redis_backend) -> None:
    await redis_backend.set('k', 42)
    result = await redis_backend.update_atomic('k', _no_op)
    assert result == 42


@_redis_skip
@pytest.mark.asyncio
async def test_redis_update_atomic_concurrent_serializes(redis_backend) -> None:
    await redis_backend.set('counter', 0)

    async def hammer() -> None:
        for _ in range(25):
            await redis_backend.update_atomic('counter', functools.partial(_increment_or_init, delta=1))

    await asyncio.gather(*(hammer() for _ in range(4)))
    assert await redis_backend.get('counter') == 25 * 4


@_redis_skip
@pytest.mark.asyncio
async def test_redis_update_atomic_respects_ttl(redis_backend) -> None:
    await redis_backend.update_atomic('leased', functools.partial(_replace_with, value='holder'), ttl=1)
    assert await redis_backend.get('leased') == 'holder'
    await asyncio.sleep(1.5)
    assert await redis_backend.get('leased') is None


@_redis_skip
@pytest.mark.asyncio
async def test_redis_retry_exhaustion_maps_to_http_503(redis_backend) -> None:
    import redis
    from starlette.requests import Request

    sync_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    real_key = redis_backend._make_key('always-contended')
    calls = 0

    def force_watch_conflict(current: Any | None) -> int:
        nonlocal calls
        calls += 1
        sync_client.set(real_key, json.dumps(calls))
        return int(current or 0) + 1

    try:
        with pytest.raises(ConcurrencyError) as caught:
            await redis_backend.update_atomic('always-contended', force_watch_conflict)
    finally:
        sync_client.close()

    assert calls == 16
    request = Request({'type': 'http', 'method': 'POST', 'path': '/', 'headers': []})
    request.state.request_id = 'req-contention'
    response = await twinkle_server_error_handler(request, caught.value)
    payload = json.loads(response.body)
    assert response.status_code == 503
    assert payload == {
        'error': "update_atomic exhausted 16 retries on key 'always-contended'",
        'category': 'server',
        'error_code': 503,
        'request_id': 'req-contention',
    }
