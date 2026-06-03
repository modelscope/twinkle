"""``RedisBackend.keys`` must use SCAN, not KEYS — verify it still returns
the right result set at scale.

KEYS is blocking against a production-size keyspace; SCAN cursors through.
The migration to ``scan_iter`` keeps the same semantics — this test pins
that contract by writing 10k keys and asserting we get all of them back
through the pattern filter.
"""
from __future__ import annotations

import asyncio
import os
import pytest
import uuid

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


pytestmark = pytest.mark.skipif(
    not _can_reach_redis(),
    reason=f'Redis at {REDIS_URL} unreachable',
)


@pytest.mark.asyncio
async def test_keys_returns_all_matches_under_load() -> None:
    from twinkle.server.state.backend.redis_backend import RedisBackend

    prefix = f'twinkle-test-{uuid.uuid4().hex[:8]}::'
    backend = RedisBackend(REDIS_URL, key_prefix=prefix)
    try:
        # 10k keys is enough to push past SCAN's default page size of 10. Keep
        # the batch size well under the redis-py default connection pool size
        # (50) so the pool never stalls.
        n = 10_000
        batch = 20
        for offset in range(0, n, batch):
            await asyncio.gather(*[backend.set(f'item::{i}', i) for i in range(offset, offset + batch)])
        await backend.set('other::1', 'x')
        keys = await backend.keys('item::*')
        assert len(keys) == n
        assert all(k.startswith('item::') for k in keys)
    finally:
        # Quick teardown via SCAN+DELETE without re-using backend.keys() (which
        # holds a connection through the whole iter). Just FLUSHDB the test
        # prefix one batch at a time.
        leftover_keys = []
        async for k in backend._client.scan_iter(match=f'{prefix}*', count=500):
            leftover_keys.append(k)
        for offset in range(0, len(leftover_keys), batch):
            await backend._client.delete(*leftover_keys[offset:offset + batch])
        await backend.close()
