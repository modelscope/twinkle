from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from typing import Any

from .base import ConcurrencyError, StateBackend

try:
    import redis.asyncio as aioredis
    from redis.exceptions import WatchError

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

# Bound on WATCH+MULTI+EXEC retries. Tuned for the worst real-world contender
# we expect on a single key (cleanup-leader lease + cascading session touches
# at burst); each retry adds an extra round-trip so going higher than ~16
# just delays the error without making it less likely.
_UPDATE_ATOMIC_MAX_RETRIES = 16
# Jittered exponential backoff cap; first retry sleeps ~5 ms, last around 80 ms.
_UPDATE_ATOMIC_BASE_BACKOFF_SECONDS = 0.005
_UPDATE_ATOMIC_MAX_BACKOFF_SECONDS = 0.080


class RedisBackend(StateBackend):
    """Redis-based persistent state backend implementation.

    Uses ``redis.asyncio`` client, values are stored as Redis strings via JSON serialization.
    TTL is managed by Redis native EXPIRE mechanism.
    """

    def __init__(self, redis_url: str, key_prefix: str = '') -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError('redis package required. Install with: pip install redis')
        self._client = aioredis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix

    def _make_key(self, key: str) -> str:
        """Add namespace prefix to key."""
        return f'{self._prefix}{key}' if self._prefix else key

    def _strip_prefix(self, key: str) -> str:
        """Remove namespace prefix from full key."""
        return key[len(self._prefix):] if self._prefix else key

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store key-value pair with optional TTL in seconds."""
        real_key = self._make_key(key)
        data = json.dumps(value)
        if ttl is not None:
            await self._client.set(real_key, data, ex=ttl)
        else:
            await self._client.set(real_key, data)

    async def get(self, key: str) -> Any | None:
        """Retrieve value, return None if not found or expired."""
        real_key = self._make_key(key)
        raw = await self._client.get(real_key)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete(self, key: str) -> None:
        """Delete key, silently ignore if not found."""
        real_key = self._make_key(key)
        await self._client.delete(real_key)

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        real_key = self._make_key(key)
        return bool(await self._client.exists(real_key))

    async def keys(self, pattern: str) -> list[str]:
        """Return all key names matching the pattern using SCAN.

        SCAN is non-blocking — production Redis instances may hold millions of
        keys; the KEYS command walks the keyspace in one step and can stall
        the server for seconds. ``scan_iter`` cursors through in batches.
        """
        real_pattern = self._make_key(pattern)
        out: list[str] = []
        async for key in self._client.scan_iter(match=real_pattern, count=500):
            out.append(self._strip_prefix(key))
        return out

    async def count(self, pattern: str) -> int:
        """Count keys matching the pattern."""
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set if not exists. Return True if successfully set, False if key already exists."""
        real_key = self._make_key(key)
        data = json.dumps(value)
        if ttl is not None:
            result = await self._client.set(real_key, data, nx=True, ex=ttl)
        else:
            result = await self._client.set(real_key, data, nx=True)
        return result is not None

    async def update_atomic(
        self,
        key: str,
        transform: Callable[[Any | None], Any | None],
        ttl: int | None = None,
    ) -> Any | None:
        """WATCH/GET/MULTI/SET/EXEC with bounded retries + jittered backoff on WatchError.

        Under sustained contention each retry waits a jittered exponential
        backoff so concurrent writers spread out and stop trampling the same
        WATCH window. After the retry budget is spent we surface
        :class:`ConcurrencyError`; callers must decide whether to skip (touch
        / heartbeat) or reraise (lease renewal).
        """
        real_key = self._make_key(key)
        for attempt in range(_UPDATE_ATOMIC_MAX_RETRIES):
            async with self._client.pipeline() as pipe:
                try:
                    await pipe.watch(real_key)
                    raw = await pipe.get(real_key)
                    current = json.loads(raw) if raw is not None else None
                    new_value = transform(current)
                    if new_value is None:
                        await pipe.unwatch()
                        return current
                    pipe.multi()
                    if ttl is not None:
                        pipe.set(real_key, json.dumps(new_value), ex=ttl)
                    else:
                        pipe.set(real_key, json.dumps(new_value))
                    await pipe.execute()
                    return new_value
                except WatchError:
                    backoff = min(
                        _UPDATE_ATOMIC_BASE_BACKOFF_SECONDS * (2**attempt),
                        _UPDATE_ATOMIC_MAX_BACKOFF_SECONDS,
                    )
                    await asyncio.sleep(backoff * random.random())
                    continue
        raise ConcurrencyError(f'update_atomic exhausted {_UPDATE_ATOMIC_MAX_RETRIES} retries on key {key!r}')

    async def mget(self, keys: list[str]) -> list[Any | None]:
        """Batch-read via native Redis MGET — single round-trip."""
        if not keys:
            return []
        real_keys = [self._make_key(k) for k in keys]
        raw_values = await self._client.mget(real_keys)
        return [json.loads(v) if v is not None else None for v in raw_values]

    async def close(self) -> None:
        """Close Redis connection."""
        await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if Redis is healthy and available."""
        try:
            return await self._client.ping()
        except Exception:
            return False
