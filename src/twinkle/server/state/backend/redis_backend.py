from __future__ import annotations

import json
from typing import Any

from .base import StateBackend

try:
    import redis.asyncio as aioredis

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


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
        return f"{self._prefix}{key}" if self._prefix else key

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
        """Return all key names matching the pattern. Supports * wildcard.

        Note: For high key volumes in production, consider using SCAN to avoid blocking.
        """
        real_pattern = self._make_key(pattern)
        raw_keys = await self._client.keys(real_pattern)
        return [self._strip_prefix(k) for k in raw_keys]

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

    async def close(self) -> None:
        """Close Redis connection."""
        await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if Redis is healthy and available."""
        try:
            return await self._client.ping()
        except Exception:
            return False
