from __future__ import annotations

import time
from fnmatch import fnmatch
from typing import Any

from .base import StateBackend


class MemoryBackend(StateBackend):
    """In-memory dictionary-based state backend implementation.

    Uses ``dict[str, tuple[Any, float | None]]`` to store (value, expire_at).
    Expiration is checked lazily during get/exists, suitable for Ray Actor single-threaded model.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _is_expired(self, key: str) -> bool:
        """Check if key is expired. If expired, delete and return True."""
        entry = self._store.get(key)
        if entry is None:
            return True
        _, expire_at = entry
        if expire_at is not None and time.time() >= expire_at:
            del self._store[key]
            return True
        return False

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store key-value pair with optional TTL in seconds."""
        expire_at = (time.time() + ttl) if ttl is not None else None
        self._store[key] = (value, expire_at)

    async def get(self, key: str) -> Any | None:
        """Retrieve value, return None if not found or expired."""
        if self._is_expired(key):
            return None
        value, _ = self._store[key]
        return value

    async def delete(self, key: str) -> None:
        """Delete key, silently ignore if not found."""
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return not self._is_expired(key)

    async def keys(self, pattern: str) -> list[str]:
        """Return all key names matching the pattern. Supports * wildcard."""
        result: list[str] = []
        # Collect expired keys during iteration to avoid modifying dict while iterating
        expired_keys: list[str] = []
        for key, (_, expire_at) in self._store.items():
            if expire_at is not None and time.time() >= expire_at:
                expired_keys.append(key)
                continue
            if fnmatch(key, pattern):
                result.append(key)
        for key in expired_keys:
            del self._store[key]
        return result

    async def count(self, pattern: str) -> int:
        """Count keys matching the pattern."""
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any) -> bool:
        """Set if not exists. Return True if successfully set, False if key already exists."""
        if not self._is_expired(key):
            return False
        self._store[key] = (value, None)
        return True

    async def close(self) -> None:
        """Close backend, clear storage."""
        self._store.clear()

    async def health_check(self) -> bool:
        """Check if backend is healthy and available. Memory backend always returns True."""
        return True
