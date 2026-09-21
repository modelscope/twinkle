# Copyright (c) ModelScope Contributors. All rights reserved.
"""In-process ``StateBackend`` test double.

Both shipped backends need infrastructure -- ``memory`` starts a detached Ray actor,
``redis`` needs a server -- so state-logic tests that only care about
``FutureManager`` / manager semantics use this dict-backed fake instead. Running on
a single event loop, ``set_nx`` / ``update_atomic`` are atomic simply by not
awaiting between read and write, mirroring the real backends' guarantee.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from fnmatch import fnmatch
from typing import Any

from twinkle.server.state.backend.base import StateBackend


class FakeBackend(StateBackend):
    """Dict-backed backend with TTL support, for tests only."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _is_expired(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return True
        _, expire_at = entry
        if expire_at is not None and time.time() >= expire_at:
            del self._store[key]
            return True
        return False

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = (value, (time.time() + ttl) if ttl is not None else None)

    async def get(self, key: str) -> Any | None:
        if self._is_expired(key):
            return None
        return self._store[key][0]

    async def mget(self, keys: list[str]) -> list[Any | None]:
        return [await self.get(key) for key in keys]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return not self._is_expired(key)

    async def keys(self, pattern: str) -> list[str]:
        return [k for k in list(self._store) if not self._is_expired(k) and fnmatch(k, pattern)]

    async def count(self, pattern: str) -> int:
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any, ttl: int | None = None) -> bool:
        if not self._is_expired(key):
            return False
        await self.set(key, value, ttl)
        return True

    async def update_atomic(
        self,
        key: str,
        transform: Callable[[Any | None], Any | None],
        ttl: int | None = None,
    ) -> Any | None:
        current = await self.get(key)
        updated = transform(current)
        if updated is None:
            return current
        await self.set(key, updated, ttl)
        return updated

    async def flush_all(self) -> None:
        self._store.clear()

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True
