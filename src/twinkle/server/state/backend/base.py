from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class ConcurrencyError(RuntimeError):
    """Raised by ``StateBackend.update_atomic`` when contention exhausts retries."""


class StateBackend(ABC):
    """Unified interface for state storage backends.

    All state management operations go through this interface, supporting
    multiple backend implementations (memory, file, Redis).
    """

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store key-value pair with optional TTL in seconds."""
        ...

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieve value, return None if not found or expired."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key, silently ignore if not found."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        ...

    @abstractmethod
    async def keys(self, pattern: str) -> list[str]:
        """Return all key names matching the pattern. Supports * wildcard (e.g. 'session::*')."""
        ...

    @abstractmethod
    async def count(self, pattern: str) -> int:
        """Count keys matching the pattern."""
        ...

    @abstractmethod
    async def set_nx(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set if not exists. Return True if successfully set, False if key already exists.

        Optional ``ttl`` (seconds) gives the new value a bounded lifetime; required
        for lease-style leader election (see ``ServerState`` cleanup leader).
        """
        ...

    @abstractmethod
    async def update_atomic(
        self,
        key: str,
        transform: Callable[[Any | None], Any | None],
        ttl: int | None = None,
    ) -> Any | None:
        """Read key, call ``transform(current_value)``, write the result atomically.

        Semantics:
        - If ``transform`` returns ``None`` the key is left unchanged and the
          current value (which may itself be ``None``) is returned.
        - Otherwise the returned value is written (with optional ``ttl``) and
          itself returned.
        - The read/transform/write triple is atomic against concurrent callers
          on the same backend; Redis-backed implementations use WATCH+MULTI+EXEC
          and may raise :class:`ConcurrencyError` after exhausting internal
          retries (default 3).

        ``transform`` must be picklable when running against a Ray-backed
        backend — pass module-level functions wrapped with ``functools.partial``,
        not lambdas or local closures.
        """
        ...

    async def mget(self, keys: list[str]) -> list[Any | None]:
        """Batch-read multiple keys. Returns values in the same order as *keys*.

        Default implementation falls back to serial ``get()`` calls.
        Backends should override for efficiency (e.g. Redis MGET).
        """
        return [await self.get(key) for key in keys]

    @abstractmethod
    async def close(self) -> None:
        """Close backend connection / release resources."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy and available."""
        ...
