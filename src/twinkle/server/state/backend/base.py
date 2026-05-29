from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StateBackend(ABC):
    """Unified interface for state storage backends.

    All state management operations go through this interface, supporting multiple backend implementations (memory, file, Redis).
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
    async def set_nx(self, key: str, value: Any) -> bool:
        """Set if not exists. Return True if successfully set, False if key already exists."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close backend connection / release resources."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy and available."""
        ...
