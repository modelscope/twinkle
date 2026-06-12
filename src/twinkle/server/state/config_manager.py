# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any

from .backend import StateBackend

# Key prefix used to namespace configuration entries inside the backend.
_CONFIG_PREFIX = 'config::'
_CONFIG_PATTERN = f'{_CONFIG_PREFIX}*'


class ConfigManager:
    """
    Manages key-value configuration entries via a :class:`StateBackend`.

    Configuration entries have no expiry; they persist until explicitly removed
    or cleared.  This manager does not inherit from BaseManager because config
    values are arbitrary Python objects rather than Pydantic models, and all
    storage is delegated to the injected backend.

    Methods are ``async`` because :class:`StateBackend` operations are async.
    Atomicity for read-modify-write entries comes from the backend's own
    primitives (``set_nx`` / ``update_atomic``), not from any single-threaded
    actor assumption — each worker holds its own ``ConfigManager`` bound to the
    shared backend, so no additional locking is layered on top of the backend.
    """

    def __init__(self, backend: StateBackend) -> None:
        self._backend = backend

    @staticmethod
    def _make_key(key: str) -> str:
        return f'{_CONFIG_PREFIX}{key}'

    # ----- CRUD -----

    async def add(self, key: str, value: Any) -> None:
        """Add or overwrite a configuration value."""
        await self._backend.set(self._make_key(key), value)

    async def add_or_get(self, key: str, value: Any) -> Any:
        """Add a value if the key does not exist; otherwise return the existing value.

        Args:
            key: Configuration key.
            value: Value to store if the key is absent.

        Returns:
            The existing or newly stored value.
        """
        backend_key = self._make_key(key)
        existing = await self._backend.get(backend_key)
        if existing is not None:
            return existing
        # Use set_nx for atomicity within a single backend; if another
        # writer already populated the key we return the winning value.
        if await self._backend.set_nx(backend_key, value):
            return value
        return await self._backend.get(backend_key)

    async def get(self, key: str) -> Any | None:
        """Return the configuration value for key, or None."""
        return await self._backend.get(self._make_key(key))

    async def pop(self, key: str) -> Any | None:
        """Remove and return the configuration value for key, or None.

        Note: get-then-delete is not atomic (TOCTOU); a concurrent pop may
        return the same value twice. This is acceptable for config entries
        where double-return is harmless.
        """
        backend_key = self._make_key(key)
        value = await self._backend.get(backend_key)
        if value is None:
            return None
        await self._backend.delete(backend_key)
        return value

    async def clear(self) -> None:
        """Remove all configuration entries."""
        keys = await self._backend.keys(_CONFIG_PATTERN)
        for backend_key in keys:
            await self._backend.delete(backend_key)

    async def count(self) -> int:
        """Return the number of stored configuration entries."""
        return await self._backend.count(_CONFIG_PATTERN)
