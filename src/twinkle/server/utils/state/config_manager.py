# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any


class ConfigManager:
    """
    Manages key-value configuration entries.

    Configuration entries have no expiry; they persist until explicitly removed
    or cleared.  This manager does not inherit from BaseManager because config
    values are arbitrary Python objects rather than Pydantic models.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    # ----- CRUD -----

    def add(self, key: str, value: Any) -> None:
        """Add or overwrite a configuration value."""
        self._store[key] = value

    def add_or_get(self, key: str, value: Any) -> Any:
        """Add a value if the key does not exist; otherwise return the existing value.

        Args:
            key: Configuration key.
            value: Value to store if the key is absent.

        Returns:
            The existing or newly stored value.
        """
        if key not in self._store:
            self._store[key] = value
        return self._store[key]

    def get(self, key: str) -> Any | None:
        """Return the configuration value for key, or None."""
        return self._store.get(key)

    def pop(self, key: str) -> Any | None:
        """Remove and return the configuration value for key, or None."""
        return self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all configuration entries."""
        self._store.clear()

    def count(self) -> int:
        """Return the number of stored configuration entries."""
        return len(self._store)
