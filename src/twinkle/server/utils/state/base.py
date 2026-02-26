# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar('T', bound=BaseModel)


class BaseManager(ABC, Generic[T]):
    """
    Abstract base class for resource managers.

    Provides common CRUD operations and timestamp parsing.
    Subclasses must implement `cleanup_expired`.
    """

    def __init__(self, expiration_timeout: float) -> None:
        self._store: dict[str, T] = {}
        self.expiration_timeout = expiration_timeout

    # ----- CRUD -----

    def add(self, resource_id: str, record: T) -> None:
        """Store a record under the given ID."""
        self._store[resource_id] = record

    def get(self, resource_id: str) -> T | None:
        """Return the record for the given ID, or None."""
        return self._store.get(resource_id)

    def remove(self, resource_id: str) -> bool:
        """Remove a record by ID. Returns True if it existed."""
        return self._store.pop(resource_id, None) is not None

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._store)

    # ----- Cleanup -----

    @abstractmethod
    def cleanup_expired(self, cutoff_time: float) -> int:
        """
        Remove all records older than cutoff_time.

        Args:
            cutoff_time: Unix timestamp; records with activity before this are removed.

        Returns:
            Number of records removed.
        """

    # ----- Helpers -----

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse an ISO-format timestamp string to a Unix timestamp.

        Falls back to the current time so that unparseable entries are
        never accidentally kept alive forever.
        """
        try:
            return datetime.fromisoformat(timestamp_str).timestamp()
        except (ValueError, AttributeError):
            return time.time()
