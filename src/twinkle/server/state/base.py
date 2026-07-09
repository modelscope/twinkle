# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Generic, TypeVar

from twinkle.server.state.backend.base import StateBackend
from twinkle.utils import get_logger

T = TypeVar('T', bound=BaseModel)
logger = get_logger()


class BaseManager(ABC, Generic[T]):
    """Abstract base class for resource managers using StateBackend.

    Provides common async CRUD operations and timestamp parsing.
    Subclasses must implement `cleanup_expired`.
    """

    def __init__(self, backend: StateBackend, key_prefix: str, record_type: type[T], expiration_timeout: float):
        self._backend = backend
        self._prefix = key_prefix  # e.g. 'session::', 'model::', 'sampling::', 'future::'
        self._record_type = record_type
        self.expiration_timeout = expiration_timeout

    def _make_key(self, resource_id: str) -> str:
        return f'{self._prefix}{resource_id}'

    def _strip_prefix(self, key: str) -> str:
        return key[len(self._prefix):]

    # ----- CRUD -----

    async def add(self, resource_id: str, record: T) -> None:
        """Store a record in the backend."""
        await self._backend.set(self._make_key(resource_id), record.model_dump())

    async def get(self, resource_id: str) -> T | None:
        """Retrieve a record by ID."""
        data = await self._backend.get(self._make_key(resource_id))
        if data is None:
            return None
        return self._record_type.model_validate(data)

    async def remove(self, resource_id: str) -> bool:
        """Remove a record. Returns True if it existed."""
        key = self._make_key(resource_id)
        exists = await self._backend.exists(key)
        if exists:
            await self._backend.delete(key)
        return exists

    async def count(self) -> int:
        """Count all records managed by this manager."""
        return await self._backend.count(f'{self._prefix}*')

    async def keys(self) -> list[str]:
        """Get all resource IDs (without prefix)."""
        raw_keys = await self._backend.keys(f'{self._prefix}*')
        return [self._strip_prefix(k) for k in raw_keys]

    async def get_all(self) -> dict[str, T]:
        """Load all records from backend. Uses batch mget for efficiency."""
        all_keys = await self._backend.keys(f'{self._prefix}*')
        if not all_keys:
            return {}
        values = await self._backend.mget(all_keys)
        result = {}
        for key, data in zip(all_keys, values):
            if data is not None:
                resource_id = self._strip_prefix(key)
                result[resource_id] = self._record_type.model_validate(data)
        return result

    # ----- Cleanup -----

    @abstractmethod
    async def cleanup_expired(self, cutoff_time: float, **kwargs) -> int:
        """Remove all records older than cutoff_time.

        Args:
            cutoff_time: Unix timestamp; records with activity before this are removed.

        Returns:
            Number of records removed.
        """
        ...

    # ----- Helpers -----

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse an ISO-format timestamp string to a Unix timestamp.

        Falls back to the current time so that unparseable entries are
        never accidentally kept alive forever.
        """
        try:
            dt = datetime.fromisoformat(timestamp_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError, AttributeError):
            logger.warning('Failed to parse timestamp %r, treating as current time', timestamp_str)
            return time.time()
