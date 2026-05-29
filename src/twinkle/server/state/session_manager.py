# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time

from .backend.base import StateBackend
from .base import BaseManager
from .models import SessionRecord


class SessionManager(BaseManager[SessionRecord]):
    """Manages client sessions.

    Expiry is based on `last_heartbeat`; falls back to `created_at` if no
    heartbeat has been recorded yet.
    """

    def __init__(self, backend: StateBackend, expiration_timeout: float) -> None:
        super().__init__(backend, "session::", SessionRecord, expiration_timeout)

    # ----- Session-specific operations -----

    async def touch(self, session_id: str) -> bool:
        """Update the heartbeat timestamp for a session.

        Returns:
            True if the session exists and was updated, False otherwise.
        """
        record = await self.get(session_id)
        if record is None:
            return False
        record.last_heartbeat = time.time()
        await self.add(session_id, record)
        return True

    async def get_last_heartbeat(self, session_id: str) -> float | None:
        """Return the last heartbeat timestamp, or None if the session does not exist."""
        record = await self.get(session_id)
        if record is None:
            return None
        return record.last_heartbeat

    # ----- Cleanup -----

    async def cleanup_expired(self, cutoff_time: float, **kwargs) -> int:
        """Remove sessions whose last activity is older than cutoff_time.

        Args:
            cutoff_time: Unix timestamp threshold.

        Returns:
            Number of sessions removed.
        """
        all_records = await self.get_all()
        expired_ids = []
        for session_id, record in all_records.items():
            last_activity = record.last_heartbeat
            if last_activity == 0.0:
                last_activity = self._parse_timestamp(record.created_at)
            if last_activity < cutoff_time:
                expired_ids.append(session_id)

        for session_id in expired_ids:
            await self.remove(session_id)

        return len(expired_ids)

    async def get_expired_ids(self, cutoff_time: float) -> list[str]:
        """Return IDs of sessions that would be removed at the given cutoff.

        Used by ServerState to cascade-expire dependent resources before
        actually deleting the sessions.
        """
        all_records = await self.get_all()
        expired_ids = []
        for session_id, record in all_records.items():
            last_activity = record.last_heartbeat
            if last_activity == 0.0:
                last_activity = self._parse_timestamp(record.created_at)
            if last_activity < cutoff_time:
                expired_ids.append(session_id)
        return expired_ids
