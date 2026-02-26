# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time

from .base import BaseManager
from .models import SessionRecord


class SessionManager(BaseManager[SessionRecord]):
    """
    Manages client sessions.

    Expiry is based on `last_heartbeat`; falls back to `created_at` if no
    heartbeat has been recorded yet.
    """

    # ----- Session-specific operations -----

    def touch(self, session_id: str) -> bool:
        """Update the heartbeat timestamp for a session.

        Returns:
            True if the session exists and was updated, False otherwise.
        """
        record = self._store.get(session_id)
        if record is None:
            return False
        record.last_heartbeat = time.time()
        return True

    def get_last_heartbeat(self, session_id: str) -> float | None:
        """Return the last heartbeat timestamp, or None if the session does not exist."""
        record = self._store.get(session_id)
        if record is None:
            return None
        return record.last_heartbeat

    # ----- Cleanup -----

    def cleanup_expired(self, cutoff_time: float) -> int:
        """Remove sessions whose last activity is older than cutoff_time.

        Args:
            cutoff_time: Unix timestamp threshold.

        Returns:
            Number of sessions removed.
        """
        expired_ids = []
        for session_id, record in self._store.items():
            last_activity = record.last_heartbeat
            if last_activity == 0.0:
                # Fallback: parse created_at
                last_activity = self._parse_timestamp(record.created_at)
            if last_activity < cutoff_time:
                expired_ids.append(session_id)

        for session_id in expired_ids:
            del self._store[session_id]

        return len(expired_ids)

    def get_expired_ids(self, cutoff_time: float) -> list[str]:
        """Return IDs of sessions that would be removed at the given cutoff.

        Used by ServerState to cascade-expire dependent resources before
        actually deleting the sessions.
        """
        expired_ids = []
        for session_id, record in self._store.items():
            last_activity = record.last_heartbeat
            if last_activity == 0.0:
                last_activity = self._parse_timestamp(record.created_at)
            if last_activity < cutoff_time:
                expired_ids.append(session_id)
        return expired_ids
