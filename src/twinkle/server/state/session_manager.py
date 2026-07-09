# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import functools
import time

from twinkle.utils import get_logger
from .backend.base import ConcurrencyError, StateBackend
from .base import BaseManager
from .models import SessionRecord

logger = get_logger()


def _session_touch_transform(existing: dict | None, *, now: float) -> dict | None:
    """Atomic transform body for :meth:`SessionManager.touch`.

    Module-level for pickling across the Ray actor boundary. Returns ``None``
    when the session does not exist so :meth:`StateBackend.update_atomic`
    treats it as a no-op (matching the legacy ``False`` return).
    """
    if existing is None:
        return None
    updated = dict(existing)
    updated['last_heartbeat'] = now
    return updated


class SessionManager(BaseManager[SessionRecord]):
    """Manages client sessions.

    Expiry is based on `last_heartbeat`; falls back to `created_at` if no
    heartbeat has been recorded yet.
    """

    def __init__(self, backend: StateBackend, expiration_timeout: float) -> None:
        super().__init__(backend, 'session::', SessionRecord, expiration_timeout)

    # ----- Session-specific operations -----

    async def touch(self, session_id: str) -> bool:
        """Update the heartbeat timestamp for a session.

        Returns:
            True if the session exists and was updated, False if the session
            is absent OR the backend's atomic-update path failed under sustained
            contention. The next heartbeat from the same client will retry.
        """
        # _session_touch_transform returns None only when the session is
        # absent; in that case update_atomic also returns None.
        try:
            result = await self._backend.update_atomic(
                self._make_key(session_id),
                functools.partial(_session_touch_transform, now=time.time()),
            )
        except ConcurrencyError as e:
            # touch() is per-heartbeat best-effort — losing one round under
            # contention is recoverable as soon as the next heartbeat arrives.
            logger.warning('SessionManager.touch dropped due to contention: %s', e)
            return False
        return result is not None

    async def get_last_heartbeat(self, session_id: str) -> float | None:
        """Return the last heartbeat timestamp, or None if the session does not exist."""
        record = await self.get(session_id)
        if record is None:
            return None
        return record.last_heartbeat

    # ----- Cleanup -----

    def _is_expired(self, record: SessionRecord, cutoff_time: float) -> bool:
        """Whether a session's last activity is older than ``cutoff_time``."""
        last_activity = record.last_heartbeat
        if last_activity == 0.0:
            last_activity = self._parse_timestamp(record.created_at)
        return last_activity < cutoff_time

    async def collect_and_remove_expired(self, cutoff_time: float) -> tuple[list[str], int]:
        """Determine expired sessions in ONE pass and remove exactly those.

        Returns ``(expired_ids, removed_count)``. The returned ``expired_ids``
        is the single authoritative set used by :class:`ServerState` to cascade
        the same sessions' child models and sampling sessions, so a session
        touched between two separate scans can no longer survive removal while
        its children are cascade-deleted (the prior TOCTOU window).
        """
        all_records = await self.get_all()
        expired_ids = [sid for sid, record in all_records.items() if self._is_expired(record, cutoff_time)]
        removed = await self.remove_many(expired_ids)
        return expired_ids, removed

    async def remove_many(self, ids: list[str]) -> int:
        """Remove the supplied session IDs. Returns the number actually removed."""
        removed = 0
        for session_id in ids:
            if await self.remove(session_id):
                removed += 1
        return removed

    async def cleanup_expired(self, cutoff_time: float, **kwargs) -> int:
        """Remove sessions whose last activity is older than ``cutoff_time``.

        Returns:
            Number of sessions removed.
        """
        _, removed = await self.collect_and_remove_expired(cutoff_time)
        return removed
