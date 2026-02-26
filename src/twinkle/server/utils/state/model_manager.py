# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from .base import BaseManager
from .models import ModelRecord


class ModelManager(BaseManager[ModelRecord]):
    """
    Manages registered models.

    Expiry is based on `created_at`.  A model is also considered expired if
    its owning session has already been removed (cascade expiry).
    """

    # ----- Cleanup -----

    def cleanup_expired(self, cutoff_time: float, expired_session_ids: list[str] | None = None) -> int:
        """Remove models that are older than cutoff_time, or whose owning
        session has already been expired.

        Args:
            cutoff_time: Unix timestamp threshold.
            expired_session_ids: Optional list of session IDs that have just
                been expired; any model belonging to one of these sessions will
                also be removed regardless of its own age.

        Returns:
            Number of models removed.
        """
        session_set = set(expired_session_ids or [])
        expired_ids = []

        for model_id, record in self._store.items():
            # Cascade: owner session was expired
            if record.session_id and record.session_id in session_set:
                expired_ids.append(model_id)
                continue
            # Own age
            created_at = self._parse_timestamp(record.created_at)
            if created_at < cutoff_time:
                expired_ids.append(model_id)

        for model_id in expired_ids:
            del self._store[model_id]

        return len(expired_ids)
