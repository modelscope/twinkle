# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import functools
from datetime import datetime
from typing import Any

from .backend.base import StateBackend
from .base import BaseManager
from .models import FutureRecord

# Status sets used by the do-not-regress guard inside the atomic transform.
_TERMINAL_STATUSES = frozenset({'completed', 'failed'})
_NON_TERMINAL_STATUSES = frozenset({'pending', 'queued', 'running'})


def _future_record_transform(
    existing: dict | None,
    *,
    new_status: str,
    model_id: str | None,
    reason: str | None,
    result: Any,
    queue_state: str | None,
    queue_state_reason: str | None,
    now: str,
) -> dict | None:
    """Atomic transform body for :meth:`FutureManager.store_status`.

    Module-level so it remains picklable when forwarded across the Ray actor
    boundary (closures and lambdas cannot be).

    Drops the write entirely (returns ``None``) when ``new_status`` would
    regress a terminal status — the StateBackend.update_atomic contract treats
    a ``None`` return as "keep the current value", which is what stops stale
    retries from clobbering a freshly committed terminal state.
    """
    if (existing is not None and existing.get('status') in _TERMINAL_STATUSES and new_status in _NON_TERMINAL_STATUSES):
        return None

    if existing is None:
        record = FutureRecord(
            status=new_status,
            model_id=model_id,
            reason=reason,
            result=result,
            queue_state=queue_state,
            queue_state_reason=queue_state_reason,
            created_at=now,
            updated_at=now,
        )
        return record.model_dump()

    updated = dict(existing)
    updated['status'] = new_status
    updated['model_id'] = model_id
    updated['updated_at'] = now
    if reason is not None:
        updated['reason'] = reason
    if result is not None:
        updated['result'] = result
    if queue_state is not None:
        updated['queue_state'] = queue_state
    if queue_state_reason is not None:
        updated['queue_state_reason'] = queue_state_reason
    return updated


class FutureManager(BaseManager[FutureRecord]):
    """Manages async task futures / request statuses.

    Expiry is based on `updated_at` (falls back to `created_at`).
    """

    def __init__(self, backend: StateBackend, expiration_timeout: float) -> None:
        super().__init__(backend, 'future::', FutureRecord, expiration_timeout)

    # ----- Future-specific operations -----

    async def store_status(
        self,
        request_id: str,
        status: str,
        model_id: str | None,
        reason: str | None = None,
        result: Any = None,
        queue_state: str | None = None,
        queue_state_reason: str | None = None,
    ) -> None:
        """Create or update a future record with the latest status.

        Uses :meth:`StateBackend.update_atomic` so that a slow retry writing
        ``pending`` cannot clobber a freshly committed terminal status — the
        backend serializes the read-transform-write triple for us.

        If the result object has a ``model_dump`` method (i.e. it is a Pydantic
        model) it is serialized to a plain dict before storage.
        """
        if result is not None and hasattr(result, 'model_dump'):
            result = result.model_dump()

        now = datetime.now().isoformat()
        await self._backend.update_atomic(
            self._make_key(request_id),
            functools.partial(
                _future_record_transform,
                new_status=status,
                model_id=model_id,
                reason=reason,
                result=result,
                queue_state=queue_state,
                queue_state_reason=queue_state_reason,
                now=now,
            ),
        )

    # ----- Cleanup -----

    async def cleanup_expired(self, cutoff_time: float, **kwargs) -> int:
        """Remove futures whose last update is older than cutoff_time.

        Args:
            cutoff_time: Unix timestamp threshold.

        Returns:
            Number of futures removed.
        """
        all_records = await self.get_all()
        expired_ids = []
        for request_id, record in all_records.items():
            timestamp_str = record.updated_at or record.created_at
            timestamp = self._parse_timestamp(timestamp_str)
            if timestamp < cutoff_time:
                expired_ids.append(request_id)

        for request_id in expired_ids:
            await self.remove(request_id)

        return len(expired_ids)
