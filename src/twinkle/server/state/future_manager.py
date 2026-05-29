# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from datetime import datetime
from typing import Any

from .backend.base import StateBackend
from .base import BaseManager
from .models import FutureRecord


class FutureManager(BaseManager[FutureRecord]):
    """Manages async task futures / request statuses.

    Expiry is based on `updated_at` (falls back to `created_at`).
    """

    def __init__(self, backend: StateBackend, expiration_timeout: float) -> None:
        super().__init__(backend, "future::", FutureRecord, expiration_timeout)

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

        If the result object has a `model_dump` method (i.e. it is a Pydantic
        model) it is serialized to a plain dict before storage.

        Args:
            request_id: Unique identifier for the request.
            status: Task status string (pending/queued/running/completed/failed/rate_limited).
            model_id: Optional associated model_id.
            reason: Optional reason string (used for rate_limited status).
            result: Optional result data (used for completed/failed status).
            queue_state: Optional queue state (active/paused_rate_limit/paused_capacity).
            queue_state_reason: Optional reason for the queue state.
        """
        if result is not None and hasattr(result, 'model_dump'):
            result = result.model_dump()

        now = datetime.now().isoformat()
        existing = await self.get(request_id)

        if existing is not None:
            existing.status = status
            existing.model_id = model_id
            existing.updated_at = now
            if reason is not None:
                existing.reason = reason
            if result is not None:
                existing.result = result
            if queue_state is not None:
                existing.queue_state = queue_state
            if queue_state_reason is not None:
                existing.queue_state_reason = queue_state_reason
            await self.add(request_id, existing)
        else:
            record = FutureRecord(
                status=status,
                model_id=model_id,
                reason=reason,
                result=result,
                queue_state=queue_state,
                queue_state_reason=queue_state_reason,
                created_at=now,
                updated_at=now,
            )
            await self.add(request_id, record)

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
