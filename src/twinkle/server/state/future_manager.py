# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import functools
from datetime import datetime
from typing import Any

from twinkle.server.utils.task_errors import task_error_payload
from twinkle.utils.logger import get_logger
from .backend.base import StateBackend
from .base import BaseManager
from .models import FutureRecord

logger = get_logger()

# Status sets used by the do-not-regress guard inside the atomic transform.
_TERMINAL_STATUSES = frozenset({'completed', 'failed'})
_NON_TERMINAL_STATUSES = frozenset({'pending', 'queued', 'running'})


def _future_record_transform(
    existing: dict | None,
    *,
    request_id: str,
    new_status: str,
    model_id: str | None,
    reason: str | None,
    result: Any,
    queue_state: str | None,
    queue_state_reason: str | None,
    replica_id: str | None,
    now: str,
) -> dict | None:
    """Atomic transform body for :meth:`FutureManager.store_status`.

    Module-level so it remains picklable when forwarded across the Ray actor
    boundary (closures and lambdas cannot be).

    A record already in a terminal state is never overwritten (returns ``None``,
    which ``update_atomic`` treats as "keep current value"). A write of a
    *different* terminal state is logged; a write of the *same* terminal state is
    dropped silently (State_Backend idempotent retries produce these and they
    indicate no defect).
    """
    existing_status = existing.get('status') if existing is not None else None
    if existing_status in _TERMINAL_STATUSES:
        if new_status != existing_status:
            logger.warning('future %s already terminal as %r; refusing %r', request_id, existing_status, new_status)
        return None

    if existing is None:
        record = FutureRecord(
            status=new_status,
            model_id=model_id,
            reason=reason,
            result=result,
            queue_state=queue_state,
            queue_state_reason=queue_state_reason,
            replica_id=replica_id,
            created_at=now,
            updated_at=now,
        )
        return record.model_dump()

    updated = dict(existing)
    updated['status'] = new_status
    updated['model_id'] = model_id
    updated['updated_at'] = now
    # replica_id is set at creation and is deliberately NOT overwritten here.
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
        replica_id: str | None = None,
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
                request_id=request_id,
                new_status=status,
                model_id=model_id,
                reason=reason,
                result=result,
                queue_state=queue_state,
                queue_state_reason=queue_state_reason,
                replica_id=replica_id,
                now=now,
            ),
        )

    # ----- Cleanup -----

    async def cleanup_expired(
        self,
        cutoff_time: float,
        *,
        alive_replica_ids: set[str] | None = None,
        absolute_ttl: float | None = None,
    ) -> int:
        """Expire future records without ever deleting a non-terminal one.

        Processing matrix (design §5.2):

        | status       | replica alive | over absolute_ttl | action            |
        |--------------|---------------|-------------------|-------------------|
        | Terminal     | —             | ts < cutoff       | delete            |
        | non-Terminal | yes           | no                | keep (untouched)  |
        | non-Terminal | yes           | yes               | write ``failed``  |
        | non-Terminal | no            | —                 | write ``failed``  |

        Args:
            cutoff_time: Unix timestamp; terminal records older than it are deleted.
            alive_replica_ids: replicas currently considered alive. ``None`` disables
                the orphan check (every non-terminal record is treated as owned).
            absolute_ttl: seconds; a non-terminal record whose ``created_at`` is older
                than this (regardless of ``updated_at``) is failed. ``None`` disables
                the absolute-survival bound.

        Returns:
            Number of terminal records removed (records written ``failed`` are not
            counted here; they are removed on a later pass once terminal).
        """
        all_records = await self.get_all()
        # Use the same clock convention as the stored timestamps (_parse_timestamp on
        # an ISO string) so the age computation is not skewed by _now_iso writing
        # local time while _parse_timestamp reads naive ISO as UTC.
        now = self._parse_timestamp(datetime.now().isoformat())
        expired_ids: list[str] = []
        for request_id, record in all_records.items():
            if record.status in _TERMINAL_STATUSES:
                timestamp = self._parse_timestamp(record.updated_at or record.created_at)
                if timestamp < cutoff_time:
                    expired_ids.append(request_id)
                continue

            # Non-terminal records are never deleted -- only ever written ``failed``.
            # replica_id None (pre-upgrade) => ownership unknown => treated as alive.
            replica_id = record.replica_id
            replica_alive = (replica_id is None or alive_replica_ids is None or replica_id in alive_replica_ids)
            if not replica_alive:
                await self.store_status(
                    request_id,
                    'failed',
                    record.model_id,
                    result=task_error_payload(
                        'The replica that owned this task is no longer available.',
                        request_id=request_id,
                        error_code=503,
                    ),
                    replica_id=replica_id,
                )
                continue
            if absolute_ttl is not None:
                created = self._parse_timestamp(record.created_at)
                if (now - created) > absolute_ttl:
                    await self.store_status(
                        request_id,
                        'failed',
                        record.model_id,
                        result=task_error_payload(
                            'Task exceeded the absolute survival bound without reaching a terminal state.',
                            request_id=request_id,
                            error_code=500,
                        ),
                        replica_id=replica_id,
                    )

        for request_id in expired_ids:
            await self.remove(request_id)

        return len(expired_ids)
