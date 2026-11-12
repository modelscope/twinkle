# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import functools
import time
from typing import Any

from twinkle.protocol.types.lifecycle import TERMINAL_STATUSES
from twinkle.utils.logger import get_logger
from .backend.base import StateBackend
from .base import BaseManager
from .models import FutureFailureRecord, FutureRecord, _now_iso

logger = get_logger()


class FutureTransitions:
    """Pure, picklable atomic transitions; the manager owns I/O and retention."""

    terminal = TERMINAL_STATUSES
    cancellable = frozenset({'pending', 'queued'})
    order = {'pending': 0, 'queued': 1, 'running': 2}

    @staticmethod
    def update(
        existing: dict | None,
        *,
        request_id: str,
        new_status: str,
        model_id: str | None,
        reason: str | None,
        result: Any,
        failure: dict[str, Any] | None,
        queue_state: str | None,
        queue_state_reason: str | None,
        replica_id: str | None,
        absolute_deadline: float | None,
        now: str,
        owner_token: str | None = None,
    ) -> dict | None:
        """Atomic transform body for :meth:`FutureManager.store_status`.

        A static method remains picklable across the Ray actor boundary.

        A record already in a terminal state is never overwritten (returns ``None``,
        which ``update_atomic`` treats as "keep current value"). A write of a
        *different* terminal state is logged; a write of the *same* terminal state is
        dropped silently (State_Backend idempotent retries produce these and they
        indicate no defect).
        """
        existing_status = existing.get('status') if existing is not None else None
        if existing_status in FutureTransitions.terminal:
            if new_status != existing_status:
                logger.warning('future %s already terminal as %r; refusing %r', request_id, existing_status, new_status)
            return None
        if new_status in FutureTransitions.order and existing_status in FutureTransitions.order:
            if FutureTransitions.order[new_status] <= FutureTransitions.order[existing_status]:
                return None
        if new_status == 'cancelled' and existing_status == 'running':
            return None

        if existing is None:
            record = FutureRecord(
                owner_token=owner_token,
                status=new_status,
                model_id=model_id,
                reason=reason,
                result=result,
                failure=failure,
                queue_state=queue_state,
                queue_state_reason=queue_state_reason,
                replica_id=replica_id,
                absolute_deadline=absolute_deadline,
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
        if new_status in FutureTransitions.terminal:
            if new_status in ('failed', 'cancelled'):
                updated['result'] = None
                updated['failure'] = failure
            else:
                updated['failure'] = None
                updated['result'] = result
        elif result is not None:
            updated['result'] = result
        if queue_state is not None:
            updated['queue_state'] = queue_state
        if queue_state_reason is not None:
            updated['queue_state_reason'] = queue_state_reason
        return updated

    @staticmethod
    def start(existing: dict | None, *, now: str) -> dict | None:
        """Acquire execution only from pending/queued, using the existing record."""
        if existing is None or existing.get('status') not in FutureTransitions.cancellable:
            return None
        return {**existing, 'status': 'running', 'queue_state': 'active', 'updated_at': now}

    @staticmethod
    def cancel(existing: dict | None, *, failure: dict, now: str) -> dict | None:
        """Atomic transform: cancel iff the task has not started (pending/queued).

        Returns ``None`` (no change) for running/terminal/missing records so a task
        already executing is never interrupted -- cancel is best-effort on the queue.
        """
        status = existing.get('status') if existing is not None else None
        if status not in FutureTransitions.cancellable:
            return None
        updated = dict(existing)
        updated['status'] = 'cancelled'
        updated['result'] = None
        updated['failure'] = failure
        updated['updated_at'] = now
        return updated


class FutureManager(BaseManager[FutureRecord]):
    """Manage future state, terminal retention, and immutable task deadlines."""

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
        failure: FutureFailureRecord | None = None,
        queue_state: str | None = None,
        queue_state_reason: str | None = None,
        replica_id: str | None = None,
        absolute_deadline: float | None = None,
        owner_token: str | None = None,
    ) -> None:
        """Create or update a future record with the latest status.

        Uses :meth:`StateBackend.update_atomic` so that a slow retry writing
        ``pending`` cannot clobber a freshly committed terminal status — the
        backend serializes the read-transform-write triple for us.

        If the result object has a ``model_dump`` method (i.e. it is a Pydantic
        model) it is serialized to a plain dict before storage.
        """
        is_failure = status in ('failed', 'cancelled')
        if is_failure and failure is None:
            raise ValueError(f'{status} future requires a FutureFailureRecord')
        if is_failure and result is not None:
            raise ValueError(f'{status} future cannot carry result')
        if not is_failure and failure is not None:
            raise ValueError(f'{status} future cannot carry failure')
        if result is not None and hasattr(result, 'model_dump'):
            result = result.model_dump()
        failure_data = failure.model_dump() if failure is not None else None

        now = _now_iso()
        await self._backend.update_atomic(
            self._make_key(request_id),
            functools.partial(
                FutureTransitions.update,
                request_id=request_id,
                new_status=status,
                model_id=model_id,
                reason=reason,
                result=result,
                failure=failure_data,
                queue_state=queue_state,
                queue_state_reason=queue_state_reason,
                replica_id=replica_id,
                absolute_deadline=absolute_deadline,
                owner_token=owner_token,
                now=now,
            ),
        )

    async def try_start(self, request_id: str) -> bool:
        """True only when this attempt atomically acquired execution rights."""
        return await self._backend.update_atomic(
            self._make_key(request_id),
            functools.partial(FutureTransitions.start, now=_now_iso()),
            return_changed=True,
        )

    async def cancel_if_pending(self, request_id: str) -> str | None:
        """Cancel a task iff it has not started; return the resulting status.

        Writes a terminal ``cancelled`` record with a domain failure only when
        the current status is pending/queued -- a running task is left alone.
        Returns the record's status after the attempt, or ``None`` if there is no
        record for ``request_id``.
        """
        failure = FutureFailureRecord(
            reason_code='cancelled',
            message='Task cancelled by client',
            attribution='user',
        )
        result = await self._backend.update_atomic(
            self._make_key(request_id),
            functools.partial(FutureTransitions.cancel, failure=failure.model_dump(), now=_now_iso()),
        )
        return result.get('status') if result else None

    # ----- Cleanup -----

    async def cleanup_expired(
        self,
        cutoff_time: float,
        *,
        alive_replica_ids: set[str] | None = None,
    ) -> int:
        """Expire future records without ever deleting a non-terminal one.

        Processing matrix:

        | status       | replica alive | past deadline | action            |
        |--------------|---------------|---------------|-------------------|
        | Terminal     | —             | ts < cutoff       | delete            |
        | non-Terminal | yes           | no                | keep (untouched)  |
        | non-Terminal | yes           | yes               | write ``failed``  |
        | non-Terminal | no            | —                 | write ``failed``  |

        Args:
            cutoff_time: Unix timestamp; terminal records older than it are deleted.
            alive_replica_ids: replicas currently considered alive. ``None`` disables
                the orphan check (every non-terminal record is treated as owned).

        Returns:
            Number of terminal records removed (records written ``failed`` are not
            counted here; they are removed on a later pass once terminal).
        """
        all_records = await self.get_all()
        now = time.time()
        expired_ids: list[str] = []
        for request_id, record in all_records.items():
            if record.status in FutureTransitions.terminal:
                timestamp = self._parse_timestamp(record.updated_at or record.created_at)
                if timestamp < cutoff_time:
                    expired_ids.append(request_id)
                continue

            # Non-terminal records are never deleted -- only ever written ``failed``.
            # Unowned background tasks have no replica; only explicit owners can be declared dead.
            replica_id = record.replica_id
            replica_alive = (replica_id is None or alive_replica_ids is None or replica_id in alive_replica_ids)
            if not replica_alive:
                await self.store_status(
                    request_id,
                    'failed',
                    record.model_id,
                    failure=FutureFailureRecord(
                        reason_code='orphaned_replica',
                        message='The replica that owned this task is no longer available.',
                        attribution='server',
                    ),
                    replica_id=replica_id,
                )
                continue
            deadline = record.absolute_deadline
            if deadline is None:
                deadline = self._parse_timestamp(record.created_at) + self.expiration_timeout
            if now > deadline:
                await self.store_status(
                    request_id,
                    'failed',
                    record.model_id,
                    failure=FutureFailureRecord(
                        reason_code='deadline_exceeded',
                        message='Task exceeded the absolute survival bound without reaching a terminal state.',
                        attribution='server',
                    ),
                    replica_id=replica_id,
                )

        for request_id in expired_ids:
            await self.remove(request_id)

        return len(expired_ids)
