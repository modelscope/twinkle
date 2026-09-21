# Copyright (c) ModelScope Contributors. All rights reserved.
"""State-hygiene tests for FutureManager cleanup and the do-not-regress guard.

Both shipped backends need infrastructure (``memory`` starts a detached Ray actor,
``redis`` needs a server),
so these pure ``FutureManager`` semantics run against the dict-backed fake below.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from fnmatch import fnmatch
from typing import Any
from unittest import mock

import pytest

from twinkle.server.state.backend.base import StateBackend
from twinkle.server.state.future_manager import FutureManager
from twinkle.server.state.models import FutureFailureRecord


class _FakeBackend(StateBackend):
    """Dict-backed StateBackend. Single event loop, so ``update_atomic`` is atomic
    simply by not awaiting between read and write -- the real backends' guarantee."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _is_expired(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return True
        _, expire_at = entry
        if expire_at is not None and time.time() >= expire_at:
            del self._store[key]
            return True
        return False

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = (value, (time.time() + ttl) if ttl is not None else None)

    async def get(self, key: str) -> Any | None:
        return None if self._is_expired(key) else self._store[key][0]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return not self._is_expired(key)

    async def keys(self, pattern: str) -> list[str]:
        return [k for k in list(self._store) if not self._is_expired(k) and fnmatch(k, pattern)]

    async def count(self, pattern: str) -> int:
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any, ttl: int | None = None) -> bool:
        if not self._is_expired(key):
            return False
        await self.set(key, value, ttl)
        return True

    async def update_atomic(
        self,
        key: str,
        transform: Callable[[Any | None], Any | None],
        ttl: int | None = None,
    ) -> Any | None:
        current = await self.get(key)
        updated = transform(current)
        if updated is None:
            return current
        await self.set(key, updated, ttl)
        return updated

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def manager():
    return FutureManager(_FakeBackend(), expiration_timeout=300.0)


async def _store(manager, request_id, status, *, replica_id=None, absolute_deadline=None):
    failure = None
    if status in ('failed', 'cancelled'):
        failure = FutureFailureRecord(
            reason_code='internal_error', message='boom', attribution='server')
    await manager.store_status(
        request_id,
        status,
        model_id='m1',
        failure=failure,
        replica_id=replica_id,
        absolute_deadline=absolute_deadline,
    )


@pytest.mark.asyncio
async def test_non_terminal_with_live_replica_is_kept(manager):
    await _store(manager, 'r1', 'running', replica_id='replica-A')
    removed = await manager.cleanup_expired(
        cutoff_time=time.time() + 10, alive_replica_ids={'replica-A'})
    assert removed == 0
    rec = await manager.get('r1')
    assert rec is not None and rec.status == 'running'


@pytest.mark.asyncio
async def test_non_terminal_orphan_is_failed_not_deleted(manager):
    await _store(manager, 'r2', 'running', replica_id='dead-replica')
    await manager.cleanup_expired(cutoff_time=time.time() + 10, alive_replica_ids={'replica-A'})
    rec = await manager.get('r2')
    assert rec is not None # NOT deleted
    assert rec.status == 'failed'
    assert rec.result is None
    assert rec.failure.reason_code == 'orphaned_replica'
    assert rec.failure.attribution == 'server'


@pytest.mark.asyncio
async def test_non_terminal_past_absolute_deadline_is_failed(manager):
    await _store(manager, 'r3', 'running', replica_id='replica-A', absolute_deadline=time.time() - 1)
    await manager.cleanup_expired(cutoff_time=time.time() + 10, alive_replica_ids={'replica-A'})
    rec = await manager.get('r3')
    assert rec is not None and rec.status == 'failed'
    assert rec.failure.reason_code == 'deadline_exceeded'


@pytest.mark.asyncio
async def test_record_without_deadline_uses_expiration_timeout(manager):
    await _store(manager, 'without-deadline', 'running', replica_id=None)
    with mock.patch('twinkle.server.state.future_manager.time.time', return_value=time.time() + 301):
        await manager.cleanup_expired(cutoff_time=time.time() + 10, alive_replica_ids=set())
    rec = await manager.get('without-deadline')
    assert rec is not None and rec.status == 'failed'
    assert rec.failure.reason_code == 'deadline_exceeded'


@pytest.mark.asyncio
async def test_terminal_expired_is_deleted(manager):
    await _store(manager, 'r4', 'completed', replica_id='replica-A')
    removed = await manager.cleanup_expired(
        cutoff_time=time.time() + 10, alive_replica_ids={'replica-A'})
    assert removed == 1
    assert await manager.get('r4') is None


@pytest.mark.asyncio
async def test_terminal_to_terminal_different_is_refused_and_warns(manager):
    await _store(manager, 'r5', 'failed', replica_id='replica-A')
    with mock.patch('twinkle.server.state.future_manager.logger') as log:
        await manager.store_status('r5', 'completed', model_id='m1')
    rec = await manager.get('r5')
    assert rec.status == 'failed'  # not overwritten
    assert log.warning.called


@pytest.mark.asyncio
async def test_terminal_to_terminal_same_is_dropped_without_warning(manager):
    await _store(manager, 'r6', 'completed', replica_id='replica-A')
    with mock.patch('twinkle.server.state.future_manager.logger') as log:
        await manager.store_status('r6', 'completed', model_id='m1')
    rec = await manager.get('r6')
    assert rec.status == 'completed'
    assert not log.warning.called


@pytest.mark.asyncio
async def test_replica_id_and_deadline_set_at_creation_not_overwritten(manager):
    deadline = time.time() + 100
    await _store(manager, 'r7', 'pending', replica_id='replica-A', absolute_deadline=deadline)
    await manager.store_status(
        'r7', 'running', model_id='m1', replica_id='replica-B', absolute_deadline=time.time() + 999)
    rec = await manager.get('r7')
    assert rec.replica_id == 'replica-A'
    assert rec.absolute_deadline == deadline


@pytest.mark.asyncio
async def test_stored_timestamps_align_with_wall_clock_regardless_of_host_tz(manager):
    """Writer (_now_iso), reader (_parse_timestamp) and time.time() must agree.

    A record written now must parse to within a second of time.time() on any host,
    not skewed by the host's UTC offset (the former naive-local / read-as-UTC bug).
    """
    before = time.time()
    await _store(manager, 'r8', 'running', replica_id='replica-A')
    after = time.time()
    rec = await manager.get('r8')
    parsed = manager._parse_timestamp(rec.created_at)
    assert before - 1 <= parsed <= after + 1


@pytest.mark.asyncio
async def test_claim_seq_dedups_then_release_readmits():
    from twinkle.server.state.server_state import ServerState
    state = ServerState(backend=_FakeBackend())
    # First claim of a (session, seq_id) is unseen -> None, caller proceeds to enqueue.
    assert await state.claim_seq('seq::s1::1', 'reqA', ttl=60) is None
    # A duplicate claim returns the original request_id -> caller returns its envelope.
    assert await state.claim_seq('seq::s1::1', 'reqB', ttl=60) == 'reqA'
    # Releasing (e.g. the original was preflight-rejected) re-admits the same seq_id.
    await state.release_seq('seq::s1::1')
    assert await state.claim_seq('seq::s1::1', 'reqC', ttl=60) is None
    # Different session with the same seq_id never collides.
    assert await state.claim_seq('seq::s2::1', 'reqD', ttl=60) is None


@pytest.mark.asyncio
async def test_cancel_drops_pending_but_never_running():
    from twinkle.server.state.server_state import ServerState
    state = ServerState(backend=_FakeBackend())
    # pending -> cancel drops it to a terminal domain failure.
    await state.store_future_status('rp', 'pending', 'm1')
    assert await state.cancel_future('rp') == {'cancelled': True, 'state': 'cancelled'}
    rec = await state.get_future('rp')
    assert rec['status'] == 'cancelled'
    assert rec['result'] is None
    assert rec['failure']['reason_code'] == 'cancelled'
    assert rec['failure']['attribution'] == 'user'
    # running -> cancel is a no-op; in-flight work is never interrupted.
    await state.store_future_status('rr', 'running', 'm1')
    assert await state.cancel_future('rr') == {'cancelled': False, 'state': 'running'}
    # unknown request_id -> not_found.
    assert await state.cancel_future('nope') == {'cancelled': False, 'state': 'not_found'}


def test_cancelled_record_maps_to_error_envelope():
    from twinkle.server.lifecycle.envelope import envelope_from_record
    rec = {
        'status': 'cancelled',
        'failure': {
            'reason_code': 'cancelled',
            'message': 'Task cancelled by client',
            'attribution': 'user',
        },
    }
    env = envelope_from_record('rc', rec)
    assert env.status == 'cancelled'
    assert env.error is not None and env.error.error_code == 499
