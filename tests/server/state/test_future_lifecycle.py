# Copyright (c) ModelScope Contributors. All rights reserved.
"""State-hygiene tests for FutureManager cleanup and the do-not-regress guard.

Both shipped backends need infrastructure (``memory`` starts a detached Ray actor,
``redis`` needs a server),
so these pure ``FutureManager`` semantics run against the dict-backed fake below.
"""
from __future__ import annotations

import asyncio
import time
from unittest import mock

import pytest

from tests.server.state.fake_backend import FakeBackend
from twinkle.server.state.future_manager import FutureManager
from twinkle.server.state.models import FutureFailureRecord


@pytest.fixture
def manager():
    return FutureManager(FakeBackend(), expiration_timeout=300.0)


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
    state = ServerState(backend=FakeBackend())
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
    state = ServerState(backend=FakeBackend())
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


@pytest.mark.asyncio
async def test_late_queued_and_repeated_running_cannot_regress_or_renew(manager):
    await _store(manager, 'race', 'pending')
    assert await manager.try_start('race')
    before = (await manager.get('race')).model_dump()
    await _store(manager, 'race', 'queued')
    await _store(manager, 'race', 'running')
    assert (await manager.get('race')).model_dump() == before
    assert await manager.cancel_if_pending('race') == 'running'
    assert not await manager.try_start('race')
    assert not await manager.try_start('missing')


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_first', [False, True])
async def test_start_and_cancel_have_one_atomic_winner(manager, cancel_first):
    await _store(manager, 'race', 'queued')
    operations = [manager.try_start('race'), manager.cancel_if_pending('race')]
    if cancel_first:
        operations.reverse()
    results = await asyncio.gather(*operations)
    started = results[1] if cancel_first else results[0]
    status = (await manager.get('race')).status
    assert started == (status == 'running')
    assert status in ('running', 'cancelled')
    assert not await manager.try_start('race')


@pytest.mark.asyncio
async def test_concurrent_starters_only_one_obtains_execution(manager):
    await _store(manager, 'race', 'pending')
    assert sum(await asyncio.gather(*(manager.try_start('race') for _ in range(20)))) == 1


@pytest.mark.asyncio
async def test_future_owner_outlives_admission_retention_until_result_cleanup(monkeypatch):
    from twinkle.server.exceptions import ResourceNotFoundError
    from twinkle.server.state import ServerState
    state = ServerState(backend=FakeBackend(), expiration_timeout=10)
    await state.store_future_status('owned', 'pending', 'training_test', owner_token='owner')
    await state.store_future_status('owned', 'completed', 'training_test', result={'value': 42})
    now = time.time()
    monkeypatch.setattr(time, 'time', lambda: now + 11)
    assert await state.get_future('owned') is not None
    await state.assert_future_owner('owned', 'owner')
    with pytest.raises(ResourceNotFoundError):
        await state.assert_future_owner('owned', 'other')
    await state._future_mgr.cleanup_expired(now + 11)
    assert await state.get_future('owned') is None
    assert not await state._backend.keys('future*')


@pytest.mark.asyncio
async def test_future_owner_is_immutable_and_not_part_of_wire_envelope():
    from twinkle.server.state import ServerState
    from twinkle.server.lifecycle.envelope import envelope_from_record
    state = ServerState(backend=FakeBackend())
    await state.store_future_status('owned', 'pending', None, owner_token='secret-owner')
    await state.store_future_status('owned', 'completed', None, result=42, owner_token='replacement')
    record = await state.get_future('owned')
    assert record['owner_token'] == 'secret-owner'
    assert 'owner_token' not in envelope_from_record('owned', record).model_dump()
    assert 'secret-owner' not in envelope_from_record('owned', record).model_dump_json()
