# Copyright (c) ModelScope Contributors. All rights reserved.
"""State-hygiene tests for FutureManager cleanup and the do-not-regress guard.

Spec: T5.6 / R9#5 / R9#6 / Property 6 / Property 7. Uses the Ray-free FileBackend.
"""
from __future__ import annotations

import time
from unittest import mock

import pytest

from twinkle.server.state.future_manager import FutureManager


@pytest.fixture
def manager(tmp_path):
    from twinkle.server.state.backend.file_backend import FileBackend
    backend = FileBackend(str(tmp_path / 'state.json'))
    return FutureManager(backend, expiration_timeout=300.0)


async def _store(manager, request_id, status, *, replica_id=None, absolute_deadline=None):
    await manager.store_status(
        request_id,
        status,
        model_id='m1',
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
    assert rec is not None  # NOT deleted (Property 6)
    assert rec.status == 'failed'
    assert rec.result['category'] == 'server'


@pytest.mark.asyncio
async def test_non_terminal_past_absolute_deadline_is_failed(manager):
    await _store(manager, 'r3', 'running', replica_id='replica-A', absolute_deadline=time.time() - 1)
    await manager.cleanup_expired(cutoff_time=time.time() + 10, alive_replica_ids={'replica-A'})
    rec = await manager.get('r3')
    assert rec is not None and rec.status == 'failed'


@pytest.mark.asyncio
async def test_legacy_record_without_deadline_uses_expiration_timeout(manager):
    await _store(manager, 'legacy', 'running', replica_id=None)
    with mock.patch('twinkle.server.state.future_manager.time.time', return_value=time.time() + 301):
        await manager.cleanup_expired(cutoff_time=time.time() + 10, alive_replica_ids=set())
    rec = await manager.get('legacy')
    assert rec is not None and rec.status == 'failed'


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
