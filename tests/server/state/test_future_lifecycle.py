# Copyright (c) ModelScope Contributors. All rights reserved.
"""State-hygiene tests for FutureManager cleanup and the do-not-regress guard.

Spec: T5.6 / R9#5 / R9#6 / Property 6 / Property 7. Uses the Ray-free FileBackend.
"""
from __future__ import annotations

from datetime import datetime
from unittest import mock

import pytest

from twinkle.server.state.future_manager import FutureManager


@pytest.fixture
def manager(tmp_path):
    from twinkle.server.state.backend.file_backend import FileBackend
    backend = FileBackend(str(tmp_path / 'state.json'))
    return FutureManager(backend, expiration_timeout=300.0)


def _clock(manager) -> float:
    """Now under the same convention the stored ISO timestamps use."""
    return manager._parse_timestamp(datetime.now().isoformat())


async def _store(manager, request_id, status, *, replica_id=None):
    await manager.store_status(request_id, status, model_id='m1', replica_id=replica_id)


@pytest.mark.asyncio
async def test_non_terminal_with_live_replica_is_kept(manager):
    await _store(manager, 'r1', 'running', replica_id='replica-A')
    removed = await manager.cleanup_expired(
        cutoff_time=_clock(manager) + 10, alive_replica_ids={'replica-A'}, absolute_ttl=None)
    assert removed == 0
    rec = await manager.get('r1')
    assert rec is not None and rec.status == 'running'


@pytest.mark.asyncio
async def test_non_terminal_orphan_is_failed_not_deleted(manager):
    await _store(manager, 'r2', 'running', replica_id='dead-replica')
    await manager.cleanup_expired(cutoff_time=_clock(manager) + 10, alive_replica_ids={'replica-A'}, absolute_ttl=None)
    rec = await manager.get('r2')
    assert rec is not None  # NOT deleted (Property 6)
    assert rec.status == 'failed'
    assert rec.result['category'] == 'Server'


@pytest.mark.asyncio
async def test_non_terminal_over_absolute_ttl_is_failed(manager):
    await _store(manager, 'r3', 'running', replica_id='replica-A')
    # absolute_ttl=0 makes any positive age exceed the bound.
    await manager.cleanup_expired(cutoff_time=_clock(manager) + 10, alive_replica_ids={'replica-A'}, absolute_ttl=0.0)
    rec = await manager.get('r3')
    assert rec is not None and rec.status == 'failed'


@pytest.mark.asyncio
async def test_terminal_expired_is_deleted(manager):
    await _store(manager, 'r4', 'completed', replica_id='replica-A')
    removed = await manager.cleanup_expired(
        cutoff_time=_clock(manager) + 10, alive_replica_ids={'replica-A'}, absolute_ttl=None)
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
async def test_replica_id_set_at_creation_not_overwritten(manager):
    await _store(manager, 'r7', 'pending', replica_id='replica-A')
    await manager.store_status('r7', 'running', model_id='m1', replica_id='replica-B')
    rec = await manager.get('r7')
    assert rec.replica_id == 'replica-A'  # creation value preserved
