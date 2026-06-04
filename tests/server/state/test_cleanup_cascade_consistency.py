# Copyright (c) ModelScope Contributors. All rights reserved.
"""Cascade-cleanup consistency regression tests (Requirement 17).

These pin the TOCTOU fix: ``ServerState.cleanup_expired_resources`` determines
the set of expired sessions exactly ONCE per pass and uses that single set both
to remove the sessions and to cascade-delete their child models / sampling
sessions. The prior code scanned for expiry twice (``get_expired_ids`` then a
second ``get_all`` walk inside ``cleanup_expired``), so a session that was
touched between the two scans could survive removal while its children were
still cascade-deleted from the first snapshot.
"""
from __future__ import annotations

import pytest
import time
from unittest import mock

from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import MemoryBackend


@pytest.mark.asyncio
async def test_cascade_set_matches_removed_sessions() -> None:
    """The sessions removed in a pass are exactly the ones whose children are
    cascade-deleted — no session is removed for cascade purposes but retained
    in the store, or vice versa."""
    backend = MemoryBackend()
    # expiration_timeout=0 → every session is immediately past the cutoff.
    state = ServerState(backend=backend, expiration_timeout=0.0)

    sid = await state.create_session({'session_id': 's-expired'})
    await state.register_model({'base_model': 'b'}, token='t1', model_id='m1', session_id=sid)

    # Let wall-clock advance so the session's last_heartbeat < cutoff.
    time.sleep(0.01)
    stats = await state.cleanup_expired_resources()

    # Session and its cascaded child model are both gone — consistently.
    assert stats['sessions'] == 1
    assert stats['models'] == 1
    assert await state.get_session_last_heartbeat('s-expired') is None
    assert await state.get_model_metadata('m1') is None


@pytest.mark.asyncio
async def test_touch_between_scans_cannot_orphan_children() -> None:
    """A session touched AFTER expiry is determined must not have its children
    cascade-deleted while it survives.

    We simulate the historical race by refreshing the session's heartbeat the
    instant ``get_all`` is observed during the cleanup pass. With the one-pass
    design the cascade set is taken from the same snapshot that drives session
    removal, so the session and its model are removed together — there is no
    second scan that could spare the session yet keep cascading its model.
    """
    backend = MemoryBackend()
    state = ServerState(backend=backend, expiration_timeout=0.0)

    sid = await state.create_session({'session_id': 's-race'})
    await state.register_model({'base_model': 'b'}, token='t1', model_id='m-race', session_id=sid)
    time.sleep(0.01)

    session_mgr = state._session_mgr
    real_get_all = session_mgr.get_all
    touched = {'done': False}

    async def get_all_then_touch():
        records = await real_get_all()
        # Refresh the heartbeat right after the snapshot is read, mimicking a
        # client heartbeat landing mid-cleanup. In the OLD two-scan code this
        # would let the session survive the second scan while its model was
        # already cascaded from the first snapshot.
        if not touched['done']:
            touched['done'] = True
            await session_mgr.touch(sid)
        return records

    with mock.patch.object(session_mgr, 'get_all', side_effect=get_all_then_touch):
        stats = await state.cleanup_expired_resources()

    session_alive = await state.get_session_last_heartbeat('s-race') is not None
    model_alive = await state.get_model_metadata('m-race') is not None

    # Consistency invariant: the session and its child model share the same
    # fate. Never "model cascaded away but session left behind".
    assert session_alive == model_alive, (
        f'inconsistent cascade: session_alive={session_alive}, model_alive={model_alive}')
    # And the pass reports a cascade count consistent with the sessions removed.
    if stats['sessions'] == 0:
        assert stats['models'] == 0


@pytest.mark.asyncio
async def test_collect_and_remove_expired_returns_single_authoritative_set() -> None:
    """``collect_and_remove_expired`` returns the one set of IDs it removed."""
    backend = MemoryBackend()
    state = ServerState(backend=backend, expiration_timeout=0.0)
    await state.create_session({'session_id': 's1'})
    await state.create_session({'session_id': 's2'})
    time.sleep(0.01)

    expired_ids, removed = await state._session_mgr.collect_and_remove_expired(time.time())
    assert removed == len(expired_ids)
    assert set(expired_ids) == {'s1', 's2'}
    # All removed — a follow-up pass finds nothing.
    expired_again, removed_again = await state._session_mgr.collect_and_remove_expired(time.time())
    assert expired_again == []
    assert removed_again == 0
