# Copyright (c) ModelScope Contributors. All rights reserved.
"""Phase 0d Redis integration tests (R19.4, R19.5).

Properties covered:
- # Feature: server-config-observability-refactor, Property 26: Cross-worker write visibility
- # Feature: server-config-observability-refactor, Property 27: Concurrent-write consistency

Both run against a real Redis instance reached at ``REDIS_URL`` (default
``redis://localhost:6379/0``). When the URL is unreachable the whole module
is skipped — these tests are explicitly Docker-dependent and must run
against the local stack rather than the in-process mock.
"""
from __future__ import annotations

import asyncio
import os
import pytest
import uuid

from twinkle.server.state import ServerState
from twinkle.server.state.backend.factory import PersistenceConfig, create_backend
from twinkle.server.state.backend.redis_backend import RedisBackend

REDIS_URL = os.environ.get('TWINKLE_TEST_REDIS_URL', 'redis://localhost:6379/0')


def _can_reach_redis() -> bool:

    async def _check() -> bool:
        backend = RedisBackend(REDIS_URL)
        try:
            return await backend.health_check()
        except Exception:
            return False
        finally:
            try:
                await backend.close()
            except Exception:
                pass

    try:
        return asyncio.run(_check())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _can_reach_redis(),
    reason=f'Redis at {REDIS_URL} unreachable — start docker compose / `docker run -p 6379:6379 redis`',
)


@pytest.fixture
def isolation_prefix() -> str:
    """Fresh key namespace per test so parallel runs don't collide."""
    return f'twinkle-test-{uuid.uuid4().hex[:8]}::'


@pytest.fixture
async def shared_backend(isolation_prefix: str):
    backend = create_backend(PersistenceConfig(mode='redis', redis_url=REDIS_URL, key_prefix=isolation_prefix))
    yield backend
    # Tear down everything we wrote.
    try:
        keys = await backend.keys('*')
        for k in keys:
            await backend.delete(k)
    finally:
        await backend.close()


@pytest.fixture
def make_state(isolation_prefix: str):
    """Factory for fresh ``ServerState`` instances over the same shared key prefix.

    Each call returns a NEW ``RedisBackend`` (separate connection pool) so the
    tests genuinely exercise cross-instance behaviour rather than two views
    of the same client.
    """
    created: list[ServerState] = []

    def _make() -> ServerState:
        backend = create_backend(PersistenceConfig(mode='redis', redis_url=REDIS_URL, key_prefix=isolation_prefix))
        state = ServerState(backend=backend)
        created.append(state)
        return state

    yield _make

    async def _cleanup() -> None:
        for s in created:
            try:
                await s._backend.close()
            except Exception:
                pass

    asyncio.run(_cleanup())


# ---------- Property 26: cross-worker visibility (R19.4) ------------------ #


@pytest.mark.asyncio
async def test_property_26_replica_write_via_a_visible_via_b(make_state) -> None:
    """One worker registers a replica; a second worker on the same shared
    backend sees the same capacity / availability view."""
    a = make_state()
    b = make_state()
    rid = f'r-{uuid.uuid4().hex[:6]}'
    await a.register_replica(rid, max_loras=4)

    cap = await b.get_capacity_info()
    assert cap['max_loras'] >= 4
    assert rid in await b.get_available_replica_ids([rid])


@pytest.mark.asyncio
async def test_property_26_model_write_visible(make_state) -> None:
    a = make_state()
    b = make_state()
    rid = f'r-{uuid.uuid4().hex[:6]}'
    await a.register_replica(rid, max_loras=2)
    mid = await a.register_model({'base_model': 'mock'}, token='tok-A', model_id='mid-A', replica_id=rid)

    meta = await b.get_model_metadata(mid)
    assert meta is not None
    assert meta['token'] == 'tok-A'
    assert meta['replica_id'] == rid


@pytest.mark.asyncio
async def test_property_26_session_and_config(make_state) -> None:
    a = make_state()
    b = make_state()
    sid = await a.create_session({'session_id': f'sess-{uuid.uuid4().hex[:6]}'})
    assert await b.get_session_last_heartbeat(sid) is not None

    await a.add_config('feature_flag', {'value': 42})
    assert await b.get_config('feature_flag') == {'value': 42}


# ---------- Property 27: concurrent-write consistency (R19.5) ------------- #


@pytest.mark.asyncio
async def test_property_27_concurrent_config_writes_no_torn_records(make_state) -> None:
    """Many concurrent writes of distinct keys complete and every record
    equals one of the writes (no torn / partial value)."""
    a = make_state()
    b = make_state()
    n = 40
    payload = {f'k-{i}': {'idx': i, 'note': 'x' * 32} for i in range(n)}

    async def writer(state: ServerState, items: dict) -> None:
        await asyncio.gather(*(state.add_config(k, v) for k, v in items.items()))

    half = list(payload.items())[:n // 2]
    other = list(payload.items())[n // 2:]
    await asyncio.gather(writer(a, dict(half)), writer(b, dict(other)))

    # Every key must read back equal to its expected payload from either side.
    for k, v in payload.items():
        assert await a.get_config(k) == v, k
        assert await b.get_config(k) == v, k


@pytest.mark.asyncio
async def test_property_27_concurrent_same_key_lands_one_of_committed(make_state) -> None:
    """Two writers race on the same key — final value equals one of the
    writes; no torn record (R19.5)."""
    a = make_state()
    b = make_state()
    write_a = {'who': 'a', 'payload': list(range(8))}
    write_b = {'who': 'b', 'payload': list(range(8, 16))}

    await asyncio.gather(a.add_config('contended', write_a), b.add_config('contended', write_b))
    final = await a.get_config('contended')
    assert final in (write_a, write_b)


@pytest.mark.asyncio
async def test_property_27_concurrent_replica_registration(make_state) -> None:
    a = make_state()
    b = make_state()
    rid = f'r-{uuid.uuid4().hex[:6]}'
    await asyncio.gather(
        a.register_replica(rid, 4),
        b.register_replica(rid, 4),
    )
    cap = await a.get_capacity_info()
    # Capacity row stores ``max_loras``; both writers wrote 4, no torn write.
    assert cap['max_loras'] == 4


# ---------- Read-modify-write race exposure (memory-actor revert PR) ----- #
#
# These tests don't fix anything — they just exercise the RMW patterns inside
# managers (`SessionManager.touch`, `ConfigManager.add_or_get`,
# `FutureManager.store_status`) under a real Redis backend so that any
# resulting lost-write surfaces as a hard test failure rather than as a
# silent production drift. Each test is allowed to xfail if the race is real:
# the goal is to make the symptom visible to the next PR that actually adds
# WATCH/MULTI/EXEC or a Lua script around the path.


@pytest.mark.asyncio
async def test_concurrent_session_touch_lost_write_exposure(make_state) -> None:
    """Two workers race ``touch`` on the same session for ~100 rounds.

    Properties checked:
    (a) every ``touch`` returns ``True`` because the session exists (no
        ``False`` from a stale read followed by a delete-races-write);
    (b) the final ``last_heartbeat`` is plausibly close to ``now`` — i.e. a
        write that landed after another write was not silently swallowed.

    The implementation is read-modify-write, so a *specific* heartbeat update
    can still get clobbered by a slower writer — but for these two assertions
    the race is benign (last-write-wins still lands a recent heartbeat). If
    either assertion ever flips on a real Redis the manager needs a
    server-side atomic update (Lua / WATCH+EXEC), not just an extra retry.
    """
    import time
    a = make_state()
    b = make_state()
    sid = await a.create_session({'session_id': f'sess-{uuid.uuid4().hex[:6]}'})

    rounds = 100
    touched: list[bool] = []

    async def hammer(state: ServerState) -> None:
        for _ in range(rounds):
            ok = await state.touch_session(sid)
            touched.append(ok)

    start = time.time()
    await asyncio.gather(hammer(a), hammer(b))

    assert all(touched), f'{touched.count(False)} touch() calls returned False — RMW lost the session row'
    final = await a.get_session_last_heartbeat(sid)
    assert final is not None
    # Allow generous slack for actor RPC + Redis round-trips during the
    # hammer phase, but it must still be close to "now".
    assert final >= start, 'final heartbeat is older than the test start — write was lost'


@pytest.mark.asyncio
async def test_concurrent_add_or_get_consistent_value(make_state) -> None:
    """``ConfigManager.add_or_get`` is implemented on top of ``set_nx``,
    which is atomic in Redis. Two writers racing distinct values for the
    same key must return the *same* committed value."""
    a = make_state()
    b = make_state()
    key = f'cfg-{uuid.uuid4().hex[:6]}'
    write_a = {'who': 'a'}
    write_b = {'who': 'b'}

    got_a, got_b = await asyncio.gather(
        a.add_or_get_config(key, write_a),
        b.add_or_get_config(key, write_b),
    )
    # Both calls must observe the same committed value — that's the whole
    # point of the SETNX-backed contract.
    assert got_a == got_b
    final = await a.get_config(key)
    assert final == got_a
    assert final in (write_a, write_b)


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason=('FutureManager.store_status is read-modify-write; on a real Redis '
            'a concurrent retry that writes "pending" can clobber a freshly '
            'committed terminal status. Tracked for follow-up.'),
    strict=False,
)
async def test_concurrent_future_update_no_state_regression(make_state) -> None:
    """Once a future is recorded as ``completed`` a concurrent ``pending``
    write must not regress it back. With the current RMW implementation
    this race is real on Redis — the test is xfail until the manager grows
    a server-side atomic update path."""
    a = make_state()
    b = make_state()
    request_id = f'req-{uuid.uuid4().hex[:6]}'
    await a.store_future_status(request_id=request_id, status='pending', model_id=None)

    async def writer_completed() -> None:
        await a.store_future_status(
            request_id=request_id,
            status='completed',
            model_id=None,
            result={'ok': True},
        )

    async def writer_pending_retry() -> None:
        # Simulate a stale retry path that re-sends "pending" after a network
        # hiccup; the manager must not lose the terminal status.
        await b.store_future_status(request_id=request_id, status='pending', model_id=None)

    await asyncio.gather(writer_completed(), writer_pending_retry())

    final = await a.get_future(request_id)
    assert final is not None
    assert final['status'] == 'completed', f'completed status was clobbered by a concurrent pending write: {final}'
