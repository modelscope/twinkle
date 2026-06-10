# Copyright (c) ModelScope Contributors. All rights reserved.
"""Redis integration tests for cross-worker visibility and concurrent writes.

Pins two contracts of the shared ``RedisBackend``:
- writes from one ``ServerState`` are visible to a second instance over the
  same key prefix (cross-worker visibility);
- concurrent writes of distinct keys never tear, and concurrent writes of the
  same key always commit one of the writers' values intact.

Runs against a real Redis instance at ``REDIS_URL`` (default
``redis://localhost:6379/0``). When the URL is unreachable the whole module
is skipped — these tests are explicitly Docker-dependent and must run
against the local stack rather than the in-process mock.
"""
from __future__ import annotations

import asyncio
import os
import pytest
import uuid

from twinkle.server.config.persistence import PersistenceConfig
from twinkle.server.state import ServerState
from twinkle.server.state.backend.factory import create_backend
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


# ---------- Cross-worker visibility -------------------------------------- #


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


# ---------- Concurrent-write consistency --------------------------------- #


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
    writes; no torn record."""
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


# ---------- Manager-level atomic-update guarantees ----------------------- #
#
# These tests pin the contract that the manager-level RMW paths
# (``SessionManager.touch``, ``ConfigManager.add_or_get``,
# ``FutureManager.store_status``) now go through ``StateBackend.update_atomic``
# or ``set_nx``, so a concurrent retry cannot lose a freshly committed write.


@pytest.mark.asyncio
async def test_concurrent_session_touch_monotonic_heartbeat(make_state) -> None:
    """Two workers racing ``touch`` on the same session leave the persisted
    ``last_heartbeat`` monotonically advancing — the final value must be ≥ the
    test start time, proving no write got permanently swallowed.

    ``SessionManager.touch`` rides on :meth:`StateBackend.update_atomic`, which
    serializes read+write under one critical section. Under heavy WATCH
    contention a small fraction of touches may surface as ``False`` (backend
    exhausted its retry budget); those are best-effort heartbeats — the next
    heartbeat from the same client succeeds and the persisted timestamp keeps
    advancing, which is the contract we actually care about.
    """
    import time
    a = make_state()
    b = make_state()
    sid = await a.create_session({'session_id': f'sess-{uuid.uuid4().hex[:6]}'})

    rounds = 200
    touched: list[bool] = []

    async def hammer(state: ServerState) -> None:
        for _ in range(rounds):
            ok = await state.touch_session(sid)
            touched.append(ok)

    start = time.time()
    await asyncio.gather(hammer(a), hammer(b))

    # At least 80% of touches must commit; a small contention-skip fraction is
    # expected when two workers hammer the same key with no delay between calls.
    success_rate = sum(touched) / len(touched)
    assert success_rate >= 0.8, (f'only {success_rate:.1%} of touches committed — backend dropped too many writes')
    final = await a.get_session_last_heartbeat(sid)
    assert final is not None
    assert final >= start, 'final heartbeat predates the test start — every write was lost'


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
async def test_concurrent_future_update_no_state_regression(make_state) -> None:
    """Once a future is recorded as ``completed`` a concurrent ``pending``
    write must not regress it back. ``FutureManager.store_status`` goes
    through :meth:`StateBackend.update_atomic`, whose transform drops the
    write entirely when ``new_status`` would walk a terminal status back to
    a non-terminal one."""
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
