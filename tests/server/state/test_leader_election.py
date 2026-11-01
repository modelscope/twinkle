"""Tests for the cleanup-leader election loop and metrics gauge in ``ServerState``.

Covers:
- exactly one ``ServerState`` instance over the same backend becomes leader;
- the cleanup task is gated on leadership (non-leaders never call cleanup);
- when the leader stops, another instance can take over;
- the ObservableGauge resource counters publish correct values without
  inflation across multiple instances.
"""
from __future__ import annotations

import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.server.state.fake_backend import FakeBackend

from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import RayActorBackend
from twinkle.server.state.cleanup_coordinator import LEADER_KEY, LEASE_RENEW, ResourceCleanupCoordinator
from twinkle.server.telemetry import MetricsRegistry


@pytest.mark.asyncio
async def test_only_one_of_many_instances_becomes_leader() -> None:
    backend = RayActorBackend()
    instances = [ServerState(backend=backend, cleanup_interval=600.0) for _ in range(4)]
    try:
        for s in instances:
            await s.start_cleanup_task()
        # Give the leader_loop enough time to run its first acquire pass.
        await asyncio.sleep(LEASE_RENEW * 0.2 + 0.5)
        leaders = [s for s in instances if s._is_leader]
        assert len(leaders) == 1, f'expected exactly one leader, got {len(leaders)}'
    finally:
        for s in instances:
            await s.stop_cleanup_task()


@pytest.mark.asyncio
async def test_leader_publishes_resource_counts() -> None:
    from twinkle.server.telemetry import MetricsRegistry

    MetricsRegistry.reset()
    backend = RayActorBackend()
    state = ServerState(
        backend=backend,
        cleanup_interval=600.0,
        metrics_update_interval=0.5,
    )
    try:
        await state.start_cleanup_task()
        # Create a session and wait for the publish loop to push the count.
        await state.create_session({})
        await asyncio.sleep(1.5)
        assert state._is_leader
        registry = MetricsRegistry.get()
        assert registry.get_resource_count('active_sessions') == 1
    finally:
        await state.stop_cleanup_task()
        MetricsRegistry.reset()


@pytest.mark.asyncio
async def test_leader_handover_after_stop() -> None:
    backend = RayActorBackend()
    a = ServerState(backend=backend, cleanup_interval=600.0)
    b = ServerState(backend=backend, cleanup_interval=600.0)
    try:
        await a.start_cleanup_task()
        await b.start_cleanup_task()
        await asyncio.sleep(LEASE_RENEW * 0.2 + 0.5)
        leader, other = (a, b) if a._is_leader else (b, a)
        assert leader._is_leader is True
        await leader.stop_cleanup_task()
        # stop_cleanup_task can't atomically release the lease (update_atomic
        # has no "delete" form), so the key would normally linger for LEASE_TTL
        # (30s). Skip that wait by clearing the key directly — the in-flight
        # _leader_loop on the other instance picks it up on its next renew tick,
        # but we drive the path manually so the test doesn't have to wait.
        await backend.delete(LEADER_KEY)
        await other._try_acquire_or_renew()
        assert other._is_leader is True
    finally:
        await a.stop_cleanup_task()
        await b.stop_cleanup_task()


@pytest.mark.asyncio
async def test_renew_keeps_leader() -> None:
    backend = RayActorBackend()
    state = ServerState(backend=backend, cleanup_interval=600.0)
    try:
        await state.start_cleanup_task()
        await asyncio.sleep(LEASE_RENEW * 0.2 + 0.5)
        assert state._is_leader
        # Manually trigger a renew; lease key still mine.
        await state._try_acquire_or_renew()
        assert state._is_leader
    finally:
        await state.stop_cleanup_task()


@pytest.mark.asyncio
async def test_leader_recovers_after_renewal_failure() -> None:
    """An unconfirmed renewal fences cleanup until the original lease expires."""
    backend = RayActorBackend()
    state = ServerState(backend=backend, cleanup_interval=600.0)

    # Become leader via the normal path.
    await state._try_acquire_or_renew()
    assert state._is_leader is True
    assert await backend.get(LEADER_KEY) == state._leader_id

    # Next renewal raises — simulate a transient backend error during renew.
    real_update_atomic = backend.update_atomic
    calls = {'n': 0}

    async def flaky_update_atomic(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise RuntimeError('transient backend error during renew')
        return await real_update_atomic(*args, **kwargs)

    from unittest import mock
    with mock.patch.object(backend, 'update_atomic', side_effect=flaky_update_atomic):
        await state._try_acquire_or_renew()
        assert state._is_leader is False

    assert await backend.get(LEADER_KEY) == state._leader_id
    await state._try_acquire_or_renew()
    assert state._is_leader is False
    await backend.set(LEADER_KEY, state._leader_id, ttl=1)
    await asyncio.sleep(1.1)
    await state._try_acquire_or_renew()
    assert state._is_leader is True


@pytest.mark.asyncio
async def test_renewal_failure_does_not_steal_other_leader_lease() -> None:
    """A non-leader whose election attempt raises must NOT delete a lease that
    another replica legitimately holds."""
    backend = RayActorBackend()
    leader = ServerState(backend=backend, cleanup_interval=600.0)
    follower = ServerState(backend=backend, cleanup_interval=600.0)

    await leader._try_acquire_or_renew()
    assert leader._is_leader is True
    leader_value = await backend.get(LEADER_KEY)

    # Follower's set_nx raises; since it was NOT leader, it must not delete the
    # lease the real leader owns.
    from unittest import mock
    with mock.patch.object(backend, 'set_nx', side_effect=RuntimeError('boom')):
        await follower._try_acquire_or_renew()
        assert follower._is_leader is False

    assert await backend.get(LEADER_KEY) == leader_value, 'follower stole the real leader lease'


def coordinator(backend=None):
    models = SimpleNamespace(rebuild_indexes=AsyncMock())
    return ResourceCleanupCoordinator(backend or FakeBackend(), {'models': models},
                                      cleanup_interval=600, expiration_timeout=300,
                                      sweep_processor_quotas=AsyncMock())


@pytest.mark.asyncio
async def test_stale_leader_renewal_failure_preserves_successor(monkeypatch):
    backend = FakeBackend()
    cleanup = coordinator(backend)
    await cleanup._try_acquire_or_renew()
    assert cleanup.is_leader

    async def successor_then_fail(*args, **kwargs):
        await backend.set(LEADER_KEY, 'successor', ttl=30)
        raise ConnectionError('renewal acknowledgement lost')

    monkeypatch.setattr(backend, 'update_atomic', successor_then_fail)
    await cleanup._try_acquire_or_renew()
    assert not cleanup.is_leader
    assert await backend.get(LEADER_KEY) == 'successor'


@pytest.mark.asyncio
async def test_cleanup_concurrent_start_rebuilds_once_and_stop_joins():
    cleanup = coordinator()
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def rebuild():
        entered.set()
        await proceed.wait()

    cleanup._managers['models'].rebuild_indexes.side_effect = rebuild
    first = asyncio.create_task(cleanup.start())
    await entered.wait()
    second = asyncio.create_task(cleanup.start())
    proceed.set()
    assert await asyncio.gather(first, second) == [True, False]
    tasks = (cleanup._cleanup_task, cleanup._leader_task)
    assert await cleanup.stop()
    assert all(task.done() for task in tasks)
    assert not await cleanup.stop()
    cleanup._managers['models'].rebuild_indexes.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_cleanup_start_can_retry():
    cleanup = coordinator()
    cleanup._managers['models'].rebuild_indexes.side_effect = [RuntimeError('offline'), None]
    with pytest.raises(RuntimeError, match='offline'):
        await cleanup.start()
    assert cleanup._cleanup_task is None and cleanup._leader_task is None
    assert await cleanup.start()
    await cleanup.stop()


@pytest.mark.asyncio
async def test_lazy_cleanup_retries_after_failed_start():
    from twinkle.server.deployment import LazyCleanupMixin
    client = LazyCleanupMixin()
    client.state = SimpleNamespace(start_cleanup_task=AsyncMock(side_effect=[RuntimeError('offline'), True]))
    await client._ensure_state_cleanup_started()
    assert not client._state_cleanup_started
    await client._ensure_state_cleanup_started()
    assert client._state_cleanup_started
    assert client.state.start_cleanup_task.await_count == 2


# ============================================================
# Metrics ObservableGauge (merged from test_metrics_observable_gauge)
# ============================================================


@pytest.mark.asyncio
async def test_four_instances_no_4x_gauge_inflation() -> None:
    """Four ServerState instances sharing one backend: only the leader publishes
    gauge counts, so the cache reads 5 (not 20)."""
    MetricsRegistry.reset()
    backend = RayActorBackend()
    instances = [ServerState(backend=backend, cleanup_interval=600.0, metrics_update_interval=0.5) for _ in range(4)]
    try:
        for s in instances:
            await s.start_cleanup_task()
        for _ in range(5):
            await instances[0].create_session({})
        await asyncio.sleep(1.5)
        leaders = [s for s in instances if s._is_leader]
        assert len(leaders) == 1
        registry = MetricsRegistry.get()
        assert registry.get_resource_count('active_sessions') == 5
    finally:
        for s in instances:
            await s.stop_cleanup_task()
        MetricsRegistry.reset()


@pytest.mark.asyncio
async def test_gauge_cache_zeroed_on_leadership_handover() -> None:
    """When a leader loses leadership its gauge cache is zeroed."""
    MetricsRegistry.reset()
    backend = RayActorBackend()
    a = ServerState(backend=backend, cleanup_interval=600.0, metrics_update_interval=0.3)
    b = ServerState(backend=backend, cleanup_interval=600.0, metrics_update_interval=0.3)
    try:
        await a.start_cleanup_task()
        await b.start_cleanup_task()
        for _ in range(3):
            await a.create_session({})
        await asyncio.sleep(1.0)

        leader = a if a._is_leader else b
        registry = MetricsRegistry.get()
        assert registry.get_resource_count('active_sessions') == 3

        # Force handover
        await backend.set(LEADER_KEY, 'someone-else')
        await leader._try_acquire_or_renew()
        assert leader._is_leader is False
        assert registry.get_resource_count('active_sessions') == 0
    finally:
        await a.stop_cleanup_task()
        await b.stop_cleanup_task()
        MetricsRegistry.reset()
