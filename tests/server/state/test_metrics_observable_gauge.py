"""Tests for the ObservableGauge-backed resource counters in MetricsRegistry.

Pins three contracts that prevent the regressions the metrics refactor was
meant to fix:
- the gauge callback reads whatever was last pushed via ``set_resource_count``;
- four ``ServerState`` instances over the same backend do NOT inflate the
  gauge by 4×: only one (the leader) pushes counts;
- ``clear_resource_counts`` zeroes the cache so a former leader stops
  contributing as soon as it loses the lease.
"""
from __future__ import annotations

import asyncio
import pytest

from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import MemoryBackend
from twinkle.server.telemetry import MetricsRegistry


@pytest.fixture(autouse=True)
def _reset_registry():
    MetricsRegistry.reset()
    yield
    MetricsRegistry.reset()


def test_set_and_get_resource_count() -> None:
    registry = MetricsRegistry.get()
    registry.set_resource_count('active_sessions', 5)
    assert registry.get_resource_count('active_sessions') == 5


def test_clear_resource_counts_zeroes_cache() -> None:
    registry = MetricsRegistry.get()
    for name in ('active_sessions', 'active_models', 'active_sampling_sessions', 'active_futures'):
        registry.set_resource_count(name, 17)
    registry.clear_resource_counts()
    for name in ('active_sessions', 'active_models', 'active_sampling_sessions', 'active_futures'):
        assert registry.get_resource_count(name) == 0


def test_unknown_name_is_ignored() -> None:
    registry = MetricsRegistry.get()
    registry.set_resource_count('does_not_exist', 99)
    # Should not raise and should not poison the known counters.
    assert registry.get_resource_count('active_sessions') == 0


@pytest.mark.asyncio
async def test_four_instances_no_4x_inflation() -> None:
    """Simulate the four-deployment topology: four ``ServerState`` instances
    against one backend, five sessions created. The leader pushes the gauge
    once and the others stay silent — the cache reads back 5, not 20.
    """
    backend = MemoryBackend()
    instances = [ServerState(backend=backend, cleanup_interval=600.0, metrics_update_interval=0.5) for _ in range(4)]
    try:
        for s in instances:
            await s.start_cleanup_task()
        # Use any one instance to create the resources — they all share the backend.
        for _ in range(5):
            await instances[0].create_session({})
        # 1.5s ≥ 2× metrics_update_interval ⇒ leader has surely published once.
        await asyncio.sleep(1.5)

        leaders = [s for s in instances if s._is_leader]
        assert len(leaders) == 1
        registry = MetricsRegistry.get()
        assert registry.get_resource_count('active_sessions') == 5
    finally:
        for s in instances:
            await s.stop_cleanup_task()


@pytest.mark.asyncio
async def test_gauge_cache_zeroed_on_leadership_handover() -> None:
    """Regression (Requirement 20): when a leader loses leadership its
    resource-gauge cache is zeroed, so a former leader stops emitting the
    counts it published while it held the lease.

    On unfixed code ``_on_lose_leader`` left the cache intact, so after handover
    (when the publish loop is cancelled and never overwrites again) the stale
    worker would keep emitting its last counts forever.
    """
    from twinkle.server.state.server_state import LEADER_KEY

    backend = MemoryBackend()
    a = ServerState(backend=backend, cleanup_interval=600.0, metrics_update_interval=0.3)
    b = ServerState(backend=backend, cleanup_interval=600.0, metrics_update_interval=0.3)
    try:
        await a.start_cleanup_task()
        await b.start_cleanup_task()

        # Create resources and let the leader publish them.
        for _ in range(3):
            await a.create_session({})
        await asyncio.sleep(1.0)

        leader, other = (a, b) if a._is_leader else (b, a)
        registry = MetricsRegistry.get()
        assert registry.get_resource_count('active_sessions') == 3

        # Force a handover: overwrite the lease with another owner, then let the
        # current leader run one election tick. Its renew sees a foreign owner
        # and it drops leadership, triggering the cache clear.
        await backend.set(LEADER_KEY, 'someone-else')
        await leader._try_acquire_or_renew()
        assert leader._is_leader is False

        # The former leader zeroed its cache on losing leadership.
        assert registry.get_resource_count('active_sessions') == 0
    finally:
        await a.stop_cleanup_task()
        await b.stop_cleanup_task()
