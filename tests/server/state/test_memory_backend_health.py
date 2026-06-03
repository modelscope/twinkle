"""Tests for MemoryBackend's actor failure handling.

If the detached ``_StateActor`` dies (OOM, node failure), ``health_check``
must surface that as ``False`` so external readiness probes can deny
traffic. Crucially, ``MemoryBackend`` must NOT silently re-create the actor:
that would erase all in-memory state without anyone noticing.
"""
from __future__ import annotations

import pytest
import ray

from twinkle.server.state.backend.memory_backend import MemoryBackend


@pytest.mark.asyncio
async def test_health_check_returns_false_after_actor_death() -> None:
    backend = MemoryBackend()
    assert await backend.health_check() is True

    # Kill the actor and verify health_check surfaces the failure.
    ray.kill(backend._actor, no_restart=True)
    assert await backend.health_check() is False


@pytest.mark.asyncio
async def test_health_check_handles_healthy_actor() -> None:
    """Sanity check: a freshly created backend reports healthy."""
    backend = MemoryBackend()
    assert await backend.health_check() is True
