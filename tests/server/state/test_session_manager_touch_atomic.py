"""Concurrent ``SessionManager.touch`` hammer test.

Pins the contract that ``SessionManager.touch`` goes through
:meth:`StateBackend.update_atomic` so concurrent updates from many "workers"
never lose a write — every observed touch must succeed (``True``) and the
final heartbeat must reflect a write that landed during the hammer phase.
"""
from __future__ import annotations

import asyncio
import pytest
import time

from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import RayActorBackend


@pytest.mark.asyncio
async def test_concurrent_touch_no_lost_write() -> None:
    backend = RayActorBackend()
    state_a = ServerState(backend=backend)
    state_b = ServerState(backend=backend)
    sid = await state_a.create_session({})

    rounds = 500
    touched: list[bool] = []

    async def hammer(state: ServerState) -> None:
        for _ in range(rounds):
            touched.append(await state.touch_session(sid))

    start = time.time()
    await asyncio.gather(hammer(state_a), hammer(state_b))

    assert all(touched), f'{touched.count(False)} touch() calls returned False'
    final = await state_a.get_session_last_heartbeat(sid)
    assert final is not None and final >= start, \
        'final heartbeat predates the test start — at least one write was lost'
