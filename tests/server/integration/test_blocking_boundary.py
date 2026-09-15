# Copyright (c) ModelScope Contributors. All rights reserved.
"""Blocking_Call_Boundary integration tests (T3.8 / R9#2 / Property 3-4).

These exercise the real ``TaskQueueMixin.call_backend`` through a minimal harness
that sets only the two attributes it uses (a dedicated executor and the optional
Admission_Gate), constructed exactly as ``_init_task_queue`` does. The backend is a
deliberately slow plain callable -- no GPU, Megatron, or Ray involved.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from twinkle.server.utils.task_queue.mixin import TaskQueueMixin
from twinkle.server.utils.task_queue.types import BackendBusyError


class _Harness(TaskQueueMixin):
    """Minimal holder exposing the real call_backend with a chosen gate setting."""

    def __init__(self, gate_enabled: bool) -> None:
        self._backend_executor = ThreadPoolExecutor(thread_name_prefix='twinkle-backend')
        self._backend_admission = asyncio.Semaphore(1) if gate_enabled else None

    def close(self) -> None:
        self._backend_executor.shutdown(wait=False)


@pytest.mark.asyncio
async def test_healthz_style_probe_responsive_during_slow_backend():
    """Property 3: while a slow backend call is in flight, an admit=False probe
    (as /healthz uses) returns well within 5 seconds."""
    h = _Harness(gate_enabled=True)
    try:
        slow = asyncio.create_task(h.call_backend(lambda: time.sleep(3.0)))
        await asyncio.sleep(0.05)  # let the slow call take the gate + a thread

        loop = asyncio.get_running_loop()
        start = loop.time()
        probe = await h.call_backend(lambda: 'pong', admit=False)  # no gate, like the ping probe
        elapsed = loop.time() - start

        assert probe == 'pong'
        assert elapsed < 5.0
        await slow
    finally:
        h.close()


@pytest.mark.asyncio
async def test_gate_held_by_leaked_call_fast_fails_next_task():
    """Property 4 / R2#4: a call that outlives its wait_for keeps the gate; the next
    admitting call fails fast with BackendBusyError instead of entering the backend."""
    h = _Harness(gate_enabled=True)
    entered = {'count': 0}

    def slow():
        time.sleep(1.5)

    def would_enter_backend():
        entered['count'] += 1
        return 'should-not-run'

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(h.call_backend(slow), timeout=0.3)

        # The leaked thread still holds the gate.
        with pytest.raises(BackendBusyError):
            await h.call_backend(would_enter_backend)
        assert entered['count'] == 0  # never reached the backend

        # After the leaked thread truly finishes, the gate frees on its own.
        await asyncio.sleep(1.6)
        assert await h.call_backend(would_enter_backend) == 'should-not-run'
        assert entered['count'] == 1
    finally:
        h.close()


@pytest.mark.asyncio
async def test_sampler_without_gate_runs_two_calls_concurrently():
    """R9#2 case 3 / opt-in: with the gate disabled (SamplerManagement), two backend
    calls are in flight at once rather than serialized."""
    h = _Harness(gate_enabled=False)
    try:
        loop = asyncio.get_running_loop()
        start = loop.time()
        results = await asyncio.gather(
            h.call_backend(lambda: (time.sleep(1.0), 'a')[1]),
            h.call_backend(lambda: (time.sleep(1.0), 'b')[1]),
        )
        elapsed = loop.time() - start

        assert sorted(results) == ['a', 'b']
        assert elapsed < 1.8  # concurrent, not ~2.0s serialized
    finally:
        h.close()
