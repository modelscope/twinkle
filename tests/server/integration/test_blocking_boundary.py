# Copyright (c) ModelScope Contributors. All rights reserved.
"""Blocking_Call_Boundary integration tests.

These exercise the real ``TaskQueueMixin.call_backend`` through a minimal harness
that sets only the two attributes it uses (a dedicated executor and the optional
Admission_Gate), constructed exactly as ``_init_task_queue`` does. The backend is a
deliberately slow plain callable -- no GPU, Megatron, or Ray involved.
"""
from __future__ import annotations

import asyncio
import httpx
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.responses import JSONResponse

ray = pytest.importorskip('ray')

from twinkle.server.task_queue.backend_gate import BackendGate  # noqa: E402
from twinkle.server.task_queue.mixin import TaskQueueMixin  # noqa: E402
from twinkle.server.task_queue.types import BackendBusyError  # noqa: E402


class _Harness(TaskQueueMixin):
    """Minimal holder exposing the real call_backend with a chosen gate setting."""

    def __init__(self, gate_enabled: bool, *, max_workers: int | None = None) -> None:
        # ``call_backend`` delegates to the extracted ``BackendGate``; build one and
        # alias its internals so the assertions below still read the same names.
        self._backend_gate = BackendGate(enable_admission_gate=gate_enabled)
        if max_workers is not None:
            self._backend_gate._executor.shutdown(wait=False)
            self._backend_gate._executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix='twinkle-backend')
        self._backend_executor = self._backend_gate._executor
        self._backend_probe_executor = self._backend_gate._probe_executor
        self._backend_admission = self._backend_gate._admission
        self._backend_poisoned = self._backend_gate._poisoned

    def close(self) -> None:
        self._backend_gate.shutdown()


@pytest.mark.asyncio
async def test_healthz_style_probe_responsive_during_slow_backend():
    """While a slow backend call is in flight, an admit=False probe
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
async def test_normal_gate_contention_waits_instead_of_failing():
    h = _Harness(gate_enabled=True)
    try:
        first = asyncio.create_task(h.call_backend(lambda: (time.sleep(0.2), 'first')[1]))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(h.call_backend(lambda: 'second'))
        assert await first == 'first'
        assert await second == 'second'
    finally:
        h.close()


@pytest.mark.asyncio
async def test_cancelled_gate_waiter_does_not_steal_lock():
    h = _Harness(gate_enabled=True)
    release = threading.Event()

    def wait_for_release():
        while not release.is_set():
            time.sleep(0.01)

    try:
        first = asyncio.create_task(h.call_backend(wait_for_release))
        await asyncio.sleep(0.05)
        waiter = asyncio.create_task(h.call_backend(lambda: 'cancelled'))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await first
        assert await h.call_backend(lambda: 'next') == 'next'
    finally:
        release.set()
        h.close()


@pytest.mark.asyncio
async def test_cancelled_queued_backend_call_releases_gate():
    h = _Harness(gate_enabled=True, max_workers=1)
    release_worker = threading.Event()
    occupied = h._backend_executor.submit(release_worker.wait)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(h.call_backend(lambda: 'never-started'), timeout=0.05)
        await asyncio.sleep(0)
        assert not h._backend_admission.locked()
        release_worker.set()
        occupied.result(timeout=5)
        assert await h.call_backend(lambda: 'next') == 'next'
    finally:
        release_worker.set()
        h.close()


@pytest.mark.asyncio
async def test_gate_held_by_leaked_call_fast_fails_next_task():
    """A call that outlives its wait_for keeps the gate; the next
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
async def test_probe_times_out_while_same_serial_actor_is_busy():

    @ray.remote
    class SerialActor:

        def slow(self):
            time.sleep(1.0)

        def ping(self):
            return True

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(num_cpus=1, logging_level='ERROR')
    actor = SerialActor.remote()
    h = _Harness(gate_enabled=True)
    app = FastAPI()

    @app.get('/healthz')
    async def healthz():
        try:
            await h.call_backend(lambda: ray.get(actor.ping.remote(), timeout=0.2), admit=False)
            return {'healthy': True}
        except ray.exceptions.GetTimeoutError:
            return JSONResponse(status_code=503, content={'healthy': False})

    try:
        slow = asyncio.create_task(h.call_backend(lambda: ray.get(actor.slow.remote(), timeout=10)))
        await asyncio.sleep(0.1)
        start = time.monotonic()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get('/healthz')
        assert response.status_code == 503
        assert time.monotonic() - start < 5
        await slow
    finally:
        h.close()
        ray.kill(actor)
        if started_ray:
            ray.shutdown()


@pytest.mark.asyncio
async def test_sampler_without_gate_runs_two_calls_concurrently():
    """Case 3 / opt-in: with the gate disabled (SamplerManagement), two backend
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
