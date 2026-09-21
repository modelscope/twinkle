# Copyright (c) ModelScope Contributors. All rights reserved.
"""The blocking-backend-call boundary and the per-replica admission gate.

Conceptually unrelated to "task queue": this is where an event-loop coroutine hands
work to a thread and waits. It was welded onto ``TaskQueueMixin`` because
``tests/server/static/test_no_direct_backend_call.py`` forces every backend call
through ``call_backend``, which made that mixin the sole path to the backend -- one
static check binding two concepts.

This class is a pure callable wrapper: it receives an already-bound ``fn`` from the
caller and only does ``executor.submit(functools.partial(fn, ...))``. It never writes a
``self.model.<method>`` attribute chain and never holds a backend reference, so it is
invisible to the AST scan in ``test_no_direct_backend_call.py`` and needs no exemption
entry.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .types import BackendBusyError


class BackendGate:
    """Owns the two executors, the admission lock, and the poison event.

    Construction must happen inside a running event loop: ``asyncio.Lock()`` and
    ``asyncio.Event()`` bind to the loop that creates them. This holds today because
    ``_init_task_queue`` is called from Ray Serve's ``async def __init__``.
    """

    def __init__(self, *, enable_admission_gate: bool = False) -> None:
        self._executor = ThreadPoolExecutor(thread_name_prefix='twinkle-backend')
        self._probe_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='twinkle-backend-probe')
        self._admission: asyncio.Lock | None = asyncio.Lock() if enable_admission_gate else None
        self._poisoned = asyncio.Event()

    async def _acquire(self, gate: asyncio.Lock) -> None:
        if self._poisoned.is_set():
            raise BackendBusyError('This replica is waiting for a timed-out backend call to exit.')
        if not gate.locked():
            await gate.acquire()
        else:
            acquire_task = asyncio.create_task(gate.acquire())
            poison_task = asyncio.create_task(self._poisoned.wait())
            try:
                done, _ = await asyncio.wait((acquire_task, poison_task), return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                acquire_task.cancel()
                poison_task.cancel()
                await asyncio.gather(acquire_task, poison_task, return_exceptions=True)
                if acquire_task.done() and not acquire_task.cancelled() and acquire_task.result():
                    gate.release()
                raise
            if poison_task in done and self._poisoned.is_set():
                if acquire_task.done() and not acquire_task.cancelled() and acquire_task.result():
                    gate.release()
                else:
                    acquire_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await acquire_task
                raise BackendBusyError('This replica is waiting for a timed-out backend call to exit.')
            poison_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poison_task
            await acquire_task
        if self._poisoned.is_set():
            gate.release()
            raise BackendBusyError('This replica is waiting for a timed-out backend call to exit.')

    async def call(self, fn: Callable[..., Any], /, *args: Any, admit: bool = True, **kwargs: Any) -> Any:
        """Run one backend call outside the event loop.

        Normal model calls serialize through the admission gate. If the awaiting task
        times out while its thread is still running, the gate is poisoned: waiters fail
        immediately until that thread exits. Health probes bypass the gate and use a
        reserved executor thread. Sampler deployments disable the gate because their
        backend owns request concurrency.
        """
        loop = asyncio.get_running_loop()
        gate = self._admission if admit else None
        if gate is not None:
            await self._acquire(gate)

        executor = self._executor if admit else self._probe_executor
        try:
            concurrent_future = executor.submit(functools.partial(fn, *args, **kwargs))
        except Exception:
            if gate is not None and gate.locked():
                gate.release()
            raise

        if gate is not None:

            def release_gate(_future) -> None:

                def release() -> None:
                    self._poisoned.clear()
                    if gate.locked():
                        gate.release()

                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(release)

            concurrent_future.add_done_callback(release_gate)

        try:
            return await asyncio.wrap_future(concurrent_future, loop=loop)
        except asyncio.CancelledError:
            if gate is not None and concurrent_future.running():
                self._poisoned.set()
            raise

    def shutdown(self) -> None:
        # Do not wait on threads that may be leaked on a timed-out backend call.
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._probe_executor.shutdown(wait=False, cancel_futures=True)
