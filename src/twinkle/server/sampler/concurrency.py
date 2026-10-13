# Copyright (c) ModelScope Contributors. All rights reserved.
"""Let each background sampling job own access until its backend has exited."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from typing import Any

from twinkle.server.exceptions import ResourceExhaustedError
from twinkle.server.task_queue.types import BackendBusyError


class SamplerConcurrency:
    """Stable weights admit readers; changing base weights requires one writer.

    A job owns its entire load/generate/cleanup sequence. Cancelling its waiter
    leaves that job running, so a request timeout cannot release live weights.
    All bookkeeping runs on the replica's event loop.
    """

    def __init__(self, limit: int = 128):
        self._limit = limit
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._changed = asyncio.Event()
        self._jobs: set[asyncio.Task] = set()
        self._draining: set[asyncio.Task] = set()
        # Nested polling tasks inherit the owner, without passing lock handles
        # through every backend call. Membership prevents use after job exit.
        self._owner: ContextVar[asyncio.Task | None] = ContextVar('sampler_job', default=None)
        self._uncertain = False
        self._closed = False

    def owns_job(self) -> bool:
        owner = self._owner.get()
        return owner is not None and not owner.done() and owner in self._jobs

    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def check_available(self, *, admitted: bool = False) -> None:
        if self._closed:
            raise BackendBusyError('The sampler is shutting down.')
        if self._uncertain:
            raise BackendBusyError('Sampler weights or backend completion are uncertain; restart this replica.')
        if self._draining and not admitted:
            raise BackendBusyError('The sampler is waiting for backend resource release.')

    def poison(self) -> None:
        self._uncertain = True
        self._notify()

    def close(self) -> None:
        self._closed = True
        self._notify()

    async def _acquire(self, exclusive: bool) -> None:
        if exclusive:
            self._waiting_writers += 1
        try:
            while True:
                self.check_available(admitted=True)
                if asyncio.current_task() in self._draining:
                    raise BackendBusyError('Sampling request exited before its job could start.')
                if not self._writer and (self._readers == 0 if exclusive else self._waiting_writers == 0):
                    if exclusive:
                        self._writer = True
                    else:
                        self._readers += 1
                    return
                await self._changed.wait()
        finally:
            if exclusive:
                self._waiting_writers -= 1
                self._notify()

    async def _execute(self, factory, exclusive: bool):
        await self._acquire(exclusive)
        token = self._owner.set(asyncio.current_task())
        try:
            return await factory()
        finally:
            self._owner.reset(token)
            if exclusive:
                self._writer = False
            else:
                self._readers -= 1
            self._notify()

    def _finished(self, job: asyncio.Task) -> None:
        self._jobs.discard(job)
        self._draining.discard(job)
        if not job.cancelled():
            job.exception()  # Observe failures even when its waiter already left.
        self._notify()

    async def run(self, factory: Callable[[], Coroutine[Any, Any, Any]], *, exclusive: bool = False):
        """Wait for one owned job without transferring cancellation to it."""
        self.check_available()
        # Unlike the future queue's admission count, this capacity is retained
        # until actual backend cleanup, including jobs outliving request timeout.
        if len(self._jobs) >= self._limit:
            raise ResourceExhaustedError('Concurrent sampling capacity is unavailable on this replica.')
        job = asyncio.create_task(self._execute(factory, exclusive))
        self._jobs.add(job)
        job.add_done_callback(self._finished)
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            if not job.done():
                self._draining.add(job)
                self._notify()
            raise
