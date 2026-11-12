# Copyright (c) ModelScope Contributors. All rights reserved.
"""Coordinate sampling with changes to shared sampler state."""
from __future__ import annotations

import asyncio

from twinkle.server.exceptions import BackendBusyError


class SamplerConcurrency:
    """Readers share stable weights; mutations wait for every reader to finish.

    Jobs retain their lock if an HTTP waiter leaves. Admission and execution
    timeouts belong to the caller; the pool may also bound lock admission.
    """

    def __init__(self):
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._condition = asyncio.Condition()
        self._jobs: set[asyncio.Task] = set()
        self._weights_error: Exception | None = None
        self._closed = False

    def check_available(self):
        if self._closed:
            raise BackendBusyError('The sampler is shutting down.')
        if self._weights_error is not None:
            raise BackendBusyError('A checkpoint write failed; the sampler is restarting.')

    def invalidate_weights(self, error: Exception):
        self._weights_error = error

    def close(self):
        self._closed = True

    async def _execute(self, factory, exclusive: bool, queue_timeout):
        async with self._condition:
            if exclusive:
                self._waiting_writers += 1
            try:

                def available():
                    return not self._writer and (self._readers == 0 if exclusive else self._waiting_writers == 0)

                if not available():
                    await asyncio.wait_for(self._condition.wait_for(available), queue_timeout)
                self.check_available()
                if exclusive:
                    self._writer = True
                else:
                    self._readers += 1
            finally:
                if exclusive:
                    self._waiting_writers -= 1
                    self._condition.notify_all()
        try:
            return await factory()
        finally:
            async with self._condition:
                if exclusive:
                    self._writer = False
                else:
                    self._readers -= 1
                self._condition.notify_all()

    async def run(self, factory, *, exclusive: bool = False, queue_timeout: float | None = None):
        self.check_available()
        started = False

        async def call():
            nonlocal started
            started = True
            return await factory()

        job = asyncio.create_task(self._execute(call, exclusive, queue_timeout))
        self._jobs.add(job)

        def finished(task):
            self._jobs.discard(task)
            if not task.cancelled():
                task.exception()

        job.add_done_callback(finished)
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            if not started:
                job.cancel()
            # The blocking backend may still be using the locked weights.
            await asyncio.gather(job, return_exceptions=True)
            raise

    async def wait_closed(self):
        await asyncio.gather(*self._jobs, return_exceptions=True)
