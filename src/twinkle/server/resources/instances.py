# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared instance supervision and bounds, with feature-owned records and policies."""
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class InstanceLifecycle:
    """Own tasks, not a second copy of instance records or persisted status."""

    def __init__(self, label, *, clock=time.monotonic):
        self.label = label
        self.clock = clock
        self.unreachable_since = {}
        self.jobs = set()
        self.startups = {}
        self.closing = {}
        self.named_jobs = {}

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.jobs.add(task)

        def done(completed):
            self.jobs.discard(completed)
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    logger.error(
                        '%s background job failed', self.label, exc_info=(type(error), error, error.__traceback__))

        task.add_done_callback(done)
        return task

    def spawn_once(self, key, factory):
        task = self.named_jobs.get(key)
        if task is None:
            task = self.spawn(factory())
            self.named_jobs[key] = task
            task.add_done_callback(lambda completed: self.named_jobs.pop(key, None))
        return task

    def start(self, identity, factory):
        if identity not in self.startups:
            self.startups[identity] = self.spawn(factory())
        return self.startups[identity]

    async def cancel_start(self, identity):
        task = self.startups.get(identity)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self, identity, stop, timeout, stopped, quarantined):
        task = self.closing.get(identity)
        if task is None:

            async def close_once():
                try:
                    await asyncio.wait_for(stop(), timeout)
                except Exception as error:
                    logger.warning(
                        '%s instance %s shutdown is unconfirmed',
                        self.label,
                        identity,
                        exc_info=(type(error), error, error.__traceback__))
                    await quarantined(error)
                else:
                    # Only the feature can retire its users/quota and commit STOPPED.
                    await stopped()
                    self.unreachable_since.pop(identity, None)

            task = self.spawn(close_once())
            self.closing[identity] = task

            def clear(completed):
                if self.closing.get(identity) is completed:
                    self.closing.pop(identity, None)

            task.add_done_callback(clear)
        await asyncio.shield(task)

    async def health(self, identity, probe, *, unreachable_timeout=30):
        """Distinguish explicit backend failure from a transient control-plane timeout."""
        try:
            result = await probe()
        except Exception:
            since = self.unreachable_since.setdefault(identity, self.clock())
            return False if self.clock() - since >= unreachable_timeout else None
        self.unreachable_since.pop(identity, None)
        return not result['unhealthy']

    @staticmethod
    def owned_count(instances):
        return sum(instance.state != 'STOPPED' for instance in instances)

    def require_released(self, instances):
        """A shutdown ACK must certify release before its owner can be removed."""
        owned = self.owned_count(instances)
        if owned:
            raise RuntimeError(f'{self.label} shutdown is unconfirmed: {owned} instances remain owned')

    @classmethod
    def headroom(cls, instances, maximum):
        return float('inf') if maximum is None else max(0, maximum - cls.owned_count(instances))

    @classmethod
    def missing(cls, instances, minimum, maximum, eligible):
        instances = tuple(instances)
        needed = max(0, minimum - sum(eligible(instance) for instance in instances))
        return min(needed, cls.headroom(instances, maximum))

    async def join(self):
        while self.jobs:
            tasks = tuple(self.jobs)
            await asyncio.gather(*tasks, return_exceptions=True)
            self.jobs.difference_update(task for task in tasks if task.done())
