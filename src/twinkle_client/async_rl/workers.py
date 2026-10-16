# Copyright (c) ModelScope Contributors. All rights reserved.
"""Small client-side lifecycle primitives for composable async-RL workers.

Workers remain concrete long-running roles.  This module deliberately does not
introduce an algorithm graph or a data-dependency DSL; role implementations
coordinate through ordinary asyncio queues and server-side DataRefs.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence


class Worker(ABC):
    """One long-running client-side computation role."""

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError('worker name must not be empty')
        self.name = name

    @abstractmethod
    async def run(self) -> None:
        """Run until this role has drained its input or fails."""


class WorkerPipeline:
    """Run a concrete set of worker roles and propagate failures as one unit."""

    def __init__(self, workers: Sequence[Worker]) -> None:
        self.workers = tuple(workers)
        if not self.workers:
            raise ValueError('at least one worker is required')
        names = [worker.name for worker in self.workers]
        if len(names) != len(set(names)):
            raise ValueError(f'worker names must be unique, got {names}')

    async def run(self) -> None:
        tasks = {
            asyncio.create_task(worker.run(), name=worker.name): worker
            for worker in self.workers
        }
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            failure = next(
                (task.exception() for task in done if not task.cancelled() and task.exception() is not None),
                None,
            )
            if failure is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failure
            await asyncio.gather(*pending)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
