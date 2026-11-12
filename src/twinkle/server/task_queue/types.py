# Copyright (c) ModelScope Contributors. All rights reserved.
"""In-process execution records, never HTTP/Ray wire models.

Coroutine factories, asyncio tasks, and callbacks belong here. Public task and
queue state enums live in twinkle.protocol.types and are shared by SDK/server.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field


@dataclass
class ExecutionScope:
    """Completion witnesses for an operation, independent of its client future."""
    jobs: set[asyncio.Task] = field(default_factory=set)
    request_ids: set[str] = field(default_factory=set)

    def track(self, job: asyncio.Task, request_id: str | None = None) -> None:
        self.jobs.add(job)
        if request_id is not None:
            self.request_ids.add(request_id)

    async def wait(self):
        observed = set()
        while pending := self.jobs - observed:
            await asyncio.gather(*pending, return_exceptions=True)
            observed.update(pending)


@dataclass
class QueuedTask:
    """Dataclass representing a task waiting in the compute queue."""
    request_id: str
    coro_factory: Callable[[], Coroutine]
    model_id: str | None
    token: str | None
    input_tokens: int
    task_type: str | None
    created_at: float
    on_finished: Callable[[bool, bool], Coroutine] | None = None
    execution_scope: ExecutionScope | None = None
