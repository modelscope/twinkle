# Copyright (c) ModelScope Contributors. All rights reserved.
"""In-process execution records, never HTTP/Ray wire models.

Coroutine factories, asyncio tasks, and callbacks belong here. Public task and
queue state enums live in twinkle.protocol.types and are shared by SDK/server.
"""
from __future__ import annotations
import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from twinkle_server.telemetry.correlation import BASE_MODEL, TASK_ID, TENANT_FINGERPRINT, TRAINING_ID
from twinkle_server.telemetry.tracing import get_trace_id


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
    tenant_fingerprint: str | None = None
    trace_context: Any = None
    base_model: str | None = None

    training_id: str | None = None

    def correlation_attrs(self) -> dict[str, str]:
        fields = {
            TRAINING_ID: self.training_id,
            TASK_ID: self.request_id,
            TENANT_FINGERPRINT: self.tenant_fingerprint,
            BASE_MODEL: self.base_model,
            'twinkle.task_type': self.task_type or 'unknown',
        }
        return {key: value for key, value in fields.items() if value is not None}

    def log_fields(self, outcome: str, trace_id: str | None = None) -> dict[str, str]:
        fields = {key.replace('.', '_'): value for key, value in self.correlation_attrs().items()}
        fields['twinkle_task_outcome'] = outcome
        if trace_id := trace_id or get_trace_id(self.trace_context):
            fields['twinkle_trace_id'] = trace_id
        return fields
