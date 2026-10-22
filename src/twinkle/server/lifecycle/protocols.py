# Copyright (c) ModelScope Contributors. All rights reserved.
"""Static declarations of what a deployment class must provide.

Not a new layer and not a base class: these are ``typing.Protocol`` declarations that
turn the implicit host contract of ``run_submit`` / ``input_metrics`` / the queue mixins
into something a type checker can check. Today those requirements are satisfied by duck
typing and ``getattr`` fallbacks, so a missing attribute surfaces at run time -- sometimes
inside a coroutine sitting in the compute queue.

Deliberately two layers, matching the two real shapes: every queued deployment
(Gateway/Model/Sampler/Processor) satisfies ``QueuedDeployment``; only ``ModelManagement``
satisfies ``DataParallelDeployment`` (it is the one with ``data_world_size``).
``DataPlaneManagement`` satisfies neither -- it has no ``state`` and opts out of the
cleanup middleware via ``attach_cleanup_middleware=False``; the gate below is what makes
that fact visible statically instead of only via that boolean.
"""
from __future__ import annotations

from fastapi import Request
from typing import Any, Protocol, runtime_checkable

from twinkle.server.lifecycle.envelope import TaskEnvelope
from twinkle.server.state import ServerState
from twinkle.server.task_queue.config import TaskQueueConfig


@runtime_checkable
class QueuedDeployment(Protocol):
    """A deployment that admits requests through the compute queue."""

    state: ServerState
    replica_id: str

    @property
    def task_queue_config(self) -> TaskQueueConfig:
        ...

    async def _on_request_start(self, request: Request) -> str:
        ...

    def assert_resource_exists(self, resource_id: str | None) -> None:
        ...

    async def _peek_terminal(self, request_id: str, *, fallback_status: str) -> TaskEnvelope:
        ...

    async def submit_and_peek(self, *args: Any, **kwargs: Any) -> TaskEnvelope:
        ...

    async def call_backend(self, fn: Any, /, *args: Any, admit: bool = True, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class DataParallelDeployment(QueuedDeployment, Protocol):
    """A queued deployment that also shards a batch across data-parallel ranks."""

    @property
    def data_world_size(self) -> int:
        ...
