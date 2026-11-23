# Copyright (c) ModelScope Contributors. All rights reserved.
"""The request-lifecycle wire model: one envelope for submit and retrieve.

This module is a public-contract carrier imported across packages (Twinkle_Server
reverse-imports ``twinkle.protocol.types``); per the naming rulings in ``base.py`` it
therefore intentionally carries **no** underscore prefix.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .base import ResponseModel, StrictRequest
from .errors import ErrorPayload, QueueState


class TaskStatus(StrEnum):
    """Canonical task states shared by the wire envelope and server execution."""

    PENDING = 'pending'
    QUEUED = 'queued'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


TERMINAL_STATUSES = frozenset({TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value})


class RetrieveFutureRequest(StrictRequest):
    """Long-poll for a terminal task or progress newer than the optional cursor."""

    request_id: str
    after_progress: int | None = Field(default=None, ge=0)


class CancelRequest(StrictRequest):
    """Body of ``POST /twinkle/cancel``: best-effort cancel of a not-yet-started task.

    New endpoint with no legacy clients, so it takes the strict base (unknown fields
    fail loudly).
    """

    request_id: str


class CancelResponse(ResponseModel):
    """Reply to ``POST /twinkle/cancel``.

    ``cancelled`` is True only when the task is in the terminal ``cancelled`` state
    after the attempt; ``state`` is its status afterwards (``cancelled`` / ``running``
    / ``completed`` / ``failed`` / ``not_found``). A running or already-terminal task
    is never interrupted -- cancel only drops tasks that have not started.
    """

    cancelled: bool
    state: str


class UploadProgress(ResponseModel):
    """Latest upload snapshot. Transfer bytes can be lower than committed bytes after deduplication."""

    sequence: int
    stage: Literal['preparing', 'uploading', 'batch_committed', 'batch_failed']
    uploaded_bytes: int = 0
    total_bytes: int | None = None
    committed_bytes: int = 0


class TaskEnvelope(ResponseModel):
    """The one lifecycle reply, shared by Submit_Endpoint and Retrieve_Endpoint.

    Success and failure live in *different* fields, and both endpoints fill the
    same field for the same meaning. That is the whole point: if failure rode in
    ``result`` on submit but in ``error`` on retrieve, a task that fails inside the
    Inline_Fast_Path window -- which is exactly where ``step`` / ``zero_grad`` /
    ``set_loss`` fail -- would have its payload read from the wrong place and
    silently dropped.

    ``result`` is ``Optional[Any]`` rather than each endpoint's concrete response
    model: this is the lifecycle-layer model, not a per-endpoint generic.
    Deserialization to the concrete model is done by the Client_Future_Layer once
    it holds a terminal envelope, since it knows the caller's expected type.

    No ``model_id``: the caller already knows which adapter it targeted;
    ``request_id`` is the only key needed to correlate a reply.
    """

    request_id: str
    status: TaskStatus
    result: Any | None = None  # set iff status == 'completed'
    error: ErrorPayload | None = None  # set iff status == 'failed'
    queue_state: QueueState | None = None
    queue_state_reason: str | None = None
    progress: UploadProgress | None = None
