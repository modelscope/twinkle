# Copyright (c) ModelScope Contributors. All rights reserved.
"""The request-lifecycle wire model: one envelope for submit and retrieve.

This module is a public-contract carrier imported across packages (Twinkle_Server
reverse-imports ``twinkle_client.types``); per the naming rulings in ``base.py`` it
therefore intentionally carries **no** underscore prefix.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from .base import ResponseModel, StrictRequest
from .errors import ErrorPayload, QueueStateLiteral

# The lifecycle status of a queued task. Kept in sync with the server-side
# ``TaskStatus`` enum values (a consistency test asserts the two sets are equal).
TaskStatus = Literal['pending', 'queued', 'running', 'completed', 'failed', 'cancelled']

# The two states past which a task never changes again. ``frozenset`` so a caller
# cannot mutate the shared set.
TERMINAL_STATUSES: frozenset[str] = frozenset({'completed', 'failed', 'cancelled'})


class RetrieveFutureRequest(StrictRequest):
    """Body of ``POST /twinkle/retrieve_future``.

    ``request_id`` is the only field: the caller already knows which adapter it
    targeted, so no ``model_id`` is needed to correlate the reply. A brand-new
    endpoint with no legacy clients, so it takes the strict base (unknown fields
    fail loudly) rather than tolerating extras.
    """

    request_id: str


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
    queue_state: QueueStateLiteral | None = None
    queue_state_reason: str | None = None
