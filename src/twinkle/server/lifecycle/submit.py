# Copyright (c) ModelScope Contributors. All rights reserved.
"""Submit_Endpoint shell and the named seams every queued handler shares.

The Inline_Fast_Path wait itself (``submit_and_peek``) lives on
:class:`~twinkle.server.utils.task_queue.mixin.TaskQueueMixin`, since it operates on
queue state; this module owns the request-shaped pieces around it.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Coroutine
from fastapi import Request
from typing import Any

from twinkle.data_format import InputFeature, Trajectory
from twinkle.server.utils.validation import get_session_id_from_request
from twinkle_client.types.lifecycle import TaskEnvelope

# --------------------------------------------------------------------------- #
# Named seams. This spec implements the current semantics; the server-request-schema
# spec later replaces these function bodies without touching the shell or the return
# path, so the two specs edit disjoint regions.
# --------------------------------------------------------------------------- #


def to_backend_inputs(inputs: Any, *, single: bool = False) -> Any:
    """Seam A: convert raw dict/list inputs to InputFeature / Trajectory objects.

    With ``single=False`` (default) a *batch* is returned: a list of parsed objects
    for a list input, a one-element list for a single dict, and the value unchanged
    otherwise. With ``single=True`` exactly one parsed object is returned (the
    streaming path accepts only one input): a list must contain exactly one element
    or a ``ValueError`` is raised, a dict is parsed to a single object, and anything
    else is passed through. Element typing is unchanged: a dict with ``input_ids``
    becomes an ``InputFeature``, otherwise a ``Trajectory``.
    """
    if single:
        if isinstance(inputs, list):
            if len(inputs) != 1:
                raise ValueError('Streaming only supports a single input')
            inputs = inputs[0]
        if isinstance(inputs, dict):
            return InputFeature(**inputs) if 'input_ids' in inputs else Trajectory(**inputs)
        return inputs
    if isinstance(inputs, list) and inputs:
        first = inputs[0]
        if isinstance(first, dict) and 'input_ids' in first:
            return [InputFeature(**item) for item in inputs]
        return [Trajectory(**item) for item in inputs]
    if isinstance(inputs, dict):
        if 'input_ids' in inputs:
            return [InputFeature(**inputs)]
        return [Trajectory(**inputs)]
    return inputs


def backend_kwargs(body: Any) -> dict[str, Any]:
    """Seam B: the passthrough kwargs forwarded to the backend call."""
    return body.model_extra or {}


def input_metrics(self, body: Any, *, data_parallel: bool = False) -> dict[str, Any]:
    """Seam C: scheduling metrics (input_tokens, and batch_size/data_world_size).

    Defensive shape (isinstance guards + .get defaults) because the body is not yet
    strictly validated; a non-dict element must not raise here.
    """
    inputs = body.inputs
    inputs_list = inputs if isinstance(inputs, list) else [inputs]
    input_tokens = sum(len(inp.get('input_ids', [])) if isinstance(inp, dict) else 0 for inp in inputs_list)
    metrics: dict[str, Any] = {'input_tokens': input_tokens}
    if data_parallel:
        metrics['batch_size'] = len(inputs_list)
        metrics['data_world_size'] = self.data_world_size
    return metrics


def resolve_twinkle_adapter_name(request: Request, adapter_name: str | None) -> str | None:
    """Build a stable per-session adapter name, falling back to request_id for older clients."""
    if adapter_name is None or adapter_name == '':
        return None
    owner_id = get_session_id_from_request(request) or request.state.request_id
    return owner_id + '-' + adapter_name


async def run_submit(
    self,
    request: Request,
    body: Any,
    *,
    task_type: str,
    backend_call: Callable[..., Coroutine],
    metrics: Callable[[Any, Any], dict[str, Any]] | None = None,
    assert_resource: bool = True,
) -> TaskEnvelope:
    """The common Submit_Endpoint judgment sequence, called by every queued
    twinkle-native handler instead of being repeated in each.

    Order is load-bearing: request start -> adapter resolution -> ``submit_and_peek``
    (whose ``schedule_task`` runs preflight). Every admission check runs before any
    state write, so a rejected request writes nothing.

    A plain helper, not a signature-rewriting decorator: each handler keeps its natural
    FastAPI signature so the app stays shallow enough for Ray Serve to cloudpickle (a
    signature-patching wrapper once deepened the route graph past CPython's C-stack
    recursion guard during ``serve.ingress``).

    Four queued endpoints deliberately do NOT route through here and call
    ``submit_and_peek`` / ``submit_background_and_peek`` directly: model
    ``add_adapter_to_model`` (creates the adapter, so the resource assertion cannot
    apply and it owns the train_mode/full-mode checks), model ``upload_to_hub`` (pure
    I/O, background task), and sampler ``sample`` / ``sample_to_data_plane`` (no adapter
    semantics). Zero-write and admission still hold for them because both live inside
    ``schedule_task`` -> ``_perform_preflight_checks``, not in this shell.

    ``backend_call(self, body, adapter_name, token)`` runs the endpoint-specific call
    and returns the JSON-safe task result. ``metrics(self, body)`` supplies scheduling
    kwargs; omit it for control-plane ops. ``assert_resource`` guards on the adapter
    existing before the work runs; set it False for endpoints that create/drop it.
    """
    token = await self._on_request_start(request)
    adapter_name = resolve_twinkle_adapter_name(request, body.adapter_name)

    schedule_kwargs = metrics(self, body) if metrics is not None else {}

    async def _task():
        if assert_resource:
            self.assert_resource_exists(adapter_name)
        return await backend_call(self, body, adapter_name, token)

    # ---- Idempotent dedup: a client-supplied seq_id makes a retried stateful op
    # apply at most once. Claim (session_id, adapter, seq_id) -> request_id atomically
    # before enqueue; a hit returns the original task's envelope instead of re-enqueuing.
    # Only grad-mutating client calls set seq_id, so other endpoints skip this.
    #
    # The adapter must be part of the key. Each client model object owns its own seq
    # counter starting at 1, while session_id is process-global -- so two adapters
    # trained from one process would collide on (session, seq) and the second
    # forward_backward would be dropped as a duplicate AND handed the first adapter's
    # loss. That is a silent wrong-result bug, i.e. the exact failure this dedup
    # exists to prevent, one level up. ----
    request_id = f'req_{uuid.uuid4().hex}'
    seq_id = getattr(body, 'seq_id', None)
    dedup_key = None
    if seq_id is not None:
        session_id = get_session_id_from_request(request) or request.state.request_id
        dedup_key = f'seq::{session_id}::{adapter_name or "-"}::{seq_id}'
        ttl = int(self._task_queue_config.effective_execution_timeout) + 60
        prior_request_id = await self.state.claim_seq(dedup_key, request_id, ttl)
        if prior_request_id is not None:
            return await self._peek_terminal(prior_request_id, fallback_status='pending')

    # ---- Decision_Boundary: preflight (in schedule_task) then peek ----
    try:
        return await self.submit_and_peek(
            _task, model_id=adapter_name, token=token, task_type=task_type, request_id=request_id, **schedule_kwargs)
    except Exception:
        # Release the seq claim only when the task never made it onto the queue --
        # decided by whether a future record exists, NOT by the exception type. A
        # preflight rejection raises before any record is written, so releasing lets
        # a retry re-enqueue. But a failure *after* enqueue (e.g. a transient state
        # error inside the peek) leaves a live task that will still run; releasing
        # there would let a retry enqueue a duplicate -> the exact double-apply this
        # dedup prevents. When unsure (record exists), keep the claim.
        if dedup_key is not None and await self.state.get_future(request_id) is None:
            await self.state.release_seq(dedup_key)
        raise
