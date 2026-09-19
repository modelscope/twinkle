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

from twinkle.data_format import InputFeature, Trajectory, is_encoded
from twinkle.server.utils.auth import get_session_id_from_request
from twinkle.server.validation import assert_request_supported
from twinkle_client.types.base import FieldRole, fields_with_role
from twinkle_client.types.data import export_batch
from twinkle_client.types.lifecycle import TaskEnvelope

# --------------------------------------------------------------------------- #
# Named seams shared by every queued twinkle-native handler.
# --------------------------------------------------------------------------- #


def to_backend_inputs(inputs: Any, *, single: bool = False) -> Any:
    """Seam A: export wire-validated ``inputs`` as the objects the backend consumes.

    This is an *export*, not a validation step. The request model declares ``inputs``
    as :data:`~twinkle_client.types.data.WireInputBatch`, so a malformed batch is
    already rejected during FastAPI body parsing -- before a future record exists and
    before anything reaches a GPU. Validating here instead would put the first check
    inside the queued task, where a rejection has already cost an enqueue.

    Entries arrive as wire models and are exported with ``exclude_none`` semantics, so
    unset optional fields stay absent (Twinkle_Core branches on key presence) and
    unknown keys the caller sent are preserved. ``InputFeature`` / ``Trajectory`` are
    ``TypedDict``s, so constructing them is a plain dict build.

    With ``single=True`` exactly one object is returned (the streaming path accepts
    only one input) and a batch of any other size is a ``ValueError``. Plain dicts pass
    through unchanged: the data-plane path resolves rows itself and never goes through
    the wire schema.
    """
    entries = export_batch(inputs) if isinstance(inputs, list) else inputs
    if single:
        if isinstance(entries, list):
            if len(entries) != 1:
                raise ValueError('Streaming only supports a single input')
            entries = entries[0]
        if isinstance(entries, dict):
            return _as_backend_entry(entries)
        return entries
    if isinstance(entries, list):
        return [_as_backend_entry(entry) if isinstance(entry, dict) else entry for entry in entries]
    if isinstance(entries, dict):
        return [_as_backend_entry(entries)]
    return entries


def _as_backend_entry(entry: dict[str, Any]) -> Any:
    """One exported entry as its ``TypedDict`` shape."""
    return InputFeature(**entry) if is_encoded(entry) else Trajectory(**entry)


def backend_kwargs(body: Any) -> dict[str, Any]:
    """Seam B: the keyword arguments forwarded to the backend call.

    Exactly two sources, both declared on the request model (see
    :mod:`twinkle_client.types.base`):

    1. fields whose role is ``BackendKwarg``, included iff their value is not ``None``;
    2. the contents of every ``Passthrough`` field, flattened.

    Control fields are never forwarded. That exclusion is the point of the field roles:
    forwarding *all* declared non-``None`` fields would re-send ``inputs`` /
    ``adapter_name`` / ``seq_id``, which the handlers already pass explicitly -- a
    duplicate keyword argument at best, and a protocol field leaking into a backend
    signature at worst.
    """
    model_cls = type(body)
    kwargs: dict[str, Any] = {}
    for name in fields_with_role(model_cls, FieldRole.BackendKwarg):
        value = getattr(body, name, None)
        if value is not None:
            kwargs[name] = value
    for name in fields_with_role(model_cls, FieldRole.Passthrough):
        region = getattr(body, name, None) or {}
        overlap = set(region) & set(kwargs)
        if overlap:
            raise ValueError(f'{name} collides with declared backend parameters: {", ".join(sorted(overlap))}')
        kwargs.update(region)
    return kwargs


def input_metrics(self, body: Any, *, data_parallel: bool = False) -> dict[str, Any]:
    """Seam C: scheduling metrics (input_tokens, and batch_size/data_world_size).

    Reads validated wire models, so no isinstance guards: ``inputs`` is a list and
    ``input_ids`` is either absent or a list of ints.
    """
    inputs = body.inputs
    input_tokens = sum(len(getattr(entry, 'input_ids', None) or ()) for entry in inputs)
    metrics: dict[str, Any] = {'input_tokens': input_tokens}
    if data_parallel:
        metrics['batch_size'] = len(inputs)
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
    capability: str | None = None,
) -> TaskEnvelope:
    """The common Submit_Endpoint judgment sequence, called by every queued
    twinkle-native handler instead of being repeated in each.

    Order is load-bearing: request start -> adapter resolution -> preflight ->
    ``submit_and_peek`` (whose ``schedule_task`` runs its own resource preflight).
    Every admission check runs before any state write, so a rejected request writes
    nothing.

    ``assert_request_supported`` is the one place the request is checked against *this
    deployment*: a parameter that only exists on the other backend and an endpoint this
    backend does not implement are decided here -- before the seq claim and before the
    enqueue, so a rejected request runs on zero data-parallel ranks. Putting these checks
    in the queued task instead would let an incompatible request cost a full GPU dispatch.
    Passthrough keys are forwarded unjudged (no spelling check): see
    :mod:`twinkle.server.validation.backend_compat`.

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
    ``capability`` names the backend capability the endpoint needs, when the endpoint is
    not implemented by every backend.
    """
    token = await self._on_request_start(request)
    adapter_name = resolve_twinkle_adapter_name(request, body.adapter_name)

    # ---- Preflight: decidable from the body plus this deployment's backend, so it
    # runs before any state write and before the enqueue. ----
    assert_request_supported(self, body, capability=capability)

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
