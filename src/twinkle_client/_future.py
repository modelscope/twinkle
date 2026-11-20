# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client_Future_Layer: the one polling implementation in Twinkle_Client.

Public client methods keep their synchronous signatures by calling :func:`resolve`;
no future object is ever exposed. Private module (underscore name) because it is
never imported by Twinkle_Server.
"""
from __future__ import annotations

import logging
import requests
import time
import uuid
from collections.abc import Callable
from typing import Any

from twinkle.protocol.headers import H_REQUEST_ID
from twinkle.protocol.types.lifecycle import TERMINAL_STATUSES, TaskEnvelope, UploadProgress
from twinkle_client.exceptions import TaskCancelledError, TaskFailedError, TaskRecordLostError, TaskWaitTimeoutError
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport

logger = logging.getLogger('twinkle_client')

# An independent constant, NOT derived from any server-side timeout: the server
# guarantees a task reaches a terminal state, so this is only "how long the client is
# willing to wait". Deliberately different from the server's execution_timeout fallback
# so a reader does not think the two are related.
_DEFAULT_TOTAL_TIMEOUT = 7200.0

# A 404 is Retrieve_Endpoint's verdict after a whole Long_Poll_Window, but state
# jitter (Redis blip, actor restart) can hide an in-flight record for one window,
# so bound-retry before declaring the record lost.
_NOT_FOUND_RETRY_MAX = 3

# 5xx / 408 / connection errors are retried with exponential backoff.
_TRANSPORT_RETRY_MAX = 5


def _retrieve_url(transport: ClientTransport) -> str:
    return transport.url('twinkle/retrieve_future')


def _cancel_url(transport: ClientTransport) -> str:
    return transport.url('twinkle/cancel')


def _best_effort_cancel(request_id: str, transport: ClientTransport) -> None:
    """Ask the server to drop a task when the caller abandons the wait (e.g. Ctrl-C).

    Never raises: a failed cancel must not mask the original interrupt. The server
    only drops not-yet-started tasks, so a running task is unaffected.
    """
    try:
        transport.post(_cancel_url(transport), json_data={'request_id': request_id}, timeout=2)
    except BaseException as e:  # noqa: BLE001 - best effort; never mask the interrupt
        logger.debug('[future] best-effort cancel of %s failed: %s', request_id, e)


def _post_retrieve(
    request_id: str, transport: ClientTransport, *, http_request_id: str, after_progress: int | None = None
) -> TaskEnvelope:
    """POST one retrieve and parse the reply into a TaskEnvelope.

    Raises ``requests.HTTPError`` (a :class:`TwinkleHTTPError` after the client
    error-parsing change lands) on a non-2xx response.
    """
    body: dict[str, Any] = {'request_id': request_id}
    if after_progress is not None:
        body['after_progress'] = after_progress
    response = transport.post(_retrieve_url(transport), json_data=body, headers={H_REQUEST_ID: http_request_id})
    return TaskEnvelope.model_validate(response.json())


def _status_of(error: requests.HTTPError) -> int | None:
    status = getattr(error, 'status_code', None)
    if status is None and getattr(error, 'response', None) is not None:
        status = error.response.status_code
    return status


def _is_retryable(status: int) -> bool:
    return status == 408 or 500 <= status <= 599


def _log_queue_state(reply: TaskEnvelope) -> None:
    if reply.queue_state and reply.queue_state != 'active':
        logger.info('[future] task %s waiting: queue_state=%s reason=%s', reply.request_id, reply.queue_state,
                    reply.queue_state_reason)


def _unwrap(env: TaskEnvelope, model_cls) -> Any:
    """Turn a terminal TaskEnvelope into a return value or an exception.

    Takes the envelope *whole* rather than destructured fields: the submit and
    retrieve paths must not be able to pass different subsets. A signature like
    ``(status, result, request_id, model_cls, error=None)`` would let the submit
    path simply never pass ``error`` -- and then every failure completing inside
    the Inline_Fast_Path window would raise 'no recorded payload' while its real
    payload sat unread.
    """
    if env.status == 'failed':
        p = env.error
        raise TaskFailedError(
            p.error,
            category=p.category.value,
            request_id=env.request_id,
            error_code=p.error_code,
            details=p.details,
            traceback=p.traceback,
        )
    if env.status == 'cancelled':
        p = env.error
        raise TaskCancelledError(
            p.error if p is not None else 'Task cancelled',
            request_id=env.request_id,
            error_code=p.error_code if p is not None else None,
        )
    return model_cls.model_validate(env.result) if model_cls is not None else env.result


def resolve(
    submit: TaskEnvelope,
    *,
    model_cls,
    total_timeout: float = _DEFAULT_TOTAL_TIMEOUT,
    transport: ClientTransport | None = None,
    on_progress: Callable[[UploadProgress], None] | None = None,
) -> Any:
    """Block until ``submit``'s task reaches a terminal state, then return its result.

    A terminal submit envelope is unwrapped directly, issuing no Retrieve_Endpoint
    request at all (single round trip for control-plane ops). Otherwise the same
    ``_unwrap`` is applied to each retrieve reply, so a failure inside the
    Inline_Fast_Path window and one observed via retrieve take an identical path.

    The main loop never sleeps: waiting is delegated to Retrieve_Endpoint's
    long-poll. Only transport retries back off.
    """
    cursor = 0

    def report_progress(envelope: TaskEnvelope) -> None:
        nonlocal cursor
        progress = envelope.progress
        if on_progress is not None and progress is not None and progress.sequence > cursor:
            cursor = progress.sequence
            on_progress(progress)

    if submit.status in TERMINAL_STATUSES:
        report_progress(submit)
        return _unwrap(submit, model_cls)  # same call as the retrieve path

    resolved_transport = capture_transport(transport)
    deadline = time.monotonic() + total_timeout
    transport_failures = not_found_count = 0
    poll_request_id = uuid.uuid4().hex
    try:
        report_progress(submit)
        while True:
            if time.monotonic() >= deadline:
                raise TaskWaitTimeoutError(request_id=submit.request_id, waited=total_timeout)
            try:
                cursor_args = {} if on_progress is None else {'after_progress': cursor}
                reply = _post_retrieve(
                    submit.request_id, resolved_transport, http_request_id=poll_request_id, **cursor_args)
                transport_failures = not_found_count = 0
            except (requests.ConnectionError, requests.Timeout):
                transport_failures += 1
                if transport_failures > _TRANSPORT_RETRY_MAX:
                    raise
                time.sleep(min(2**transport_failures, 30))
                continue
            except requests.HTTPError as e:
                status = _status_of(e)
                if status == 404:
                    not_found_count += 1
                    if not_found_count > _NOT_FOUND_RETRY_MAX:
                        raise TaskRecordLostError(request_id=submit.request_id) from e
                    continue
                if status is None or not _is_retryable(status):
                    raise
                transport_failures += 1
                if transport_failures > _TRANSPORT_RETRY_MAX:
                    raise
                time.sleep(min(2**transport_failures, 30))
                continue
            report_progress(reply)
            if reply.status in TERMINAL_STATUSES:
                return _unwrap(reply, model_cls)  # same call as the submit path
            _log_queue_state(reply)
            poll_request_id = uuid.uuid4().hex
    except (KeyboardInterrupt, SystemExit):
        # Caller abandoned the wait: best-effort ask the server to drop the task if it
        # has not started, then re-raise so the interrupt is never swallowed.
        _best_effort_cancel(submit.request_id, resolved_transport)
        raise


def resolve_response(
    response, model_cls, *, transport: ClientTransport | None = None,
    on_progress: Callable[[UploadProgress], None] | None = None,
) -> Any:
    """Resolve a Submit_Endpoint response using the submitter's transport."""
    payload = response.json()
    return resolve(
        TaskEnvelope.model_validate(payload),
        model_cls=model_cls,
        transport=transport,
        # Older servers reject unknown retrieve fields and omit progress in submit replies.
        on_progress=on_progress if 'progress' in payload else None,
    )
