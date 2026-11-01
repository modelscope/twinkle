# Copyright (c) ModelScope Contributors. All rights reserved.
"""The one place a FutureRecord becomes a TaskEnvelope.

Both the Submit_Endpoint and the Retrieve_Endpoint go through this function so a
``failed`` status always lands in ``error`` and never in ``result``. Duplicating
this mapping per endpoint is how a task that failed inside the Inline_Fast_Path
window loses its payload.
"""
from __future__ import annotations

from typing import Any

from twinkle.protocol.types.errors import ErrorCategory, ErrorPayload
from twinkle.protocol.types.lifecycle import TaskEnvelope
from twinkle.server.state.models import FutureFailureRecord
from twinkle.server.task_errors import FailureResponse, trim_traceback


def error_payload_from_failure(stored: Any, *, request_id: str) -> ErrorPayload:
    """Map one protocol-independent failure record to Twinkle's wire model."""
    failure = FutureFailureRecord.model_validate(stored)
    response = FailureResponse.for_reason(failure.reason_code)
    diagnostic = failure.diagnostic if response.category is ErrorCategory.Server else None
    return ErrorPayload(
        error=failure.message[:1024],
        category=response.category,
        error_code=response.error_code,
        request_id=request_id,
        traceback=trim_traceback(diagnostic) if diagnostic else None,
        details=failure.details,
    )


def envelope_from_record(
    request_id: str,
    record: dict[str, Any] | None,
    *,
    fallback_status: str = 'pending',
) -> TaskEnvelope:
    """Map a stored ``FutureRecord`` dict to the wire ``TaskEnvelope``.

    Failed and cancelled records carry a protocol-independent ``failure`` field.
    This is the only place that maps those domain reasons to Twinkle's
    ``ErrorPayload`` status/category vocabulary. Legacy failures embedded in
    ``result`` are intentionally unsupported because that format was never merged.

    ``completed`` with ``result is None`` is a valid success (``step`` /
    ``zero_grad`` / ``lr_step`` all return ``None``); it is NOT treated as a
    failure. The tinker endpoint's ``HTTPException(500, 'Task completed but no
    result found')`` is a bug that this function deliberately does not copy.

    ``failed`` and ``cancelled`` both carry an ``ErrorPayload`` in ``error`` (the
    cancel payload is stored the same way a failure payload is), so the client can
    distinguish them by ``status`` while reading one field.
    """
    record = record or {}
    status = record.get('status', fallback_status)
    common = dict(
        queue_state=record.get('queue_state'),
        queue_state_reason=record.get('queue_state_reason'),
    )
    if status in ('failed', 'cancelled'):
        return TaskEnvelope(
            request_id=request_id,
            status=status,
            error=error_payload_from_failure(record.get('failure'), request_id=request_id),
            **common,
        )
    return TaskEnvelope(
        request_id=request_id,
        status=status,
        result=record.get('result') if status == 'completed' else None,
        **common,
    )
