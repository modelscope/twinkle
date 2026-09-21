# Copyright (c) ModelScope Contributors. All rights reserved.
"""The one place a FutureRecord becomes a TaskEnvelope.

Both the Submit_Endpoint and the Retrieve_Endpoint go through this function so a
``failed`` status always lands in ``error`` and never in ``result``. Duplicating
this mapping per endpoint is how a task that failed inside the Inline_Fast_Path
window loses its payload.
"""
from __future__ import annotations

from typing import Any

from twinkle.server.state.models import FutureFailureRecord
from twinkle.server.task_errors import trim_traceback
from twinkle_client.types.errors import ErrorCategory, ErrorPayload
from twinkle_client.types.lifecycle import TaskEnvelope

# Keys must equal ``state.models.FAILURE_REASON_CODES`` (guarded by test_envelope).
_FAILURE_WIRE: dict[str, tuple[int, ErrorCategory]] = {
    'invalid_request': (400, ErrorCategory.User),
    'request_rejected': (400, ErrorCategory.User),
    'resource_not_found': (404, ErrorCategory.User),
    'full_mode_busy': (409, ErrorCategory.User),
    'input_tokens_exceeded': (422, ErrorCategory.User),
    'batch_size_invalid': (422, ErrorCategory.User),
    'rate_limit_exceeded': (429, ErrorCategory.User),
    'resource_quota_exceeded': (429, ErrorCategory.User),
    'cancelled': (499, ErrorCategory.User),
    'endpoint_unavailable': (501, ErrorCategory.Server),
    'state_contention': (503, ErrorCategory.Server),
    'backend_gate_unavailable': (503, ErrorCategory.Server),
    'orphaned_replica': (503, ErrorCategory.Server),
    'execution_timeout': (504, ErrorCategory.Server),
    'deadline_exceeded': (500, ErrorCategory.Server),
    'internal_error': (500, ErrorCategory.Server),
}


def error_payload_from_failure(stored: Any, *, request_id: str) -> ErrorPayload:
    """Map one protocol-independent failure record to Twinkle's wire model."""
    failure = FutureFailureRecord.model_validate(stored)
    error_code, category = _FAILURE_WIRE.get(
        failure.reason_code,
        (500, ErrorCategory.Server),
    )
    diagnostic = failure.diagnostic if category is ErrorCategory.Server else None
    return ErrorPayload(
        error=failure.message[:1024],
        category=category,
        error_code=error_code,
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
