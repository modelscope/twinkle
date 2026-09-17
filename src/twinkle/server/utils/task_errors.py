# Copyright (c) ModelScope Contributors. All rights reserved.
"""Construction and backward-compatible reading of failure payloads.

``ErrorPayload`` is the single representation of a failure both on the wire and in
state. This module owns the two entry points that produce/repair it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from twinkle_client.types.errors import ErrorCategory, ErrorPayload

_ERROR_MAX = 1024
_TRACEBACK_MAX = 65536
_TRUNCATION_MARKER = '...[traceback truncated, tail kept]...\n'


def _trim_traceback(text: str) -> str:
    """Keep the tail of an over-long traceback (innermost frames are densest)."""
    if len(text) <= _TRACEBACK_MAX:
        return text
    keep = _TRACEBACK_MAX - len(_TRUNCATION_MARKER)
    return _TRUNCATION_MARKER + text[-keep:]


def task_error_payload(
    error: str,
    *,
    request_id: str,
    error_code: int = 500,
    category: ErrorCategory | str = ErrorCategory.Server,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    """Build an ``ErrorPayload`` and return it as a JSON-safe dict for storage.

    Traceback splitting and length trimming happen here so over-long text is never
    written to State_Backend. A ``user`` category carries no traceback.
    """
    if isinstance(category, str):
        category = ErrorCategory(category.lower())
    tb = _trim_traceback(traceback_text) if category is ErrorCategory.Server and traceback_text else None
    lines = str(error).splitlines()
    summary = (lines[0] if lines else '')[:_ERROR_MAX]
    payload = ErrorPayload(
        error=summary,
        category=category,
        error_code=error_code,
        request_id=request_id,
        traceback=tb,
    )
    return payload.model_dump(mode='json', exclude_none=True)


def error_payload_from_stored(stored: Any, *, request_id: str) -> ErrorPayload:
    """Build an ``ErrorPayload`` from whatever is sitting in ``FutureRecord.result``.

    Records written before this spec have only ``{error, category}``. Missing
    ``error_code`` / ``request_id`` / ``category`` are backfilled with ``500`` /
    the caller-supplied value / ``Unknown`` so a rolling upgrade never raises
    ``pydantic.ValidationError``.
    """
    if isinstance(stored, Mapping):
        data = dict(stored)
    else:
        data = {'error': 'Unknown error' if stored is None else str(stored)}
    data.setdefault('category', ErrorCategory.Unknown)
    data.setdefault('error_code', 500)
    data.setdefault('request_id', request_id)
    category = str(data['category']).lower()
    if category != ErrorCategory.Server.value:
        data.pop('traceback', None)
    return ErrorPayload.model_validate(data)
