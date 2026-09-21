# Copyright (c) ModelScope Contributors. All rights reserved.
"""Helpers for constructing protocol-layer ``ErrorPayload`` values."""
from __future__ import annotations

from typing import Any

from twinkle.protocol.types.errors import ErrorCategory, ErrorPayload

_ERROR_MAX = 1024
_TRACEBACK_MAX = 65536
_TRUNCATION_MARKER = '...[traceback truncated, tail kept]...\n'


def trim_traceback(text: str) -> str:
    """Keep the tail of an over-long traceback (innermost frames are densest)."""
    if len(text) <= _TRACEBACK_MAX:
        return text
    keep = _TRACEBACK_MAX - len(_TRUNCATION_MARKER)
    return _TRUNCATION_MARKER + text[-keep:]


def build_error_payload(
    error: str,
    *,
    request_id: str,
    error_code: int = 500,
    category: ErrorCategory | str = ErrorCategory.Server,
    traceback_text: str | None = None,
    details: list[dict[str, Any]] | None = None,
) -> ErrorPayload:
    """Build a validated error payload with bounded diagnostic text."""
    if isinstance(category, str):
        category = ErrorCategory(category.lower())
    tb = trim_traceback(traceback_text) if category is ErrorCategory.Server and traceback_text else None
    lines = str(error).splitlines()
    summary = (lines[0] if lines else '')[:_ERROR_MAX]
    return ErrorPayload(
        error=summary,
        category=category,
        error_code=error_code,
        request_id=request_id,
        traceback=tb,
        details=details,
    )


def task_error_payload(
    error: str,
    *,
    request_id: str,
    error_code: int = 500,
    category: ErrorCategory | str = ErrorCategory.Server,
    traceback_text: str | None = None,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe wire payload for direct or streaming responses."""
    payload = build_error_payload(
        error,
        request_id=request_id,
        error_code=error_code,
        category=category,
        traceback_text=traceback_text,
        details=details,
    )
    return payload.model_dump(mode='json', exclude_none=True)
