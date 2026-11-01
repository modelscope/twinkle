# Copyright (c) ModelScope Contributors. All rights reserved.
"""Helpers for constructing protocol-layer ``ErrorPayload`` values."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from twinkle.protocol.types.errors import ErrorCategory, ErrorPayload


@dataclass(frozen=True)
class FailureResponse:
    """How a domain failure is presented inside a response, not its HTTP status."""

    error_code: int
    category: ErrorCategory

    @classmethod
    def for_reason(cls, reason: str) -> FailureResponse:
        policies = cls._policies()
        return policies.get(reason, policies['internal_error'])

    @classmethod
    def reasons(cls) -> frozenset[str]:
        return frozenset(cls._policies())

    @classmethod
    @lru_cache(maxsize=1)
    def _policies(cls) -> dict[str, FailureResponse]:
        return {
            'capacity_unavailable': cls(error_code=409, category=ErrorCategory.User),
            'startup_failed': cls(error_code=500, category=ErrorCategory.Server),
            'startup_timeout': cls(error_code=504, category=ErrorCategory.Server),
            'invalid_request': cls(error_code=400, category=ErrorCategory.User),
            'request_rejected': cls(error_code=400, category=ErrorCategory.User),
            'resource_not_found': cls(error_code=404, category=ErrorCategory.User),
            'full_mode_busy': cls(error_code=409, category=ErrorCategory.User),
            'input_tokens_exceeded': cls(error_code=422, category=ErrorCategory.User),
            'batch_size_invalid': cls(error_code=422, category=ErrorCategory.User),
            'rate_limit_exceeded': cls(error_code=429, category=ErrorCategory.User),
            'resource_quota_exceeded': cls(error_code=429, category=ErrorCategory.User),
            'cancelled': cls(error_code=499, category=ErrorCategory.User),
            'endpoint_unavailable': cls(error_code=501, category=ErrorCategory.Server),
            'state_contention': cls(error_code=503, category=ErrorCategory.Server),
            'backend_gate_unavailable': cls(error_code=503, category=ErrorCategory.Server),
            'orphaned_replica': cls(error_code=503, category=ErrorCategory.Server),
            'backend_cancelled': cls(error_code=503, category=ErrorCategory.Server),
            'execution_timeout': cls(error_code=504, category=ErrorCategory.Server),
            'deadline_exceeded': cls(error_code=500, category=ErrorCategory.Server),
            'internal_error': cls(error_code=500, category=ErrorCategory.Server),
        }


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
