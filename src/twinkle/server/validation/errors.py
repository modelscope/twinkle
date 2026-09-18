# Copyright (c) ModelScope Contributors. All rights reserved.
"""Deliver a request-body validation failure in the same shape as every other error.

FastAPI's default handler answers ``RequestValidationError`` with
``{"detail": [...]}``, which is a second error shape on the wire: a client that learned
to read ``error`` / ``category`` / ``error_code`` from :class:`ErrorPayload` -- what
every other twinkle failure uses -- gets nothing it recognises from a 422. Registering
this on the shared app builder makes the Model, Sampler and Processor deployments answer
identically.

The per-field ``details`` are the point of a 422: they name the offending field, its
path inside the body, and why it was rejected, so a caller can fix the request without
guessing. There is no traceback -- a rejected body is the caller's problem, not a crash,
and pydantic's error list already localises it exactly.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import Any

from twinkle_client.types.errors import ErrorCategory, ErrorPayload

# A body can produce hundreds of errors (one per element of a mis-typed tensor), and a
# response listing all of them helps nobody while costing bandwidth on every retry.
_MAX_DETAILS = 20


def _detail(error: dict[str, Any]) -> dict[str, Any]:
    """One pydantic error as a JSON-safe detail entry."""
    location = [str(part) for part in error.get('loc', ())]
    return {
        'field': location[-1] if location else '',
        'path': '.'.join(location),
        'type': error.get('type', ''),
        'message': error.get('msg', ''),
    }


def _summary(errors: list[dict[str, Any]]) -> str:
    fields = []
    for error in errors:
        path = '.'.join(str(part) for part in error.get('loc', ()))
        if path and path not in fields:
            fields.append(path)
    shown = ', '.join(fields[:_MAX_DETAILS]) or 'request body'
    suffix = '' if len(fields) <= _MAX_DETAILS else f' (+{len(fields) - _MAX_DETAILS} more)'
    return f'Request body validation failed for: {shown}{suffix}'


def _mentions_unknown_field(errors: list[dict[str, Any]]) -> bool:
    return any(error.get('type') == 'extra_forbidden' for error in errors)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Map a body validation failure to a 422 carrying an ``ErrorPayload``."""
    errors = list(exc.errors())
    message = _summary(errors)
    if _mentions_unknown_field(errors):
        # An unknown top-level field is what an older client looks like against a newer
        # server, so say so instead of leaving the caller to infer it from a field list.
        message += ('. Unknown fields are rejected; if this worked before, upgrade '
                    'twinkle-kit on the client to match the server version.')
    payload = ErrorPayload(
        error=message[:1024],
        category=ErrorCategory.User,
        error_code=422,
        request_id=getattr(request.state, 'request_id', None) or '',
        details=[_detail(error) for error in errors[:_MAX_DETAILS]],
    )
    return JSONResponse(status_code=422, content=payload.model_dump(mode='json', exclude_none=True))


def register_validation_error_handler(app: FastAPI) -> None:
    """Install the handler on one deployment app."""
    app.add_exception_handler(RequestValidationError, validation_error_handler)
