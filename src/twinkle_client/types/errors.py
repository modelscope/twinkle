# Copyright (c) ModelScope Contributors. All rights reserved.
"""Structured failure payload -- the single representation of a failure.

Twinkle <-> tinker exception mapping (verified, kept here so a future new exception
class can be lined up against its tinker counterpart):

- tinker ``RequestFailedError`` (``tinker/_exceptions.py:176-196``; carries
  ``message`` / ``request_id`` / ``category``) is the "the task completed in a failed
  terminal state" exception. Its ``category`` uses the same three values as
  :class:`ErrorCategory` (``Unknown`` / ``Server`` / ``User``).

``twinkle_client/utils/patch_tinker.py`` shows two SDKs can coexist in one process,
so a semantically-equal but differently-named exception must be lookup-able.
"""
from __future__ import annotations

from enum import StrEnum
from pydantic import Field
from typing import Any, Optional

from .base import ResponseModel


class ErrorCategory(StrEnum):
    """Error attribution. Matches tinker's ``RequestErrorCategory``."""

    Unknown = 'Unknown'
    Server = 'Server'
    User = 'User'


class ErrorPayload(ResponseModel):
    """The single representation of a failure, on the wire and in state.

    ``error_code``, not ``status_code``: once server-request-lifecycle lands, an
    execution-time failure is delivered with HTTP 200, so this value is
    *systematically* unequal to the response status code. Keeping the name
    ``status_code`` would make every reader misparse it once. The 400-599 range is
    kept to reuse HTTP's semantic space, not to align with response codes.

    Inherits ``ResponseModel`` (``extra='ignore'``), so a future added field does
    not make an old client fail to parse it.
    """

    error: str = Field(max_length=1024)
    category: ErrorCategory
    error_code: int = Field(ge=400, le=599)
    request_id: str
    traceback: Optional[str] = Field(default=None, max_length=65536)
    details: Optional[list[dict[str, Any]]] = None
