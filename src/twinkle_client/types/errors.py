# Copyright (c) ModelScope Contributors. All rights reserved.
"""Structured failure payload -- the single representation of a failure.

Twinkle <-> tinker exception mapping (verified, kept here so a future new exception
class can be lined up against its tinker counterpart):

- Tinker 0.29.0 ``RequestFailedError`` (``tinker/_exceptions.py``; carries
  ``message`` / ``request_id`` / ``category``) is the "the task completed in a failed
  terminal state" exception. Its wire values are ``unknown`` / ``server`` /
  ``user``, matching :class:`ErrorCategory`; legacy TitleCase values are normalized.

``twinkle_client/utils/patch_tinker.py`` shows two SDKs can coexist in one process,
so a semantically-equal but differently-named exception must be lookup-able.
"""
from __future__ import annotations

from enum import StrEnum
from pydantic import Field, field_validator, model_validator
from typing import Any, Literal, Optional

from .base import ResponseModel

# Closed value set, kept in sync with the server-side ``QueueState`` enum values
# (a consistency test asserts the two sets are equal). Wire fields carrying a queue
# state declare this alias, never a bare ``str`` (naming ruling 4).
QueueStateLiteral = Literal['active', 'paused_rate_limit', 'paused_capacity', 'unknown']


class ErrorCategory(StrEnum):
    """Error attribution. Matches tinker's ``RequestErrorCategory``."""

    Unknown = 'unknown'
    Server = 'server'
    User = 'user'


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

    @field_validator('category', mode='before')
    @classmethod
    def normalize_legacy_category(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode='after')
    def traceback_is_server_only(self) -> 'ErrorPayload':
        if self.traceback is not None and self.category is not ErrorCategory.Server:
            raise ValueError('traceback is only valid for server errors')
        return self
