# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal


def _now_iso() -> str:
    # UTC-aware so _parse_timestamp (which reads timestamps back as UTC) agrees with
    # it and with time.time(); a naive local string would be misread as UTC and skew
    # every expiry comparison by the host's UTC offset.
    return datetime.now(timezone.utc).isoformat()


class SessionRecord(BaseModel):
    """Represents a client session."""

    tags: list[str] = Field(default_factory=list)
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    sdk_version: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    last_heartbeat: float = Field(default_factory=time.time)


class ModelRecord(BaseModel):
    """Represents a registered model."""

    token: str
    session_id: str | None = None
    model_seq_id: Any = None
    base_model: str | None = None
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    lora_config: Any = None
    replica_id: str | None = None
    created_at: str = Field(default_factory=_now_iso)


class SamplingSessionRecord(BaseModel):
    """Represents a sampling session."""

    session_id: str | None = None
    seq_id: Any = None
    base_model: str | None = None
    model_path: str | None = None
    created_at: str = Field(default_factory=_now_iso)


class FutureFailureRecord(BaseModel):
    """Protocol-independent reason an asynchronous task failed."""

    reason_code: str
    message: str
    attribution: Literal['user', 'server']
    details: list[dict[str, Any]] | None = None
    diagnostic: str | None = None


# Canonical set of protocol-independent failure reason codes. The Twinkle and Tinker
# gateways each map this same key set to their own wire vocabularies; keeping the set
# in one place lets a consistency test catch a map that drifts out of coverage.
FAILURE_REASON_CODES: frozenset[str] = frozenset({
    'invalid_request',
    'request_rejected',
    'resource_not_found',
    'full_mode_busy',
    'input_tokens_exceeded',
    'batch_size_invalid',
    'rate_limit_exceeded',
    'resource_quota_exceeded',
    'cancelled',
    'endpoint_unavailable',
    'state_contention',
    'backend_gate_unavailable',
    'orphaned_replica',
    'execution_timeout',
    'deadline_exceeded',
    'internal_error',
})


class FutureRecord(BaseModel):
    """Represents an async task future / request status."""

    status: str
    model_id: str | None = None
    reason: str | None = None
    result: Any = None
    failure: FutureFailureRecord | None = None
    queue_state: str | None = None
    queue_state_reason: str | None = None
    # Replica ownership and deadline are fixed when the record is created.
    replica_id: str | None = None
    absolute_deadline: float | None = None
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    @model_validator(mode='after')
    def result_and_failure_match_status(self) -> FutureRecord:
        is_failure = self.status in ('failed', 'cancelled')
        if is_failure and self.failure is None:
            raise ValueError(f'{self.status} future requires failure')
        if is_failure and self.result is not None:
            raise ValueError(f'{self.status} future cannot carry result')
        if not is_failure and self.failure is not None:
            raise ValueError(f'{self.status} future cannot carry failure')
        return self
