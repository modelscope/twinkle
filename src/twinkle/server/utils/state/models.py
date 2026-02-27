# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat()


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
    created_at: str = Field(default_factory=_now_iso)


class SamplingSessionRecord(BaseModel):
    """Represents a sampling session."""

    session_id: str | None = None
    seq_id: Any = None
    base_model: str | None = None
    model_path: str | None = None
    created_at: str = Field(default_factory=_now_iso)


class FutureRecord(BaseModel):
    """Represents an async task future / request status."""

    status: str
    model_id: str | None = None
    reason: str | None = None
    result: Any = None
    queue_state: str | None = None
    queue_state_reason: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
