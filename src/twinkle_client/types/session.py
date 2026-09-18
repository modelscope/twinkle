# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pydantic models for twinkle session management endpoints."""
from typing import Any, Dict, Optional

from .base import ResponseModel, StrictRequest


class CreateSessionRequest(StrictRequest):
    """Request body for POST /twinkle/create_session."""
    metadata: Optional[Dict[str, Any]] = None


class CreateSessionResponse(ResponseModel):
    """Response body for POST /twinkle/create_session."""
    session_id: str


class SessionHeartbeatRequest(StrictRequest):
    """Request body for POST /twinkle/session_heartbeat."""
    session_id: str


class SessionHeartbeatResponse(ResponseModel):
    """Response body for POST /twinkle/session_heartbeat."""
