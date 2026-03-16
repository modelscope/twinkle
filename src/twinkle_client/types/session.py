# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pydantic models for twinkle session management endpoints."""
from pydantic import BaseModel
from typing import Any, Dict, Optional


class CreateSessionRequest(BaseModel):
    """Request body for POST /twinkle/create_session."""
    metadata: Optional[Dict[str, Any]] = None


class CreateSessionResponse(BaseModel):
    """Response body for POST /twinkle/create_session."""
    session_id: str


class SessionHeartbeatRequest(BaseModel):
    """Request body for POST /twinkle/session_heartbeat."""
    session_id: str


class SessionHeartbeatResponse(BaseModel):
    """Response body for POST /twinkle/session_heartbeat."""
    pass
