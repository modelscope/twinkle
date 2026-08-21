# Copyright (c) ModelScope Contributors. All rights reserved.
"""Minimal private HTTP tunnel models used by ModelScope and DashServing."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Any


class TunnelRequest(BaseModel):
    """HTTP request to forward to the internal `/api/v1/` server."""

    method: str = Field(min_length=1)
    path: str
    query: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None

    @field_validator('path')
    @classmethod
    def validate_path(cls, path: str) -> str:
        if not path.startswith('/api/v1/'):
            raise ValueError('path must start with /api/v1/')
        return path


class TunnelResponse(BaseModel):
    status_code: int = Field(ge=100, le=599)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None
