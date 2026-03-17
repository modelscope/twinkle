# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Pydantic request/response models for twinkle processor endpoints.

These models are used by both the server-side handler and the twinkle client.

Note: Class names are prefixed with 'Processor' to avoid name collisions when
importing from twinkle_client.types alongside model.py classes.
"""
from pydantic import BaseModel
from typing import Any


class ProcessorCreateRequest(BaseModel):
    processor_type: str
    class_type: str

    class Config:
        extra = 'allow'


class ProcessorHeartbeatRequest(BaseModel):
    processor_id: str


class ProcessorCallRequest(BaseModel):
    processor_id: str
    function: str

    class Config:
        extra = 'allow'


class ProcessorCreateResponse(BaseModel):
    """Response body for the /create endpoint."""
    processor_id: str


class ProcessorHeartbeatResponse(BaseModel):
    """Response body for the /heartbeat endpoint."""
    status: str = 'ok'


class ProcessorCallResponse(BaseModel):
    """Response body for the /call endpoint."""
    result: Any
