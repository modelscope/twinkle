# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Pydantic request/response models for twinkle processor endpoints.

These models are used by both the server-side handler and the twinkle client.

Note: Class names are prefixed with 'Processor' to avoid name collisions when
importing from twinkle_client.types alongside model.py classes.
"""
from pydantic import BaseModel


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
