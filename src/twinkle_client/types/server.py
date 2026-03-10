# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Pydantic response models for the twinkle server health/error endpoints."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class DeleteCheckpointResponse(BaseModel):
    success: bool
    message: str


class ErrorResponse(BaseModel):
    detail: str
