# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Pydantic response models for the twinkle server health/error endpoints."""
from pydantic import BaseModel
from typing import Any


class HealthResponse(BaseModel):
    status: str


class DeleteCheckpointResponse(BaseModel):
    success: bool
    message: str


class ErrorResponse(BaseModel):
    detail: str


class WeightsInfoRequest(BaseModel):
    twinkle_path: str


class WeightsInfoResponse(BaseModel):
    """Response body for the /weights_info endpoint."""
    weights_info: Any


class CheckpointPathResponse(BaseModel):
    """Response body for the /checkpoint_path endpoint."""
    path: str
    twinkle_path: str
