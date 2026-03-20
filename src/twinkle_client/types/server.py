# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Pydantic response models for the twinkle server health/error endpoints."""
from pydantic import BaseModel
from typing import Any, List, Optional


class SupportedModel(BaseModel):
    """Information about a supported model."""
    model_name: str


class GetServerCapabilitiesResponse(BaseModel):
    """Response body for the /get_server_capabilities endpoint."""
    supported_models: List[SupportedModel]


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
