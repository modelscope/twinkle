# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Pydantic response models for the twinkle server health/error endpoints."""
from pydantic import BaseModel
from typing import List

from .base import ResponseModel, StrictRequest


class SupportedModel(BaseModel):
    """Information about a supported model.

    A nested value inside a response, not a response body of its own, so it keeps the
    plain base -- the strict/ignore split applies to what crosses the wire as a whole.
    """
    model_name: str


class GetServerCapabilitiesResponse(ResponseModel):
    """Response body for the /get_server_capabilities endpoint."""
    supported_models: List[SupportedModel]


class HealthResponse(ResponseModel):
    status: str


class DeleteCheckpointResponse(ResponseModel):
    success: bool
    message: str


class ErrorResponse(ResponseModel):
    detail: str


class WeightsInfoRequest(StrictRequest):
    twinkle_path: str


class CheckpointPathResponse(ResponseModel):
    """Response body for the /checkpoint_path endpoint."""
    path: str
    twinkle_path: str


class CapacityInfoResponse(ResponseModel):
    """Response body for the /capacity_info endpoint."""
    max_loras: int
    used_loras: int
    free_loras: int
