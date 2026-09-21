# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Pydantic response models for the twinkle server health/error endpoints."""
from pydantic import BaseModel, Field
from typing import List

from .base import ResponseModel, StrictRequest


class SupportedModel(BaseModel):
    """Information about a supported model.

    A nested value inside a response, not a response body of its own, so it keeps the
    plain base -- the strict/ignore split applies to what crosses the wire as a whole.
    """
    model_name: str


class ClientFeatures(ResponseModel):
    task_envelope: bool = True
    cancel: bool = False
    data_plane: bool = False
    full_training: bool = False
    batch_retrieve: bool = False


class ProtocolLimits(ResponseModel):
    long_poll_timeout_seconds: float | None = None
    max_payload_bytes: int | None = None
    max_batch_size: int | None = None


class GetServerCapabilitiesResponse(ResponseModel):
    """Versioned Twinkle-native capabilities with old-server defaults."""
    supported_models: List[SupportedModel]
    protocol_version: int = 1
    features: ClientFeatures = Field(default_factory=ClientFeatures)
    limits: ProtocolLimits = Field(default_factory=ProtocolLimits)


class HealthResponse(ResponseModel):
    status: str


class DeleteCheckpointResponse(ResponseModel):
    success: bool
    message: str


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
