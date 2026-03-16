# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Shared Pydantic models for twinkle training runs and checkpoints.

These types are used both by twinkle_client (as request/response shapes)
and by twinkle.server.common.io_utils (as persistence models).
"""
from datetime import datetime
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class Cursor(BaseModel):
    limit: int
    offset: int
    total_count: int


class Checkpoint(BaseModel):
    """Twinkle checkpoint model."""
    checkpoint_id: str
    checkpoint_type: str
    time: datetime
    size_bytes: int
    public: bool = False
    twinkle_path: str
    # Training run info (stored for hub downloads)
    base_model: Optional[str] = None
    is_lora: bool = False
    lora_rank: Optional[int] = None
    train_unembed: Optional[bool] = None
    train_mlp: Optional[bool] = None
    train_attn: Optional[bool] = None
    user_metadata: Optional[Dict[str, Any]] = None


class TrainingRun(BaseModel):
    """Twinkle training run model."""
    training_run_id: str
    base_model: str
    model_owner: str
    is_lora: bool = False
    corrupted: bool = False
    lora_rank: Optional[int] = None
    last_request_time: Optional[datetime] = None
    last_checkpoint: Optional[Dict[str, Any]] = None
    last_sampler_checkpoint: Optional[Dict[str, Any]] = None
    user_metadata: Optional[Dict[str, Any]] = None


class TrainingRunsResponse(BaseModel):
    training_runs: List[TrainingRun]
    cursor: Cursor


class CheckpointsListResponse(BaseModel):
    checkpoints: List[Checkpoint]
    cursor: Optional[Cursor] = None


class ParsedCheckpointTwinklePath(BaseModel):
    """Twinkle-specific parsed path model."""
    path: str
    twinkle_path: str
    training_run_id: str
    checkpoint_type: str
    checkpoint_id: str


class WeightsInfoResponse(BaseModel):
    """Twinkle weights info response."""
    training_run_id: str
    base_model: str
    model_owner: str
    is_lora: bool = False
    lora_rank: Optional[int] = None


class LoraConfig(BaseModel):
    """Twinkle LoRA configuration."""
    rank: int = 8
    train_unembed: bool = False
    train_mlp: bool = True
    train_attn: bool = True


class CreateModelRequest(BaseModel):
    """Twinkle create model request."""
    base_model: str
    lora_config: Optional[LoraConfig] = None
    user_metadata: Optional[Dict[str, Any]] = None
