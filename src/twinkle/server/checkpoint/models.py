# Copyright (c) ModelScope Contributors. All rights reserved.
"""Internal Pydantic base specs used as type constraints for the generic
checkpoint / training-run managers.

Relocated from ``utils/checkpoint_base.py`` (TIER 2 consolidation). No logic change.
"""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel
from typing import Any, Dict, Optional, TypeVar


class BaseCheckpoint(BaseModel):
    """Base checkpoint model that can be extended."""
    checkpoint_id: str
    checkpoint_type: str
    time: datetime
    size_bytes: int
    public: bool = False
    # Training run info (stored for hub downloads)
    base_model: str | None = None
    is_lora: bool = False
    lora_rank: int | None = None
    train_unembed: bool | None = None
    train_mlp: bool | None = None
    train_attn: bool | None = None
    user_metadata: dict[str, Any] | None = None


class BaseTrainingRun(BaseModel):
    """Base training run model that can be extended."""
    training_run_id: str
    base_model: str
    model_owner: str
    save_dir: str | None = None
    is_lora: bool = False
    corrupted: bool = False
    lora_rank: int | None = None
    last_request_time: datetime | None = None
    last_checkpoint: dict[str, Any] | None = None
    last_sampler_checkpoint: dict[str, Any] | None = None
    user_metadata: dict[str, Any] | None = None


class BaseLoraConfig(BaseModel):
    """Base LoRA configuration model."""
    rank: int = 8
    train_unembed: bool = False
    train_mlp: bool = True
    train_attn: bool = True


class BaseCreateModelRequest(BaseModel):
    """Base request model for creating a model."""
    base_model: str
    lora_config: BaseLoraConfig | None = None
    save_dir: str | None = None
    user_metadata: dict[str, Any] | None = None


class BaseParsedCheckpointPath(BaseModel):
    """Base model for parsed checkpoint paths."""
    path: str
    training_run_id: str
    checkpoint_type: str
    checkpoint_id: str


class BaseWeightsInfoResponse(BaseModel):
    """Base model for weights info response."""
    training_run_id: str
    base_model: str
    model_owner: str
    is_lora: bool = False
    lora_rank: int | None = None


# Type variables for generic types
TCheckpoint = TypeVar('TCheckpoint', bound=BaseCheckpoint)
TTrainingRun = TypeVar('TTrainingRun', bound=BaseTrainingRun)
TCreateModelRequest = TypeVar('TCreateModelRequest', bound=BaseCreateModelRequest)
TParsedPath = TypeVar('TParsedPath', bound=BaseParsedCheckpointPath)
TWeightsInfo = TypeVar('TWeightsInfo', bound=BaseWeightsInfoResponse)
