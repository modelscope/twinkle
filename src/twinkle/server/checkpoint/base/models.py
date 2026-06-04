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
    base_model: Optional[str] = None
    is_lora: bool = False
    lora_rank: Optional[int] = None
    train_unembed: Optional[bool] = None
    train_mlp: Optional[bool] = None
    train_attn: Optional[bool] = None
    user_metadata: Optional[Dict[str, Any]] = None


class BaseTrainingRun(BaseModel):
    """Base training run model that can be extended."""
    training_run_id: str
    base_model: str
    model_owner: str
    save_dir: Optional[str] = None
    is_lora: bool = False
    corrupted: bool = False
    lora_rank: Optional[int] = None
    last_request_time: Optional[datetime] = None
    last_checkpoint: Optional[Dict[str, Any]] = None
    last_sampler_checkpoint: Optional[Dict[str, Any]] = None
    user_metadata: Optional[Dict[str, Any]] = None


class BaseLoraConfig(BaseModel):
    """Base LoRA configuration model."""
    rank: int = 8
    train_unembed: bool = False
    train_mlp: bool = True
    train_attn: bool = True


class BaseCreateModelRequest(BaseModel):
    """Base request model for creating a model."""
    base_model: str
    lora_config: Optional[BaseLoraConfig] = None
    save_dir: Optional[str] = None
    user_metadata: Optional[Dict[str, Any]] = None


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
    lora_rank: Optional[int] = None


# Type variables for generic types
TCheckpoint = TypeVar('TCheckpoint', bound=BaseCheckpoint)
TTrainingRun = TypeVar('TTrainingRun', bound=BaseTrainingRun)
TCreateModelRequest = TypeVar('TCreateModelRequest', bound=BaseCreateModelRequest)
TParsedPath = TypeVar('TParsedPath', bound=BaseParsedCheckpointPath)
TWeightsInfo = TypeVar('TWeightsInfo', bound=BaseWeightsInfoResponse)
