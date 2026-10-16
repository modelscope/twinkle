# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Spectral Hybrid configuration."""
import math
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpectralHybridArgs(BaseModel):
    """Strict server-owned Spectral Hybrid allocation and optimizer defaults."""

    model_config = ConfigDict(extra='forbid')

    allocation_path: str = Field(min_length=1)
    default_lr_lora: float = Field(default=2.5e-5, gt=0)
    default_lr_fft: float = Field(default=1.0e-6, gt=0)

    @field_validator('default_lr_lora', 'default_lr_fft')
    @classmethod
    def _finite_learning_rate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError('learning rate must be finite')
        return value
