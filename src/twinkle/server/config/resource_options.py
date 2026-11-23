# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared instance startup, allocation, shutdown and monitoring budgets."""

from pydantic import BaseModel, ConfigDict, Field


class ResourceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allocation_timeout_s: float = Field(default=3, gt=0, allow_inf_nan=False)
    startup_timeout_s: float = Field(default=1800, gt=0, allow_inf_nan=False)
    shutdown_timeout_s: float = Field(default=60, gt=0, allow_inf_nan=False)
    sweep_interval_s: float = Field(default=5, gt=0, allow_inf_nan=False)
