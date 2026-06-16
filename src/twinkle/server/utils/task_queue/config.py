# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Task queue configuration.

Provides TaskQueueConfig (Pydantic) for controlling rate limits, timeouts,
and queue behavior. Constraints are validated at construction time so an
invalid YAML/dict value is rejected before the deployment reaches a ready
state.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TaskQueueConfig(BaseModel):
    """Configuration for task queue and rate limiting.

    Attributes:
        rps_limit: Maximum requests per second per user token. ``0`` disables.
        tps_limit: Maximum input tokens per second per user token. ``0`` disables.
        window_seconds: Sliding window for rate-limit calculations. Must be > 0.
        queue_timeout: Maximum time a task can wait in queue (seconds).
        execution_timeout: Maximum time a task can execute (seconds). 0 means no limit.
        enabled: Whether rate limiting is enabled.
        token_cleanup_multiplier: Multiplier for token cleanup threshold.
        token_cleanup_interval: How often to run cleanup task (seconds).
        max_input_tokens: Maximum allowed input tokens per request.
    """

    model_config = ConfigDict(extra='forbid')

    rps_limit: float = Field(default=100.0, ge=0)
    tps_limit: float = Field(default=16000.0, ge=0)
    window_seconds: float = Field(default=1.0, gt=0)
    queue_timeout: float = Field(default=300.0, ge=0)
    execution_timeout: float = Field(default=120.0, ge=0)
    enabled: bool = True
    token_cleanup_multiplier: float = Field(default=10.0, ge=0)
    token_cleanup_interval: float = Field(default=60.0, ge=0)
    max_input_tokens: int = Field(default=16000, ge=1)
