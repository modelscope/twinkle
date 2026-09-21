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

# Finite bounds used when configuration omits a limit and by long-running methods.
_ZERO_EXECUTION_TIMEOUT_FALLBACK: float = 3600.0
_MAX_DECLARED_BACKEND_TIMEOUT: float = 3600.0
_ABSOLUTE_TTL_MULTIPLIER: int = 2


class TaskQueueConfig(BaseModel):
    """Configuration for task queue and rate limiting.

    Attributes:
        rps_limit: Maximum requests per second per user token. ``0`` disables.
        tps_limit: Maximum input tokens per second per user token. ``0`` disables.
        window_seconds: Sliding window for rate-limit calculations. Must be > 0.
        queue_timeout: Maximum time a task can wait in queue (seconds).
        execution_timeout: Maximum time a task can execute (seconds). ``0`` means "no
            configured limit"; a finite bound of 3600s is substituted instead of
            unbounded waiting (see ``effective_execution_timeout``).
        enabled: Whether rate limiting is enabled.
        token_cleanup_multiplier: Multiplier for token cleanup threshold.
        token_cleanup_interval: How often to run cleanup task (seconds).
        max_input_tokens: Maximum allowed input tokens per request.
        inline_fast_path_timeout: Upper bound (seconds) on how long submit briefly
            polls the record so a millisecond-scale control-plane op (step / zero_grad)
            completes in a single HTTP round trip instead of forcing a retrieve.
            Must remain < Long_Poll_Window.
    """

    model_config = ConfigDict(extra='forbid')

    rps_limit: float = Field(default=100.0, ge=0)
    tps_limit: float = Field(default=16000.0, ge=0)
    window_seconds: float = Field(default=1.0, gt=0)
    queue_timeout: float = Field(default=300.0, ge=0)
    execution_timeout: float = Field(default=1800.0, ge=0)
    enabled: bool = True
    token_cleanup_multiplier: float = Field(default=10.0, ge=0)
    token_cleanup_interval: float = Field(default=60.0, ge=0)
    max_input_tokens: int = Field(default=16000, ge=1)
    inline_fast_path_timeout: float = Field(default=0.05, gt=0)

    @property
    def effective_execution_timeout(self) -> float:
        """The single source of the execution time bound.

        ``0`` is not rejected (that would fail existing deployments); it is read
        as "no configured limit" and replaced by a finite fallback so the bound
        is always positive. This value feeds both ``_ray_get_timeout`` and the
        ComputeWorker's ``asyncio.wait_for`` -- there is no second, independently
        configurable timeout.
        """
        if self.execution_timeout > 0:
            return self.execution_timeout
        return _ZERO_EXECUTION_TIMEOUT_FALLBACK

    def absolute_future_ttl(self, collect_width: int) -> float:
        """Conservative lifetime for a non-terminal future record."""
        ray_timeout = max(self.effective_execution_timeout, _MAX_DECLARED_BACKEND_TIMEOUT)
        resource_bound = max(1, collect_width) * ray_timeout
        return _ABSOLUTE_TTL_MULTIPLIER * (self.queue_timeout + resource_bound)
