# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Core type definitions for the task queue system.

Provides:
- TaskStatus: Enum for tracking task lifecycle states
- QueueState: Enum for tinker client compatibility (retry behavior hints)
- QueuedTask: Dataclass representing a queued work item
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Coroutine


class TaskStatus(Enum):
    """Task lifecycle status."""
    PENDING = 'pending'  # Task created, waiting to be processed
    QUEUED = 'queued'  # Task in queue waiting for execution
    RUNNING = 'running'  # Task currently executing
    COMPLETED = 'completed'  # Task completed successfully
    FAILED = 'failed'  # Task failed with error
    RATE_LIMITED = 'rate_limited'  # Task rejected due to rate limiting


class QueueState(Enum):
    """Queue state for tinker client compatibility.

    These states are returned to the tinker client to indicate the current
    state of the task queue and help the client adjust its retry behavior.
    """
    ACTIVE = 'active'  # Queue is actively processing tasks
    PAUSED_RATE_LIMIT = 'paused_rate_limit'  # Queue paused due to rate limiting
    PAUSED_CAPACITY = 'paused_capacity'  # Queue paused due to capacity limits
    UNKNOWN = 'unknown'  # Unknown or unspecified state


@dataclass
class QueuedTask:
    """Dataclass representing a task waiting in the compute queue."""
    request_id: str
    coro_factory: Callable[[], Coroutine]
    model_id: str | None
    token: str | None
    input_tokens: int
    task_type: str | None
    created_at: float
    first_rate_limited_at: float | None = None
