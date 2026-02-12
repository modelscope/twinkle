# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Task Queue Management for Tinker Server.

This module provides:
1. TaskStatus - Enum for tracking task lifecycle states
2. TaskQueueConfig - Configuration for rate limits and queue behavior
3. TaskQueueMixin - Mixin class for serial task execution with rate limiting
"""
from __future__ import annotations

import asyncio
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional, Tuple

from twinkle.utils.logger import get_logger
from .rate_limiter import RateLimiter

if TYPE_CHECKING:
    from twinkle.server.utils.state import ServerStateProxy

logger = get_logger()


class TaskStatus(Enum):
    """Task lifecycle status."""
    PENDING = "pending"           # Task created, waiting to be processed
    QUEUED = "queued"             # Task in queue waiting for execution
    RUNNING = "running"           # Task currently executing
    COMPLETED = "completed"       # Task completed successfully
    FAILED = "failed"             # Task failed with error
    RATE_LIMITED = "rate_limited"  # Task rejected due to rate limiting


class QueueState(Enum):
    """Queue state for tinker client compatibility.

    These states are returned to the tinker client to indicate the current
    state of the task queue and help the client adjust its retry behavior.
    """
    ACTIVE = "active"                     # Queue is actively processing tasks
    PAUSED_RATE_LIMIT = "paused_rate_limit"  # Queue paused due to rate limiting
    PAUSED_CAPACITY = "paused_capacity"   # Queue paused due to capacity limits
    UNKNOWN = "unknown"                   # Unknown or unspecified state


@dataclass
class TaskQueueConfig:
    """Configuration for task queue and rate limiting.

    Attributes:
        rps_limit: Maximum requests per second per user token.
        tps_limit: Maximum input tokens per second per user token.
        window_seconds: Time window for rate limiting calculations.
        queue_timeout: Maximum time a task can wait in queue (seconds).
        enabled: Whether rate limiting is enabled.
        token_cleanup_multiplier: Multiplier for token cleanup threshold.
        token_cleanup_interval: How often to run cleanup task (seconds).
        per_token_adapter_limit: Maximum number of adapters per user token.
        adapter_timeout: Timeout in seconds for inactive adapters (default 30 minutes).
    """
    rps_limit: float = 100.0           # 10 requests per second
    tps_limit: float = 10000.0        # 10000 input tokens per second
    window_seconds: float = 1.0       # 1 second sliding window
    queue_timeout: float = 300.0      # 5 minutes queue timeout
    enabled: bool = True              # Rate limiting enabled by default
    # Remove tokens after 10x window inactivity
    token_cleanup_multiplier: float = 10.0
    token_cleanup_interval: float = 60.0    # Run cleanup every 60 seconds

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]] = None) -> 'TaskQueueConfig':
        """Create TaskQueueConfig from a dictionary.

        Args:
            config_dict: Dictionary with configuration values. Supports keys:
                - rps_limit: requests per second limit
                - tps_limit: input tokens per second limit
                - window_seconds: sliding window duration
                - queue_timeout: queue timeout in seconds
                - enabled: whether rate limiting is enabled
                - token_cleanup_multiplier: multiplier for token cleanup threshold
                - token_cleanup_interval: cleanup task interval in seconds

        Returns:
            TaskQueueConfig instance with values from dict merged with defaults.
        """
        config = cls()
        if config_dict:
            if 'rps_limit' in config_dict:
                config.rps_limit = float(config_dict['rps_limit'])
            if 'tps_limit' in config_dict:
                config.tps_limit = float(config_dict['tps_limit'])
            if 'window_seconds' in config_dict:
                config.window_seconds = float(config_dict['window_seconds'])
            if 'queue_timeout' in config_dict:
                config.queue_timeout = float(config_dict['queue_timeout'])
            if 'enabled' in config_dict:
                config.enabled = bool(config_dict['enabled'])
            if 'token_cleanup_multiplier' in config_dict:
                config.token_cleanup_multiplier = float(
                    config_dict['token_cleanup_multiplier'])
            if 'token_cleanup_interval' in config_dict:
                config.token_cleanup_interval = float(
                    config_dict['token_cleanup_interval'])
        return config


class TaskQueueMixin:
    """Mixin providing task queue management, rate limiting, and status tracking.

    This mixin should be inherited by classes that need to:
    1. Execute async tasks serially through a queue
    2. Apply per-user rate limiting (rps and tps)
    3. Track task lifecycle status for proper client polling

    Requirements:
        - Inheriting class must have `self.state: ServerStateProxy` attribute
        - Call `_init_task_queue()` in `__init__` to initialize the queue
        - Call `await _start_worker()` to start the background worker

    Example:
        class MyService(TaskQueueMixin):
            def __init__(self):
                self.state = get_server_state()
                self._init_task_queue(TaskQueueConfig.from_dict(config_dict))

            async def my_endpoint(self, request, body):
                async def _do_work():
                    return await some_operation()
                return await self.schedule_task(
                    _do_work(),
                    model_id=body.model_id,
                    token=request.state.token,
                    input_tokens=len(body.tokens)
                )
    """

    # Type hint for state attribute that inheriting classes must provide
    state: 'ServerStateProxy'

    def _init_task_queue(self, config: Optional[TaskQueueConfig] = None) -> None:
        """Initialize the task queue system.

        Args:
            config: Optional TaskQueueConfig. If None, uses default config.
        """
        self._task_queue_config = config or TaskQueueConfig()
        self._task_queue: asyncio.Queue = asyncio.Queue()

        # Initialize rate limiter for RPS/TPS control
        self._rate_limiter = RateLimiter(
            rps_limit=self._task_queue_config.rps_limit,
            tps_limit=self._task_queue_config.tps_limit,
            window_seconds=self._task_queue_config.window_seconds,
            token_cleanup_multiplier=self._task_queue_config.token_cleanup_multiplier,
            token_cleanup_interval=self._task_queue_config.token_cleanup_interval,
        )
        # Start the rate limiter cleanup task
        self._rate_limiter.start_cleanup_task()

        self._worker_task: Optional[asyncio.Task] = None
        self._worker_started = False
        self._worker_start_lock = asyncio.Lock()

    async def _ensure_worker_started(self) -> None:
        """Ensure the background worker is running.

        Thread-safe: Uses asyncio.Lock to prevent race conditions when
        multiple concurrent requests try to start the worker simultaneously.
        """
        # Fast path: avoid lock if already started
        if self._worker_started:
            return

        # Slow path: acquire lock to safely check and start
        async with self._worker_start_lock:
            # Double-check after acquiring lock (another coroutine might have started it)
            if not self._worker_started:
                logger.debug(f"[TaskQueue] Starting background worker...")
                self._worker_task = asyncio.create_task(self._queue_worker())
                self._worker_started = True
                logger.debug(
                    f"[TaskQueue] Background worker started: {self._worker_task}")

    async def _queue_worker(self) -> None:
        """Background worker that processes tasks from the queue serially.

        This worker runs indefinitely, pulling tasks from the queue and
        executing them one at a time. This ensures thread-safe execution
        of model operations that cannot be parallelized.
        """
        logger.debug(f"[TaskQueue] Worker started")
        while True:
            try:
                # Wait for a task from the queue
                logger.debug(
                    f"[TaskQueue] Waiting for task... (queue size: {self._task_queue.qsize()})")
                request_id, coro, model_id = await self._task_queue.get()

                logger.debug(f"[TaskQueue] Processing task {request_id}")
                try:
                    # Update status to RUNNING
                    self.state.store_future_status(
                        request_id, TaskStatus.RUNNING.value, model_id,
                        queue_state=QueueState.ACTIVE.value
                    )

                    # Execute the task
                    result = await coro

                    logger.debug(
                        f"[TaskQueue] Task {request_id} completed successfully")
                    # Store completed result
                    self.state.store_future_status(
                        request_id, TaskStatus.COMPLETED.value, model_id, result=result
                    )
                except Exception:
                    # Store error result
                    logger.debug(
                        f"[TaskQueue] Task {request_id} failed with error")
                    error_payload = {
                        'error': traceback.format_exc(),
                        'category': 'Server'
                    }
                    self.state.store_future_status(
                        request_id, TaskStatus.FAILED.value, model_id, result=error_payload
                    )
                finally:
                    self._task_queue.task_done()

            except asyncio.CancelledError:
                logger.warning(f"[TaskQueue] Worker cancelled")
                break
            except Exception:
                # Log but don't crash the worker
                logger.warning("Error in task queue worker")
                continue

    async def schedule_task(
        self,
        coro: Coroutine,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
        input_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Schedule an async task with rate limiting and status tracking.

        This method replaces the old `schedule_task` function with proper
        status tracking to fix the race condition where clients would receive
        404 instead of 408 when polling before task execution started.

        Key improvements:
        1. Register PENDING status BEFORE creating the task
        2. Apply rate limiting per user token
        3. Execute tasks serially through a queue

        Args:
            coro: The coroutine to execute.
            model_id: Optional model_id to associate with the result.
            token: Optional user token for rate limiting.
            input_tokens: Number of input tokens for tps rate limiting.

        Returns:
            Dict containing request_id and model_id for future retrieval.
        """
        request_id = f"req_{uuid.uuid4().hex}"

        logger.debug(
            f"[TaskQueue] Scheduling task {request_id}, rps_limit={self._task_queue_config.rps_limit}, enabled={self._task_queue_config.enabled}")

        # 1. Register PENDING status FIRST (fixes race condition)
        self.state.store_future_status(
            request_id, TaskStatus.PENDING.value, model_id,
            queue_state=QueueState.ACTIVE.value
        )

        # 2. Check rate limiting if enabled and token provided
        if self._task_queue_config.enabled and token:
            logger.debug(
                f"[TaskQueue] Checking rate limit for token={token[:8]}... input_tokens={input_tokens}")
            allowed, reason = await self._rate_limiter.check_and_record(token, input_tokens)
            if not allowed:
                logger.debug(f"[TaskQueue] Rate limited: {reason}")
                self.state.store_future_status(
                    request_id, TaskStatus.RATE_LIMITED.value, model_id,
                    reason=reason,
                    queue_state=QueueState.PAUSED_RATE_LIMIT.value,
                    queue_state_reason=reason
                )
                return {'request_id': request_id, 'model_id': model_id}
            logger.debug(f"[TaskQueue] Rate limit check passed")

        # 3. Ensure worker is started
        await self._ensure_worker_started()

        # 4. Put task in queue and update status
        logger.debug(
            f"[TaskQueue] Adding task {request_id} to queue (current size: {self._task_queue.qsize()})")
        await self._task_queue.put((request_id, coro, model_id))
        self.state.store_future_status(
            request_id, TaskStatus.QUEUED.value, model_id,
            queue_state=QueueState.ACTIVE.value
        )
        logger.debug(
            f"[TaskQueue] Task {request_id} queued, new queue size: {self._task_queue.qsize()}")

        return {'request_id': request_id, 'model_id': model_id}

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics.

        Returns:
            Dict with queue size and worker status.
        """
        return {
            'queue_size': self._task_queue.qsize(),
            'worker_running': self._worker_started and self._worker_task is not None,
            'rate_limit_config': {
                'rps_limit': self._task_queue_config.rps_limit,
                'tps_limit': self._task_queue_config.tps_limit,
                'enabled': self._task_queue_config.enabled,
            }
        }

    def get_rate_limit_stats(self, token: str) -> Dict[str, Any]:
        """Get rate limiting stats for a specific user token.

        Args:
            token: User token to get stats for.

        Returns:
            Dict with current and available rate limits.
        """
        return self._rate_limiter.get_stats(token)

    def get_rate_limiter_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics from the rate limiter.

        Returns:
            Dict with active token count and cleanup configuration.
        """
        return self._rate_limiter.get_memory_stats()

    async def shutdown_task_queue(self) -> None:
        """Gracefully shutdown the task queue and cleanup tasks.

        This should be called when shutting down the server to ensure
        proper cleanup of background tasks.
        """
        # Stop the rate limiter cleanup task
        await self._rate_limiter.stop_cleanup_task()

        # Cancel the worker task if running
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.debug("[TaskQueue] Task queue shutdown complete")
