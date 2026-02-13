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
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Deque, Dict, Optional

from twinkle.utils.logger import get_logger
from .rate_limiter import RateLimiter

if TYPE_CHECKING:
    from twinkle.server.utils.state import ServerStateProxy

logger = get_logger()


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
        max_input_tokens: Maximum allowed input tokens per request (default 10000).
    """
    rps_limit: float = 100.0  # 10 requests per second
    tps_limit: float = 16000.0  # 10000 input tokens per second
    window_seconds: float = 1.0  # 1 second sliding window
    queue_timeout: float = 300.0  # 5 minutes queue timeout
    enabled: bool = True  # Rate limiting enabled by default
    # Remove tokens after 10x window inactivity
    token_cleanup_multiplier: float = 10.0
    token_cleanup_interval: float = 60.0  # Run cleanup every 60 seconds
    max_input_tokens: int = 16000  # Maximum input tokens per request

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None = None) -> TaskQueueConfig:
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
                - max_input_tokens: maximum input tokens per request

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
                config.token_cleanup_multiplier = float(config_dict['token_cleanup_multiplier'])
            if 'token_cleanup_interval' in config_dict:
                config.token_cleanup_interval = float(config_dict['token_cleanup_interval'])
            if 'max_input_tokens' in config_dict:
                config.max_input_tokens = int(config_dict['max_input_tokens'])
        return config


@dataclass
class _QueuedTask:
    request_id: str
    coro_factory: Callable[[], Coroutine]
    model_id: str | None
    token: str | None
    input_tokens: int
    task_type: str | None
    created_at: float
    first_rate_limited_at: float | None = None


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
                    _do_work,
                    model_id=body.model_id,
                    token=request.state.token,
                    input_tokens=len(body.tokens)
                )
    """

    # Type hint for state attribute that inheriting classes must provide
    state: ServerStateProxy

    def _init_task_queue(self, config: TaskQueueConfig | None = None) -> None:
        """Initialize the task queue system.

        Args:
            config: Optional TaskQueueConfig. If None, uses default config.
        """
        self._task_queue_config = config or TaskQueueConfig()
        # Per-key queues, but executed by a single global worker.
        self._task_queues: dict[str, asyncio.Queue] = {}
        self._queue_order: Deque[str] = deque()
        self._new_task_event: asyncio.Event = asyncio.Event()

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

        # Single worker to ensure model operations remain serial.
        self._worker_task: asyncio.Task | None = None
        self._worker_started = False
        self._worker_start_lock = asyncio.Lock()

        # Event loop reference for thread-safe callbacks (e.g., adapter expiration thread)
        self._event_loop: asyncio.AbstractEventLoop | None = None

    @staticmethod
    def _queue_key(
        model_id: str | None,
        token: str | None,
    ) -> str:
        if model_id:
            return f'model:{model_id}'
        if token:
            return f'token:{token}'
        return 'default'

    async def _ensure_worker_started(self) -> None:
        """Ensure the single background worker is running."""
        if self._worker_started and self._worker_task is not None and not self._worker_task.done():
            return

        async with self._worker_start_lock:
            if self._worker_started and self._worker_task is not None and not self._worker_task.done():
                return
            self._worker_task = asyncio.create_task(self._queue_worker())
            self._worker_started = True

    def _ensure_queue_registered(self, queue_key: str) -> None:
        if queue_key not in self._task_queues:
            self._task_queues[queue_key] = asyncio.Queue()
        if queue_key not in self._queue_order:
            self._queue_order.append(queue_key)

    async def _queue_worker(self) -> None:
        """Single background worker that processes tasks serially across all queues.

        Selection policy: round-robin across queue keys. If a task is rate-limited
        at execution time, it is requeued and the worker tries other queues.
        """
        logger.debug('[TaskQueue] Worker started')
        while True:
            try:
                # Wait until there is at least one queue with a task
                while True:
                    if any(q.qsize() > 0 for q in self._task_queues.values()):
                        break
                    self._new_task_event.clear()
                    await self._new_task_event.wait()

                executed_any = False
                # Try each queue at most once per loop for fairness
                for _ in range(len(self._queue_order)):
                    queue_key = self._queue_order[0]
                    self._queue_order.rotate(-1)

                    q = self._task_queues.get(queue_key)
                    if q is None:
                        continue

                    try:
                        task: _QueuedTask = q.get_nowait()
                    except asyncio.QueueEmpty:
                        continue

                    now = time.monotonic()

                    # Global queue timeout
                    if (now - task.created_at) > self._task_queue_config.queue_timeout:
                        error_payload = {
                            'error': f'Queue timeout exceeded: waited {now - task.created_at:.2f}s',
                            'category': 'Server'
                        }
                        self.state.store_future_status(
                            task.request_id,
                            TaskStatus.FAILED.value,
                            task.model_id,
                            result=error_payload,
                            queue_state=QueueState.PAUSED_CAPACITY.value,
                            queue_state_reason=error_payload['error'],
                        )
                        q.task_done()
                        continue

                    # Rate limiting check has been moved to schedule_task(), so tasks here should pass rate limits

                    # Execute
                    executed_any = True
                    self.state.store_future_status(
                        task.request_id, TaskStatus.RUNNING.value, task.model_id, queue_state=QueueState.ACTIVE.value)

                    try:
                        coro = task.coro_factory()
                        result = await coro
                        self.state.store_future_status(
                            task.request_id,
                            TaskStatus.COMPLETED.value,
                            task.model_id,
                            result=result,
                            queue_state=QueueState.ACTIVE.value)
                    except Exception:
                        error_payload = {'error': traceback.format_exc(), 'category': 'Server'}
                        self.state.store_future_status(
                            task.request_id,
                            TaskStatus.FAILED.value,
                            task.model_id,
                            result=error_payload,
                            queue_state=QueueState.ACTIVE.value)
                    finally:
                        q.task_done()

                    # Keep serial semantics: execute at most one runnable task per loop
                    break

                if not executed_any:
                    # All available tasks were rate-limited; avoid busy looping.
                    await asyncio.sleep(min(self._task_queue_config.window_seconds, 0.1))

            except asyncio.CancelledError:
                logger.warning('[TaskQueue] Worker cancelled')
                break
            except Exception:
                logger.warning('Error in task queue worker')
                continue

    async def _fail_queue_tasks_async(self, queue_key: str, reason: str) -> None:
        q = self._task_queues.get(queue_key)
        if q is None:
            return

        drained: list[_QueuedTask] = []
        while True:
            try:
                drained.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break

        for task in drained:
            error_payload = {'error': reason, 'category': 'Server'}
            self.state.store_future_status(
                task.request_id,
                TaskStatus.FAILED.value,
                task.model_id,
                result=error_payload,
                queue_state=QueueState.UNKNOWN.value,
                queue_state_reason=reason,
            )
            q.task_done()

        # Remove queue structures
        self._task_queues.pop(queue_key, None)
        try:
            while queue_key in self._queue_order:
                self._queue_order.remove(queue_key)
        except ValueError:
            pass

    def fail_pending_tasks_for_model(self, model_id: str, reason: str) -> None:
        """Fail and drop queued tasks for a model. Safe to call from non-async threads."""
        queue_key = self._queue_key(model_id=model_id, token=None)
        if self._event_loop is None:
            # Best-effort: nothing we can do safely without a loop.
            logger.warning(f'[TaskQueue] fail_pending_tasks_for_model called without event loop: {queue_key}')
            return

        def _schedule() -> None:
            asyncio.create_task(self._fail_queue_tasks_async(queue_key, reason))

        self._event_loop.call_soon_threadsafe(_schedule)

    async def _perform_preflight_checks(
        self,
        request_id: str,
        model_id: str | None,
        token: str | None,
        input_tokens: int,
        batch_size: int | None = None,
        data_world_size: int | None = None,
    ) -> dict[str, Any] | None:
        """Perform pre-flight checks including rate limiting and token validation.

        Args:
            request_id: The request ID for status tracking.
            model_id: Optional model_id for error reporting.
            token: Optional user token for rate limiting.
            input_tokens: Number of input tokens for validation.
            batch_size: Optional batch size for validation.
            data_world_size: Optional data world size for batch size validation.

        Returns:
            None if checks pass, or error response dict if checks fail.
        """
        if not token or not self._task_queue_config.enabled:
            return None

        # Check max input tokens
        if input_tokens > self._task_queue_config.max_input_tokens:
            error_msg = f'Input tokens ({input_tokens}) exceed maximum allowed ({self._task_queue_config.max_input_tokens})'  # noqa: E501
            error_payload = {'error': error_msg, 'category': 'User'}
            self.state.store_future_status(
                request_id,
                TaskStatus.FAILED.value,
                model_id,
                result=error_payload,
                queue_state=QueueState.UNKNOWN.value,
                queue_state_reason=error_msg,
            )
            return {'request_id': request_id, 'model_id': model_id}

        # Check batch size if provided
        if batch_size is not None and data_world_size is not None:
            if batch_size < data_world_size:
                error_msg = f'Batch size {batch_size} must be greater than or equal to data world size {data_world_size}'  # noqa: E501
                error_payload = {'error': error_msg, 'category': 'User'}
                self.state.store_future_status(
                    request_id,
                    TaskStatus.FAILED.value,
                    model_id,
                    result=error_payload,
                    queue_state=QueueState.UNKNOWN.value,
                    queue_state_reason=error_msg,
                )
                return {'request_id': request_id, 'model_id': model_id}

        # Check rate limits
        allowed, reason = await self._rate_limiter.check_and_record(token, input_tokens)
        if not allowed:
            error_msg = f'Rate limit exceeded: {reason}'
            error_payload = {'error': error_msg, 'category': 'User'}
            self.state.store_future_status(
                request_id,
                TaskStatus.FAILED.value,
                model_id,
                result=error_payload,
                queue_state=QueueState.PAUSED_RATE_LIMIT.value,
                queue_state_reason=error_msg,
            )
            return {'request_id': request_id, 'model_id': model_id}

        return None

    async def schedule_task(
        self,
        coro_factory: Callable[[], Coroutine],
        model_id: str | None = None,
        token: str | None = None,
        input_tokens: int = 0,
        batch_size: int | None = None,
        data_world_size: int | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Schedule an async task with rate limiting and status tracking.

        This method replaces the old `schedule_task` function with proper
        status tracking to fix the race condition where clients would receive
        404 instead of 408 when polling before task execution started.

        Key improvements:
        1. Register PENDING status BEFORE creating the task
        2. Apply rate limiting per user token
        3. Execute tasks serially through a queue

        Args:
            coro_factory: Factory that creates the coroutine to execute. The coroutine
                will be created only after passing rate limiting and when it's time
                to execute the queued task.
            model_id: Optional model_id to associate with the result.
            token: Optional user token for rate limiting.
            input_tokens: Number of input tokens for tps rate limiting.
            batch_size: Optional batch size for validation.
            data_world_size: Optional data world size for batch size validation.
            task_type: Optional task type for logging/observability.

        Returns:
            Dict containing request_id and model_id for future retrieval.
        """
        # Generate request_id first so it can be included in error responses
        request_id = f'req_{uuid.uuid4().hex}'

        # 1. Pre-flight checks: rate limiting, max token validation, and batch size validation
        preflight_result = await self._perform_preflight_checks(request_id, model_id, token, input_tokens, batch_size,
                                                                data_world_size)
        if preflight_result is not None:
            return preflight_result

        if self._event_loop is None:
            self._event_loop = asyncio.get_running_loop()

        logger.debug(
            f'[TaskQueue] Scheduling task {request_id}, rps_limit={self._task_queue_config.rps_limit}, enabled={self._task_queue_config.enabled}'  # noqa: E501
        )

        # 2. Register PENDING status FIRST
        self.state.store_future_status(
            request_id, TaskStatus.PENDING.value, model_id, queue_state=QueueState.ACTIVE.value)

        # 3. Route to per-model/per-token queue
        queue_key = self._queue_key(model_id=model_id, token=token)
        self._ensure_queue_registered(queue_key)

        # 4. Ensure worker is started
        await self._ensure_worker_started()

        # 5. Put task in queue and update status
        q = self._task_queues[queue_key]
        logger.debug(
            f'[TaskQueue] Adding task {request_id} to queue key={queue_key} (current size: {q.qsize()}) type={task_type}'  # noqa: E501
        )
        await q.put(
            _QueuedTask(
                request_id=request_id,
                coro_factory=coro_factory,
                model_id=model_id,
                token=token,
                input_tokens=input_tokens,
                task_type=task_type,
                created_at=time.monotonic(),
            ))
        self.state.store_future_status(
            request_id, TaskStatus.QUEUED.value, model_id, queue_state=QueueState.ACTIVE.value)
        logger.debug(f'[TaskQueue] Task {request_id} queued, new queue size: {q.qsize()} key={queue_key}')

        self._new_task_event.set()

        return {'request_id': request_id, 'model_id': model_id}

    def get_queue_stats(self) -> dict[str, Any]:
        """Get current queue statistics.

        Returns:
            Dict with queue size and worker status.
        """
        return {
            'queue_size': sum(q.qsize() for q in self._task_queues.values()),
            'queue_count': len(self._task_queues),
            'worker_running': self._worker_task is not None and not self._worker_task.done(),
            'rate_limit_config': {
                'rps_limit': self._task_queue_config.rps_limit,
                'tps_limit': self._task_queue_config.tps_limit,
                'enabled': self._task_queue_config.enabled,
            }
        }

    def get_rate_limit_stats(self, token: str) -> dict[str, Any]:
        """Get rate limiting stats for a specific user token.

        Args:
            token: User token to get stats for.

        Returns:
            Dict with current and available rate limits.
        """
        return self._rate_limiter.get_stats(token)

    def get_rate_limiter_memory_stats(self) -> dict[str, Any]:
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

        self._worker_task = None
        self._worker_started = False

        self._task_queues.clear()
        self._queue_order.clear()

        logger.debug('[TaskQueue] Task queue shutdown complete')
