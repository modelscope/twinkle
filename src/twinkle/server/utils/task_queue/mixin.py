# Copyright (c) ModelScope Contributors. All rights reserved.
"""TaskQueueMixin: serial compute queue plus admitted concurrent tasks.

``schedule_task`` serializes stateful Model operations. Detached tasks are
used for I/O and for engines such as vLLM that own their compute concurrency
and continuous batching internally.
"""
from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from twinkle.server.telemetry.middleware import get_task_metrics
from twinkle.server.utils.task_errors import task_error_payload
from twinkle.utils.logger import get_logger
from .config import TaskQueueConfig
from .rate_limiter import RateLimiter
from .types import QueuedTask, QueueState, TaskStatus
from .worker import ComputeWorker

if TYPE_CHECKING:
    from twinkle.server.state import ServerState

logger = get_logger()


class TaskQueueMixin:
    """Mixin providing two task execution paths.

    Execution paths
    ---------------
    1. Compute queue (schedule_task / schedule_task_and_wait):
       Single background worker, serial execution, round-robin across queues.
       Use for GPU operations: forward, backward, step, save, load, etc.

    2. Background task (schedule_background_task):
       asyncio.create_task, runs concurrently with compute queue. Use for I/O
       or an engine that provides its own safe concurrency and batching.
       Status is still tracked; clients can poll the same status endpoints.

    Requirements
    ------------
    Inheriting class must expose self.state: ServerState and call
    _init_task_queue() during __init__.
    """

    state: ServerState

    def _init_task_queue(self, config: TaskQueueConfig | None = None, deployment_name: str = '') -> None:
        """Initialise the task queue, rate limiter, and compute worker.

        ``config`` must be a typed :class:`TaskQueueConfig` (the launcher
        passes the instance straight through). ``None`` constructs a default
        config.
        """
        self._task_queue_config = config if config is not None else TaskQueueConfig()
        self._deployment_name = deployment_name
        self._task_metrics = get_task_metrics(deployment_name) if deployment_name else None

        self._rate_limiter = RateLimiter(
            rps_limit=self._task_queue_config.rps_limit,
            tps_limit=self._task_queue_config.tps_limit,
            window_seconds=self._task_queue_config.window_seconds,
            token_cleanup_multiplier=self._task_queue_config.token_cleanup_multiplier,
            token_cleanup_interval=self._task_queue_config.token_cleanup_interval,
            active_tokens_gauge=self._task_metrics.rate_limiter_active_tokens if self._task_metrics else None,
            deployment_name=deployment_name,
        )
        self._rate_limiter.start_cleanup_task()

        self._compute_worker = ComputeWorker(
            state=self.state,
            config=self._task_queue_config,
            task_metrics=self._task_metrics,
            deployment_name=deployment_name,
        )

        self._event_loop: asyncio.AbstractEventLoop | None = None

    @staticmethod
    def _queue_key(model_id: str | None, token: str | None) -> str:
        if model_id:
            return f'model:{model_id}'
        if token:
            return f'token:{token}'
        return 'default'

    async def _perform_preflight_checks(
        self,
        request_id: str,
        model_id: str | None,
        token: str | None,
        input_tokens: int,
        batch_size: int | None = None,
        data_world_size: int | None = None,
        batch_size_multiple: int | None = None,
        persist_failure: bool = True,
    ) -> dict[str, Any] | None:
        """Run rate-limit and validation checks before queuing a task.

        Returns None if all checks pass, or an error-response dict on failure.
        """
        if not token or not self._task_queue_config.enabled:
            return None

        async def reject(error_msg: str, queue_state: str) -> dict[str, Any]:
            error_payload = {'error': error_msg, 'category': 'User'}
            if persist_failure:
                await self.state.store_future_status(
                    request_id,
                    TaskStatus.FAILED.value,
                    model_id,
                    result=error_payload,
                    queue_state=queue_state,
                    queue_state_reason=error_msg,
                )
                return {'request_id': request_id, 'model_id': model_id}
            # Private marker consumed by schedule_task_and_wait().  It is not
            # returned by the public polling-style schedule_task() API.
            return {
                'request_id': request_id,
                'model_id': model_id,
                '_error': error_msg,
            }

        if input_tokens > self._task_queue_config.max_input_tokens:
            error_msg = (f'Input tokens ({input_tokens}) exceed maximum allowed '
                         f'({self._task_queue_config.max_input_tokens})')
            return await reject(error_msg, QueueState.UNKNOWN.value)

        if batch_size is not None and data_world_size is not None:
            if batch_size < data_world_size:
                error_msg = (f'Batch size {batch_size} must be >= data world size {data_world_size}')
                return await reject(error_msg, QueueState.UNKNOWN.value)
            if batch_size_multiple is not None:
                required_multiple = data_world_size * batch_size_multiple
                if batch_size % required_multiple != 0:
                    error_msg = (f'Batch size {batch_size} must be divisible by {required_multiple} '
                                 f'so each data-parallel shard gets a multiple of '
                                 f'{batch_size_multiple} examples')
                    return await reject(error_msg, QueueState.UNKNOWN.value)

        allowed, reason = await self._rate_limiter.check_and_record(token, input_tokens)
        if not allowed:
            if self._task_metrics:
                self._task_metrics.rate_limit_rejections.inc(tags={'deployment': self._deployment_name})
            error_msg = f'Rate limit exceeded: {reason}'
            return await reject(error_msg, QueueState.PAUSED_RATE_LIMIT.value)

        return None

    async def _schedule_task(
        self,
        coro_factory: Callable[[], Coroutine],
        model_id: str | None = None,
        token: str | None = None,
        input_tokens: int = 0,
        batch_size: int | None = None,
        data_world_size: int | None = None,
        batch_size_multiple: int | None = None,
        task_type: str | None = None,
        *,
        completion: asyncio.Future[Any] | None = None,
        persist_status: bool,
    ) -> dict[str, Any]:
        """Common enqueue path for polling and in-process wait callers."""
        request_id = f'req_{uuid.uuid4().hex}'

        preflight_result = await self._perform_preflight_checks(
            request_id=request_id,
            model_id=model_id,
            token=token,
            input_tokens=input_tokens,
            batch_size=batch_size,
            data_world_size=data_world_size,
            batch_size_multiple=batch_size_multiple,
            persist_failure=persist_status,
        )
        if preflight_result is not None:
            return preflight_result

        if self._event_loop is None:
            self._event_loop = asyncio.get_running_loop()

        if persist_status:
            await self.state.store_future_status(
                request_id,
                TaskStatus.PENDING.value,
                model_id,
                queue_state=QueueState.ACTIVE.value,
            )

        queue_key = self._queue_key(model_id=model_id, token=token)
        self._compute_worker.ensure_queue_registered(queue_key)
        await self._compute_worker.ensure_started()

        q = self._compute_worker.task_queues[queue_key]
        await q.put(
            QueuedTask(
                request_id=request_id,
                coro_factory=coro_factory,
                model_id=model_id,
                token=token,
                input_tokens=input_tokens,
                task_type=task_type,
                created_at=time.monotonic(),
                completion=completion,
                persist_status=persist_status,
            ))
        if persist_status:
            await self.state.store_future_status(
                request_id,
                TaskStatus.QUEUED.value,
                model_id,
                queue_state=QueueState.ACTIVE.value,
            )
        logger.info(f'[TaskQueue] Task {request_id} queued, type={task_type or "unknown"}, '
                    f'model_id={model_id}, queue_key={queue_key}, '
                    f'queue_depth={q.qsize()}, input_tokens={input_tokens}')

        self._compute_worker.new_task_event.set()

        if self._task_metrics:
            total_depth = sum(q.qsize() for q in self._compute_worker.task_queues.values())
            self._task_metrics.queue_depth.set(total_depth, tags={'deployment': self._deployment_name})

        return {'request_id': request_id, 'model_id': model_id}

    async def schedule_task(
        self,
        coro_factory: Callable[[], Coroutine],
        model_id: str | None = None,
        token: str | None = None,
        input_tokens: int = 0,
        batch_size: int | None = None,
        data_world_size: int | None = None,
        batch_size_multiple: int | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Schedule a GPU compute task through the serial compute queue.

        Tasks are processed one at a time in round-robin order across all
        per-adapter/per-token queues. Use for any operation that touches GPU
        state: forward, backward, step, save, load, add_adapter, etc.

        Args:
            coro_factory: Zero-argument callable that creates the coroutine.
            model_id: Adapter/model id for queue routing and result association.
            token: User token for rate limiting.
            input_tokens: Token count for TPS rate limiting.
            batch_size: Optional batch size, validated against data_world_size.
            data_world_size: Optional data world size for batch validation.
            task_type: Label for logging and metrics.

        Returns:
            {'request_id': str, 'model_id': str | None}
        """
        return await self._schedule_task(
            coro_factory,
            model_id=model_id,
            token=token,
            input_tokens=input_tokens,
            batch_size=batch_size,
            data_world_size=data_world_size,
            batch_size_multiple=batch_size_multiple,
            task_type=task_type,
            persist_status=True,
        )

    async def schedule_task_and_wait(
        self,
        coro_factory: Callable[[], Coroutine],
        model_id: str | None = None,
        token: str | None = None,
        input_tokens: int = 0,
        batch_size: int | None = None,
        data_world_size: int | None = None,
        batch_size_multiple: int | None = None,
        task_type: str | None = None,
    ) -> Any:
        """Schedule a compute task and block until it completes.

        Twinkle-side counterpart to schedule_task(). Enqueues the task through
        the same serial worker but delivers the result through an in-process
        Future. Large model outputs therefore never enter ServerState.

        Raises:
            RuntimeError: If the task fails or scheduling is rejected.
        """
        completion = asyncio.get_running_loop().create_future()
        task_ref = await self._schedule_task(
            coro_factory,
            model_id=model_id,
            token=token,
            input_tokens=input_tokens,
            batch_size=batch_size,
            data_world_size=data_world_size,
            batch_size_multiple=batch_size_multiple,
            task_type=task_type,
            completion=completion,
            persist_status=False,
        )
        if error := task_ref.get('_error'):
            completion.cancel()
            raise RuntimeError(error)
        return await completion

    async def schedule_background_task(
        self,
        coro_factory: Callable[[], Coroutine],
        model_id: str | None = None,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        """Schedule a fire-and-forget task outside the serial queue.

        The task is launched immediately as an asyncio task. This is suitable
        for pure I/O and for an inference engine such as vLLM that performs its
        own safe request concurrency and continuous batching. Stateful Model
        operations must continue to use :meth:`schedule_task`.

        Status is tracked via state.store_future_status so clients can poll
        progress through the same status endpoints as schedule_task().

        Args:
            coro_factory: Zero-argument callable that creates the coroutine.
            model_id: Optional model id for result association.
            task_type: Label for logging.

        Returns:
            {'request_id': str, 'model_id': str | None}
        """
        request_id = f'req_{uuid.uuid4().hex}'
        logger.info(f'[TaskQueue] Scheduling background task {request_id}, '
                    f'type={task_type or "unknown"}, model_id={model_id}')

        await self.state.store_future_status(
            request_id,
            TaskStatus.RUNNING.value,
            model_id,
            queue_state=QueueState.ACTIVE.value,
        )

        async def _run() -> None:
            try:
                result = await coro_factory()
                await self.state.store_future_status(
                    request_id,
                    TaskStatus.COMPLETED.value,
                    model_id,
                    result=result,
                    queue_state=QueueState.ACTIVE.value,
                )
                logger.info(f'[TaskQueue] Background task {request_id} completed, type={task_type or "unknown"}')
            except Exception:
                error_payload = task_error_payload(traceback.format_exc())
                await self.state.store_future_status(
                    request_id,
                    TaskStatus.FAILED.value,
                    model_id,
                    result=error_payload,
                    queue_state=QueueState.ACTIVE.value,
                )
                logger.error(f'[TaskQueue] Background task {request_id} FAILED, type={task_type or "unknown"}:\n'
                             f'{traceback.format_exc(limit=3)}')

        asyncio.create_task(_run())
        return {'request_id': request_id, 'model_id': model_id}

    async def _fail_queue_tasks_async(self, queue_key: str, reason: str) -> None:
        await self._compute_worker.fail_queue_tasks(queue_key, reason)

    def fail_pending_tasks_for_model(self, model_id: str, reason: str) -> None:
        """Fail and drop all queued tasks for a model. Thread-safe."""
        queue_key = self._queue_key(model_id=model_id, token=None)
        if self._event_loop is None:
            logger.warning(f'[TaskQueue] fail_pending_tasks_for_model called without event loop: {queue_key}')
            return

        def _schedule() -> None:
            asyncio.create_task(self._fail_queue_tasks_async(queue_key, reason))

        self._event_loop.call_soon_threadsafe(_schedule)

    def get_queue_stats(self) -> dict[str, Any]:
        """Return current compute queue statistics."""
        return {
            'queue_size':
            sum(q.qsize() for q in self._compute_worker.task_queues.values()),
            'queue_count':
            len(self._compute_worker.task_queues),
            'worker_running': (self._compute_worker._worker_task is not None
                               and not self._compute_worker._worker_task.done()),
            'rate_limit_config': {
                'rps_limit': self._task_queue_config.rps_limit,
                'tps_limit': self._task_queue_config.tps_limit,
                'enabled': self._task_queue_config.enabled,
            },
        }

    def get_rate_limit_stats(self, token: str) -> dict[str, Any]:
        """Return rate-limiting stats for a user token."""
        return self._rate_limiter.get_stats(token)

    def get_rate_limiter_memory_stats(self) -> dict[str, Any]:
        """Return memory usage statistics from the rate limiter."""
        return self._rate_limiter.get_memory_stats()

    async def shutdown_task_queue(self) -> None:
        """Gracefully shut down the compute queue and release resources."""
        await self._rate_limiter.stop_cleanup_task()
        await self._compute_worker.stop()
        logger.debug('[TaskQueue] Task queue shutdown complete')
