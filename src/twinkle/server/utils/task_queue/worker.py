# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Compute queue worker.

Provides ComputeWorker: a single background asyncio Task that processes
GPU compute tasks serially across all per-adapter/per-token queues using
a round-robin policy.
"""
from __future__ import annotations

import asyncio
import time
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, Deque

from twinkle.utils.logger import get_logger
from .config import TaskQueueConfig
from .types import QueuedTask, QueueState, TaskStatus

if TYPE_CHECKING:
    from twinkle.server.utils.metrics import TaskMetrics
    from twinkle.server.utils.state import ServerStateProxy

logger = get_logger()


class ComputeWorker:
    """Serial background worker that processes GPU compute tasks.

    Implements a round-robin scheduling policy across per-adapter/per-token
    queues so that no single user's long-running task (e.g. save/load) can
    starve other users waiting in different queues.

    Only one task is executed at a time to preserve serial GPU semantics.
    Upload and other pure-I/O tasks should NOT be submitted here; use the
    background-task path in TaskQueueMixin instead.
    """

    def __init__(
        self,
        state: ServerStateProxy,
        config: TaskQueueConfig,
        task_metrics: TaskMetrics | None,
        deployment_name: str,
    ) -> None:
        self._state = state
        self._config = config
        self._task_metrics = task_metrics
        self._deployment_name = deployment_name

        self.task_queues: dict[str, asyncio.Queue] = {}
        self.queue_order: Deque[str] = deque()
        self.new_task_event: asyncio.Event = asyncio.Event()

        self._worker_task: asyncio.Task | None = None
        self._started = False
        self._start_lock = asyncio.Lock()

    async def ensure_started(self) -> None:
        """Ensure the background worker coroutine is running."""
        if self._started and self._worker_task is not None and not self._worker_task.done():
            return
        async with self._start_lock:
            if self._started and self._worker_task is not None and not self._worker_task.done():
                return
            self._worker_task = asyncio.create_task(self._worker_loop())
            self._started = True

    async def stop(self) -> None:
        """Cancel the worker and wait for it to exit cleanly."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None
        self._started = False
        self.task_queues.clear()
        self.queue_order.clear()

    def ensure_queue_registered(self, queue_key: str) -> None:
        """Register a new per-key queue if it does not yet exist."""
        if queue_key not in self.task_queues:
            self.task_queues[queue_key] = asyncio.Queue()
        if queue_key not in self.queue_order:
            self.queue_order.append(queue_key)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _record_tasks_total(self, task_type: str, status: str) -> None:
        """Increment the tasks_total counter if metrics are enabled."""
        if self._task_metrics:
            self._task_metrics.tasks_total.inc(tags={
                'deployment': self._deployment_name,
                'task_type': task_type,
                'status': status,
            })

    def _record_execution_time(self, task_type: str, exec_time: float) -> None:
        """Observe execution duration in the execution_seconds histogram if metrics are enabled."""
        if self._task_metrics:
            self._task_metrics.execution_seconds.observe(
                exec_time, tags={
                    'deployment': self._deployment_name,
                    'task_type': task_type,
                })

    def _record_queue_metrics(self, task_type: str, queue_wait: float) -> None:
        """Observe queue wait time and update current queue depth if metrics are enabled."""
        if self._task_metrics:
            self._task_metrics.queue_wait_seconds.observe(
                queue_wait, tags={
                    'deployment': self._deployment_name,
                    'task_type': task_type,
                })
            total_depth = sum(qq.qsize() for qq in self.task_queues.values())
            self._task_metrics.queue_depth.set(total_depth, tags={'deployment': self._deployment_name})

    # ------------------------------------------------------------------

    async def _store_task_failed(
        self,
        task: QueuedTask,
        error: str,
        queue_state: str,
        queue_state_reason: str | None = None,
    ) -> None:
        """Store FAILED status with a standardised error payload."""
        await self._state.store_future_status(
            task.request_id,
            TaskStatus.FAILED.value,
            task.model_id,
            result={
                'error': error,
                'category': 'Server'
            },
            queue_state=queue_state,
            queue_state_reason=queue_state_reason,
        )

    async def fail_queue_tasks(self, queue_key: str, reason: str) -> None:
        """Drain a queue and mark all pending tasks as FAILED."""
        q = self.task_queues.get(queue_key)
        if q is None:
            return

        drained: list[QueuedTask] = []
        while True:
            try:
                drained.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break

        for task in drained:
            await self._store_task_failed(task, reason, QueueState.UNKNOWN.value, queue_state_reason=reason)
            q.task_done()

        self.task_queues.pop(queue_key, None)
        try:
            while queue_key in self.queue_order:
                self.queue_order.remove(queue_key)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Worker loop helpers
    # ------------------------------------------------------------------

    async def _wait_for_work(self) -> None:
        """Block until at least one queue has a pending task."""
        while True:
            if any(q.qsize() > 0 for q in self.task_queues.values()):
                return
            self.new_task_event.clear()
            await self.new_task_event.wait()

    async def _fail_timed_out_task(self, task: QueuedTask, waited: float, q: asyncio.Queue) -> None:
        """Mark a queue-timed-out task as FAILED and release it from the queue."""
        error = f'Queue timeout exceeded: waited {waited:.2f}s'
        await self._store_task_failed(task, error, QueueState.PAUSED_CAPACITY.value, queue_state_reason=error)
        self._record_tasks_total(task.task_type or 'unknown', 'timeout')
        q.task_done()

    async def _execute_task(self, task: QueuedTask, queue_key: str, q: asyncio.Queue) -> None:
        """Execute a single task: update status, run coroutine, record metrics.

        Handles execution timeout, general exceptions, and always calls
        q.task_done() in the finally block.
        """
        await self._state.store_future_status(
            task.request_id, TaskStatus.RUNNING.value, task.model_id, queue_state=QueueState.ACTIVE.value)

        task_type = task.task_type or 'unknown'
        exec_start = time.monotonic()
        task_status = 'completed'
        exec_time = 0.0
        try:
            coro = task.coro_factory()
            logger.debug(f'[ComputeWorker] Task {task.request_id} executing, '
                         f'type={task_type}, queue_key={queue_key}')
            if self._config.execution_timeout > 0:
                result = await asyncio.wait_for(coro, timeout=self._config.execution_timeout)
            else:
                result = await coro
            exec_time = time.monotonic() - exec_start
            logger.info(f'[ComputeWorker] Task {task.request_id} completed in {exec_time:.2f}s, type={task_type}')
            await self._state.store_future_status(
                task.request_id,
                TaskStatus.COMPLETED.value,
                task.model_id,
                result=result,
                queue_state=QueueState.ACTIVE.value)
        except asyncio.TimeoutError:
            task_status = 'timeout'
            exec_time = time.monotonic() - exec_start
            error = (f'Execution timeout exceeded: {self._config.execution_timeout}s, '
                     f'actual execution time: {exec_time:.2f}s')
            logger.error(f'[ComputeWorker] Task {task.request_id} TIMEOUT after {exec_time:.2f}s, '
                         f'type={task_type}, queue_key={queue_key}')
            await self._store_task_failed(task, error, QueueState.ACTIVE.value)
        except Exception:
            task_status = 'failed'
            exec_time = time.monotonic() - exec_start
            error = traceback.format_exc()
            logger.error(f'[ComputeWorker] Task {task.request_id} FAILED after {exec_time:.2f}s, '
                         f'type={task_type}:\n{traceback.format_exc(limit=3)}')
            await self._store_task_failed(task, error, QueueState.ACTIVE.value)
        finally:
            q.task_done()
            self._record_execution_time(task_type, exec_time)
            self._record_tasks_total(task_type, task_status)

    async def _try_run_one(self) -> bool:
        """Round-robin: pick one runnable task and execute it.

        Iterates queues in round-robin order (at most once each).
        - Queue-timed-out tasks are failed and skipped; the next queue is tried.
        - The first runnable task is executed and the method returns True.
        - Returns False if all queues were empty or every dequeued task had timed out.
        """
        for _ in range(len(self.queue_order)):
            queue_key = self.queue_order[0]
            self.queue_order.rotate(-1)

            q = self.task_queues.get(queue_key)
            if q is None:
                continue

            try:
                task: QueuedTask = q.get_nowait()
            except asyncio.QueueEmpty:
                continue

            # Record queue-wait metrics and check queue-level timeout
            queue_wait = time.monotonic() - task.created_at
            self._record_queue_metrics(task.task_type or 'unknown', queue_wait)

            if queue_wait > self._config.queue_timeout:
                await self._fail_timed_out_task(task, queue_wait, q)
                continue  # try the next queue

            # Execute the task (serial: stops after the first execution)
            await self._execute_task(task, queue_key, q)
            return True

        return False

    # ------------------------------------------------------------------
    # Main worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Main worker loop: wait for work, run one task, repeat."""
        logger.info(f'[ComputeWorker] Started, queue_timeout={self._config.queue_timeout}, '
                    f'execution_timeout={self._config.execution_timeout}')
        while True:
            try:
                await self._wait_for_work()
                executed = await self._try_run_one()
                if not executed:
                    # All dequeued tasks were timed-out; yield briefly to avoid busy spin.
                    await asyncio.sleep(min(self._config.window_seconds, 0.1))
            except asyncio.CancelledError:
                logger.info('[ComputeWorker] Worker stopped')
                break
            except Exception:
                logger.error(f'[ComputeWorker] Unexpected error:\n{traceback.format_exc(limit=3)}')
                continue
