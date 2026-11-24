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
from dataclasses import replace
from ray.exceptions import RayTaskError
from typing import TYPE_CHECKING, Any, Callable, Deque

from twinkle.protocol.types.errors import ErrorCategory, QueueState
from twinkle.protocol.types.lifecycle import TaskStatus
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.exceptions import (BackendBusyError, BatchSizeError, EndpointUnavailableError, FullModeBusyError,
                                       InputTokensExceededError, RateLimitExceededError, RequestRejectedError,
                                       ResourceExhaustedError, ResourceNotFoundError, ResourceQuotaExceededError,
                                       StateBackendError, TrainModeMismatchError, TwinkleServerError)
from twinkle.server.state.backend.base import ConcurrencyError
from twinkle.server.state.models import FutureFailureRecord
from twinkle.server.task_queue.backend_gate import BACKEND_TIMEOUT_EXCEPTIONS
from twinkle.server.telemetry.correlation import BASE_MODEL, TASK_ID, TENANT_FINGERPRINT, TRAINING_ID
from twinkle.server.telemetry.identity import current_tenant_fingerprint
from twinkle.server.telemetry.metrics import METERED_TASK_TYPES
from twinkle.server.telemetry.tracing import get_trace_id, set_span_error, traced_operation
from twinkle.utils.logger import get_logger
from .types import QueuedTask

if TYPE_CHECKING:
    from twinkle.server.state import ServerState
    from twinkle.server.telemetry.metrics import TaskMetrics

logger = get_logger()


def _exception_summary(exc: BaseException) -> str:
    """Build a single-line summary, unwrapping Ray's transport wrapper.

    A RayTaskError's own ``str`` starts with the Ray call header (actor id,
    pid, repr); the actionable error (e.g. the loss assertion) is in ``cause``.
    Preferring it keeps the summary useful without enlarging the wire field,
    while the full traceback still travels separately.

    Only Ray's wrapper is unwrapped: elsewhere ``raise X from e`` is a
    deliberate message and must not be replaced by its inner cause.
    """
    cause = None
    if isinstance(exc, RayTaskError):
        cause = getattr(exc, 'cause', None)
        if not isinstance(cause, BaseException):
            cause = exc.__cause__
    if isinstance(cause, BaseException):
        return f'{type(cause).__name__}: {cause}'
    return f'{type(exc).__name__}: {exc}'


def _reason_code_for_server_error(exc: TwinkleServerError) -> str:
    """Map typed execution failures to protocol-independent reason codes."""
    mappings: tuple[tuple[type[TwinkleServerError], str], ...] = (
        (TrainModeMismatchError, 'invalid_request'),
        (ResourceNotFoundError, 'resource_not_found'),
        (FullModeBusyError, 'full_mode_busy'),
        (InputTokensExceededError, 'input_tokens_exceeded'),
        (BatchSizeError, 'batch_size_invalid'),
        (RateLimitExceededError, 'rate_limit_exceeded'),
        (ResourceQuotaExceededError, 'resource_quota_exceeded'),
        (EndpointUnavailableError, 'endpoint_unavailable'),
        (ConcurrencyError, 'state_contention'),
        (StateBackendError, 'backend_gate_unavailable'),
        (RequestRejectedError, 'request_rejected'),
    )
    for error_type, reason_code in mappings:
        if isinstance(exc, error_type):
            return reason_code
    return 'internal_error'


class ComputeWorker:
    """Serial background worker that processes GPU compute tasks.

    Implements a round-robin scheduling policy across per-adapter/per-token
    queues so that no single user's long-running task (e.g. save/load) can
    starve other users waiting in different queues.

    Only one queued task is executed at a time to preserve serial GPU semantics;
    :meth:`run_detached` runs a task outside that order. Upload and other
    pure-I/O tasks should NOT be submitted here; use the background-task path
    in TaskQueueMixin instead.
    """

    def __init__(
        self,
        state: ServerState,
        config: TaskQueueConfig,
        task_metrics: TaskMetrics | None,
        deployment_name: str,
        on_backend_timeout: Callable[[], Any] | None = None,
    ) -> None:
        self._state = state
        self._config = config
        self._task_metrics = task_metrics
        self._deployment_name = deployment_name
        # Optional coroutine-returning callback fired on a backend timeout.
        self._on_backend_timeout = on_backend_timeout

        self.task_queues: dict[str, asyncio.Queue] = {}
        self._request_ids: set[str] = set()
        self._task_outcomes: dict[str, dict[str, Any]] = {}
        self.queue_order: Deque[str] = deque()
        self.new_task_event: asyncio.Event = asyncio.Event()

        self._worker_task: asyncio.Task | None = None
        self._started = False
        self._stopping = False
        self._start_lock = asyncio.Lock()
        # Strong references to tasks started by run_detached(); the event loop
        # only keeps weak ones.
        self._detached_tasks: dict[asyncio.Task, QueuedTask] = {}
        self._detached_reserved = 0

    def ensure_accepting(self) -> None:
        if self._stopping:
            raise ResourceExhaustedError('The replica is shutting down.')

    async def ensure_started(self) -> None:
        """Ensure the background worker coroutine is running; never reopen a stopped replica."""
        self.ensure_accepting()
        if self._started and self._worker_task is not None and not self._worker_task.done():
            return
        async with self._start_lock:
            self.ensure_accepting()
            if self._started and self._worker_task is not None and not self._worker_task.done():
                return
            self._worker_task = asyncio.create_task(self._worker_loop())
            self._started = True

    async def stop(self) -> None:
        """Cancel the worker and any detached tasks, and wait for them to exit."""
        self._stopping = True
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        detached = list(self._detached_tasks.items())
        for running, _ in detached:
            running.cancel()
        await asyncio.gather(*(running for running, _ in detached), return_exceptions=True)
        for running, task in detached:
            if running.cancelled() and not await self._is_record_terminal(task.request_id):
                logger.warning(f'[ComputeWorker] Task {task.request_id} cancelled during replica shutdown',
                               extra=task.log_fields('cancelled'))
                await self._store_task_failed(
                    task,
                    'Sampler task stopped during replica shutdown.',
                    QueueState.ACTIVE.value,
                    reason_code='orphaned_replica')
        for queue_key in list(self.task_queues):
            await self.fail_queue_tasks(queue_key, 'Replica stopped before the task could execute.')
        self._worker_task = None
        self._started = False
        self._request_ids.clear()

    def ensure_queue_registered(self, queue_key: str) -> None:
        """Register a new per-key queue if it does not yet exist."""
        if queue_key not in self.task_queues:
            self.task_queues[queue_key] = asyncio.Queue()
        if queue_key not in self.queue_order:
            self.queue_order.append(queue_key)

    def get_queue(self, queue_key: str) -> asyncio.Queue:
        """Return the registered queue for ``queue_key``.

        Callers must have registered it first via :meth:`ensure_queue_registered`.
        Exposed so producers (the mixin) enqueue through a method instead of
        indexing the worker's internal ``task_queues`` container directly.
        """
        return self.task_queues[queue_key]

    def track_task(self, request_id: str) -> None:
        self._request_ids.add(request_id)

    def has_task(self, request_id: str) -> bool:
        """True while a task is queued or executing, including a terminal-future race."""
        return request_id in self._request_ids

    def get_task_outcome(self, request_id: str) -> dict[str, Any] | None:
        """Read bounded local execution facts, never infer them from a future status."""
        cutoff = time.time() - self._config.absolute_future_ttl(1)
        for identity, outcome in list(self._task_outcomes.items()):
            if outcome['finished_at'] <= cutoff:
                self._task_outcomes.pop(identity)
        outcome = self._task_outcomes.get(request_id)
        return dict(outcome) if outcome is not None else None

    async def finish_task(self,
                          task: QueuedTask,
                          executed: bool,
                          unsafe: bool = False,
                          finished_at: float | None = None) -> None:
        if task.on_finished is not None:
            self.get_task_outcome(task.request_id)
            self._task_outcomes[task.request_id] = {
                'state': 'unsafe' if unsafe else 'finished',
                'executed': executed,
                'finished_at': time.time() if finished_at is None else finished_at,
            }
        self._request_ids.discard(task.request_id)
        if task.on_finished is not None:
            acknowledged = await task.on_finished(executed, unsafe)
            if acknowledged is True:
                self._task_outcomes.pop(task.request_id, None)

    def total_queued(self) -> int:
        """Total number of pending tasks across all per-key queues."""
        return sum(q.qsize() for q in self.task_queues.values())

    def notify_new_task(self) -> None:
        """Wake the worker loop and refresh the queue-depth gauge.

        Producers (the mixin) call this instead of reaching into the worker's
        ``new_task_event`` directly; it is also the single write point of the
        queue-depth gauge, so the gauge has one writer rather than two.
        """
        self.new_task_event.set()
        self._record_queue_depth()

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    def _task_tags(self, task: QueuedTask) -> dict[str, str]:
        tags = {'deployment': self._deployment_name, 'task_type': task.task_type or 'unknown'}
        if task.tenant_fingerprint:
            tags['tenant_fingerprint'] = task.tenant_fingerprint
        if task.base_model:
            tags['base_model'] = task.base_model
        return tags

    def record_admission(self, task: QueuedTask) -> None:
        if self._task_metrics and task.tenant_fingerprint and task.task_type in METERED_TASK_TYPES:
            tags = self._task_tags(task)
            self._task_metrics.usage_requests.inc(tags=tags)
            self._task_metrics.usage_input_tokens.inc(task.input_tokens, tags=tags)

    def _record_tasks_total(self, task: QueuedTask, status: str) -> None:
        if self._task_metrics:
            self._task_metrics.tasks_total.inc(tags={**self._task_tags(task), 'status': status})

    def _record_execution_time(self, task: QueuedTask, exec_time: float) -> None:
        if self._task_metrics:
            tags = self._task_tags(task)
            self._task_metrics.execution_seconds.observe(exec_time, tags=tags)

    def _record_queue_depth(self) -> None:
        """Single writer of the queue-depth gauge."""
        if self._task_metrics:
            total_depth = sum(qq.qsize() for qq in self.task_queues.values())
            self._task_metrics.queue_depth.set(total_depth, tags={'deployment': self._deployment_name})

    def _record_queue_metrics(self, task: QueuedTask, queue_wait: float) -> None:
        if self._task_metrics:
            self._task_metrics.queue_wait_seconds.observe(queue_wait, tags=self._task_tags(task))

    # ------------------------------------------------------------------

    async def _store_task_failed(
        self,
        task: QueuedTask,
        error: str,
        queue_state: str,
        queue_state_reason: str | None = None,
        *,
        reason_code: str = 'internal_error',
        attribution: str = 'server',
        traceback_text: str | None = None,
    ) -> None:
        """Store FAILED status with protocol-independent failure details."""
        failure = FutureFailureRecord(
            reason_code=reason_code,
            message=(error.splitlines() or [''])[0][:1024],
            attribution='user' if attribution == 'user' else 'server',
            diagnostic=traceback_text,
        )
        await self._state.store_future_status(
            task.request_id,
            TaskStatus.FAILED.value,
            task.model_id,
            failure=failure,
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
            logger.warning(f'[ComputeWorker] Task {task.request_id} failed before execution: {reason}',
                           extra=task.log_fields('failed'))
            await self._store_task_failed(task, reason, QueueState.UNKNOWN.value, queue_state_reason=reason)
            q.task_done()
            await self.finish_task(task, False)

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
        logger.warning(f'[ComputeWorker] Task {task.request_id} {error}', extra=task.log_fields('timeout'))
        await self._store_task_failed(task, error, QueueState.PAUSED_CAPACITY.value, queue_state_reason=error)
        self._record_tasks_total(task, 'timeout')
        q.task_done()
        await self.finish_task(task, False)

    async def _execute_task(self, task: QueuedTask, queue_wait: float, q: asyncio.Queue) -> None:
        """Execute a dequeued task, then release its queue slot."""
        try:
            await self._process_task(task, queue_wait)
        finally:
            q.task_done()

    async def submit_detached(self, task: QueuedTask, *, replica_id: str | None,
                              absolute_deadline: float, submission_fingerprint: str | None = None) -> bool:
        """Own admission, future persistence and startup as one lifecycle."""
        self.reserve_detached()
        wrote_status = False
        try:
            fields = dict(
                model_id=task.model_id,
                queue_state=QueueState.ACTIVE.value,
                replica_id=replica_id,
                absolute_deadline=absolute_deadline,
                owner_token=task.token)
            if submission_fingerprint is None:
                await self._state.store_future_status(task.request_id, TaskStatus.PENDING.value, **fields)
            elif not await self._state.create_pending_future(
                    task.request_id, submission_fingerprint=submission_fingerprint, **fields):
                self.release_detached()
                return False
            wrote_status = True
            self.run_detached(task)
        except BaseException:
            self.release_detached()
            if wrote_status:
                logger.warning(f'[ComputeWorker] Task {task.request_id} could not start',
                               extra=task.log_fields('failed'))
                await self._store_task_failed(
                    task,
                    'Replica stopped before the concurrent task could start.',
                    QueueState.ACTIVE.value,
                    reason_code='orphaned_replica')
            raise
        self.record_admission(task)
        return True

    def reserve_detached(self) -> None:
        """Reserve before the future write, so capacity rejection leaves no record."""
        self.ensure_accepting()
        if self._detached_reserved >= self._config.concurrent_task_limit:
            raise ResourceExhaustedError('Concurrent compute capacity is unavailable on this replica.')
        self._detached_reserved += 1

    def release_detached(self) -> None:
        self._detached_reserved -= 1

    def run_detached(self, task: QueuedTask) -> None:
        """Start ``task`` now, beside the serial loop instead of in its turn.

        For backends that batch concurrent requests themselves (vLLM). The task
        gets the same status records, execution timeout and failure mapping as a
        queued one.
        """
        if self._stopping:
            raise ResourceExhaustedError('The replica is shutting down.')
        execution = None

        def start():
            nonlocal execution
            execution = asyncio.create_task(task.coro_factory())
            return asyncio.shield(execution)

        async def run():
            try:
                queue_wait = time.monotonic() - task.created_at
                self._record_queue_metrics(task, queue_wait)
                await self._process_task(replace(task, coro_factory=start), queue_wait)
            finally:
                # A timeout ends the client future, not the backend operation.
                # Retain this task's sole capacity reservation until it exits.
                if execution is not None:
                    if not execution.done():
                        execution.cancel()
                    while not execution.done():
                        try:
                            await asyncio.shield(execution)
                        except asyncio.CancelledError:
                            if not execution.done() and not execution.cancelling():
                                execution.cancel()
                            continue
                        except Exception:
                            break
                    if not execution.cancelled():
                        execution.exception()

        running = asyncio.create_task(run())
        self._detached_tasks[running] = task
        scope = task.execution_scope
        if scope is not None:
            scope.track(running, task.request_id)

        def done(completed: asyncio.Task) -> None:
            self._detached_tasks.pop(completed, None)
            self.release_detached()
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    logger.error('Detached task failed', exc_info=(type(error), error, error.__traceback__))

        running.add_done_callback(done)

    async def _process_task(self, task: QueuedTask, queue_wait: float) -> None:
        """Run a single task: update status, run coroutine, record metrics.

        Handles execution timeout and general exceptions.
        """
        task_type = task.task_type or 'unknown'
        task_status = 'completed'
        exec_time = 0.0
        executed = False
        finished_at = None
        queue_attrs = {
            TENANT_FINGERPRINT: task.tenant_fingerprint,
            TRAINING_ID: task.model_id,
            BASE_MODEL: task.base_model,
            'twinkle.task_type': task_type,
            'twinkle.deployment': self._deployment_name or 'unknown',
            TASK_ID: task.request_id,
            'twinkle.wait_seconds': queue_wait,
        }
        with traced_operation('task_queue.execute', attrs=queue_attrs, parent_context=task.trace_context) as span:
            trace_id = get_trace_id()
            try:
                if not await self._state.try_start_future(task.request_id):
                    task_status = 'cancelled'
                    logger.info(f'[ComputeWorker] Task {task.request_id} was not acquired; skipping.',
                                extra=task.log_fields(task_status))
                    return
                binding = current_tenant_fingerprint.set(task.tenant_fingerprint)
                exec_start = time.monotonic()
                try:
                    executed = True
                    result = await asyncio.wait_for(
                        task.coro_factory(), timeout=self._config.effective_execution_timeout)
                    finished_at = time.time()
                finally:
                    exec_time = time.monotonic() - exec_start
                    current_tenant_fingerprint.reset(binding)
                    span.set_attribute('twinkle.execution_seconds', exec_time)
                await self._state.store_future_status(
                    task.request_id,
                    TaskStatus.COMPLETED.value,
                    task.model_id,
                    result=result,
                    queue_state=QueueState.ACTIVE.value,
                )
                logger.info(f'[ComputeWorker] Task {task.request_id} completed in {exec_time:.2f}s, type={task_type}',
                            extra=task.log_fields(task_status, trace_id))
            except asyncio.CancelledError:
                task_status = 'cancelled'
                logger.warning('[ComputeWorker] Task %s cancelled, type=%s', task.request_id, task_type,
                               extra=task.log_fields(task_status, trace_id))
                if not await self._is_record_terminal(task.request_id):
                    await self._store_task_failed(
                        task,
                        ('Task stopped during replica shutdown.' if self._stopping
                         else 'Backend generation was cancelled.'),
                        QueueState.ACTIVE.value,
                        reason_code='orphaned_replica' if self._stopping else 'backend_cancelled')
                raise
            except BACKEND_TIMEOUT_EXCEPTIONS:
                task_status = 'timeout'
                set_span_error(span, 'execution_timeout')
                error = (f'Backend call timed out (bound {self._config.effective_execution_timeout}s), '
                         f'actual execution time: {exec_time:.2f}s')
                logger.error(f'[ComputeWorker] Task {task.request_id} TIMEOUT after {exec_time:.2f}s, '
                             f'type={task_type}', extra=task.log_fields(task_status, trace_id))
                # asyncio.TimeoutError and Ray_Get_Timeout are 504/Server.
                await self._store_task_failed(task, error, QueueState.ACTIVE.value, reason_code='execution_timeout')
                # Probe actor liveness after a timeout so an operator learns the replica's
                # state without waiting for a second request to also time out.
                if self._on_backend_timeout is not None:
                    try:
                        await self._on_backend_timeout()
                    except Exception as exc:
                        logger.error('[ComputeWorker] Backend-timeout probe failed (%s)', type(exc).__name__)
            except BackendBusyError as exc:
                task_status = 'failed'
                set_span_error(span, 'backend_gate_unavailable')
                error = str(exc)
                logger.error('[ComputeWorker] Task %s refused after %.2fs, type=%s, reason=backend_gate_unavailable',
                             task.request_id, exec_time, task_type, extra=task.log_fields('refused', trace_id))
                # Gate held by a leaked timed-out call -> 503/Server.
                await self._store_task_failed(
                    task, error, QueueState.ACTIVE.value, reason_code='backend_gate_unavailable')
            except TwinkleServerError as exc:
                # A typed server error carries its own status + category (e.g.
                # ResourceNotFoundError = 404/User from a deferred
                # assert_resource_exists). Honour them instead of collapsing every
                # such failure to 500/Server. Only Server-category errors keep a
                # traceback; a User rejection does not.
                task_status = 'failed'
                is_server = exc.category is ErrorCategory.Server
                reason_code = _reason_code_for_server_error(exc)
                if is_server:
                    set_span_error(span, reason_code)
                    logger.error(f'[ComputeWorker] Task {task.request_id} FAILED after {exec_time:.2f}s, '
                                 f'type={task_type}, reason={reason_code}',
                                 extra=task.log_fields(task_status, trace_id))
                else:
                    logger.info(f'[ComputeWorker] Task {task.request_id} rejected, type={task_type}, '
                                f'reason={reason_code}', extra=task.log_fields('refused', trace_id))
                await self._store_task_failed(
                    task,
                    f'{type(exc).__name__}: {exc}',
                    QueueState.ACTIVE.value,
                    reason_code=reason_code,
                    attribution=exc.category.value,
                    traceback_text=traceback.format_exc() if is_server else None,
                )
            except Exception as exc:
                task_status = 'failed'
                set_span_error(span, type(exc).__name__)
                error = _exception_summary(exc)
                logger.error(f'[ComputeWorker] Task {task.request_id} FAILED after {exec_time:.2f}s, '
                             f'type={task_type}, error={type(exc).__name__}',
                             extra=task.log_fields(task_status, trace_id))
                await self._store_task_failed(
                    task,
                    error,
                    QueueState.ACTIVE.value,
                    reason_code='internal_error',
                    traceback_text=traceback.format_exc(),
                )
            finally:
                span.set_attribute('twinkle.task_outcome', task_status)
                if executed:
                    self._record_execution_time(task, exec_time)
                self._record_tasks_total(task, task_status)
                await self.finish_task(
                    task, executed, executed and task_status in ('timeout', 'cancelled'), finished_at)

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
            self._record_queue_metrics(task, queue_wait)
            self._record_queue_depth()

            if queue_wait > self._config.queue_timeout:
                await self._fail_timed_out_task(task, queue_wait, q)
                continue  # try the next queue

            # Execute the task (serial: stops after the first execution)
            await self._execute_task(task, queue_wait, q)
            return True

        return False

    async def _is_record_terminal(self, request_id: str) -> bool:
        """True if the future record already holds a Terminal_State."""
        record = await self._state.get_future(request_id)
        if record is None:
            return False
        return record.get('status') in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value)

    # ------------------------------------------------------------------
    # Main worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Main worker loop: wait for work, run one task, repeat."""
        logger.info(f'[ComputeWorker] Started, queue_timeout={self._config.queue_timeout}, '
                    f'execution_timeout={self._config.execution_timeout}')
        # Python 3.10/3.11 wait_for can swallow cancellation when its child
        # completes simultaneously. Do not wait for more work after stop().
        while not self._stopping:
            try:
                await self._wait_for_work()
                executed = await self._try_run_one()
                if not executed:
                    # All dequeued tasks were timed-out; yield briefly to avoid busy spin.
                    await asyncio.sleep(min(self._config.window_seconds, 0.1))
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error('[ComputeWorker] Unexpected worker error', exc_info=True)
                continue
        logger.info('[ComputeWorker] Worker stopped')
