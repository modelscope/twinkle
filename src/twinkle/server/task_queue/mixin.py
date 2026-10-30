# Copyright (c) ModelScope Contributors. All rights reserved.
"""TaskQueueMixin: serial compute queue plus admitted concurrent tasks.

``schedule_task`` serializes stateful Model operations. Detached tasks are
used for I/O and for engines such as vLLM that own their compute concurrency
and continuous batching internally.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from twinkle.protocol.types.errors import QueueState
from twinkle.protocol.types.lifecycle import TERMINAL_STATUSES, TaskEnvelope, TaskStatus
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.exceptions import BatchSizeError, ConfigError, InputTokensExceededError, RateLimitExceededError
from twinkle.server.lifecycle.envelope import envelope_from_record
from twinkle.server.lifecycle.poll_config import long_poll_window
from twinkle.server.state.models import FutureFailureRecord
from twinkle.server.telemetry.metrics import get_task_metrics
from twinkle.utils.logger import get_logger
from .backend_gate import BackendGate
from .rate_limiter import RateLimiter
from .types import ExecutionScope, QueuedTask
from .worker import ComputeWorker

if TYPE_CHECKING:
    from twinkle.server.state import ServerState

logger = get_logger()


class TaskQueueMixin:
    """Mixin providing three task execution paths.

    Execution paths
    ---------------
    1. Compute queue (schedule_task / submit_and_peek):
       Single background worker, serial execution, round-robin across queues.
       Use for GPU operations: forward, backward, step, save, load, etc.

    2. Concurrent compute task (schedule_concurrent_task):
       Same admission, status tracking and failure handling as the compute
       queue, but starts immediately. Use for a backend that batches
       concurrent requests itself, such as vLLM sampling.

    3. Background task (schedule_background_task):
       Reuses the supervised concurrent path for pure I/O. Status, capacity
       and shutdown are owned by the same worker as concurrent compute.

    Requirements
    ------------
    Inheriting class must expose self.state: ServerState and call
    _init_task_queue() during __init__.
    """

    state: ServerState
    replica_id: str

    def _init_task_queue(
        self,
        config: TaskQueueConfig | None = None,
        deployment_name: str = '',
        *,
        enable_admission_gate: bool = False,
        on_backend_timeout: Callable[[], Coroutine[Any, Any, None]] | None = None,
        collect_width: int = 1,
    ) -> None:
        """Initialise the task queue, rate limiter, and compute worker.

        ``config`` must be a typed :class:`TaskQueueConfig` (the launcher
        passes the instance straight through). ``None`` constructs a default
        config.

        ``enable_admission_gate`` turns on the per-replica Admission_Gate
        (:meth:`call_backend`). ``ModelManagement`` enables it; ``SamplerManagement``
        does not (vllm sampler owns its own concurrency and the weight-update /
        generation mutual exclusion is covered by infra ``_cw_barrier``).

        ``on_backend_timeout`` runs after a backend timeout. ``collect_width`` is the
        number of actor results a backend call may collect and determines the persisted
        future deadline.
        """
        self._task_queue_config = config if config is not None else TaskQueueConfig()
        if self._task_queue_config.execution_timeout == 0:
            logger.warning(
                '[TaskQueue] execution_timeout=0: a finite %.0fs bound has replaced unbounded waiting '
                '(deployment=%s).', self._task_queue_config.effective_execution_timeout, deployment_name or 'unknown')
        # The Inline_Fast_Path window must stay strictly under Long_Poll_Window: a
        # submit that peeks longer than a retrieve would wait makes no sense (D4).
        # Raised, not asserted -- `python -O` strips asserts and would drop this
        # invariant silently.
        _inline = self._task_queue_config.inline_fast_path_timeout
        _window = long_poll_window()
        if _inline >= _window:
            raise ConfigError(
                'inline_fast_path_timeout',
                _inline,
                message=(f'inline_fast_path_timeout ({_inline}s) must be < Long_Poll_Window '
                         f'({_window}s); lower it or raise TWINKLE_LONG_POLL_TIMEOUT.'))
        self._deployment_name = deployment_name
        self._task_metrics = get_task_metrics(deployment_name) if deployment_name else None
        self._future_absolute_ttl = self._task_queue_config.absolute_future_ttl(collect_width)

        self._rate_limiter = RateLimiter(
            rps_limit=self._task_queue_config.rps_limit,
            tps_limit=self._task_queue_config.tps_limit,
            window_seconds=self._task_queue_config.window_seconds,
            token_cleanup_multiplier=self._task_queue_config.token_cleanup_multiplier,
            token_cleanup_interval=self._task_queue_config.token_cleanup_interval,
            active_tokens_gauge=self._task_metrics.rate_limiter_active_tokens if self._task_metrics else None,
            deployment_name=deployment_name,
            replica_id=self.replica_id,
        )
        self._rate_limiter.start_cleanup_task()

        self._compute_worker = ComputeWorker(
            state=self.state,
            config=self._task_queue_config,
            task_metrics=self._task_metrics,
            deployment_name=deployment_name,
            on_backend_timeout=on_backend_timeout,
        )

        self._backend_gate = BackendGate(enable_admission_gate=enable_admission_gate)
        #: Strong references to in-flight fire-and-forget background tasks (see schedule_background_task).
        self._background_tasks: set = set()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._queue_tasks: set[asyncio.Task] = set()

    @property
    def task_queue_config(self) -> TaskQueueConfig:
        """The deployment's validated queue config.

        Public because four modules outside this file read it; it was
        ``_task_queue_config`` (name-private, interface-public), which meant the
        Host_Protocol could not declare it without declaring a private name.
        """
        return self._task_queue_config

    @property
    def sequence_retention_s(self) -> float:
        return self.state.expiration_timeout + self._future_absolute_ttl

    async def call_backend(self, fn: Callable[..., Any], /, *args: Any, admit: bool = True, **kwargs: Any) -> Any:
        """Run one backend call outside the event loop.

        Single delegation to :class:`BackendGate`. The name and call shape are
        load-bearing: ``test_no_direct_backend_call.py`` keys on ``self.call_backend``
        call sites, so keeping it here means the check's predicate and its exemption
        table need no change.
        """
        return await self._backend_gate.call(fn, *args, admit=admit, **kwargs)

    def _future_deadline(self) -> float:
        ttl = getattr(self, '_future_absolute_ttl', self._task_queue_config.absolute_future_ttl(1))
        return time.time() + ttl

    @staticmethod
    def _queue_key(model_id: str | None, token: str | None) -> str:
        if model_id:
            return f'model:{model_id}'
        if token:
            return f'token:{token}'
        return 'default'

    async def _perform_preflight_checks(
        self,
        model_id: str | None,
        token: str | None,
        input_tokens: int,
        batch_size: int | None = None,
        data_world_size: int | None = None,
        batch_size_multiple: int | None = None,
    ) -> None:
        """Run rate-limit and validation checks before queuing a task.

        Returns ``None`` when every check passes. On failure it RAISES a
        ``RequestRejectedError`` subclass -- the Decision_Boundary is this line, and
        raising before any ``store_future_status`` call is what guarantees zero
        future writes for a rejected request. It writes no FAILED
        record and returns no ``_error`` marker.
        """
        if not token or not self._task_queue_config.enabled:
            return

        if input_tokens > self._task_queue_config.max_input_tokens:
            raise InputTokensExceededError(f'Input tokens ({input_tokens}) exceed maximum allowed '
                                           f'({self._task_queue_config.max_input_tokens})')

        if batch_size is not None and data_world_size is not None:
            if batch_size < data_world_size:
                raise BatchSizeError(f'Batch size {batch_size} must be >= data world size {data_world_size}')
            if batch_size_multiple is not None:
                required_multiple = data_world_size * batch_size_multiple
                if batch_size % required_multiple != 0:
                    raise BatchSizeError(f'Batch size {batch_size} must be divisible by {required_multiple} '
                                         f'so each data-parallel shard gets a multiple of '
                                         f'{batch_size_multiple} examples')

        allowed, reason = await self._rate_limiter.check_and_record(token, input_tokens)
        if not allowed:
            if self._task_metrics:
                self._task_metrics.rate_limit_rejections.inc(tags={'deployment': self._deployment_name})
            raise RateLimitExceededError(f'Rate limit exceeded: {reason}')

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
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Common enqueue path. Always persists status: the future record is the
        single delivery channel for both result and failure."""
        request_id = request_id or f'req_{uuid.uuid4().hex}'

        # Decision_Boundary: raises RequestRejectedError before any state write.
        await self._perform_preflight_checks(
            model_id=model_id,
            token=token,
            input_tokens=input_tokens,
            batch_size=batch_size,
            data_world_size=data_world_size,
            batch_size_multiple=batch_size_multiple,
        )

        task = QueuedTask(
            request_id=request_id,
            coro_factory=coro_factory,
            model_id=model_id,
            token=token,
            input_tokens=input_tokens,
            task_type=task_type,
            created_at=time.monotonic())
        await self._prepare_queued_task(task)

        if self._event_loop is None:
            self._event_loop = asyncio.get_running_loop()

        try:
            await self._enqueue_prepared_task(task)
        except BaseException:
            if task.on_finished is not None and await self.state.get_future(request_id) is None:
                await self._compute_worker.finish_task(task, False)
            raise
        return {'request_id': request_id, 'model_id': model_id}

    async def _prepare_queued_task(self, task: QueuedTask) -> None:
        """Optional owner admission, after preflight and before any future write."""

    async def _enqueue_prepared_task(self, task: QueuedTask) -> None:
        request_id, model_id = task.request_id, task.model_id
        self._compute_worker.ensure_accepting()
        await self.state.store_future_status(
            request_id,
            TaskStatus.PENDING.value,
            model_id,
            queue_state=QueueState.ACTIVE.value,
            replica_id=getattr(self, 'replica_id', None),
            absolute_deadline=self._future_deadline(),
            owner_token=task.token,
        )

        try:
            await self._compute_worker.ensure_started()
        except BaseException:
            try:
                await self.state.store_future_status(
                    request_id,
                    TaskStatus.FAILED.value,
                    model_id,
                    failure=FutureFailureRecord(
                        reason_code='orphaned_replica',
                        attribution='server',
                        message='Replica stopped before the task could be enqueued.'))
            finally:
                await self._compute_worker.finish_task(task, False)
            raise
        queue_key = self._queue_key(model_id=model_id, token=task.token)
        self._compute_worker.ensure_queue_registered(queue_key)
        q = self._compute_worker.get_queue(queue_key)
        self._compute_worker.track_task(request_id)
        await q.put(task)
        try:
            await self.state.store_future_status(
                request_id,
                TaskStatus.QUEUED.value,
                model_id,
                queue_state=QueueState.ACTIVE.value,
            )
            logger.info(f'[TaskQueue] Task {request_id} queued, type={task.task_type or "unknown"}, '
                        f'model_id={model_id}, queue_key={queue_key}, '
                        f'queue_depth={q.qsize()}, input_tokens={task.input_tokens}')
        finally:
            # Once q.put succeeds the sleeping worker must always be woken, even
            # if an acknowledgement or status write fails afterwards.
            self._compute_worker.notify_new_task()

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
        request_id: str | None = None,
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
            request_id=request_id,
        )

    async def schedule_concurrent_task(
        self,
        coro_factory: Callable[[], Coroutine],
        model_id: str | None = None,
        token: str | None = None,
        input_tokens: int = 0,
        task_type: str | None = None,
        request_id: str | None = None,
        execution_scope: ExecutionScope | None = None,
    ) -> dict[str, Any]:
        """Schedule a compute task that runs now instead of in serial order.

        Use for a backend that batches concurrent requests itself, such as the
        vLLM sampler: through the serial queue it would only ever see one
        request at a time. Preflight checks, the persisted future, the execution
        timeout and failure mapping are the same as :meth:`schedule_task`.

        Returns:
            {'request_id': str, 'model_id': str | None}
        """
        request_id = request_id or f'req_{uuid.uuid4().hex}'

        # Decision_Boundary: raises RequestRejectedError before any state write.
        await self._perform_preflight_checks(model_id=model_id, token=token, input_tokens=input_tokens)

        await self._compute_worker.submit_detached(
            QueuedTask(
                request_id=request_id,
                coro_factory=coro_factory,
                model_id=model_id,
                token=token,
                input_tokens=input_tokens,
                task_type=task_type,
                created_at=time.monotonic(),
                execution_scope=execution_scope),
            self._queue_key(model_id=model_id, token=token),
            replica_id=getattr(self, 'replica_id', None),
            absolute_deadline=self._future_deadline())
        logger.info(f'[TaskQueue] Task {request_id} started concurrently, type={task_type or "unknown"}, '
                    f'model_id={model_id}, input_tokens={input_tokens}')
        return {'request_id': request_id, 'model_id': model_id}

    # Poll cadence *inside* the Inline_Fast_Path window. Much smaller than the window
    # itself so a task that finishes early is noticed promptly; the window bound
    # (config.inline_fast_path_timeout) is what actually caps submit latency.
    _INLINE_FAST_PATH_POLL = 0.005

    async def submit_concurrent_and_peek(self, coro_factory: Callable[[], Coroutine],
                                         **schedule_kwargs) -> TaskEnvelope:
        """Concurrent compute submission with the existing Twinkle future envelope."""
        ref = await self.schedule_concurrent_task(coro_factory, **schedule_kwargs)
        return await self._peek_terminal(ref['request_id'], fallback_status='pending')

    async def _peek_terminal(self, request_id: str, *, fallback_status: str) -> TaskEnvelope:
        """Poll a record up to the Inline_Fast_Path window; return a terminal envelope
        if it settled, else a non-terminal envelope with ``fallback_status``."""
        deadline = time.monotonic() + self._task_queue_config.inline_fast_path_timeout
        record = None
        while time.monotonic() < deadline:
            record = await self.state.get_future(request_id)
            if record is not None and record.get('status') in TERMINAL_STATUSES:
                return envelope_from_record(request_id, record)
            await asyncio.sleep(self._INLINE_FAST_PATH_POLL)
        return envelope_from_record(request_id, record, fallback_status=fallback_status)

    async def submit_and_peek(
        self,
        coro_factory: Callable[[], Coroutine],
        *,
        model_id: str | None = None,
        token: str | None = None,
        task_type: str | None = None,
        request_id: str | None = None,
        **schedule_kwargs: Any,
    ) -> TaskEnvelope:
        """Enqueue a task, then briefly wait so a fast op finishes in one round trip.

        Exceeding the window is not an error: the caller polls
        ``/twinkle/retrieve_future`` instead. That is what makes this loop
        fundamentally different from the deleted in-process blocking wait -- it owes
        nothing to failure handling, so it needs no terminal write, no missing-record
        branch, and no race with the worker. A terminal record inside the window is
        returned as a terminal envelope (success or failure); a window that elapses
        still non-terminal returns a non-terminal envelope carrying ``queue_state``.
        """
        ref = await self.schedule_task(
            coro_factory, model_id=model_id, token=token, task_type=task_type, request_id=request_id, **schedule_kwargs)
        return await self._peek_terminal(ref['request_id'], fallback_status='pending')

    async def submit_background_and_peek(
        self,
        coro_factory: Callable[[], Coroutine],
        *,
        model_id: str | None = None,
        task_type: str | None = None,
    ) -> TaskEnvelope:
        """Fire-and-forget variant of :meth:`submit_and_peek` for pure-I/O tasks.

        Uses ``schedule_background_task`` (outside the serial compute queue) but returns
        the same Task_Envelope, so ``upload_to_hub`` shares the retrieve/future machinery
        instead of its own status endpoint. The task is already RUNNING on return, so the
        non-terminal fallback is ``running``.
        """
        ref = await self.schedule_background_task(coro_factory, model_id=model_id, task_type=task_type)
        return await self._peek_terminal(ref['request_id'], fallback_status='running')

    async def schedule_background_task(
        self,
        coro_factory: Callable[[], Coroutine],
        model_id: str | None = None,
        task_type: str | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Run pure I/O outside serial compute, with the same supervised lifecycle.

        Capacity, timeout, cancellation and shutdown are owned by ComputeWorker;
        background jobs must not create a second unsupervised execution path.
        """
        return await self.schedule_concurrent_task(coro_factory, model_id=model_id, task_type=task_type, token=token)

    async def _fail_queue_tasks_async(self, queue_key: str, reason: str) -> None:
        await self._compute_worker.fail_queue_tasks(queue_key, reason)

    def fail_pending_tasks_for_model(self, model_id: str, reason: str) -> None:
        """Fail and drop all queued tasks for a model. Thread-safe."""
        queue_key = self._queue_key(model_id=model_id, token=None)
        if self._event_loop is None:
            logger.warning(f'[TaskQueue] fail_pending_tasks_for_model called without event loop: {queue_key}')
            return

        def _schedule() -> None:
            task = asyncio.create_task(self._fail_queue_tasks_async(queue_key, reason))
            self._queue_tasks.add(task)
            task.add_done_callback(self._queue_task_done)

        self._event_loop.call_soon_threadsafe(_schedule)

    def _queue_task_done(self, task: asyncio.Task) -> None:
        self._queue_tasks.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error('Queue maintenance failed', exc_info=(type(error), error, error.__traceback__))

    async def shutdown_task_queue(self) -> None:
        """Gracefully shut down the compute queue and release resources."""
        await self._rate_limiter.stop_cleanup_task()
        await self._compute_worker.stop()
        await asyncio.gather(*self._queue_tasks, return_exceptions=True)
        self._backend_gate.shutdown()
        logger.debug('[TaskQueue] Task queue shutdown complete')
