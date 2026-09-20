# Copyright (c) ModelScope Contributors. All rights reserved.
"""TaskQueueMixin: serial compute queue plus admitted concurrent tasks.

``schedule_task`` serializes stateful Model operations. Detached tasks are
used for I/O and for engines such as vLLM that own their compute concurrency
and continuous batching internally.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import time
import traceback
import uuid
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from twinkle.server.exceptions import BatchSizeError, ConfigError, InputTokensExceededError, RateLimitExceededError
from twinkle.server.lifecycle.envelope import envelope_from_record
from twinkle.server.lifecycle.poll_config import long_poll_window
from twinkle.server.state.models import FutureFailureRecord
from twinkle.server.telemetry.middleware import get_task_metrics
from twinkle.utils.logger import get_logger
from twinkle_client.types.lifecycle import TERMINAL_STATUSES, TaskEnvelope
from .config import TaskQueueConfig
from .rate_limiter import RateLimiter
from .types import BackendBusyError, QueuedTask, QueueState, TaskStatus
from .worker import ComputeWorker

if TYPE_CHECKING:
    from twinkle.server.state import ServerState

logger = get_logger()


class TaskQueueMixin:
    """Mixin providing two task execution paths.

    Execution paths
    ---------------
    1. Compute queue (schedule_task / submit_and_peek):
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
        )
        self._rate_limiter.start_cleanup_task()

        self._compute_worker = ComputeWorker(
            state=self.state,
            config=self._task_queue_config,
            task_metrics=self._task_metrics,
            deployment_name=deployment_name,
            on_backend_timeout=on_backend_timeout,
        )

        self._backend_executor = ThreadPoolExecutor(thread_name_prefix='twinkle-backend')
        self._backend_probe_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='twinkle-backend-probe')
        self._backend_admission: asyncio.Lock | None = asyncio.Lock() if enable_admission_gate else None
        self._backend_poisoned = asyncio.Event()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    async def _acquire_backend_gate(self, gate: asyncio.Lock) -> None:
        if self._backend_poisoned.is_set():
            raise BackendBusyError('This replica is waiting for a timed-out backend call to exit.')
        if not gate.locked():
            await gate.acquire()
        else:
            acquire_task = asyncio.create_task(gate.acquire())
            poison_task = asyncio.create_task(self._backend_poisoned.wait())
            try:
                done, _ = await asyncio.wait((acquire_task, poison_task), return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                acquire_task.cancel()
                poison_task.cancel()
                await asyncio.gather(acquire_task, poison_task, return_exceptions=True)
                if acquire_task.done() and not acquire_task.cancelled() and acquire_task.result():
                    gate.release()
                raise
            if poison_task in done and self._backend_poisoned.is_set():
                if acquire_task.done() and not acquire_task.cancelled() and acquire_task.result():
                    gate.release()
                else:
                    acquire_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await acquire_task
                raise BackendBusyError('This replica is waiting for a timed-out backend call to exit.')
            poison_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poison_task
            await acquire_task
        if self._backend_poisoned.is_set():
            gate.release()
            raise BackendBusyError('This replica is waiting for a timed-out backend call to exit.')

    async def call_backend(self, fn: Callable[..., Any], /, *args: Any, admit: bool = True, **kwargs: Any) -> Any:
        """Run one backend call outside the event loop.

        Normal model calls serialize through the admission gate. If the awaiting
        task times out while its thread is still running, the gate is poisoned:
        waiters fail immediately until that thread exits. Health probes bypass the
        gate and use a reserved executor thread. Sampler deployments disable the
        gate because their backend owns request concurrency.
        """
        loop = asyncio.get_running_loop()
        gate = self._backend_admission if admit else None
        if gate is not None:
            await self._acquire_backend_gate(gate)

        executor = self._backend_executor if admit else self._backend_probe_executor
        try:
            concurrent_future = executor.submit(functools.partial(fn, *args, **kwargs))
        except Exception:
            if gate is not None and gate.locked():
                gate.release()
            raise

        if gate is not None:

            def release_gate(_future) -> None:

                def release() -> None:
                    self._backend_poisoned.clear()
                    if gate.locked():
                        gate.release()

                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(release)

            concurrent_future.add_done_callback(release_gate)

        try:
            return await asyncio.wrap_future(concurrent_future, loop=loop)
        except asyncio.CancelledError:
            if gate is not None and concurrent_future.running():
                self._backend_poisoned.set()
            raise

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
        future writes for a rejected request (Property 3). It writes no FAILED
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

        if self._event_loop is None:
            self._event_loop = asyncio.get_running_loop()

        await self.state.store_future_status(
            request_id,
            TaskStatus.PENDING.value,
            model_id,
            queue_state=QueueState.ACTIVE.value,
            replica_id=getattr(self, 'replica_id', None),
            absolute_deadline=self._future_deadline(),
        )

        queue_key = self._queue_key(model_id=model_id, token=token)
        self._compute_worker.ensure_queue_registered(queue_key)
        await self._compute_worker.ensure_started()

        q = self._compute_worker.get_queue(queue_key)
        await q.put(
            QueuedTask(
                request_id=request_id,
                coro_factory=coro_factory,
                model_id=model_id,
                token=token,
                input_tokens=input_tokens,
                task_type=task_type,
                created_at=time.monotonic(),
            ))
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
            total_depth = self._compute_worker.total_queued()
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

    # Poll cadence *inside* the Inline_Fast_Path window. Much smaller than the window
    # itself so a task that finishes early is noticed promptly; the window bound
    # (config.inline_fast_path_timeout) is what actually caps submit latency.
    _INLINE_FAST_PATH_POLL = 0.005

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
            replica_id=getattr(self, 'replica_id', None),
            absolute_deadline=self._future_deadline(),
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
            except Exception as exc:
                failure = FutureFailureRecord(
                    reason_code='internal_error',
                    message=f'{type(exc).__name__}: {exc}'[:1024],
                    attribution='server',
                    diagnostic=traceback.format_exc(),
                )
                await self.state.store_future_status(
                    request_id,
                    TaskStatus.FAILED.value,
                    model_id,
                    failure=failure,
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

    async def shutdown_task_queue(self) -> None:
        """Gracefully shut down the compute queue and release resources."""
        await self._rate_limiter.stop_cleanup_task()
        await self._compute_worker.stop()
        # Do not wait on threads that may be leaked on a timed-out backend call.
        if getattr(self, '_backend_executor', None) is not None:
            self._backend_executor.shutdown(wait=False, cancel_futures=True)
        if getattr(self, '_backend_probe_executor', None) is not None:
            self._backend_probe_executor.shutdown(wait=False, cancel_futures=True)
        logger.debug('[TaskQueue] Task queue shutdown complete')
