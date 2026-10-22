import asyncio
import pytest
import pytest_asyncio

from twinkle.server.task_queue.config import TaskQueueConfig
from twinkle.server.task_queue.mixin import TaskQueueMixin
from twinkle.server.task_queue.types import UserTaskError
from twinkle.server.task_queue.worker import ComputeWorker


class _DummyState:

    def __init__(self):
        self.records = []
        self._latest = {}

    async def store_future_status(self, *args, **kwargs):
        self.records.append((args, kwargs))
        request_id, status = args[0], args[1]
        self._latest[request_id] = {
            'status': status,
            'result': kwargs.get('result'),
            'failure': (kwargs['failure'].model_dump() if kwargs.get('failure') is not None else None),
            'queue_state': kwargs.get('queue_state'),
            'queue_state_reason': kwargs.get('queue_state_reason'),
        }

    async def get_future(self, request_id):
        return self._latest.get(request_id)


class _AllowingRateLimiter:

    async def check_and_record(self, token, input_tokens):
        return True, None


class _DummyQueue(TaskQueueMixin):

    def __init__(self):
        self.state = _DummyState()
        # A generous Inline_Fast_Path window keeps "task settles inside submit"
        # deterministic for the trivial in-process coroutines used here.
        self._task_queue_config = TaskQueueConfig(inline_fast_path_timeout=5.0)
        self._rate_limiter = _AllowingRateLimiter()
        self._task_metrics = None
        self._deployment_name = 'test'

    def enable_compute_worker(self):
        self._compute_worker = ComputeWorker(
            state=self.state,
            config=self._task_queue_config,
            task_metrics=None,
            deployment_name=self._deployment_name,
        )
        self._event_loop = None


@pytest.mark.asyncio
async def test_worker_stop_exits_when_wait_for_swallows_cancellation(monkeypatch):
    queue = _DummyQueue()
    queue.enable_compute_worker()
    worker = queue._compute_worker
    completed = asyncio.Event()
    stop_task = None

    async def work():
        return {'ok': True}

    async def wait_for_with_completion_race(coro, timeout):
        # Python 3.10/3.11 wait_for can return a completed child's result
        # instead of propagating a simultaneous cancellation of its caller.
        result = await coro
        completed.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return result

    monkeypatch.setattr(asyncio, 'wait_for', wait_for_with_completion_race)
    ready_task = asyncio.create_task(completed.wait())
    try:
        ref = await queue.schedule_task(work, model_id='model1', token='token1')
        done, _ = await asyncio.wait({ready_task}, timeout=1)
        assert ready_task in done, 'worker did not execute the task'

        stop_task = asyncio.create_task(worker.stop())
        done, _ = await asyncio.wait({stop_task}, timeout=1)
        assert stop_task in done, 'worker waited for new work after cancellation was swallowed'
        await stop_task
        assert (await queue.state.get_future(ref['request_id']))['result'] == {'ok': True}
    finally:
        ready_task.cancel()
        if worker._worker_task is not None:
            worker._worker_task.cancel()
        if stop_task is not None:
            await stop_task
        else:
            await worker.stop()


@pytest.mark.asyncio
async def test_worker_can_restart_after_stopping_while_idle():
    queue = _DummyQueue()
    queue.enable_compute_worker()
    worker = queue._compute_worker

    async def work():
        return {'ok': True}

    try:
        await worker.ensure_started()
        await asyncio.sleep(0)  # Let the idle worker wait for new work.
        await asyncio.wait_for(worker.stop(), timeout=1)
        env = await queue.submit_and_peek(work, model_id='model1', token='token1')
        assert env.status == 'completed'
        assert env.result == {'ok': True}
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_stop_cancels_running_task():
    queue = _DummyQueue()
    queue.enable_compute_worker()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def work():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    try:
        await queue.schedule_task(work, model_id='model1', token='token1')
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(queue._compute_worker.stop(), timeout=1)
        assert cancelled.is_set()
    finally:
        await queue._compute_worker.stop()


@pytest.mark.asyncio
async def test_preflight_rejects_batch_without_per_dp_multiple():
    queue = _DummyQueue()
    from twinkle.server.exceptions import BatchSizeError

    with pytest.raises(BatchSizeError, match='must be divisible by 4'):
        await queue._perform_preflight_checks(
            model_id='model1',
            token='token1',
            input_tokens=0,
            batch_size=2,
            data_world_size=2,
            batch_size_multiple=2,
        )

    # A rejection writes no future record.
    assert queue.state.records == []


@pytest.mark.asyncio
async def test_preflight_accepts_batch_with_per_dp_multiple():
    queue = _DummyQueue()

    result = await queue._perform_preflight_checks(
        model_id='model1',
        token='token1',
        input_tokens=0,
        batch_size=4,
        data_world_size=2,
        batch_size_multiple=2,
    )

    assert result is None
    assert queue.state.records == []


@pytest.mark.asyncio
async def test_background_task_tracks_status():
    queue = _DummyQueue()

    async def work():
        return {'ok': True}

    await queue.schedule_background_task(
        work,
        model_id='model1',
    )
    await asyncio.sleep(0)

    assert [args[1] for args, _ in queue.state.records] == ['running', 'completed']
    assert queue.state.records[0][1]['absolute_deadline'] > 0
    assert queue.state.records[-1][1]['result'] == {'ok': True}


@pytest.mark.asyncio
async def test_submit_and_peek_returns_completed_envelope_and_persists():
    queue = _DummyQueue()
    queue.replica_id = 'replica-1'
    queue.enable_compute_worker()
    result = {'logps': [[float(index) for index in range(128)]]}

    async def work():
        return result

    try:
        env = await queue.submit_and_peek(
            work,
            model_id='model1',
            token='token1',
            task_type='forward_backward',
        )
    finally:
        await queue._compute_worker.stop()

    assert env.status == 'completed'
    assert env.result == result
    # The future record is now the single delivery channel: the result IS persisted.
    assert any(args[1] == 'completed' for args, _ in queue.state.records)


@pytest.mark.asyncio
async def test_polling_schedule_task_still_persists_its_result():
    queue = _DummyQueue()
    queue.replica_id = 'replica-1'
    queue.enable_compute_worker()
    result = {'value': 42}

    async def work():
        return result

    try:
        await queue.schedule_task(work, model_id='model1', token='token1')
        for _ in range(100):
            completed = [kwargs for args, kwargs in queue.state.records if args[1] == 'completed']
            if completed:
                break
            await asyncio.sleep(0)
    finally:
        await queue._compute_worker.stop()

    pending = next(kwargs for args, kwargs in queue.state.records if args[1] == 'pending')
    assert pending['replica_id'] == 'replica-1'
    assert pending['absolute_deadline'] > 0
    assert completed[-1]['result'] is result


@pytest.mark.asyncio
async def test_submit_and_peek_failure_returns_failed_envelope_and_persists():
    queue = _DummyQueue()
    queue.replica_id = 'replica-1'
    queue.enable_compute_worker()

    async def work():
        raise ValueError('model failed')

    try:
        env = await queue.submit_and_peek(
            work,
            model_id='model1',
            token='token1',
            task_type='forward_backward',
        )
    finally:
        await queue._compute_worker.stop()

    # The failure payload rides the envelope's `error` field.
    assert env.status == 'failed'
    assert env.error is not None and 'model failed' in env.error.error
    assert any(args[1] == 'failed' for args, _ in queue.state.records)


@pytest.mark.asyncio
async def test_user_task_error_is_stored_as_user_failure():
    queue = _DummyQueue()
    queue.enable_compute_worker()

    async def work():
        raise UserTaskError('invalid request')

    try:
        await queue.schedule_task(work, model_id='model1', token='token1')
        for _ in range(100):
            failed = [kwargs for args, kwargs in queue.state.records if args[1] == 'failed']
            if failed:
                break
            await asyncio.sleep(0)
    finally:
        await queue._compute_worker.stop()

    failure = failed[-1]['failure']
    assert failure.reason_code == 'request_rejected'
    assert failure.attribution == 'user'
    assert failure.diagnostic is None


@pytest.mark.asyncio
async def test_typed_server_error_keeps_its_domain_reason_and_attribution():
    """A typed user failure must retain its domain meaning in persisted state."""
    from twinkle.server.exceptions import ResourceNotFoundError

    queue = _DummyQueue()
    queue.enable_compute_worker()

    async def work():
        raise ResourceNotFoundError('adapter foo not found')

    try:
        await queue.schedule_task(work, model_id='model1', token='token1')
        for _ in range(100):
            failed = [kwargs for args, kwargs in queue.state.records if args[1] == 'failed']
            if failed:
                break
            await asyncio.sleep(0)
    finally:
        await queue._compute_worker.stop()

    failure = failed[-1]['failure']
    assert failure.reason_code == 'resource_not_found'
    assert failure.attribution == 'user'
    assert failure.diagnostic is None


@pytest.mark.asyncio
async def test_submit_and_peek_preflight_rejection_raises_without_writing_or_queuing():
    from twinkle.server.exceptions import BatchSizeError
    queue = _DummyQueue()
    queue.enable_compute_worker()

    async def work():
        raise AssertionError('preflight rejection must not execute the task')

    with pytest.raises(BatchSizeError, match='must be divisible by 4'):
        await queue.submit_and_peek(
            work,
            model_id='model1',
            token='token1',
            batch_size=2,
            data_world_size=2,
            batch_size_multiple=2,
        )

    assert queue.state.records == []
    assert queue._compute_worker._worker_task is None


@pytest.mark.asyncio
async def test_submit_and_peek_honors_explicit_request_id():
    # run_submit generates the request_id up front (to claim the seq dedup key
    # atomically) and threads it through submit_and_peek; the future record and the
    # returned envelope must use that exact id, not a freshly generated one.
    queue = _DummyQueue()
    queue.enable_compute_worker()

    async def _factory():
        return {'ok': True}

    env = await queue.submit_and_peek(_factory, task_type='step', request_id='req_fixed123')
    assert env.request_id == 'req_fixed123'
    assert env.status == 'completed'


@pytest_asyncio.fixture
async def make_concurrent_queue():
    """Keep concurrent worker setup and teardown in one place."""
    queues = []

    def create(config=None):
        queue = _DummyQueue()
        if config is not None:
            queue._task_queue_config = config
        queue.enable_compute_worker()
        queues.append(queue)
        return queue

    yield create
    for queue in reversed(queues):
        await queue._compute_worker.stop()


@pytest.mark.asyncio
async def test_concurrent_tasks_run_side_by_side(make_concurrent_queue):
    queue = make_concurrent_queue()
    started = []
    both_started = asyncio.Event()

    async def work(index):
        started.append(index)
        if len(started) == 2:
            both_started.set()
        # Under serial execution the first task would wait here forever.
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return {'index': index}

    refs = [
        await queue.schedule_concurrent_task(lambda i=i: work(i), token='token1', task_type='sample')
        for i in range(2)
    ]
    for _ in range(100):
        records = [await queue.state.get_future(ref['request_id']) for ref in refs]
        if all(record['status'] == 'completed' for record in records):
            break
        await asyncio.sleep(0.01)

    assert [record['result'] for record in records] == [{'index': 0}, {'index': 1}]
    # The serial worker loop is never involved.
    assert queue._compute_worker._worker_task is None


@pytest.mark.asyncio
async def test_concurrent_task_keeps_queue_failure_mapping(make_concurrent_queue):
    queue = make_concurrent_queue()

    async def work():
        raise UserTaskError('invalid request')

    await queue.schedule_concurrent_task(work, token='token1')
    for _ in range(100):
        failed = [kwargs for args, kwargs in queue.state.records if args[1] == 'failed']
        if failed:
            break
        await asyncio.sleep(0)

    assert [args[1] for args, _ in queue.state.records] == ['pending', 'running', 'failed']
    failure = failed[-1]['failure']
    assert failure.reason_code == 'request_rejected'
    assert failure.attribution == 'user'


@pytest.mark.asyncio
async def test_concurrent_task_preflight_rejection_raises_without_writing(make_concurrent_queue):
    from twinkle.server.exceptions import InputTokensExceededError
    queue = make_concurrent_queue()

    async def work():
        raise AssertionError('preflight rejection must not execute the task')

    with pytest.raises(InputTokensExceededError):
        await queue.schedule_concurrent_task(
            work, token='token1', input_tokens=queue.task_queue_config.max_input_tokens + 1)

    assert queue.state.records == []
    assert not queue._compute_worker._detached_tasks


@pytest.mark.asyncio
async def test_worker_stop_cancels_concurrent_task(make_concurrent_queue):
    queue = make_concurrent_queue()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def work():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    await queue.schedule_concurrent_task(work, token='token1')
    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.wait_for(queue._compute_worker.stop(), timeout=1)
    assert cancelled.is_set()
    assert not queue._compute_worker._detached_tasks


@pytest.mark.asyncio
async def test_concurrent_capacity_rejects_before_future_write_and_is_reusable(make_concurrent_queue):
    from twinkle.server.exceptions import ResourceExhaustedError
    queue = make_concurrent_queue(TaskQueueConfig(concurrent_task_limit=1))
    release = asyncio.Event()

    async def work():
        await release.wait()
        return {'ok': True}

    try:
        first = await queue.schedule_concurrent_task(work, token='token1')
        with pytest.raises(ResourceExhaustedError):
            await queue.schedule_concurrent_task(work, token='token2')
        assert set(queue.state._latest) == {first['request_id']}
        release.set()
        for _ in range(100):
            if queue._compute_worker._detached_reserved == 0:
                break
            await asyncio.sleep(0.001)
        envelope = await queue.submit_concurrent_and_peek(work, token='token2')
        assert envelope.status == 'completed'
    finally:
        release.set()


@pytest.mark.asyncio
async def test_concurrent_reservation_is_released_when_state_write_fails(make_concurrent_queue):
    queue = make_concurrent_queue()

    async def fail(*args, **kwargs):
        raise RuntimeError('state unavailable')

    queue.state.store_future_status = fail
    with pytest.raises(RuntimeError, match='state unavailable'):
        await queue.schedule_concurrent_task(lambda: asyncio.sleep(0), token='token1')
    assert queue._compute_worker._detached_reserved == 0
    assert not queue._compute_worker._detached_tasks


@pytest.mark.asyncio
async def test_stop_between_future_write_and_start_leaves_terminal_record(make_concurrent_queue):
    from twinkle.server.exceptions import ResourceExhaustedError
    queue = make_concurrent_queue()
    original = queue.state.store_future_status

    async def write_then_stop(*args, **kwargs):
        await original(*args, **kwargs)
        if args[1] == 'pending':
            await queue._compute_worker.stop()

    queue.state.store_future_status = write_then_stop
    with pytest.raises(ResourceExhaustedError):
        await queue.schedule_concurrent_task(lambda: asyncio.sleep(0), token='token1', request_id='req-stop')
    assert queue.state._latest['req-stop']['status'] == 'failed'
    assert queue._compute_worker._detached_reserved == 0


@pytest.mark.asyncio
async def test_shutdown_cancel_before_task_start_leaves_failed_future(make_concurrent_queue):
    queue = make_concurrent_queue()
    executed = False
    async def work():
        nonlocal executed
        executed = True
    ref = await queue.schedule_concurrent_task(work, token='token1')
    await queue._compute_worker.stop()  # No yield between admission and shutdown.
    record = queue.state._latest[ref['request_id']]
    assert record['status'] == 'failed'
    assert record['failure']['reason_code'] == 'orphaned_replica'
    assert not executed and queue._compute_worker._detached_reserved == 0


@pytest.mark.asyncio
async def test_cancel_preserves_existing_terminal_record(make_concurrent_queue):
    queue = make_concurrent_queue()
    entered = asyncio.Event()
    async def work():
        entered.set()
        await asyncio.Event().wait()
    ref = await queue.schedule_concurrent_task(work, token='token1')
    await entered.wait()
    await queue.state.store_future_status(ref['request_id'], 'failed', None, reason='already terminal')
    record = queue.state._latest[ref['request_id']].copy()
    await queue._compute_worker.stop()
    assert queue.state._latest[ref['request_id']] == record


@pytest.mark.asyncio
async def test_cancel_during_running_status_write_leaves_failed_future(make_concurrent_queue):
    queue = make_concurrent_queue()
    entered = asyncio.Event()
    original = queue.state.store_future_status
    async def write(*args, **kwargs):
        if args[1] == 'running':
            entered.set()
            await asyncio.Event().wait()
        await original(*args, **kwargs)
    queue.state.store_future_status = write
    ref = await queue.schedule_concurrent_task(lambda: asyncio.sleep(0), token='token1')
    await entered.wait()
    await queue._compute_worker.stop()
    record = queue.state._latest[ref['request_id']]
    assert record['status'] == 'failed'
    assert record['failure']['reason_code'] == 'orphaned_replica'
    assert [args[1] for args, _ in queue.state.records] == ['pending', 'failed']


@pytest.mark.asyncio
async def test_cancel_winning_running_write_race_skips_execution(make_concurrent_queue):
    queue = make_concurrent_queue()
    executed = False
    original = queue.state.store_future_status

    async def cancel_before_running(*args, **kwargs):
        if args[1] == 'running':
            # Mirror FutureManager: cancel_if_pending wins, then the terminal record refuses 'running'.
            queue.state._latest[args[0]] = {'status': 'cancelled', 'result': None, 'failure': None}
            return
        await original(*args, **kwargs)

    async def work():
        nonlocal executed
        executed = True
        return {'ok': True}

    queue.state.store_future_status = cancel_before_running
    ref = await queue.schedule_concurrent_task(work, token='token1')
    for _ in range(100):
        if queue._compute_worker._detached_reserved == 0:
            break
        await asyncio.sleep(0.001)

    assert not executed
    assert queue.state._latest[ref['request_id']]['status'] == 'cancelled'
    assert queue._compute_worker._detached_reserved == 0
    assert not queue._compute_worker._detached_tasks
