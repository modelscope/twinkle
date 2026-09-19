import asyncio
import pytest

from twinkle.server.utils.task_queue.config import TaskQueueConfig
from twinkle.server.utils.task_queue.mixin import TaskQueueMixin
from twinkle.server.utils.task_queue.types import UserTaskError
from twinkle.server.utils.task_queue.worker import ComputeWorker


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

    # Property 3: a rejection writes no future record.
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

    # Property 0: the failure payload rides the envelope's `error` field.
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

    assert failed[-1]['result']['category'] == 'user'
    assert 'traceback' not in failed[-1]['result']


@pytest.mark.asyncio
async def test_typed_server_error_keeps_its_status_and_category():
    """A TwinkleServerError (e.g. ResourceNotFoundError) must keep its own 404/user
    classification instead of collapsing to a generic 500/server."""
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

    assert failed[-1]['result']['error_code'] == 404
    assert failed[-1]['result']['category'] == 'user'
    # A user rejection carries no traceback.
    assert 'traceback' not in failed[-1]['result']


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
