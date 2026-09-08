import asyncio

import pytest

from twinkle.server.utils.task_queue.config import TaskQueueConfig
from twinkle.server.utils.task_queue.mixin import TaskQueueMixin
from twinkle.server.utils.task_queue.worker import ComputeWorker


class _DummyState:

    def __init__(self):
        self.records = []

    async def store_future_status(self, *args, **kwargs):
        self.records.append((args, kwargs))


class _AllowingRateLimiter:

    async def check_and_record(self, token, input_tokens):
        return True, None


class _DummyQueue(TaskQueueMixin):

    def __init__(self):
        self.state = _DummyState()
        self._task_queue_config = TaskQueueConfig()
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

    result = await queue._perform_preflight_checks(
        request_id='req1',
        model_id='model1',
        token='token1',
        input_tokens=0,
        batch_size=2,
        data_world_size=2,
        batch_size_multiple=2,
    )

    assert result == {'request_id': 'req1', 'model_id': 'model1'}
    _, kwargs = queue.state.records[-1]
    assert kwargs['result']['category'] == 'User'
    assert 'Batch size 2 must be divisible by 4' in kwargs['result']['error']


@pytest.mark.asyncio
async def test_preflight_accepts_batch_with_per_dp_multiple():
    queue = _DummyQueue()

    result = await queue._perform_preflight_checks(
        request_id='req1',
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
    assert queue.state.records[-1][1]['result'] == {'ok': True}


@pytest.mark.asyncio
async def test_schedule_task_and_wait_returns_large_result_without_persisting_it():
    queue = _DummyQueue()
    queue.enable_compute_worker()
    result = {'logps': [[float(index) for index in range(128)]]}

    async def work():
        return result

    try:
        actual = await queue.schedule_task_and_wait(
            work,
            model_id='model1',
            token='token1',
            task_type='forward_backward',
        )
    finally:
        await queue._compute_worker.stop()

    assert actual is result
    assert queue.state.records == []


@pytest.mark.asyncio
async def test_polling_schedule_task_still_persists_its_result():
    queue = _DummyQueue()
    queue.enable_compute_worker()
    result = {'value': 42}

    async def work():
        return result

    try:
        await queue.schedule_task(work, model_id='model1', token='token1')
        for _ in range(100):
            completed = [
                kwargs
                for args, kwargs in queue.state.records
                if args[1] == 'completed'
            ]
            if completed:
                break
            await asyncio.sleep(0)
    finally:
        await queue._compute_worker.stop()

    assert completed[-1]['result'] is result


@pytest.mark.asyncio
async def test_schedule_task_and_wait_propagates_failure_without_persisting_it():
    queue = _DummyQueue()
    queue.enable_compute_worker()

    async def work():
        raise ValueError('model failed')

    try:
        with pytest.raises(RuntimeError, match='ValueError: model failed'):
            await queue.schedule_task_and_wait(
                work,
                model_id='model1',
                token='token1',
                task_type='forward_backward',
            )
    finally:
        await queue._compute_worker.stop()

    assert queue.state.records == []


@pytest.mark.asyncio
async def test_schedule_task_and_wait_reports_preflight_failure_without_persisting_it():
    queue = _DummyQueue()
    queue.enable_compute_worker()

    async def work():
        raise AssertionError('preflight rejection must not execute the task')

    with pytest.raises(RuntimeError, match='Batch size 2 must be divisible by 4'):
        await queue.schedule_task_and_wait(
            work,
            model_id='model1',
            token='token1',
            batch_size=2,
            data_world_size=2,
            batch_size_multiple=2,
        )

    assert queue.state.records == []
    assert queue._compute_worker._worker_task is None
