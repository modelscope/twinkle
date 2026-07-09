import pytest

from twinkle.server.utils.task_queue.config import TaskQueueConfig
from twinkle.server.utils.task_queue.mixin import TaskQueueMixin


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
