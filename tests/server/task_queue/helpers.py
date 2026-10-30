from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.task_queue.mixin import TaskQueueMixin
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

    async def try_start_future(self, request_id):
        record = self._latest.get(request_id)
        if record is None or record['status'] not in ('pending', 'queued'):
            return False
        await self.store_future_status(request_id, 'running', None, queue_state='active')
        return self._latest[request_id]['status'] == 'running'

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
        self._queue_tasks = set()
