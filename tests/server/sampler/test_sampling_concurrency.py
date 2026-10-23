"""Sampler route concurrency, weight ownership and request failure recovery."""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI
from ray.exceptions import GetTimeoutError
from starlette.requests import Request
from tinker import types as tinker_types

from twinkle.data_format import SampledSequence, SampleResponse
from twinkle.exceptions import FullWeightLoadError
from twinkle.protocol.types import SampleRequest
from twinkle.server.sampler.app import SamplerManagement
from twinkle.server.sampler.concurrency import SamplerConcurrency
from twinkle.server.sampler.tinker_handlers import _register_tinker_sampler_routes
from twinkle.server.sampler.twinkle_handlers import _register_twinkle_sampler_routes
from twinkle.server.task_queue.backend_gate import BackendGate
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.exceptions import BackendBusyError
from twinkle.server.task_queue.worker import ComputeWorker


async def _until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), 3)


class _State:
    def __init__(self):
        self.records = {}

    async def store_future_status(self, request_id, status, model_id=None, **kwargs):
        self.records[request_id] = {'status': status, **kwargs}

    async def try_start_future(self, request_id):
        record = self.records.get(request_id)
        if record is None or record['status'] not in ('pending', 'queued'):
            return False
        record['status'] = 'running'
        return True

    async def get_future(self, request_id):
        return self.records.get(request_id)


class _Limiter:
    async def check_and_record(self, *args):
        return True, None


class _Sampler:
    def __init__(self):
        self.weight = 0
        self.loads = []
        self.active = 0
        self.peak = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self._lock = threading.Lock()
        self.template_calls = 0

    def set_template(self, *args, **kwargs):
        self.template_calls += 1

    def load_full_weights_from_path(self, path):
        self.weight = int(path.rsplit('/', 1)[-1])
        self.loads.append(self.weight)

    def sample(self, inputs, sampling_params=None, **kwargs):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.entered.set()
        try:
            if not self.release.wait(3):
                raise TimeoutError('test did not release generation')
            token = self.weight or inputs[0]['input_ids'][0]
            return [SampleResponse(
                prompt_token_ids=list(item['input_ids']),
                sequences=[SampledSequence(stop_reason='length', tokens=[token], logprobs=[[(token, -0.1)]])],
            ) for item in inputs]
        finally:
            with self._lock:
                self.active -= 1


class _Service(SamplerManagement):
    def __init__(self, *, timeout=2, limit=128):
        self.model_id = 'mock-model'
        self.replica_id = 'sampler-test'
        self.state = _State()
        self.sampler = _Sampler()
        self.sampling_guard = SamplerConcurrency()
        self._task_queue_config = TaskQueueConfig(execution_timeout=timeout, concurrent_task_limit=limit)
        self._rate_limiter = _Limiter()
        self._task_metrics = None
        self._deployment_name = None
        self._event_loop = None
        self._backend_gate = BackendGate()
        self._compute_worker = ComputeWorker(
            state=self.state, config=self._task_queue_config, task_metrics=None, deployment_name=None)
        app = FastAPI()
        _register_tinker_sampler_routes(app, lambda: self)
        _register_twinkle_sampler_routes(app, lambda: self)
        self.endpoints = {route.path: route.endpoint for route in app.routes}

    async def _on_request_start(self, request):
        return 'token'

    async def submit(self, dialect, token=7, path=None):
        request = Request({'type': 'http', 'headers': []})
        if dialect == 'tinker':
            body = tinker_types.SampleRequest(
                prompt=tinker_types.ModelInput.from_ints([token]), model_path=path,
                sampling_params=tinker_types.SamplingParams(max_tokens=1, temperature=0))
            ref = await self.endpoints['/tinker/asample'](request, body, self)
            return ref['request_id']
        body = SampleRequest(inputs=[{'input_ids': [token]}], adapter_uri=path,
                             sampling_params={'max_tokens': 1, 'temperature': 0})
        envelope = await self.endpoints['/twinkle/sample'](request, body, self)
        return envelope.request_id

    async def result(self, request_id):
        await _until(lambda: self.state.records[request_id]['status'] in ('completed', 'failed'))
        return self.state.records[request_id]

    async def close(self):
        self.sampler.release.set()
        self.sampling_guard.close()
        await self._compute_worker.stop()
        await _until(lambda: not self.sampling_guard._jobs)
        self._backend_gate.shutdown()


def _tokens(record, dialect):
    assert record['status'] == 'completed', record
    result = record['result']
    if dialect == 'tinker':
        return result.sequences[0].tokens
    return result['samples'][0]['sequences'][0]['tokens']


@pytest_asyncio.fixture
async def make_service():
    """Create services and close them even when a test assertion fails."""
    services = []

    def create(**kwargs):
        service = _Service(**kwargs)
        services.append(service)
        return service

    yield create
    for service in reversed(services):
        await service.close()


@pytest.mark.asyncio
async def test_standalone_checkpoint_load_waits_for_ordinary_sampling(make_service, tmp_path):
    from twinkle.server.sampler.weights import resolve_sampler_weights
    checkpoint = tmp_path / '2'
    checkpoint.mkdir()
    (checkpoint / 'model.safetensors').touch()
    service = make_service()
    service.sampler.release.clear()
    ref = await service.submit('twinkle')
    await _until(service.sampler.entered.is_set)
    loading = asyncio.create_task(resolve_sampler_weights(service, str(checkpoint)))
    try:
        await asyncio.sleep(.02)
        assert service.sampler.loads == []
        service.sampler.release.set()
        assert _tokens(await service.result(ref), 'twinkle') == [7]
        await loading
        assert service.sampler.loads == [2]
    finally:
        service.sampler.release.set()
        await asyncio.gather(loading, return_exceptions=True)

@pytest.mark.asyncio
@pytest.mark.parametrize('dialects', [('tinker', 'tinker'), ('twinkle', 'twinkle'), ('tinker', 'twinkle')])
@pytest.mark.parametrize('count', [1, 8, 32])
async def test_real_routes_overlap_and_keep_request_results(make_service, dialects, count):
    service = make_service()
    service.sampler.release.clear()
    jobs = [dialects[i % 2] for i in range(count)]
    refs = await asyncio.gather(*(service.submit(dialect, token=i + 1) for i, dialect in enumerate(jobs)))
    await _until(lambda: service.sampler.peak >= min(count, 2))
    service.sampler.release.set()
    records = await asyncio.gather(*(service.result(ref) for ref in refs))
    assert [_tokens(record, dialect) for record, dialect in zip(records, jobs)] == [[i + 1] for i in range(count)]
    assert service._compute_worker._worker_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize('first,second', [('tinker', 'tinker'), ('twinkle', 'twinkle'),
                                       ('tinker', 'twinkle'), ('twinkle', 'tinker')])
async def test_full_checkpoint_generation_is_one_exclusive_transaction(make_service, tmp_path, first, second):
    for number in ['1', '2']:
        (tmp_path / number).mkdir()
        (tmp_path / number / 'model.safetensors').touch()
    service = make_service()
    service.sampler.release.clear()
    a = await service.submit(first, path=str(tmp_path / '1'))
    await _until(service.sampler.entered.is_set)
    b = await service.submit(second, path=str(tmp_path / '2'))
    await _until(lambda: service.sampling_guard._waiting_writers == 1)
    assert service.sampler.loads == [1]
    service.sampler.release.set()
    assert _tokens(await service.result(a), first) == [1]
    assert _tokens(await service.result(b), second) == [2]
    assert service.sampler.loads == [1, 2]


@pytest.mark.asyncio
async def test_mutation_waits_for_both_dialects_generation(make_service):
    service = make_service()
    service.sampler.release.clear()
    mutation = None
    try:
        a, b = await asyncio.gather(service.submit('tinker'), service.submit('twinkle'))
        await _until(lambda: service.sampler.peak == 2)
        mutation = asyncio.create_task(service.sampling_guard.run(lambda: service.call_backend(service.sampler.set_template, 'template'), exclusive=True))
        await _until(lambda: service.sampling_guard._waiting_writers == 1)
        assert service.sampler.template_calls == 0
        service.sampler.release.set()
        await asyncio.gather(service.result(a), service.result(b), mutation)
        assert service.sampler.template_calls == 1
    finally:
        service.sampler.release.set()
        if mutation is not None:
            await asyncio.gather(mutation, return_exceptions=True)


@pytest.mark.asyncio
async def test_timeout_keeps_actual_load_exclusive_and_recovers_after_exit(make_service, tmp_path):
    for number in ['1', '2']:
        (tmp_path / number).mkdir()
        (tmp_path / number / 'model.safetensors').touch()
    service = make_service(timeout=0.05)
    entered = threading.Event()
    release = threading.Event()
    original = service.sampler.load_full_weights_from_path
    def slow_load(path):
        entered.set()
        assert release.wait(3)
        original(path)
    service.sampler.load_full_weights_from_path = slow_load
    try:
        ref = await service.submit('tinker', path=str(tmp_path / '1'))
        await _until(entered.is_set)
        failed = await service.result(ref)
        assert failed['failure'].reason_code == 'execution_timeout'
        assert service.sampling_guard._writer
        submitting = asyncio.create_task(service.submit('twinkle', path=str(tmp_path / '2')))
        await _until(lambda: service.sampling_guard._waiting_writers == 1)
        assert service.sampler.loads == []
        release.set()
        next_ref = await submitting
        assert _tokens(await service.result(next_ref), 'twinkle') == [2]
        await _until(lambda: not service._compute_worker._detached_tasks)
        assert service.state.records[ref] is failed
        assert service.sampler.loads == [1, 2]
    finally:
        release.set()




@pytest.mark.asyncio
async def test_waiting_writer_blocks_new_readers_until_the_writer_finishes():
    guard = SamplerConcurrency()
    readers_release = asyncio.Event()
    writer_release = asyncio.Event()
    order = []

    async def reader():
        order.append('first-reader')
        await readers_release.wait()

    async def writer():
        order.append('writer')
        await writer_release.wait()

    async def next_reader():
        order.append('next-reader')

    first = asyncio.create_task(guard.run(reader))
    await _until(lambda: guard._readers == 1)
    write = asyncio.create_task(guard.run(writer, exclusive=True))
    await _until(lambda: guard._waiting_writers == 1)
    second = asyncio.create_task(guard.run(next_reader))
    await asyncio.sleep(0)
    assert order == ['first-reader']
    readers_release.set()
    await _until(lambda: guard._writer)
    assert order == ['first-reader', 'writer']
    writer_release.set()
    await asyncio.gather(first, write, second)
    assert order == ['first-reader', 'writer', 'next-reader']
    assert guard._readers == 0


@pytest.mark.asyncio
async def test_shutdown_cancels_futures_without_freeing_running_backend_early(make_service):
    service = make_service()
    service.sampler.release.clear()
    a, b = await asyncio.gather(service.submit('tinker'), service.submit('twinkle'))
    await _until(lambda: service.sampler.peak == 2)
    service.sampling_guard.close()
    stopping = asyncio.create_task(service._compute_worker.stop())
    try:
        await _until(lambda: service.state.records[a]['status'] == 'failed')
        assert service.state.records[a]['failure'].reason_code == 'orphaned_replica'
        assert service.state.records[b]['failure'].reason_code == 'orphaned_replica'
        assert service.sampling_guard._readers == 2
        assert not stopping.done()
        with pytest.raises(BackendBusyError):
            await service.sampling_guard.run(lambda: asyncio.sleep(0))
    finally:
        service.sampler.release.set()
        await stopping
    assert service.sampling_guard._readers == 0
    assert service._compute_worker._detached_reserved == 0


@pytest.mark.asyncio
async def test_data_plane_generation_blocks_mutations_through_collection(make_service):
    service = make_service()
    admitted = threading.Event()
    completed = threading.Event()
    stored = []
    service.data_plane = SimpleNamespace(enabled=True)

    def submit_generation(*args, **kwargs):
        admitted.set()

    def status(submission_id):
        return {'status': 'completed' if completed.is_set() else 'running'}

    service.sampler.submit_generation = submit_generation
    service.sampler.get_generation_status = status
    service.sampler.collect_generation = lambda submission_id: service.sampler.sample([{'input_ids': [1]}])
    service.sampler.cancel_generation = lambda submission_id: None

    async def put(rows, **kwargs):
        from twinkle.protocol.types import DataRef
        stored.extend(rows)
        return DataRef(ref_id='ref', size=len(rows), fields=['tokens'], kind='rollout')

    service.data_plane.put = put
    service.submit_concurrent_and_peek = lambda factory, **kwargs: factory()
    from twinkle.protocol.types import DataPlaneSampleRequest
    body = DataPlaneSampleRequest(inputs=[{'input_ids': [1]}], sampling_params={'max_tokens': 1})
    request = Request({'type': 'http', 'headers': []})
    generation = asyncio.create_task(service.endpoints['/twinkle/sample_to_data_plane'](request, body, service))
    mutation = None
    try:
        await _until(admitted.is_set)
        mutation = asyncio.create_task(service.sampling_guard.run(lambda: service.call_backend(service.sampler.set_template, 'template'), exclusive=True))
        await _until(lambda: service.sampling_guard._waiting_writers == 1)
        assert service.sampler.template_calls == 0
        completed.set()
        await asyncio.gather(generation, mutation)
        assert stored[0]['tokens'] == [1]
        assert service.sampler.template_calls == 1
    finally:
        completed.set()
        await asyncio.gather(generation, *([mutation] if mutation is not None else []), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
async def test_partial_checkpoint_failure_quarantines_changed_weights(make_service, tmp_path, dialect):
    service = make_service()
    (tmp_path / 'model.safetensors').touch()
    def partial_load(path):
        service.sampler.weight = 99
        raise FullWeightLoadError('load failed after a weight was changed')
    service.sampler.load_full_weights_from_path = partial_load
    ref = await service.submit(dialect, path=str(tmp_path))
    assert (await service.result(ref))['status'] == 'failed'
    ref = await service.submit('twinkle')
    record = await service.result(ref)
    assert record['failure'].reason_code == 'backend_gate_unavailable'
    assert service.sampler.active == 0
    with pytest.raises(BackendBusyError, match='checkpoint write failed'):
        await service.check_health()




@pytest.mark.asyncio
async def test_completed_generation_failure_does_not_quarantine_replica(make_service):
    from twinkle.server.sampler.twinkle_handlers import _await_generation
    service = make_service()
    service.sampler.get_generation_status = lambda _: {'status': 'failed', 'error': 'invalid input'}
    service.sampler.cancel_generation = lambda _: {'status': 'completed'}
    with pytest.raises(RuntimeError, match='invalid input'):
        await service.sampling_guard.run(lambda: _await_generation(service, 'bad'))
    ref = await service.submit('twinkle')
    assert _tokens(await service.result(ref), 'twinkle') == [7]


@pytest.mark.asyncio
async def test_timeout_keeps_job_capacity_and_does_not_overwrite_failed_future(make_service):
    from twinkle.server.exceptions import ResourceExhaustedError
    service = make_service(timeout=0.05, limit=1)
    service.sampler.release.clear()
    ref = await service.submit('tinker')
    await _until(service.sampler.entered.is_set)
    failed = await service.result(ref)
    assert failed['failure'].reason_code == 'execution_timeout'
    assert service._compute_worker._detached_reserved == 1
    with pytest.raises(ResourceExhaustedError):
        await service.submit('twinkle')
    service.sampler.release.set()
    await _until(lambda: not service._compute_worker._detached_tasks)
    assert service.state.records[ref] is failed
    assert (await service.result(await service.submit('twinkle')))['status'] == 'completed'




@pytest.mark.asyncio
async def test_admitted_writer_waits_for_abandoned_reader_to_finish():
    guard = SamplerConcurrency()
    release = asyncio.Event()
    wrote = asyncio.Event()
    async def read():
        await release.wait()
    async def write():
        wrote.set()
    reader = asyncio.create_task(guard.run(read))
    await _until(lambda: guard._readers == 1)
    writer = asyncio.create_task(guard.run(write, exclusive=True))
    await _until(lambda: guard._waiting_writers == 1)
    reader.cancel()
    await asyncio.sleep(0)
    assert not reader.done() and not wrote.is_set()
    release.set()
    await asyncio.gather(reader, return_exceptions=True)
    await asyncio.wait_for(writer, 1)
    assert wrote.is_set()


@pytest.mark.asyncio
async def test_cancelled_writer_wait_releases_its_lock_bookkeeping():
    guard = SamplerConcurrency()
    release = asyncio.Event()
    wrote = asyncio.Event()
    reader = asyncio.create_task(guard.run(release.wait))
    await _until(lambda: guard._readers == 1)
    async def write():
        wrote.set()
    writer = asyncio.create_task(guard.run(write, exclusive=True))
    await _until(lambda: guard._waiting_writers == 1)
    writer.cancel()
    await asyncio.gather(writer, return_exceptions=True)
    assert guard._waiting_writers == 0 and not wrote.is_set()
    release.set()
    await reader


@pytest.mark.asyncio
@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
async def test_backend_cancellation_leaves_terminal_failure(make_service, dialect):
    from concurrent.futures import CancelledError
    service = make_service()

    def cancelled(*args, **kwargs):
        raise CancelledError('engine coroutine cancelled')

    service.sampler.sample = cancelled
    ref = await service.submit(dialect)
    record = await service.result(ref)
    assert record['status'] == 'failed'
    assert record['failure'].reason_code == 'backend_cancelled'
    await _until(lambda: not service._compute_worker._detached_tasks)
    assert service._compute_worker._detached_reserved == 0
    await service._compute_worker.stop()
    assert service.state.records[ref] == record


@pytest.mark.asyncio
@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
async def test_invalid_checkpoint_rejects_request_without_disabling_sampling(make_service, tmp_path, dialect):
    service = make_service()
    ref = await service.submit(dialect, path=str(tmp_path / 'missing'))
    failed = await service.result(ref)
    assert failed['failure'].reason_code == 'request_rejected'
    assert failed['failure'].attribution == 'user'
    assert _tokens(await service.result(await service.submit(dialect)), dialect) == [7]


@pytest.mark.asyncio
async def test_ordinary_backend_timeout_does_not_disable_replica(make_service):
    service = make_service()
    sample = service.sampler.sample
    service.sampler.sample = lambda *a, **k: (_ for _ in ()).throw(GetTimeoutError('one request timed out'))
    with pytest.raises(GetTimeoutError):
        await service.call_backend(service.sampler.sample)
    service.sampler.sample = sample
    assert _tokens(await service.result(await service.submit('twinkle')), 'twinkle') == [7]


@pytest.mark.asyncio
async def test_stream_route_resolves_adapter_uri(make_service, tmp_path):
    service = make_service()
    (tmp_path / 'adapter_config.json').write_text(json.dumps({'r': 1}))
    service.sampler.reset_prefix_cache = lambda: None
    service.sampler._actors = []
    request = Request({'type': 'http', 'headers': []})
    response = await service.endpoints['/twinkle/sample_stream'](
        request, SampleRequest(inputs=[{'input_ids': [7]}], adapter_uri=str(tmp_path)), service)
    rows = [json.loads(row) async for row in response.body_iterator]
    assert rows[0]['error_code'] == 503


@pytest.mark.asyncio
async def test_sampling_timeout_keeps_capacity_without_blocking_other_readers(make_service):
    import asyncio
    service = make_service(timeout=0.05, limit=2)
    service.sampler.release.clear()
    timed_out = await service.submit('tinker')
    await _until(service.sampler.entered.is_set)
    assert (await service.result(timed_out))['failure'].reason_code == 'execution_timeout'
    submitting = asyncio.create_task(service.submit('twinkle'))
    await _until(lambda: service.sampler.peak == 2)
    service.sampler.release.set()
    assert _tokens(await service.result(await submitting), 'twinkle') == [7]
    await _until(lambda: service._compute_worker._detached_reserved == 0)
    await service.check_health()
