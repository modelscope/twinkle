from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from twinkle.server.sampler.concurrency import SamplerConcurrency
from twinkle.exceptions import FullWeightLoadError
from twinkle.server.task_queue.types import BackendBusyError


from tests.server.sampler.sampling_test_helpers import _tokens, _until, make_service


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
async def test_shutdown_before_task_first_step_marks_future_failed(make_service):
    from twinkle.server.task_queue.types import QueuedTask
    service = make_service()
    worker = service._compute_worker
    worker.reserve_detached()
    await service.state.store_future_status('not-started', 'pending')
    worker.run_detached(QueuedTask('not-started', lambda: asyncio.sleep(0), None, None, 0, 'sample', 0), 'q')
    await worker.stop()
    assert service.state.records['not-started']['failure'].reason_code == 'orphaned_replica'
    assert worker._detached_reserved == 0


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
async def test_nested_status_poll_reuses_job_access(make_service):
    service = make_service()
    async def job():
        # A separate polling Task inherits the job's access context.
        await asyncio.wait_for(service.call_backend(service.sampler.set_template, 'test'), timeout=1)
    await service.sampling_guard.run(job, exclusive=True)
    assert service.sampler.template_calls == 1
    assert not service.sampling_guard._writer


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
