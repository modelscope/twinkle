# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import hashlib
import hmac
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request

from tests.server.state.fake_backend import FakeBackend
from twinkle.server.checkpoint import paths
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.exceptions import BackendBusyError
from twinkle.server.lifecycle.submit import submit_training_task
from twinkle.server.middleware.auth import get_token_from_request, verify_request_token
from twinkle.server.processor.twinkle_handlers import _register_processor_routes
from twinkle.server.state.server_state import ServerState
from twinkle.server.task_queue.mixin import TaskQueueMixin
from twinkle.server.telemetry import metrics, resource_metrics
from twinkle.server.telemetry.http_middleware import create_metrics_middleware
from twinkle.server.telemetry.identity import USER_ID_KEY_ENV, current_user_id, user_fingerprint
from twinkle.server.telemetry.tracing import create_tracing_middleware
from twinkle.utils.logger import get_logger

TOKEN = 'synthetic-credential-测试'


@pytest.mark.parametrize('salt', ['twinkle-path-salt-v1', 'custom-目录-salt'])
def test_checkpoint_fingerprint_keeps_the_previous_directory(monkeypatch, salt):
    legacy = hmac.new(salt.encode(), TOKEN.encode(), hashlib.sha256).hexdigest()[:16]
    monkeypatch.setattr(paths, '_TOKEN_SALT', salt)
    monkeypatch.setenv('TWINKLE_TOKEN_SALT', salt)
    monkeypatch.setenv(USER_ID_KEY_ENV, 'ops-only-key')
    assert paths._hash_token(TOKEN) == legacy
    assert user_fingerprint(None, TOKEN) == user_fingerprint('', TOKEN)
    assert user_fingerprint(TOKEN) != user_fingerprint(None, TOKEN)
    assert not user_fingerprint(TOKEN).startswith(legacy)
    monkeypatch.setenv(USER_ID_KEY_ENV, salt)
    assert user_fingerprint(TOKEN) is None


@pytest.mark.asyncio
async def test_usage_is_recorded_once_at_admission(monkeypatch, in_memory_span_exporter, caplog):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setenv(USER_ID_KEY_ENV, 'ops-only-key')
    monkeypatch.setattr(metrics, 'get_meter', provider.get_meter)
    metrics.MetricsRegistry.reset()
    in_memory_span_exporter.clear()

    queue = TaskQueueMixin()
    queue.state = ServerState(backend=FakeBackend())
    queue.replica_id = 'test-replica'
    queue._init_task_queue(TaskQueueConfig(), deployment_name='Compute', telemetry_model_id='base-model')

    app = FastAPI()
    app.middleware('http')(verify_request_token)
    app.middleware('http')(create_tracing_middleware('Compute'))
    app.middleware('http')(create_metrics_middleware('Compute'))

    def process():
        return 'processed'

    process._execute = True
    processor = SimpleNamespace(
        _ensure_resource_cleanup_started=AsyncMock(),
        assert_resource_exists=lambda _: None,
        resource_dict={'processor-1': SimpleNamespace(process=process)},
    )
    _register_processor_routes(app, lambda: processor)

    async def compute():
        return {'ok': True}

    @app.post('/train')
    async def train(request: Request):
        request.state.model_id = 'training-1'
        return await submit_training_task(
            queue, compute, training_id='training-1', stream='native', seq_id=1, peek=True,
            token=get_token_from_request(request), input_tokens=3, task_type='forward_backward')

    @app.post('/sample')
    async def sample(request: Request):
        return await queue.submit_concurrent_and_peek(
            compute, token=get_token_from_request(request), input_tokens=5, task_type='sample',
            request_id='sample-1', submission_fingerprint='stable-request-retry-receipt')

    @app.post('/save')
    async def save(request: Request):
        return await queue.submit_and_peek(compute, token=get_token_from_request(request), task_type='save')

    @app.post('/heartbeat')
    async def session_heartbeat():
        return {}

    @app.post('/poll')
    async def retrieve_future():
        return {}

    @app.post('/rejected', status_code=429)
    async def rejected():
        return {}

    @app.post('/register')
    async def register(request: Request):
        return await queue.state.register_model({'base_model': 'base-model'}, get_token_from_request(request))

    caplog.set_level(logging.INFO)
    logger = get_logger()
    logger.addHandler(caplog.handler)
    try:
        token = 'synthetic-header-credential'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://local',
                                     headers={'Authorization': f'Bearer {token}', 'x-request-id': 'client-call'}) as client:
            for route in ('train', 'train', 'sample', 'sample', 'save', 'heartbeat', 'poll', 'rejected', 'register'):
                response = await client.post('/' + route)
                assert response.status_code == (429 if route == 'rejected' else 200)
            response = await client.post('/twinkle/call', json={
                'processor_id': 'pid:processor-1', 'function': 'process', 'call_kwargs': {}})
            assert response.json() == {'result': 'processed'}

        instruments = {
            metric.name: metric.data.data_points
            for resource in reader.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics for metric in scope.metrics
        }
        requests = instruments['twinkle.usage.requests.total']
        assert {point.attributes['task_type']: point.value for point in requests} == {
            'forward_backward': 1, 'sample': 1}
        assert sum(point.value for point in instruments['twinkle.usage.input_tokens.total']) == 8
        assert sum(point.value for point in instruments['twinkle.usage.run_seconds.total']) > 0
        assert {point.attributes['model_id'] for point in requests} == {'training-1', 'base-model'}
        user_id = user_fingerprint(token)
        assert {point.attributes['user_id'] for point in requests} == {user_id}
        http = [point.attributes for point in instruments['twinkle.http.requests.total']]
        assert {'session_heartbeat', 'retrieve_future'}.issubset({attrs['operation'] for attrs in http})
        assert any(attrs['status'] == '429' and attrs['user_id'] == user_id for attrs in http)

        spans = in_memory_span_exporter.get_finished_spans()
        executions = [span for span in spans if span.name == 'task_queue.execute']
        assert len(executions) == 3
        parent_trace = {span.context.trace_id for span in spans if span.name in ('POST /train', 'POST /sample', 'POST /save')}
        for span in executions:
            assert span.context.trace_id in parent_trace
            assert span.attributes['twinkle.user_id'] == user_id
            assert span.attributes['twinkle.wait_seconds'] >= 0
            assert span.attributes['twinkle.execution_seconds'] > 0
        assert next(span.attributes['twinkle.queue_key'] for span in executions
                    if span.attributes['twinkle.task_type'] == 'sample') == user_id
        for name in ('server_state.register_model', 'processor.call.process'):
            assert next(span.attributes['twinkle.user_id'] for span in spans if span.name == name) == user_id
        assert token not in repr([dict(span.attributes) for span in spans])
        sample_logs = [record for record in caplog.records
                       if getattr(record, 'twinkle_request_id', None) == 'sample-1']
        assert [record.twinkle_task_outcome for record in sample_logs] == ['started', 'completed']
        assert {record.twinkle_model_id for record in sample_logs} == {'base-model'}
        assert user_id in caplog.text
        assert token not in caplog.text
    finally:
        logger.removeHandler(caplog.handler)
        await queue.shutdown_task_queue()
        metrics.MetricsRegistry.reset()
        provider.shutdown()


@pytest.mark.asyncio
async def test_training_logs_link_each_outcome_to_its_training(caplog, in_memory_span_exporter, monkeypatch):
    queue = TaskQueueMixin()
    queue.state = ServerState(backend=FakeBackend())
    queue.replica_id = 'training-logs-test'
    queue._init_task_queue(TaskQueueConfig(execution_timeout=0.1, inline_fast_path_timeout=1),
                           deployment_name='Model', telemetry_model_id='training-base')
    in_memory_span_exporter.clear()
    store_future_status = queue.state.store_future_status

    async def check_queue_log_before_status_write(request_id, status, *args, **kwargs):
        if status == 'queued':
            assert any(getattr(log, 'twinkle_request_id', None) == request_id and log.twinkle_task_outcome == 'queued'
                       for log in caplog.records)
        return await store_future_status(request_id, status, *args, **kwargs)

    monkeypatch.setattr(queue.state, 'store_future_status', check_queue_log_before_status_write)

    async def complete():
        return {'ok': True}

    async def fail():
        raise RuntimeError('intentional error')

    async def timeout():
        await asyncio.sleep(2)

    async def refuse():
        raise BackendBusyError('intentional gate refusal')

    logger = get_logger()
    caplog.set_level(logging.INFO)
    logger.addHandler(caplog.handler)
    try:
        cases = ((complete, 'completed'), (fail, 'failed'), (timeout, 'timeout'), (refuse, 'refused'))
        for number, (operation, expected) in enumerate(cases):
            model_id = f'training-{number}'
            fingerprint = f'fingerprint-{number}'
            request_id = f'req-{number}'
            binding = current_user_id.set(fingerprint)
            try:
                await queue.submit_and_peek(operation, model_id=model_id, token=TOKEN,
                                            task_type='forward_backward', request_id=request_id)
            finally:
                current_user_id.reset(binding)
            records = [record for record in caplog.records if getattr(record, 'twinkle_request_id', None) == request_id]
            assert [record.twinkle_task_outcome for record in records] == ['queued', expected]
            assert {record.twinkle_model_id for record in records} == {model_id}
            assert {record.twinkle_user_id for record in records} == {fingerprint}
            assert {record.twinkle_task_type for record in records} == {'forward_backward'}
            assert {record.twinkle_base_model for record in records} == {'training-base'}
            span, = [span for span in in_memory_span_exporter.get_finished_spans()
                     if span.attributes.get('twinkle.request_id') == request_id]
            assert records[-1].twinkle_trace_id == f'{span.context.trace_id:032x}'
            assert TOKEN not in repr([record.__dict__ for record in records])
    finally:
        logger.removeHandler(caplog.handler)
        await queue.shutdown_task_queue()


def test_same_card_index_on_two_nodes_has_distinct_observations(monkeypatch):
    monkeypatch.setattr(resource_metrics, '_nvml_handle_count', lambda: 1)
    monkeypatch.setattr(resource_metrics, 'pynvml', SimpleNamespace(
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUUID=lambda handle: b'GPU-test',
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=10),
    ), raising=False)
    for node in ('node-a', 'node-b'):
        monkeypatch.setenv('NODE_NAME', node)
        collector = resource_metrics.ResourceMetricsCollector()
        gpu, = collector._gpu_memory_callback(None)
        assert gpu.attributes == {'node': node, 'gpu_index': 0, 'gpu_uuid': 'GPU-test'}
        if resource_metrics._PSUTIL_AVAILABLE:
            cpu, = collector._cpu_utilization_callback(None)
            assert cpu.attributes == {'node': node, 'pid': os.getpid()}
