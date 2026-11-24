# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import hashlib
import hmac
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
from twinkle.server.identity import USER_ID_KEY_ENV, current_tenant_fingerprint, tenant_fingerprint
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
    assert tenant_fingerprint(None) == tenant_fingerprint('') == 'unassigned'
    assert tenant_fingerprint(TOKEN) != tenant_fingerprint(None)
    assert not tenant_fingerprint(TOKEN).startswith(legacy)
    monkeypatch.setenv(USER_ID_KEY_ENV, salt)
    assert tenant_fingerprint(TOKEN) is None


@pytest.mark.asyncio
async def test_usage_is_recorded_once_at_admission(monkeypatch, in_memory_span_exporter, caplog, isolated_metrics_registry):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setenv(USER_ID_KEY_ENV, 'ops-only-key')
    monkeypatch.setattr(metrics, 'get_meter', provider.get_meter)

    in_memory_span_exporter.clear()

    queue = TaskQueueMixin()
    queue.state = ServerState(backend=FakeBackend())
    queue.replica_id = 'test-replica'
    queue._init_task_queue(TaskQueueConfig(), deployment_name='Compute', telemetry_base_model='base-model')

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
        request.state.training_id = 'training-1'
        request.state.base_model = 'base-model'
        return await submit_training_task(
            queue, compute, training_id='training-1', stream='native', seq_id=1, peek=True,
            token=get_token_from_request(request), input_tokens=3, task_type='forward_backward')

    @app.post('/clip/{training_id}')
    async def clip_grad_and_step(request: Request, training_id: str):
        request.state.training_id = training_id
        request.state.base_model = 'base-model'
        return await submit_training_task(
            queue, compute, training_id=training_id, stream='native', seq_id=2 if training_id == 'training-1' else 1,
            peek=True, token=get_token_from_request(request), task_type='clip_grad_and_step')

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
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://local',
            headers={'Authorization': f'Bearer {token}', 'x-request-id': 'client-call'},
        ) as client:
            routes = (
                'train', 'train', 'clip/training-1', 'clip/training-1', 'clip/training-2',
                'sample', 'sample', 'save', 'heartbeat', 'poll', 'rejected', 'register',
            )
            for route in routes:
                response = await client.post('/' + route)
                assert response.status_code == (429 if route == 'rejected' else 200)
            response = await client.post(
                '/twinkle/call', json={'processor_id': 'pid:processor-1', 'function': 'process', 'call_kwargs': {}})
            assert response.json() == {'result': 'processed'}

        instruments = {
            metric.name: metric.data.data_points
            for resource in reader.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics for metric in scope.metrics
        }
        requests = instruments['twinkle.usage.requests.total']
        assert {point.attributes['task_type']: point.value for point in requests} == {
            'forward_backward': 1, 'clip_grad_and_step': 2, 'sample': 1}
        assert sum(point.value for point in instruments['twinkle.usage.input_tokens.total']) == 8
        assert sum(point.sum for point in instruments['twinkle.task.execution_seconds']) > 0
        assert {point.attributes['base_model'] for point in requests} == {'base-model'}
        user_id = tenant_fingerprint(token)
        assert {point.attributes['tenant_fingerprint'] for point in requests} == {user_id}
        http = [point.attributes for point in instruments['twinkle.http.requests.total']]
        assert {'session_heartbeat', 'retrieve_future'}.issubset({attrs['operation'] for attrs in http})
        assert any(attrs['status'] == '429' and attrs['tenant_fingerprint'] == user_id for attrs in http)

        spans = in_memory_span_exporter.get_finished_spans()
        executions = [span for span in spans if span.name == 'task_queue.execute']
        assert len(executions) == 5
        assert {s.attributes['twinkle.training_id'] for s in executions
                if s.attributes['twinkle.task_type'] == 'clip_grad_and_step'} == {'training-1', 'training-2'}
        submission_spans = {'POST /train', 'POST /clip/{training_id}', 'POST /sample', 'POST /save'}
        parent_trace = {span.context.trace_id for span in spans if span.name in submission_spans}
        for span in executions:
            assert span.context.trace_id in parent_trace
            assert span.attributes['twinkle.tenant_fingerprint'] == user_id
            assert span.attributes['twinkle.wait_seconds'] >= 0
            assert span.attributes['twinkle.execution_seconds'] > 0
        for name in ('server_state.register_model', 'processor.call.process'):
            assert next(span.attributes['twinkle.tenant_fingerprint'] for span in spans if span.name == name) == user_id
        assert token not in repr([dict(span.attributes) for span in spans])
        sample_logs = [record for record in caplog.records
                       if getattr(record, 'twinkle_task_id', None) == 'sample-1']
        assert [record.twinkle_task_outcome for record in sample_logs] == ['started', 'completed']
        assert {record.twinkle_base_model for record in sample_logs} == {'base-model'}
        assert all(not hasattr(record, 'twinkle_training_id') for record in sample_logs)
        assert user_id in caplog.text
        assert token not in caplog.text
    finally:
        logger.removeHandler(caplog.handler)
        await queue.shutdown_task_queue()

        provider.shutdown()


@pytest.mark.asyncio
async def test_training_logs_link_each_outcome_to_its_training(caplog, in_memory_span_exporter, monkeypatch):
    queue = TaskQueueMixin()
    queue.state = ServerState(backend=FakeBackend())
    queue.replica_id = 'training-logs-test'
    queue._init_task_queue(TaskQueueConfig(execution_timeout=0.1, inline_fast_path_timeout=1),
                           deployment_name='Model', telemetry_base_model='training-base')
    in_memory_span_exporter.clear()
    store_future_status = queue.state.store_future_status

    async def check_queue_log_before_status_write(request_id, status, *args, **kwargs):
        if request_id == 'req-4' and status == 'completed':
            raise RuntimeError('private result write failure')
        if status == 'queued':
            assert any(getattr(log, 'twinkle_task_id', None) == request_id and log.twinkle_task_outcome == 'queued'
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
        cases = ((complete, 'completed'), (fail, 'failed'), (timeout, 'timeout'), (refuse, 'refused'),
                 (complete, 'failed'))
        for number, (operation, expected) in enumerate(cases):
            model_id = f'training-{number}'
            fingerprint = f'fingerprint-{number}'
            request_id = f'req-{number}'
            binding = current_tenant_fingerprint.set(fingerprint)
            try:
                result = await queue.submit_and_peek(operation, model_id=model_id, token=TOKEN,
                                            task_type='forward_backward', request_id=request_id)
            finally:
                current_tenant_fingerprint.reset(binding)
            records = [record for record in caplog.records if getattr(record, 'twinkle_task_id', None) == request_id]
            assert [record.twinkle_task_outcome for record in records] == ['queued', expected]
            assert {record.twinkle_training_id for record in records} == {model_id}
            assert {record.twinkle_tenant_fingerprint for record in records} == {fingerprint}
            assert {record.twinkle_task_type for record in records} == {'forward_backward'}
            assert {record.twinkle_base_model for record in records} == {'training-base'}
            span, = [span for span in in_memory_span_exporter.get_finished_spans()
                     if span.attributes.get('twinkle.task_id') == request_id]
            assert records[-1].twinkle_trace_id == f'{span.context.trace_id:032x}'
            assert span.status.status_code.name == ('UNSET' if expected == 'completed' else 'ERROR')
            assert span.attributes['twinkle.task_outcome'] == ('failed' if expected == 'refused' else expected)
            if number == 4:
                assert result.status == 'failed'
                assert 'private result write failure' not in repr(dict(span.attributes))
                assert 'private result write failure' not in caplog.text
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
            assert cpu.attributes == {'node': node}
            rss, = collector._process_memory_callback(None)
            assert rss.attributes == {'node': node, 'pid': os.getpid()}
        competitor = resource_metrics.ResourceMetricsCollector()
        assert list(competitor._gpu_memory_callback(None)) == []
        collector.close()
        assert list(competitor._gpu_memory_callback(None)) == [gpu]
        competitor.close()


@pytest.mark.asyncio
async def test_http_failure_status_and_anonymous_attribution(monkeypatch, in_memory_span_exporter):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from twinkle.protocol.headers import H_CLIENT_ID
    from twinkle.server import deployment

    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(metrics, 'get_meter', meter_provider.get_meter)
    monkeypatch.setattr(metrics.MetricsRegistry, '_instance', None)
    monkeypatch.setenv(USER_ID_KEY_ENV, 'test-identity-secret')
    in_memory_span_exporter.clear()

    async def verify(request, call_next):
        if request.url.path == '/middleware-error':
            raise RuntimeError('private auth error')
        return await verify_request_token(request, call_next)

    monkeypatch.setattr(deployment, 'verify_request_token', verify)

    def register(app, _):
        @app.get('/v1/models', status_code=429)
        async def handler(request: Request):
            return {'fingerprint': request.state.tenant_fingerprint}

    app = deployment.build_deployment_app('Audit', register)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://test',
        ) as client:
            identities = (({}, tenant_fingerprint(None)), ({H_CLIENT_ID: 'sdk'}, tenant_fingerprint(None)))
            for headers, expected in identities:
                for request_id in ('id-one', 'id-two'):
                    response = await client.get('/v1/models', headers={**headers, 'x-request-id': request_id})
                    assert response.status_code == 429
                    assert response.json()['fingerprint'] == expected
            assert (await client.get('/middleware-error')).status_code == 500

        points = [p for r in reader.get_metrics_data().resource_metrics for scope in r.scope_metrics
                  for m in scope.metrics if m.name == 'twinkle.http.requests.total' for p in m.data.data_points]
        assert sum(p.value for p in points if p.attributes['status'] == '500') == 1
        spans = in_memory_span_exporter.get_finished_spans()
        assert [s.status.status_code.name for s in spans if s.name == 'GET /v1/models'] == ['UNSET'] * 4
        error, = [s for s in spans if s.attributes.get('http.response.status_code') == 500]
        assert error.status.status_code.name == 'ERROR'
        event, = error.events
        assert dict(event.attributes) == {'exception.type': 'RuntimeError'}
    finally:
        meter_provider.shutdown()


@pytest.mark.asyncio
async def test_proxy_client_span_connects_two_server_spans(in_memory_span_exporter, monkeypatch):
    from twinkle.server.lifecycle.proxy import ServiceProxy

    in_memory_span_exporter.clear()
    requests_total = SimpleNamespace(add=Mock())
    request_metrics = SimpleNamespace(requests_total=requests_total, request_duration_seconds=Mock())
    monkeypatch.setattr(metrics.MetricsRegistry, 'get', lambda: request_metrics)
    upstream = FastAPI()
    upstream.middleware('http')(create_tracing_middleware('Model'))
    upstream.middleware('http')(create_metrics_middleware('Model'))

    @upstream.post('/api/v1/model/base-model/twinkle/step', status_code=429)
    async def step():
        return {}

    proxy = ServiceProxy()
    await proxy.client.aclose()
    proxy.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=upstream))
    gateway = FastAPI()
    gateway.middleware('http')(create_tracing_middleware('Gateway'))
    gateway.middleware('http')(create_metrics_middleware('Gateway'))

    @gateway.post('/train')
    async def train(request: Request):
        return await proxy.proxy_request(request, 'twinkle/step', 'base-model', 'model')

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url='http://test') as client:
            assert (await client.post('/train')).status_code == 429
    finally:
        await proxy.close()

    assert [(call.kwargs['attributes']['deployment'], call.kwargs['attributes']['response_origin'])
            for call in requests_total.add.call_args_list] == [('Model', 'local'), ('Gateway', 'forwarded')]
    target, outbound, entry = in_memory_span_exporter.get_finished_spans()
    assert target.parent.span_id == outbound.context.span_id
    assert outbound.parent.span_id == entry.context.span_id
    assert len({s.context.trace_id for s in (target, outbound, entry)}) == 1
    assert outbound.kind.name == 'CLIENT' and outbound.status.status_code.name == 'ERROR'
    assert entry.status.status_code.name == target.status.status_code.name == 'UNSET'


@pytest.mark.asyncio
async def test_sampler_router_propagates_new_trace_context(in_memory_span_exporter):
    from starlette.requests import Request as ASGIRequest

    from twinkle.server.sampler.pool.router import SamplerRouter
    from twinkle.server.telemetry.tracing import traced_operation

    in_memory_span_exporter.clear()
    forwarded = AsyncMock(return_value={'status': 429, 'body': b'{}', 'headers': []})
    router = SamplerRouter.__new__(SamplerRouter)
    router.model_id = 'base-model'
    router.controller = SimpleNamespace(request=SimpleNamespace(remote=forwarded))
    request = ASGIRequest(
        {'type': 'http', 'method': 'POST', 'headers': [], 'query_string': b''},
        receive=AsyncMock(return_value={'type': 'http.request', 'body': b'{}'}),
    )
    with traced_operation('router.http'):
        assert (await router.forward(request, 'twinkle/sample')).status_code == 429
    assert request.state.forwarded_429
    hop, entry = in_memory_span_exporter.get_finished_spans()
    assert hop.parent.span_id == entry.context.span_id
    payload = forwarded.await_args.args[0]
    headers = dict(payload['headers'])
    assert headers[b'traceparent'].decode().split('-')[1:3] == [
        f'{entry.context.trace_id:032x}', f'{hop.context.span_id:016x}',
    ]
