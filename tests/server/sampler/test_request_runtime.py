# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler runtime initialization, completion ownership and cancellation."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from tests.server.task_queue.helpers import _DummyQueue
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.sampler.concurrency import SamplerConcurrency
from twinkle.server.task_queue.types import ExecutionScope


@pytest.mark.asyncio
async def test_allocated_sampler_initializes_with_typed_owner_context(monkeypatch):
    from unittest.mock import Mock
    from twinkle.infra._ray.ray_helper import RayHelper
    from twinkle.infra._ray.resource_manager import ResourceManager
    from twinkle.server.sampler import app as module
    from twinkle.server.state.models import RuntimeContext

    context = RuntimeContext(namespace='owned', registry_name='registry', instance_id='sampler-owned')
    allocation = object()
    manager = object()
    allocate = Mock(return_value=manager)
    sampler = SimpleNamespace()
    monkeypatch.setenv('TWINKLE_RAY_NAMESPACE', 'before')
    monkeypatch.setenv('TWINKLE_RAY_REGISTRY_NAME', 'before')
    monkeypatch.setattr(RayHelper, 'resource_manager', None)
    monkeypatch.setattr(ResourceManager, 'from_allocation', allocate)
    monkeypatch.setattr(module, 'init_twinkle_runtime', lambda **kwargs: None)
    monkeypatch.setattr(module, '_construct_sampler_backend', lambda *args: sampler)
    monkeypatch.setattr(module.SamplerManagement, '_init_task_queue', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'get_server_state', lambda: SimpleNamespace())
    monkeypatch.setattr(module.serve, 'get_replica_context', lambda: SimpleNamespace(
        replica_id=SimpleNamespace(unique_id='serve-replica')))
    owner = module.SamplerManagement.__new__(module.SamplerManagement)
    await owner.__init__(
        model_id='mock', nproc_per_node=1, device_group={'name': 'sampler', 'ranks': 1, 'device_type': 'cpu'},
        device_mesh={'dp_size': 1}, sampler_type='mock', sampling_context=context,
        placement_group=allocation, allocation_devices=['0'])
    assert owner.replica_id == context.instance_id
    assert RayHelper.resource_manager is manager
    allocate.assert_called_once_with(owner.device_group, allocation, ['0'])
    await owner.data_plane.close()


@pytest.mark.asyncio
async def test_worker_exports_actual_completion_handle_without_private_task_lookup():
    queue = _DummyQueue()
    queue._task_queue_config = TaskQueueConfig(execution_timeout=0.05)
    queue.enable_compute_worker()
    guard = SamplerConcurrency()
    entered, exit_gate = asyncio.Event(), asyncio.Event()
    scope = ExecutionScope()

    async def backend():
        entered.set()
        await exit_gate.wait()
        return 'late result'

    ref = await queue.schedule_concurrent_task(lambda: guard.run(backend), execution_scope=scope)
    try:
        await asyncio.wait_for(entered.wait(), 1)
        completion = asyncio.create_task(scope.wait())
        for _ in range(100):
            record = await queue.state.get_future(ref['request_id'])
            if record['status'] == 'failed':
                break
            await asyncio.sleep(0.01)
        assert record['status'] == 'failed'
        assert not completion.done() and guard._readers == 1
        assert scope.request_ids == {ref['request_id']} and len(scope.jobs) == 1
        exit_gate.set()
        await asyncio.wait_for(completion, 1)
        assert guard._readers == 0
    finally:
        exit_gate.set()
        await queue._compute_worker.stop()



@pytest.mark.asyncio
async def test_runtime_does_not_certify_an_unconfirmed_cancel_ack():
    from twinkle.server.sampler.runtime import SamplerRuntime
    from twinkle.server.exceptions import BackendBusyError
    guard = SamplerConcurrency()
    runtime = SamplerRuntime(SimpleNamespace(sampling_guard=guard))
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    from twinkle.server.sampler.runtime import SamplingOperation
    runtime._operations['cancelled'] = SamplingOperation(ExecutionScope(), task)
    guard.invalidate_weights(RuntimeError('cancel ACK preceded actual generation exit'))
    with pytest.raises(BackendBusyError):
        await runtime.wait_operation('cancelled')
    assert 'cancelled' in runtime._operations


@pytest.mark.asyncio
async def test_cold_lora_config_does_not_block_sampler_health(monkeypatch):
    import json
    import sys
    from twinkle.server.sampler.runtime import SamplerRuntime
    entered, release = threading.Event(), threading.Event()
    calls = []

    class ColdConfig:

        def __init__(self, **kwargs):
            entered.set()
            assert release.wait(2)

    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(LoraConfig=ColdConfig))

    async def call_backend(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def add_adapter(name, config):
        calls.append((name, config))

    owner = SimpleNamespace(
        model_id='mock',
        sampling_guard=SamplerConcurrency(),
        sampler=SimpleNamespace(add_adapter_to_sampler=add_adapter),
        call_backend=call_backend)
    runtime = SamplerRuntime(owner)
    payload = {
        'endpoint': 'twinkle/add_adapter_to_sampler',
        'method': 'POST',
        'headers': [(b'content-type', b'application/json'), (b'x-request-id', b'configuration')],
        'body': json.dumps({
            'adapter_name': 'a',
            'config': {
                'r': 4
            }
        }).encode()
    }
    invocation = asyncio.create_task(runtime.invoke('configuration', payload))
    try:
        began = await asyncio.to_thread(entered.wait, 5)
        assert began, invocation.result() if invocation.done() else 'Configuration route did not start'
        assert (await asyncio.wait_for(runtime.runtime_identity(), 0.1))['unhealthy'] is False
        assert not invocation.done() and not calls
        release.set()
        assert (await invocation)['status'] == 200
        await runtime.wait_operation('configuration')
        assert len(calls) == 1
    finally:
        release.set()
        await asyncio.gather(invocation, return_exceptions=True)


@pytest.mark.asyncio
async def test_managed_compute_http_cannot_bypass_pool_admission(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from twinkle.server.exceptions import RequestRejectedError
    from twinkle.server.sampler import app as module

    def scaffold(title, register_routes, **kwargs):
        app = FastAPI()
        register_routes(app, lambda: SimpleNamespace())
        return app

    monkeypatch.setattr(module, 'build_deployment_app', scaffold)
    monkeypatch.setattr(module, 'bind_deployment', lambda app, *args, **kwargs: app)
    args = {
        'model_id': 'mock',
        'nproc_per_node': 1,
        'device_group': {
            'name': 'sampler',
            'ranks': 1,
            'device_type': 'cpu'
        },
        'device_mesh': {
            'dp_size': 1
        },
        'deploy_options': {},
        'sampler_type': 'mock'
    }
    managed = module.build_sampler_app(**args, sampling_context={'instance_id': 'owned'})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=managed), base_url='http://test') as client:
        with pytest.raises(RequestRejectedError, match='stable sampler endpoint'):
            await client.post('/twinkle/create', json={})
    standalone = module.build_sampler_app(**args)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=standalone), base_url='http://test') as client:
        assert (await client.post('/twinkle/create', json={})).status_code == 200


@pytest.mark.asyncio
async def test_scope_waits_for_jobs_registered_by_an_earlier_job():
    scope = ExecutionScope()
    child_started, child_release, parent_release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def child():
        child_started.set()
        await child_release.wait()

    async def parent():
        await parent_release.wait()
        scope.track(asyncio.create_task(child()))

    scope.track(asyncio.create_task(parent()))
    waiter = asyncio.create_task(scope.wait())
    try:
        await asyncio.sleep(0)
        parent_release.set()
        await child_started.wait()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(waiter), 0.05)
        child_release.set()
        await waiter
    finally:
        child_release.set()
        await asyncio.gather(waiter, return_exceptions=True)
