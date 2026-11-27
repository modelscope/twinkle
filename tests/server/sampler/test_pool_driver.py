# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampler pool transport and failure recovery."""
import asyncio
from types import SimpleNamespace

import pytest

from tests.server.state.fake_backend import FakeBackend
from twinkle_server.sampler.concurrency import SamplerConcurrency
from twinkle_server.sampler.pool import driver as driver_module
from twinkle_server.sampler.runtime import SamplerRuntime, SamplingOperation
from twinkle_server.state.future_manager import FutureManager
from twinkle_server.task_queue.types import ExecutionScope


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    yield


@pytest.mark.asyncio
async def test_generation_waits_leave_serve_slots_available_for_health(monkeypatch):
    # Simulate the per-replica Serve limit while all five backend jobs are still running.
    replica = SamplerRuntime(SimpleNamespace(sampler=None, sampling_guard=SamplerConcurrency()))
    slots = asyncio.Semaphore(5)
    generation_done = asyncio.Event()

    def remote(method):
        async def call(*args):
            async with slots:
                return await method(*args)

        return SimpleNamespace(remote=call)

    async def handle(_instance):
        return SimpleNamespace(
            operation_done=remote(replica.operation_done),
            runtime_identity=remote(replica.runtime_identity))

    driver = driver_module.SamplerDriver(None, 'test', 'test', {}, {})
    monkeypatch.setattr(driver, 'handle', handle)
    for i in range(5):
        replica._operations[str(i)] = SamplingOperation(ExecutionScope(), asyncio.create_task(generation_done.wait()))

    instance = SimpleNamespace(identity='sampler-1')
    waits = [asyncio.create_task(driver.wait_operation(instance, str(i))) for i in range(5)]
    try:
        # Start the waits before probing, even when they return control to Serve immediately.
        await asyncio.sleep(0)
        assert await asyncio.wait_for(driver.health(instance), 0.5) == {'unhealthy': False}
        assert len(replica._operations) == 5 and all(not task.done() for task in waits)
        generation_done.set()
        await asyncio.wait_for(asyncio.gather(*waits), 1)
        assert not replica._operations
    finally:
        generation_done.set()
        await asyncio.gather(*waits, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_settles_only_nonterminal_futures_of_the_dead_replica(monkeypatch):
    futures = FutureManager(FakeBackend(), expiration_timeout=300)
    await futures.store_status('running', 'running', None, replica_id='sampler-1', owner_token='owner')
    await futures.store_status('pending', 'pending', None, replica_id='sampler-1')
    await futures.store_status('completed', 'completed', None, result={'sample': 1}, replica_id='sampler-1')
    await futures.store_status('other', 'running', None, replica_id='sampler-2')
    await futures.store_status('unowned', 'running', None)
    monkeypatch.setattr(driver_module, 'get_server_state', lambda: SimpleNamespace(futures=futures))
    driver = driver_module.SamplerDriver(None, 'test', 'test', {}, {})

    async def unreachable(_instance):
        raise ConnectionError('The replica cannot answer RPCs')

    async def shutdown(identity, name):
        assert (identity, name) == ('sampler-1', 'sampler-instance')
        # Never settle failed work before the underlying runtime confirms shutdown.
        assert (await futures.get('running')).status == 'running'

    monkeypatch.setattr(driver, 'handle', unreachable)
    monkeypatch.setattr(driver.runtime, 'shutdown', shutdown)
    await driver.shutdown(SimpleNamespace(identity='sampler-1', application='sampler-instance'))

    for request_id in ('running', 'pending'):
        record = await futures.get(request_id)
        assert record.status == 'failed' and record.failure.reason_code == 'orphaned_replica'
    assert (await futures.get('running')).owner_token == 'owner'
    assert (await futures.get('completed')).result == {'sample': 1}
    assert (await futures.get('other')).status == (await futures.get('unowned')).status == 'running'
