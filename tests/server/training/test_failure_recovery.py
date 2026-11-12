# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fault injection ensures persistence and notification failures do not leak resources."""
import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from tests.server.state.fake_backend import FakeBackend
from tests.server.training.helpers import create, request, setup
from tests.server.training.helpers import RPC, RuntimeQueue, settle
from twinkle.protocol.types.errors import ErrorCategory
from twinkle.server.exceptions import ResourceExhaustedError, ResourceNotFoundError
from twinkle.server.state.models import FutureFailureRecord


async def wait_idle(queue):
    worker = queue._compute_worker
    for _ in range(100):
        if not worker._request_ids and not worker.new_task_event.is_set():
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0.001)
    raise AssertionError('Worker did not enter idle wait')


@pytest.mark.asyncio
async def test_warm_idle_worker_wakes_after_enqueue_failure(monkeypatch):
    controller, state, _, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'queue-failure')
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)
    executed = []

    async def operation():
        executed.append(True)
        return {'ok': True}

    try:
        warmup = await queue.schedule_task(operation, model_id=identity, token='owner', task_type='step')
        await settle(state, warmup['request_id'])
        await wait_idle(queue)
        executed.clear()
        previous = state.futures.store_status

        async def broken(request_id, status, *args, **kwargs):
            if status == 'queued':
                raise RuntimeError('injected queued write failure')
            return await previous(request_id, status, *args, **kwargs)

        monkeypatch.setattr(state.futures, 'store_status', broken)
        with pytest.raises(RuntimeError, match='queued write'):
            await queue.schedule_task(operation, model_id=identity, token='owner', task_type='step',
                                      request_id='queued-failure')
        ref = {'request_id': 'queued-failure'}
        assert (await settle(state, ref['request_id']))['status'] == 'completed'
        await wait_idle(queue)
        assert executed == [True]
        assert not controller.trainings[identity].pending_operation_ids
        assert queue._compute_worker.total_queued() == 0
    finally:
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault,mode', [
    ('retire', 'lora'), ('save_draining', 'full'), ('save_stopped', 'full'),
    ('close_retire', 'full'), ('post_retire_save', 'lora')
])
async def test_cleanup_storage_failure_is_retryable(fault, mode, monkeypatch):
    controller, state, runtime, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, fault, mode=mode)
    training = controller.trainings[identity]
    instance = controller.instances[training.instance_id]
    previous_retire = controller.store.retire
    previous_save = controller.store.save_instance
    failed = False

    async def retire(record):
        nonlocal failed
        if fault in ('retire', 'close_retire') and not failed:
            failed = True
            raise RuntimeError('injected retirement failure')
        return await previous_retire(record)

    async def save(record):
        nonlocal failed
        matches = ((fault == 'save_draining' and record.state == 'DRAINING')
                   or (fault == 'save_stopped' and record.state == 'STOPPED')
                   or (fault == 'post_retire_save' and not record.training_ids))
        if matches and not failed:
            failed = True
            raise RuntimeError('injected instance write failure')
        return await previous_save(record)

    monkeypatch.setattr(controller.store, 'retire', retire)
    monkeypatch.setattr(controller.store, 'save_instance', save)
    with pytest.raises(RuntimeError, match='injected'):
        await controller.release(identity, 'owner', 'session')
    assert failed
    assert identity not in controller._cleaning
    assert instance.instance_id not in controller.lifecycle.closing
    assert training.state == 'CLOSING'
    assert instance.state != 'STOPPED'
    for _ in range(3):
        await controller.sweep()
        await controller.wait_for_jobs()
        if training.state == 'RELEASED':
            break
    assert training.state == 'RELEASED'
    assert not instance.training_ids
    assert instance.state == 'STOPPED'
    assert not runtime.allocations
    assert await state.get_model_metadata(identity) is None
    assert (await state._backend.get('token_count::owner') or 0) == 0
    await controller.shutdown()


@pytest.mark.asyncio
async def test_failure_transition_write_is_retried_and_admission_fenced(monkeypatch):
    controller, state, runtime, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'failure-transition')
    training = controller.trainings[identity]
    instance = controller.instances[training.instance_id]
    await controller.admit(identity, 'owner', 'session', 'pending-op')
    await controller.store.future('pending-op', 'pending', identity)
    original = controller.store.save_instance
    failed = False

    async def save(record):
        nonlocal failed
        if record.state == 'DRAINING' and not failed:
            failed = True
            raise RuntimeError('injected draining write failure')
        return await original(record)

    monkeypatch.setattr(controller.store, 'save_instance', save)
    failure = FutureFailureRecord(reason_code='orphaned_replica', attribution='server', message='Worker died')
    with pytest.raises(RuntimeError, match='draining write'):
        await controller.fail_instance(instance.instance_id, failure)
    assert instance.state == 'READY'
    with pytest.raises(ResourceNotFoundError):
        await controller.route(identity, 'owner')
    await controller.sweep()
    await controller.wait_for_jobs()
    assert instance.state == 'STOPPED' and training.state == 'FAILED'
    assert not runtime.allocations and not training.pending_operation_ids
    assert (await state.get_future('pending-op'))['failure']['reason_code'] == 'orphaned_replica'
    assert not controller._failed_instances
    await controller.shutdown()


@pytest.mark.asyncio
async def test_failed_resident_is_excluded_when_rebuilding_minimum_replicas(monkeypatch):
    controller, _, runtime, _ = await setup(backend=FakeBackend(), min_replicas=1)
    failed_instance = next(iter(controller.instances.values()))
    original = controller.store.save_instance
    failed = False

    async def save(record):
        nonlocal failed
        if record.state == 'DRAINING' and not failed:
            failed = True
            raise RuntimeError('injected draining write failure')
        return await original(record)

    monkeypatch.setattr(controller.store, 'save_instance', save)
    failure = FutureFailureRecord(reason_code='orphaned_replica', attribution='server', message='Worker died')
    with pytest.raises(RuntimeError, match='draining write'):
        await controller.fail_instance(failed_instance.instance_id, failure)

    assert failed_instance.state == 'READY'
    assert failed_instance.instance_id in controller._failed_instances
    await controller.sweep()
    await controller.wait_for_jobs()

    ready_residents = [
        instance for instance in controller.instances.values() if instance.resident and instance.state == 'READY'
    ]
    assert failed_instance.state == 'STOPPED'
    assert len(ready_residents) == 1
    assert ready_residents[0].instance_id != failed_instance.instance_id
    assert set(runtime.allocations) == {ready_residents[0].instance_id}
    await controller.shutdown()


@pytest.mark.asyncio
async def test_shutdown_waits_for_an_in_flight_instance_close():
    controller, _, runtime, _ = await setup(backend=FakeBackend(), min_replicas=1)
    instance = next(iter(controller.instances.values()))
    original_shutdown = runtime.shutdown
    entered, resume = asyncio.Event(), asyncio.Event()

    async def slow_shutdown(target):
        entered.set()
        await resume.wait()
        await original_shutdown(target)

    runtime.shutdown = slow_shutdown
    closing = asyncio.create_task(controller._close_instance(instance))
    await entered.wait()
    shutdown = asyncio.create_task(controller.shutdown())
    for _ in range(100):
        if instance.state == 'DRAINING' or shutdown.done():
            break
        await asyncio.sleep(0)

    assert instance.state == 'DRAINING'
    assert not shutdown.done()
    resume.set()
    await asyncio.gather(closing, shutdown)
    assert instance.state == 'STOPPED'
    assert instance.instance_id not in controller.lifecycle.closing


@pytest.mark.asyncio
async def test_cleanup_error_is_not_classified_as_startup_failure():
    controller, state, runtime, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'release-failure')
    instance = controller.instances[controller.trainings[identity].instance_id]
    runtime.release_training = AsyncMock(side_effect=TimeoutError('release timed out'))
    await controller.release(identity, 'owner')
    assert instance.failure == 'Training cleanup failed: release timed out'
    assert controller.trainings[identity].terminal_reason == 'internal_error'
    await controller.shutdown()


@pytest.mark.asyncio
async def test_unready_service_is_503_server():
    controller, _, _, _ = await setup(backend=FakeBackend())
    controller._ready = False
    with pytest.raises(ResourceExhaustedError) as error:
        await controller.create(request('unready'))
    assert error.value.error_code == 503 and error.value.category is ErrorCategory.Server
    await controller.shutdown()


@pytest.mark.asyncio
async def test_sweep_heartbeat_io_does_not_hold_route_lock(monkeypatch):
    controller, state, _, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'slow-heartbeat')
    entered, resume = asyncio.Event(), asyncio.Event()
    original = state.sessions.get_last_heartbeats

    async def slow(identities):
        entered.set()
        await resume.wait()
        return await original(identities)

    monkeypatch.setattr(state.sessions, 'get_last_heartbeats', slow)
    sweeping = asyncio.create_task(controller.sweep())
    try:
        await entered.wait()
        assert await asyncio.wait_for(controller.route(identity, 'owner'), 0.1)
    finally:
        resume.set()
        await sweeping
        await controller.wait_for_jobs()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_explicit_tinker_request_id_is_preserved():
    from twinkle.server.lifecycle.submit import submit_training_task
    controller, state, _, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'explicit-id')
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)
    try:
        ref = await submit_training_task(queue, AsyncMock(return_value='ok'), training_id=identity, seq_id=10,
                                        stream='tinker::step', token='owner',
                                        task_type='step', request_id='explicit-id')
        assert ref['request_id'] == 'explicit-id'
        assert (await settle(state, 'explicit-id'))['status'] == 'completed'
    finally:
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_shutdown_waits_for_late_serve_deployment_before_confirming_release(monkeypatch):
    import importlib
    import threading
    from types import SimpleNamespace
    from twinkle.server.config.training import TrainingConfig
    from twinkle.server.state.training_records import InstanceRecord
    from twinkle.server.model.pool.driver import RayInstanceDriver
    from ray import serve
    config = TrainingConfig(models=[{'name': 'small', 'model_id': 'weights', 'backend': 'mock', 'device_type': 'cpu', 'mode': 'lora', 'config': {'num_devices': 1, 'device_mesh': {'dp_size': 1}}}])
    store = SimpleNamespace(models=SimpleNamespace(unregister_replica_capacity=AsyncMock()))
    driver = RayInstanceDriver(config, store, 'test-late', 'controller')
    instance = InstanceRecord(instance_id='instance', controller_epoch='epoch', base_model='small',
                              mode='lora', startup_deadline=0)
    entered, release = threading.Event(), threading.Event()
    remove = AsyncMock()

    def deploy(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError('test did not release Serve')

    monkeypatch.setattr('twinkle.server.model.app.build_model_app', lambda **kwargs: 'app')
    monkeypatch.setattr(importlib.import_module('ray.util.placement_group'), 'get_placement_group', lambda name: 'pg')
    monkeypatch.setattr('ray.util.list_named_actors', lambda **kwargs: [])
    monkeypatch.setattr(serve, 'run', deploy)
    monkeypatch.setattr(serve, 'delete', lambda name: None)
    monkeypatch.setattr(driver.runtime.pool, 'remove', remove)
    deploying = asyncio.create_task(driver.deploy(instance, driver.config.get('small', 'lora')))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        deploying.cancel()
        await asyncio.gather(deploying, return_exceptions=True)
        closing = asyncio.create_task(driver.shutdown(instance))
        await asyncio.sleep(0.01)
        assert not closing.done()
        remove.assert_not_awaited()
        store.models.unregister_replica_capacity.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(deploying, return_exceptions=True)
        if closing is not None:
            await closing
        await driver.runtime.wait_for_cleanup()
    assert not driver.runtime.deployments and not driver.runtime.lifecycle.jobs
    store.models.unregister_replica_capacity.assert_awaited()


@pytest.mark.asyncio
async def test_capacity_unregister_does_not_retire_managed_model():
    controller, state, _, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'capacity-only')
    instance = controller.instances[controller.trainings[identity].instance_id]
    await state.register_replica(instance.instance_id, 5)
    await state.models.unregister_replica_capacity(instance.instance_id)
    assert await state.get_model_metadata(identity) is not None
    assert await state._backend.get('token_count::owner') == 1
    await controller.release(identity, 'owner')
    assert await state.get_model_metadata(identity) is None
    assert await state._backend.get('token_count::owner') == 0
    await controller.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['lora', 'full'])
@pytest.mark.parametrize('failure_stage', ['stopped', 'quarantined', 'persist_draining'])
async def test_late_release_ack_cannot_retire_a_failed_instance_user(mode, failure_stage, monkeypatch):
    from twinkle.server.config.training import TrainingConfig

    controller, state, runtime, _ = await setup(backend=FakeBackend())
    config = controller.config.model_dump()
    next(model['config'] for model in config['models'] if model['mode'] == mode).update(min_replicas=1, max_replicas=1)
    controller.config = TrainingConfig.model_validate(config)
    await controller.sweep()
    await controller.wait_for_jobs()
    identity = await create(controller, state, 'late-release', mode=mode)
    training = controller.trainings[identity]
    instance = controller.instances[training.instance_id]
    entered, resume = asyncio.Event(), asyncio.Event()
    release = runtime.release_training

    async def delayed_ack(target, user):
        await release(target, user)
        entered.set()
        await resume.wait()

    runtime.release_training = delayed_ack
    training.state, training.terminal_reason = 'CLOSING', 'released'
    await controller.store.save_training(training)
    cleanup = asyncio.create_task(controller._cleanup_user(training))
    save_instance = controller.store.save_instance

    async def unavailable(record):
        if record.state == 'DRAINING':
            raise ConnectionError('draining write unavailable')
        await save_instance(record)

    try:
        await asyncio.wait_for(entered.wait(), 1)
        runtime.fail_shutdown = failure_stage == 'quarantined'
        failure = FutureFailureRecord(reason_code='orphaned_replica', attribution='server', message='Worker died')
        if failure_stage == 'persist_draining':
            monkeypatch.setattr(controller.store, 'save_instance', unavailable)
            with pytest.raises(ConnectionError):
                await controller.fail_instance(instance.instance_id, failure)
        else:
            await controller.fail_instance(instance.instance_id, failure)
        state_before_ack = instance.state
        resume.set()
        await asyncio.wait_for(cleanup, 1)
        assert instance.state == state_before_ack
        if failure_stage == 'stopped':
            assert training.state == 'FAILED'
            assert not runtime.allocations
        else:
            assert training.state == 'CLOSING'
            assert identity in instance.training_ids
            assert await state.get_model_metadata(identity) is not None
            assert await state._backend.get('token_count::owner') == 1
        with pytest.raises(ResourceNotFoundError):
            await controller.route(identity, 'owner')
    finally:
        resume.set()
        await asyncio.gather(cleanup, return_exceptions=True)
        runtime.fail_shutdown = False
        monkeypatch.setattr(controller.store, 'save_instance', save_instance)
        await controller.shutdown()
