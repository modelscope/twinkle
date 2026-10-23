# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lifecycle protection through the actual compute worker and backend thread."""
import asyncio
import threading
from types import SimpleNamespace
from tests.server.training.helpers import RPC, RuntimeQueue, settle

import pytest

from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.task_queue import TaskQueueMixin
from twinkle.server.model.runtime import TrainingRuntime
from twinkle.server.lifecycle.submit import submit_training_task
from tests.server.training.helpers import create, setup


@pytest.mark.asyncio
async def test_execution_timeout_keeps_live_thread_and_allocation_quarantined():
    controller, state, runtime, _ = await setup(devices=2)
    identity = await create(controller, state, 1)
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)
    entered, unblock = threading.Event(), threading.Event()
    runtime.fail_shutdown = True
    def blocking():
        entered.set()
        unblock.wait(3)
    async def operation():
        await queue.call_backend(blocking)
    try:
        ref = await queue.schedule_task(operation, model_id=identity, token='owner', task_type='forward_backward')
        future = await settle(state, ref['request_id'])
        assert entered.is_set() and not unblock.is_set()
        assert future['failure']['reason_code'] == 'execution_timeout'
        for _ in range(100):
            if instance.state == 'QUARANTINED':break
            await asyncio.sleep(0.01)
        assert instance.state == 'QUARANTINED' and runtime.allocations
        assert controller.trainings[identity].pending_operation_ids
        assert queue._model_unhealthy
    finally:
        unblock.set()
        await queue.shutdown_task_queue()
        runtime.fail_shutdown = False
        await controller.sweep()
        await controller.wait_for_jobs()
    assert not runtime.allocations


@pytest.mark.asyncio
async def test_tinker_seq_retry_runs_once_and_different_operations_do_not_collide():
    controller, state, runtime, _ = await setup()
    identity = await create(controller, state, 1)
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)
    counter = []
    async def operation():
        counter.append('executed')
        return {'ok': True}
    try:
        first, retry = await asyncio.gather(*(
            submit_training_task(queue, operation, training_id=identity, seq_id=1,
                                 stream='tinker::forward_backward', token='owner', task_type='forward_backward')
            for _ in range(2)))
        assert first == retry
        assert (await settle(state, first['request_id']))['status'] == 'completed'
        step = await submit_training_task(queue, operation, training_id=identity, seq_id=1,
                                          stream='tinker::optim_step', token='owner', task_type='optim_step')
        assert step['request_id'] != first['request_id']
        await settle(state, step['request_id'])
        assert len(counter) == 2
    finally:
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_cancelled_queued_operation_releases_ticket_without_business_renewal(monkeypatch):
    import time
    # Freeze wall time as well as the injected business clock; RPC latency is not activity.
    now = time.time()
    monkeypatch.setattr('twinkle.server.task_queue.worker.time.time', lambda: now)
    controller, state, runtime, clock = await setup()
    identity = await create(controller, state, 1)
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)
    queue._task_queue_config = TaskQueueConfig(execution_timeout=2)
    queue._compute_worker._config = queue._task_queue_config
    entered, unblock = asyncio.Event(), asyncio.Event()
    executions = []
    async def busy():
        entered.set()
        await unblock.wait()
    async def cancelled():
        executions.append('must not run')
    try:
        first = await queue.schedule_task(busy, model_id=identity, token='owner', task_type='forward_backward')
        await entered.wait()
        second = await queue.schedule_task(cancelled, model_id=identity, token='owner', task_type='step')
        assert (await state.cancel_future(second['request_id']))['cancelled']
        clock.now = 9
        unblock.set()
        await settle(state, first['request_id'])
        for _ in range(100):
            if not controller.trainings[identity].pending_operation_ids:break
            await asyncio.sleep(0.01)
        assert not executions and not controller.trainings[identity].pending_operation_ids
        assert controller.trainings[identity].last_activity_at == 9
        assert (await state.get_future(second['request_id']))['status'] == 'cancelled'
    finally:
        unblock.set()
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_readiness_waits_for_real_ray_worker_constructor():
    import ray
    import time
    @ray.remote(num_cpus=0.1)
    class Worker:
        def __init__(self):
            time.sleep(0.3)
            self.loaded = True
        def is_loaded(self):
            return self.loaded
    worker = Worker.remote()
    runtime = TrainingRuntime()
    runtime.model = SimpleNamespace(_actors=[worker])
    try:
        await runtime.wait_for_training_workers()
        assert await worker.is_loaded.remote()
        assert ray._private.state.actors(worker._actor_id.hex())['State'] == 'ALIVE'
    finally:
        ray.kill(worker, no_restart=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['heartbeat', 'max_lifetime'])
async def test_adapter_expiry_preserves_already_queued_work(reason):
    import time
    from datetime import datetime, timedelta, timezone
    from tests.server.state.fake_backend import FakeBackend

    controller, state, runtime, _ = await setup(
        backend=FakeBackend(), adapter_config={'adapter_timeout': 30, 'adapter_max_lifetime': 60})
    identity = await create(controller, state, 'queued-expiry')
    training = controller.trainings[identity]
    queue = RuntimeQueue(controller, state, controller.instances[training.instance_id])
    queue._task_queue_config = TaskQueueConfig(execution_timeout=5)
    queue._compute_worker._config = queue._task_queue_config
    entered, resume, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    executions = []

    async def first():
        entered.set()
        await resume.wait()
        executions.append('first')

    async def second():
        executions.append('second')

    async def notify(*args, **kwargs):
        await controller.finish_operation(*args, **kwargs)
        if not training.pending_operation_ids:
            finished.set()
        return True

    queue._training_controller.finish_operation = RPC(notify)
    try:
        active = await queue.schedule_task(first, model_id=identity, token='owner', task_type='forward_backward')
        await asyncio.wait_for(entered.wait(), 2)
        queued = await queue.schedule_task(second, model_id=identity, token='owner', task_type='step')
        if reason == 'heartbeat':
            session = await state.sessions.get('session')
            session.last_heartbeat = time.time() - 31
            await state.sessions.add('session', session)
        else:
            training.created_at = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
        await controller.sweep()
        await controller.wait_for_jobs()
        assert identity in runtime.users and len(training.pending_operation_ids) == 2
        resume.set()
        assert (await settle(state, active['request_id']))['status'] == 'completed'
        assert (await settle(state, queued['request_id']))['status'] == 'completed'
        await asyncio.wait_for(finished.wait(), 2)
        await controller.sweep()
        await controller.wait_for_jobs()
        assert executions == ['first', 'second']
        assert training.state == 'EXPIRED' and not runtime.allocations
    finally:
        resume.set()
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_lost_finish_notification_is_reconciled_without_stopping_compute():
    controller, state, runtime, clock = await setup()
    identity = await create(controller, state, 'lost-notification')
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)
    async def lost(*args, **kwargs):
        raise RuntimeError('Completion RPC lost')
    queue._training_controller.finish_operation = RPC(lost)
    async def status(instance, request_id):
        return await queue.verify_operation(request_id)
    runtime.verify_operation = status
    async def compute():
        return {'updated': True}
    try:
        ref = await queue.schedule_task(compute, model_id=identity, token='owner', task_type='step')
        assert (await settle(state, ref['request_id']))['status'] == 'completed'
        assert controller.trainings[identity].pending_operation_ids
        clock.now = 6
        await controller.sweep()
        await controller.wait_for_jobs()
        assert not controller.trainings[identity].pending_operation_ids
        assert controller.trainings[identity].state == 'ACTIVE' and not queue._model_unhealthy
        clock.now += 11
        await controller.sweep()
        await controller.wait_for_jobs()
        assert not runtime.allocations
    finally:
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault,executed', [('queue_timeout', False), ('unknown_start_ack', False),
                                         ('backend_error', True)])
async def test_lost_failure_notification_uses_actual_execution_facts(fault, executed, monkeypatch):
    from tests.server.state.fake_backend import FakeBackend
    controller, state, runtime, clock = await setup(backend=FakeBackend())
    identity = await create(controller, state, fault)
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)

    async def lost(*args, **kwargs):
        raise ConnectionError('Completion RPC lost')

    queue._training_controller.finish_operation = RPC(lost)
    runtime.verify_operation = lambda instance, request_id: queue.verify_operation(request_id)
    ran = []

    async def operation():
        ran.append(True)
        raise ValueError('executed backend error')

    if fault == 'queue_timeout':
        queue._compute_worker._config = TaskQueueConfig(queue_timeout=0)
    elif fault == 'unknown_start_ack':
        start = state.try_start_future

        async def lose_ack(request_id):
            assert await start(request_id)
            raise ConnectionError('start acknowledgement lost')

        monkeypatch.setattr(state, 'try_start_future', lose_ack)
    try:
        clock.now = 7
        ref = await queue.schedule_task(operation, model_id=identity, token='owner', task_type='step')
        assert (await settle(state, ref['request_id']))['status'] == 'failed'
        for _ in range(100):
            if not queue._compute_worker.has_task(ref['request_id']):
                break
            await asyncio.sleep(0.01)
        outcome = await queue.verify_operation(ref['request_id'])
        assert outcome['state'] == 'finished' and outcome['executed'] is executed
        assert bool(ran) is executed
        assert controller.trainings[identity].pending_operation_ids
        clock.now = 13  # Admission at 7 must age past the five-second reconciliation interval.
        await controller.sweep()
        await controller.wait_for_jobs()
        training = controller.trainings[identity]
        assert not training.pending_operation_ids
        assert (training.last_activity_at > 6) is executed
        if not executed:
            clock.now = 14
            await controller.sweep()
            await controller.wait_for_jobs()
            assert training.state == 'EXPIRED'
    finally:
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_execution_facts_expire_without_future_status_guessing(monkeypatch):
    from tests.server.state.fake_backend import FakeBackend
    controller, state, runtime, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'facts-expire')
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)

    async def lost(*args, **kwargs):
        raise ConnectionError('Completion RPC lost')

    queue._training_controller.finish_operation = RPC(lost)
    try:
        ref = await queue.schedule_task(lambda: asyncio.sleep(0), model_id=identity, token='owner', task_type='step')
        await settle(state, ref['request_id'])
        worker = queue._compute_worker
        for _ in range(100):
            if not worker.has_task(ref['request_id']):
                break
            await asyncio.sleep(0.01)
        outcome = worker.get_task_outcome(ref['request_id'])
        assert outcome['executed']
        monkeypatch.setattr('twinkle.server.task_queue.worker.time.time',
                            lambda: outcome['finished_at'] + worker._config.absolute_future_ttl(1) + 1)
        assert (await queue.verify_operation(ref['request_id']))['state'] == 'missing'
        assert not worker._task_outcomes
    finally:
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_completion_write_failure_retains_ticket_and_execution_fact(monkeypatch):
    from tests.server.state.fake_backend import FakeBackend
    controller, state, runtime, clock = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'finish-write')
    instance = controller.instances[controller.trainings[identity].instance_id]
    queue = RuntimeQueue(controller, state, instance)
    runtime.verify_operation = lambda instance, request_id: queue.verify_operation(request_id)
    save = controller.store.save_training
    failed = False

    async def fail_completion(record):
        nonlocal failed
        if record.state == 'ACTIVE' and not record.pending_operation_ids and not failed:
            failed = True
            raise ConnectionError('completion write failed')
        await save(record)

    monkeypatch.setattr(controller.store, 'save_training', fail_completion)
    try:
        clock.now = 7
        ref = await queue.schedule_task(lambda: asyncio.sleep(0), model_id=identity, token='owner', task_type='step')
        await settle(state, ref['request_id'])
        for _ in range(100):
            if not queue._compute_worker.has_task(ref['request_id']):
                break
            await asyncio.sleep(0.01)
        for _ in range(100):
            if failed:
                break
            await asyncio.sleep(0.01)
        training = controller.trainings[identity]
        assert failed and ref['request_id'] in training.pending_operation_ids
        assert training.last_activity_at == 0
        assert queue._compute_worker.get_task_outcome(ref['request_id'])['executed']
        clock.now = 13
        await controller.sweep()
        await controller.wait_for_jobs()
        assert not training.pending_operation_ids and 6 < training.last_activity_at <= 13
        followup = await queue.schedule_task(lambda: asyncio.sleep(0), model_id=identity, token='owner', task_type='step')
        await settle(state, followup['request_id'])
        for _ in range(100):
            if queue._compute_worker.get_task_outcome(followup['request_id']) is None:
                break
            await asyncio.sleep(0.01)
        assert queue._compute_worker.get_task_outcome(followup['request_id']) is None
    finally:
        await queue.shutdown_task_queue()
        await controller.shutdown()
