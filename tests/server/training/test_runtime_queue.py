# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lifecycle protection through the actual compute worker and backend thread."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from twinkle.server.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.training.runtime import ManagedTrainingRuntime, _tinker_operation
from tests.server.training.test_controller import create, setup


class RPC:
    def __init__(self, method):
        self.method = method

    def remote(self, *args, **kwargs):
        return asyncio.create_task(self.method(*args, **kwargs))


class RuntimeQueue(ManagedTrainingRuntime, TaskQueueMixin):
    def __init__(self, controller, state, instance):
        self.state, self.replica_id = state, instance.instance_id
        self.training_context = {'instance_id': instance.instance_id, 'epoch': controller.epoch}
        self._model_unhealthy = False
        self._training_controller = SimpleNamespace(**{
            name: RPC(getattr(controller, name))
            for name in ('admit', 'commit_operation', 'finish_operation', 'fail_instance')
        })
        self._init_task_queue(TaskQueueConfig(execution_timeout=0.1, inline_fast_path_timeout=0.01),
                              enable_admission_gate=True)


async def settle(state, request_id):
    for _ in range(300):
        future = await state.get_future(request_id)
        if future and future['status'] in ('completed', 'failed', 'cancelled'):
            return future
        await asyncio.sleep(0.01)
    raise AssertionError('Future did not settle')


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
    context = _tinker_operation.set((identity, 1))
    try:
        first, retry = await asyncio.gather(*(
            queue.schedule_task(operation, model_id=identity, token='owner', task_type='forward_backward')
            for _ in range(2)))
        assert first == retry
        assert (await settle(state, first['request_id']))['status'] == 'completed'
        step = await queue.schedule_task(operation, model_id=identity, token='owner', task_type='optim_step')
        assert step['request_id'] != first['request_id']
        await settle(state, step['request_id'])
        assert len(counter) == 2
    finally:
        _tinker_operation.reset(context)
        await queue.shutdown_task_queue()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_cancelled_queued_operation_releases_ticket_without_business_renewal():
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
    runtime = ManagedTrainingRuntime()
    runtime.model = SimpleNamespace(_actors=[worker])
    try:
        await runtime.wait_for_training_workers()
        assert await worker.is_loaded.remote()
        assert ray._private.state.actors(worker._actor_id.hex())['State'] == 'ALIVE'
    finally:
        ray.kill(worker, no_restart=True)


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
