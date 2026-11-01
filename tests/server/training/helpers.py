# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.config.training import TrainingConfig
from twinkle.server.resources.allocation import CapacityUnavailableError
from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import RayActorBackend
from twinkle.server.state.training_records import CreateTraining
from twinkle.server.task_queue import TaskQueueMixin
from twinkle.server.model.pool.controller import TrainingController
from twinkle.server.model.runtime import TrainingRuntime


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Runtime:
    """Allocates a finite pool and stores actual per-user state for assertions."""

    def __init__(self, devices=16):
        self.devices = devices
        self.allocations = {}
        self.users = {}
        self.starts = 0
        self.stopped = []
        self.start_gate = asyncio.Event()
        self.start_gate.set()
        self.fail_start = False
        self.fail_shutdown = False

    async def allocate(self, instance, spec):
        self.starts += 1
        if self.fail_start:
            raise RuntimeError('injected startup failure')
        used = sum(self.allocations.values())
        if used + spec.config.num_devices > self.devices:
            raise CapacityUnavailableError('No devices available')
        self.allocations[instance.instance_id] = spec.config.num_devices
        return {}

    async def deploy(self, instance, spec):
        await self.start_gate.wait()

    async def create_training(self, instance, training, request):
        assert instance.instance_id in self.allocations
        assert training.training_id not in self.users
        self.users[training.training_id] = {
            'instance': instance.instance_id,
            'label': training.adapter_label,
            'steps': 0
        }

    async def release_training(self, instance, training):
        self.users.pop(training.training_id, None)

    async def shutdown(self, instance):
        if self.fail_shutdown:
            raise RuntimeError('cannot confirm process death')
        self.allocations.pop(instance.instance_id, None)
        self.users = {key: value for key, value in self.users.items() if value['instance'] != instance.instance_id}
        self.stopped.append(instance.instance_id)

    async def verify_operation(self, instance, request_id):
        return {'state': 'pending'}

    async def recover_orphans(self):
        pass


async def setup(*, devices=16, min_replicas=0, max_replicas=None, quota=100, backend=None, adapter_config=None):
    mode = {
        'num_devices': 2,
        'device_mesh': {
            'fsdp_size': 2
        },
        'max_loras': 5,
        'idle_timeout_s': 10,
        'min_replicas': min_replicas,
        'max_replicas': max_replicas
    }
    if adapter_config is not None:
        mode['adapter_config'] = adapter_config
    config = TrainingConfig(models=[{
        'name': 'small',
        'model_id': 'ms://small',
        'backend': 'mock',
        'device_type': 'cuda',
        'mode': 'lora',
        'config': mode,
    }, {
        'name': 'small-full',
        'model_id': 'ms://small',
        'backend': 'mock',
        'device_type': 'cuda',
        'mode': 'full',
        'config': {
            'num_devices': 2,
            'device_mesh': {
                'fsdp_size': 2
            },
            'idle_timeout_s': 10
        },
    }])
    state = ServerState(backend=backend or RayActorBackend(), per_token_model_limit=quota)
    await state.create_session({'session_id': 'session'})
    runtime, clock = Runtime(devices), Clock()
    controller = TrainingController(config, state.create_training_store('unit'), runtime, clock=clock)
    await controller.initialize()
    return controller, state, runtime, clock


def request(key, *, mode='lora', label='same', token='owner'):
    return CreateTraining(
        base_model='small-full' if mode == 'full' else 'small',
        mode=mode,
        token=token,
        session_id='session',
        create_key=str(key),
        client_type='twinkle',
        payload={},
        adapter_label=label)


async def create(controller, state, key, **kwargs):
    ref = await controller.create(request(key, **kwargs))
    await controller.wait_for_jobs()
    future = await state.get_future(ref['request_id'])
    assert future['status'] == 'completed', future
    return future['result']['training_id']


class RPC:

    def __init__(self, method):
        self.method = method

    def remote(self, *args, **kwargs):
        return asyncio.create_task(self.method(*args, **kwargs))


class RuntimeQueue(TrainingRuntime, TaskQueueMixin):

    def __init__(self, controller, state, instance):
        self.state, self.replica_id = state, instance.instance_id
        from twinkle.server.state.training_records import TrainingRuntimeContext
        self.training_context = TrainingRuntimeContext(
            instance_id=instance.instance_id,
            epoch=controller.epoch,
            namespace='test',
            registry_name='test',
            controller_name='test')
        self._model_unhealthy = False
        self._training_controller = SimpleNamespace(
            **{name: RPC(getattr(controller, name))
               for name in ('admit', 'finish_operation', 'fail_instance')})
        self._init_task_queue(
            TaskQueueConfig(execution_timeout=0.1, inline_fast_path_timeout=0.01), enable_admission_gate=True)


async def settle(state, request_id):
    for _ in range(300):
        future = await state.get_future(request_id)
        if future and future['status'] in ('completed', 'failed', 'cancelled'):
            return future
        await asyncio.sleep(0.01)
    raise AssertionError('Future did not settle')
