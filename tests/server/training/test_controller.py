# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise real controller/storage decisions against a controllable runtime."""
from __future__ import annotations

import asyncio

import pytest

from twinkle.server.config.training import TrainingConfig
from twinkle.server.exceptions import RequestRejectedError, ResourceNotFoundError
from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import RayActorBackend
from twinkle.server.training.controller import TrainingController
from twinkle.server.training.errors import CapacityUnavailableError, TrainingExpiredError
from twinkle.server.training.records import CreateTraining
from twinkle.server.training.store import TrainingStore


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

    async def start(self, instance, spec):
        self.starts += 1
        if self.fail_start:
            raise RuntimeError('injected startup failure')
        used = sum(self.allocations.values())
        if used + spec.config.num_devices > self.devices:
            raise CapacityUnavailableError('No devices available')
        self.allocations[instance.instance_id] = spec.config.num_devices
        await self.start_gate.wait()
        return {}

    async def create_training(self, instance, training, request):
        assert instance.instance_id in self.allocations
        assert training.training_id not in self.users
        self.users[training.training_id] = {'instance': instance.instance_id, 'label': training.adapter_label, 'steps': 0}

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


@pytest.mark.asyncio
async def test_initialization_timeout_only_retires_the_failing_tenant():
    from tests.server.state.fake_backend import FakeBackend
    controller, state, runtime, _ = await setup(backend=FakeBackend())
    active = await create(controller, state, 'active')
    runtime.users[active]['steps'] = 7
    instance = controller.instances[controller.trainings[active].instance_id]
    async def timeout(instance, training, request):
        raise TimeoutError('User initialization timed out')
    runtime.create_training = timeout
    ref = await controller.create(request('failing'))
    await controller.wait_for_jobs()
    assert (await state.get_future(ref['request_id']))['failure']['reason_code'] == 'startup_timeout'
    assert controller.trainings[active].state == 'ACTIVE'
    assert runtime.users[active]['steps'] == 7
    assert instance.state == 'READY' and instance.training_ids == [active]
    await controller.shutdown()


@pytest.mark.asyncio
async def test_late_initializer_keeps_its_slot_reserved_until_serial_cleanup():
    from tests.server.state.fake_backend import FakeBackend
    controller, state, runtime, _ = await setup(backend=FakeBackend(), max_replicas=1)
    active = await create(controller, state, 'active')
    raw = controller.config.model_dump()
    raw['startup_timeout_s'] = 0.03
    raw['models'][0]['modes']['lora']['max_loras'] = 2
    controller.config = TrainingConfig.model_validate(raw)
    from twinkle.server.training.catalog import TrainingCatalog
    controller.catalog = TrainingCatalog(controller.config)
    entered, unblock = asyncio.Event(), asyncio.Event()
    original_create, original_release = runtime.create_training, runtime.release_training
    initializer = None
    async def initialize(instance, training, req):
        nonlocal initializer
        async def late():
            entered.set()
            await unblock.wait()
            await original_create(instance, training, req)
        initializer = asyncio.create_task(late())
        await asyncio.shield(initializer)
    async def release(instance, training):
        await initializer
        await original_release(instance, training)
    runtime.create_training, runtime.release_training = initialize, release
    ref = await controller.create(request('late'))
    await entered.wait()
    for _ in range(100):
        future = await state.get_future(ref['request_id'])
        if future['status'] == 'failed':
            break
        await asyncio.sleep(0.005)
    assert future['failure']['reason_code'] == 'startup_timeout'
    instance = controller.instances[controller.trainings[active].instance_id]
    assert len(instance.training_ids) == 2 and controller.trainings[active].state == 'ACTIVE'
    refused = await controller.create(request('no-free-slot'))
    assert (await state.get_future(refused['request_id']))['failure']['reason_code'] == 'capacity_unavailable'
    unblock.set()
    await controller.wait_for_jobs()
    assert instance.training_ids == [active] and set(runtime.users) == {active}
    runtime.create_training, runtime.release_training = original_create, original_release
    replacement = await create(controller, state, 'replacement')
    assert controller.trainings[replacement].instance_id == instance.instance_id
    await controller.shutdown()


@pytest.mark.asyncio
async def test_user_gets_full_initialization_window_after_base_model_startup():
    from tests.server.state.fake_backend import FakeBackend
    controller, state, runtime, clock = await setup(backend=FakeBackend())
    runtime.start_gate.clear()
    ref = await controller.create(request('near-startup-deadline'))
    instance = next(iter(controller.instances.values()))
    clock.now = instance.startup_deadline - 1
    original = runtime.create_training
    async def initialize(instance, training, req):
        clock.now += 2
        await original(instance, training, req)
    runtime.create_training = initialize
    runtime.start_gate.set()
    await controller.wait_for_jobs()
    assert (await state.get_future(ref['request_id']))['status'] == 'completed'
    await controller.shutdown()


@pytest.mark.asyncio
async def test_resident_rebuild_does_not_block_expiry_or_another_mode():
    from tests.server.state.fake_backend import FakeBackend
    from twinkle.server.training.catalog import TrainingCatalog
    controller, state, runtime, clock = await setup(backend=FakeBackend())
    expired = await create(controller, state, 'expired-full', mode='full')
    raw = controller.config.model_dump()
    raw['models'][0]['modes']['lora']['min_replicas'] = 1
    controller.config = TrainingConfig.model_validate(raw)
    controller.catalog = TrainingCatalog(controller.config)
    runtime.start_gate.clear()
    await asyncio.wait_for(controller.sweep(), 0.1)
    for _ in range(20):
        await asyncio.sleep(0)
        if runtime.starts == 2:
            break
    clock.now = 11
    await asyncio.wait_for(controller.sweep(), 0.1)
    for _ in range(20):
        await asyncio.sleep(0)
        if controller.trainings[expired].state == 'EXPIRED':
            break
    assert controller.trainings[expired].state == 'EXPIRED'
    ref = await controller.create(request('unrelated-full', mode='full'))
    assert (await state.get_future(ref['request_id']))['status'] == 'pending'
    resident_ids = [item.instance_id for item in controller.instances.values() if item.resident]
    assert len(resident_ids) == 1
    await controller.sweep()
    assert [item.instance_id for item in controller.instances.values() if item.resident] == resident_ids
    runtime.start_gate.set()
    await controller.wait_for_jobs()
    assert (await state.get_future(ref['request_id']))['status'] == 'completed'
    await controller.shutdown()


@pytest.mark.asyncio
async def test_reconciliation_of_committed_completion_does_not_extend_idle_time():
    import time
    from tests.server.state.fake_backend import FakeBackend
    controller, state, runtime, clock = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'active')
    await controller.admit(identity, 'owner', 'session', 'lost-finish')
    await controller.commit_operation(identity, 'lost-finish', controller.epoch)
    async def finished(instance, request_id):
        return {'state': 'finished', 'executed': True, 'finished_at': time.time() - 31}
    runtime.verify_operation = finished
    clock.now = 31
    await controller.sweep()
    await controller.wait_for_jobs()
    assert not controller.trainings[identity].pending_operation_ids
    assert controller.trainings[identity].last_activity_at < 0.01
    await controller.sweep()
    await controller.wait_for_jobs()
    assert not runtime.allocations


@pytest.mark.asyncio
async def test_legacy_session_and_max_lifetime_bounds_keep_their_original_meaning():
    import time
    from datetime import datetime, timezone
    from tests.server.state.fake_backend import FakeBackend
    from twinkle.server.training.catalog import TrainingCatalog
    controller, state, runtime, clock = await setup(backend=FakeBackend())
    raw = controller.config.model_dump()
    raw['models'][0]['modes']['lora']['idle_timeout_s'] = None
    raw['models'][0]['legacy_expiration'] = {'session_timeout_s': 30, 'max_lifetime_s': 36000}
    controller.config = TrainingConfig.model_validate(raw)
    controller.catalog = TrainingCatalog(controller.config)
    identity = await create(controller, state, 'legacy')
    clock.now = 1801
    await state.touch_session('session')
    await controller.sweep()
    await controller.wait_for_jobs()
    assert controller.trainings[identity].state == 'ACTIVE'
    controller.trainings[identity].created_at = datetime.fromtimestamp(time.time() - 36001, timezone.utc).isoformat()
    await controller.sweep()
    await controller.wait_for_jobs()
    assert controller.trainings[identity].state == 'EXPIRED'
    second = await create(controller, state, 'stale-heartbeat')
    session = await state._backend.get('session::session')
    session['last_heartbeat'] = time.time() - 31
    await state._backend.set('session::session', session)
    await controller.sweep()
    await controller.wait_for_jobs()
    assert controller.trainings[second].state == 'EXPIRED'


async def setup(*, devices=16, min_replicas=0, max_replicas=None, quota=100, backend=None):
    mode = {'num_devices': 2, 'device_mesh': {'fsdp_size': 2}, 'max_loras': 5,
            'idle_timeout_s': 10, 'min_replicas': min_replicas, 'max_replicas': max_replicas}
    config = TrainingConfig(models=[{
        'name': 'small', 'model_id': 'ms://small', 'backend': 'mock', 'device_type': 'cuda',
        'modes': {'lora': mode, 'full': {'num_devices': 2, 'device_mesh': {'fsdp_size': 2}, 'idle_timeout_s': 10}},
    }])
    state = ServerState(backend=backend or RayActorBackend(), per_token_model_limit=quota)
    await state.create_session({'session_id': 'session'})
    runtime, clock = Runtime(devices), Clock()
    controller = TrainingController(config, TrainingStore(state, 'unit'), runtime, clock=clock)
    await controller.initialize()
    return controller, state, runtime, clock


def request(key, *, mode='lora', label='same', token='owner'):
    return CreateTraining(base_model='small', mode=mode, token=token, session_id='session', create_key=str(key),
                          client_type='twinkle', payload={}, adapter_label=label)


async def create(controller, state, key, **kwargs):
    ref = await controller.create(request(key, **kwargs))
    await controller.wait_for_jobs()
    future = await state.get_future(ref['request_id'])
    assert future['status'] == 'completed', future
    return future['result']['training_id']


@pytest.mark.asyncio
async def test_lazy_queries_allocate_nothing():
    controller, state, runtime, _ = await setup()
    assert runtime.starts == 0
    assert controller.catalog.get('small', 'full').capacity == 1
    with pytest.raises(ResourceNotFoundError):
        await controller.route('unknown', 'owner')
    assert not runtime.allocations
    assert await state.get_capacity_info() == {'max_loras': 0, 'used_loras': 0, 'free_loras': 0}


@pytest.mark.asyncio
async def test_32_concurrent_creates_coalesce_even_before_allocation_ready():
    controller, state, runtime, _ = await setup()
    runtime.start_gate.clear()
    refs = await asyncio.gather(*(controller.create(request(index)) for index in range(32)))
    assert len(controller.instances) == 7
    assert sorted(len(instance.training_ids) for instance in controller.instances.values()) == [2, 5, 5, 5, 5, 5, 5]
    runtime.start_gate.set()
    await controller.wait_for_jobs()
    records = [await state.get_future(ref['request_id']) for ref in refs]
    assert all(record['status'] == 'completed' for record in records)
    assert len({record['result']['training_id'] for record in records}) == 32
    assert runtime.starts == 7
    assert len(runtime.users) == 32


@pytest.mark.asyncio
async def test_full_and_lora_isolation_and_device_reuse():
    controller, state, runtime, _ = await setup(devices=6)
    shared = await create(controller, state, 1)
    full1 = await create(controller, state, 2, mode='full')
    full2 = await create(controller, state, 3, mode='full')
    identities = {runtime.users[key]['instance'] for key in (shared, full1, full2)}
    assert len(identities) == 3
    old_full = runtime.users[full1]['instance']
    await controller.release(full1, 'owner')
    replacement = await create(controller, state, 4, mode='full')
    assert runtime.users[replacement]['instance'] != old_full
    assert shared in runtime.users and full2 in runtime.users
    assert len(runtime.allocations) == 3


@pytest.mark.asyncio
async def test_capacity_failure_is_terminal_and_does_not_hold_quota():
    controller, state, runtime, _ = await setup(devices=2)
    await create(controller, state, 1, mode='full')
    ref = await controller.create(request(2, mode='full'))
    await controller.wait_for_jobs()
    record = await state.get_future(ref['request_id'])
    assert record['status'] == 'failed'
    assert record['failure']['reason_code'] == 'capacity_unavailable'
    assert await state._backend.get('token_count::owner') == 1
    assert len(runtime.allocations) == 1
    assert await controller.create(request(2, mode='full')) == ref


@pytest.mark.asyncio
async def test_create_idempotency_and_changed_payload_conflict():
    controller, state, runtime, _ = await setup()
    refs = await asyncio.gather(*(controller.create(request('one')) for _ in range(16)))
    assert len({ref['request_id'] for ref in refs}) == 1
    await controller.wait_for_jobs()
    assert len(runtime.users) == 1
    changed = request('one')
    changed.payload = {'different': True}
    with pytest.raises(RequestRejectedError, match='different parameters'):
        await controller.create(changed)


@pytest.mark.asyncio
async def test_pending_creates_count_toward_quota():
    controller, state, runtime, _ = await setup(quota=1)
    runtime.start_gate.clear()
    first = await controller.create(request(1))
    second = await controller.create(request(2))
    failure = await state.get_future(second['request_id'])
    assert failure['failure']['reason_code'] == 'resource_quota_exceeded'
    runtime.start_gate.set()
    await controller.wait_for_jobs()
    assert (await state.get_future(first['request_id']))['status'] == 'completed'
    assert len(runtime.users) == 1


@pytest.mark.asyncio
async def test_heartbeat_does_not_prevent_idle_expiry_or_affect_busy_user():
    controller, state, runtime, clock = await setup()
    idle = await create(controller, state, 1)
    busy = await create(controller, state, 2)
    await controller.admit(busy, 'owner', 'session', 'op')
    await controller.commit_operation(busy, 'op', controller.epoch)
    clock.now = 11
    await state.touch_session('session')
    await controller.sweep()
    await controller.wait_for_jobs()
    assert idle not in runtime.users and busy in runtime.users
    with pytest.raises(TrainingExpiredError):
        await controller.route(idle, 'owner')
    await controller.finish_operation(busy, 'op', controller.epoch, executed=True)
    assert controller.trainings[busy].last_activity_at == 11
    clock.now = 20
    await controller.sweep()
    await controller.wait_for_jobs()
    assert busy in runtime.users
    clock.now = 22
    await controller.sweep()
    await controller.wait_for_jobs()
    assert not runtime.allocations


@pytest.mark.asyncio
async def test_release_drains_accepted_operation_before_cleanup():
    controller, state, runtime, _ = await setup()
    identity = await create(controller, state, 1)
    await controller.admit(identity, 'owner', 'session', 'op')
    await controller.commit_operation(identity, 'op', controller.epoch)
    releasing = asyncio.create_task(controller.release(identity, 'owner'))
    while controller.trainings[identity].state == 'ACTIVE':
        await asyncio.sleep(0)
    assert identity in runtime.users
    with pytest.raises(ResourceNotFoundError):
        await controller.admit(identity, 'owner', 'session', 'next')
    await controller.finish_operation(identity, 'op', controller.epoch, executed=True)
    await controller.finish_operation(identity, 'op', controller.epoch, executed=True)
    await controller.wait_for_jobs()
    await releasing
    assert not runtime.users and not runtime.allocations
    assert len(runtime.stopped) == 1


@pytest.mark.asyncio
async def test_same_label_recreation_and_ownership_do_not_reattach_old_handle():
    controller, state, runtime, _ = await setup()
    old = await create(controller, state, 1)
    with pytest.raises(ResourceNotFoundError):
        await controller.route(old, 'other')
    with pytest.raises(ResourceNotFoundError):
        await controller.route(old, 'owner', 'other-session')
    await controller.release(old, 'owner')
    new = await create(controller, state, 2)
    assert new != old and runtime.users[new]['label'] == 'same'
    with pytest.raises(ResourceNotFoundError):
        await controller.route(old, 'owner')


@pytest.mark.asyncio
async def test_resident_survives_empty_but_extra_replica_does_not():
    controller, state, runtime, _ = await setup(min_replicas=1)
    refs = [await controller.create(request(index)) for index in range(6)]
    await controller.wait_for_jobs()
    identities = [(await state.get_future(ref['request_id']))['result']['training_id'] for ref in refs]
    for identity in identities:
        await controller.release(identity, 'owner')
    assert len(runtime.allocations) == 1
    resident = next(instance for instance in controller.instances.values() if instance.resident)
    assert resident.state == 'READY'


@pytest.mark.asyncio
async def test_startup_failure_cleans_all_waiters_and_quota():
    controller, state, runtime, _ = await setup()
    runtime.fail_start = True
    refs = await asyncio.gather(*(controller.create(request(index)) for index in range(5)))
    await controller.wait_for_jobs()
    records = [await state.get_future(ref['request_id']) for ref in refs]
    assert all(record['status'] == 'failed' for record in records)
    assert (await state._backend.get('token_count::owner') or 0) == 0
    assert not runtime.allocations


@pytest.mark.asyncio
async def test_managed_records_are_not_deleted_by_generic_training_ttl():
    controller, state, runtime, clock = await setup()
    identity = await create(controller, state, 1)
    await state._model_mgr.cleanup_expired(10**12, ['session'])
    assert await state.get_model_metadata(identity) is not None
    assert identity in runtime.users


@pytest.mark.asyncio
async def test_restart_invalidates_handles_and_cleans_old_allocation():
    controller, state, runtime, clock = await setup()
    identity = await create(controller, state, 1)
    replacement = TrainingController(controller.config, TrainingStore(state, 'unit'), runtime, clock=clock)
    await replacement.initialize()
    assert replacement.epoch != controller.epoch
    assert not runtime.allocations and not runtime.users
    assert await state.get_model_metadata(identity) is None
    assert (await state._backend.get('token_count::owner') or 0) == 0
    with pytest.raises(ResourceNotFoundError):
        await replacement.route(identity, 'owner')


@pytest.mark.asyncio
async def test_cancelled_create_never_initializes_user_and_keeps_receipt():
    controller, state, runtime, _ = await setup()
    runtime.start_gate.clear()
    ref = await controller.create(request('cancel'))
    assert (await state.cancel_future(ref['request_id']))['cancelled']
    runtime.start_gate.set()
    await controller.wait_for_jobs()
    assert (await state.get_future(ref['request_id']))['status'] == 'cancelled'
    assert not runtime.users and not runtime.allocations
    assert await controller.create(request('cancel')) == ref


@pytest.mark.asyncio
async def test_unconfirmed_worker_shutdown_holds_devices_and_quota_until_retry():
    from twinkle.server.state.models import FutureFailureRecord
    controller, state, runtime, _ = await setup(devices=2)
    identity = await create(controller, state, 1, mode='full')
    instance = controller.instances[controller.trainings[identity].instance_id]
    await controller.admit(identity, 'owner', 'session', 'running')
    await controller.commit_operation(identity, 'running', controller.epoch)
    runtime.fail_shutdown = True
    await controller.fail_instance(instance.instance_id, FutureFailureRecord(
        reason_code='execution_timeout', message='RPC timed out', attribution='server'))
    assert instance.state == 'QUARANTINED'
    assert runtime.allocations and controller.trainings[identity].pending_operation_ids
    assert await state.get_model_metadata(identity) is not None
    failed = await controller.create(request(2, mode='full'))
    await controller.wait_for_jobs()
    assert (await state.get_future(failed['request_id']))['failure']['reason_code'] == 'capacity_unavailable'
    runtime.fail_shutdown = False
    await controller.sweep()
    await controller.wait_for_jobs()
    assert not runtime.allocations
    assert instance.state == 'STOPPED'
    assert (await state._backend.get('token_count::owner') or 0) == 0


@pytest.mark.asyncio
async def test_uncertain_enqueue_is_failed_before_ticket_is_released():
    controller, state, runtime, clock = await setup()
    identity = await create(controller, state, 1)
    await controller.admit(identity, 'owner', 'session', 'lost-submit')
    await controller.store.future('lost-submit', 'pending', identity)
    async def missing(instance, request_id):
        return {'state': 'missing'}
    runtime.verify_operation = missing
    clock.now = 31
    await controller.sweep()
    await controller.wait_for_jobs()
    assert (await state.get_future('lost-submit'))['status'] == 'failed'
    assert not controller.trainings[identity].pending_operation_ids
    assert controller.trainings[identity].last_activity_at == 0
    await controller.sweep()
    await controller.wait_for_jobs()
    assert not runtime.allocations


@pytest.mark.asyncio
async def test_restart_fails_pending_operations_and_fences_old_controller():
    controller, state, runtime, clock = await setup()
    identity = await create(controller, state, 1)
    await controller.admit(identity, 'owner', 'session', 'queued')
    await controller.store.future('queued', 'queued', identity)
    replacement = TrainingController(controller.config, TrainingStore(state, 'unit'), runtime, clock=clock)
    await replacement.initialize()
    assert (await state.get_future('queued'))['failure']['reason_code'] == 'orphaned_replica'
    with pytest.raises(ResourceNotFoundError, match='epoch'):
        await controller.create(request(2))
    with pytest.raises(ResourceNotFoundError, match='epoch'):
        await controller.route(identity, 'owner')
    assert await replacement.create(request(1)) == {'request_id': controller.trainings[identity].create_request_id}
    assert not runtime.allocations


@pytest.mark.asyncio
async def test_info_does_not_renew_and_expiry_wins_before_next_sweep():
    controller, state, runtime, clock = await setup()
    identity = await create(controller, state, 1)
    clock.now = 9
    await controller.admit(identity, 'owner', 'session', 'info', renew=False)
    await controller.finish_operation(identity, 'info', controller.epoch, executed=True)
    assert controller.trainings[identity].last_activity_at == 0
    clock.now = 10
    with pytest.raises(TrainingExpiredError):
        await controller.admit(identity, 'owner', 'session', 'next')
    await controller.wait_for_jobs()
    assert not runtime.users


@pytest.mark.asyncio
async def test_full_resident_cannot_accept_user_while_resetting():
    controller, state, runtime, _ = await setup()
    original = controller.config.model_dump()
    original['models'][0]['modes']['full']['min_replicas'] = 1
    original['models'][0]['modes']['full']['max_replicas'] = 1
    controller.config = TrainingConfig.model_validate(original)
    from twinkle.server.training.catalog import TrainingCatalog
    controller.catalog = TrainingCatalog(controller.config)
    await controller.sweep()
    await controller.wait_for_jobs()
    identity = await create(controller, state, 1, mode='full')
    entered, resume = asyncio.Event(), asyncio.Event()
    previous = runtime.release_training
    async def resetting(instance, training):
        entered.set()
        await resume.wait()
        await previous(instance, training)
    runtime.release_training = resetting
    releasing = asyncio.create_task(controller.release(identity, 'owner'))
    await entered.wait()
    instance = controller.instances[controller.trainings[identity].instance_id]
    assert instance.state == 'RESETTING'
    ref = await controller.create(request(2, mode='full'))
    assert (await state.get_future(ref['request_id']))['failure']['reason_code'] == 'capacity_unavailable'
    resume.set()
    await releasing
    assert instance.state == 'READY'
    next_id = await create(controller, state, 3, mode='full')
    assert controller.trainings[next_id].instance_id == instance.instance_id


@pytest.mark.asyncio
async def test_dirty_resident_is_rebuilt_after_last_user_leaves():
    controller, state, runtime, _ = await setup(min_replicas=1)
    identity = await create(controller, state, 1)
    instance = controller.instances[controller.trainings[identity].instance_id]
    instance.dirty = True
    await controller.release(identity, 'owner')
    assert instance.state == 'STOPPED'
    await controller.sweep()
    await controller.wait_for_jobs()
    ready = [item for item in controller.instances.values() if item.state == 'READY']
    assert len(ready) == 1 and ready[0].instance_id != instance.instance_id


@pytest.mark.asyncio
@pytest.mark.parametrize('late_phase', ['runtime', 'user'])
async def test_late_startup_result_cannot_publish_success(late_phase):
    controller, state, runtime, clock = await setup()
    if late_phase == 'runtime':
        original = runtime.start
        async def late(instance, spec):
            result = await original(instance, spec)
            clock.now = instance.startup_deadline + 1
            return result
        runtime.start = late
    else:
        original = runtime.create_training
        async def late(instance, training, request):
            await original(instance, training, request)
            clock.now = training.creation_deadline + 1
        runtime.create_training = late
    ref = await controller.create(request('late'))
    await controller.wait_for_jobs()
    future = await state.get_future(ref['request_id'])
    assert future['status'] == 'failed' and future['failure']['reason_code'] == 'startup_timeout'
    assert not runtime.allocations and not runtime.users
    assert all(instance.state == 'STOPPED' for instance in controller.instances.values())


@pytest.mark.asyncio
async def test_failed_resident_startup_rolls_back_previously_loaded_residents():
    controller, state, runtime, clock = await setup(devices=2)
    raw = controller.config.model_dump()
    raw['models'][0]['modes']['lora']['min_replicas'] = 2
    replacement = TrainingController(TrainingConfig.model_validate(raw), TrainingStore(state, 'unit'), runtime, clock=clock)
    with pytest.raises(CapacityUnavailableError):
        await replacement.initialize()
    assert not replacement._ready and not runtime.allocations
    assert all(item.state == 'STOPPED' for item in replacement.instances.values())
    await replacement.wait_for_jobs()


@pytest.mark.asyncio
async def test_retired_diagnostics_are_bounded_without_discarding_live_training():
    controller, state, runtime, clock = await setup()
    old = await create(controller, state, 'old')
    await controller.release(old, 'owner', 'session')
    retired_instance = controller.trainings[old].instance_id
    live = await create(controller, state, 'live')
    controller.trainings[old].terminal_at -= state.expiration_timeout + 1
    controller.instances[retired_instance].stopped_at -= state.expiration_timeout + 1
    await controller.sweep()
    await controller.wait_for_jobs()
    assert old not in controller.trainings and retired_instance not in controller.instances
    assert retired_instance not in controller._startup
    assert (await controller.route(live, 'owner', 'session'))['instance_id'] in runtime.allocations
    await controller.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('startup_blocked', [False, True])
async def test_creation_record_write_failure_closes_new_empty_instance(monkeypatch, startup_blocked):
    controller, state, runtime, clock = await setup()
    if startup_blocked:
        runtime.start_gate.clear()
    original = state._backend.set
    failed = False
    async def failing_set(key, value, *args, **kwargs):
        nonlocal failed
        if (not failed and key.startswith('model::') and isinstance(value, dict)
                and value.get('instance_id') and value.get('state') == 'PENDING'):
            failed = True
            raise RuntimeError('injected training record write failure')
        return await original(key, value, *args, **kwargs)
    monkeypatch.setattr(state._backend, 'set', failing_set)
    ref = await controller.create(request('write-failure'))
    await controller.wait_for_jobs()
    assert (await state.get_future(ref['request_id']))['status'] == 'failed'
    assert not runtime.allocations
    assert await state._backend.get('token_count::owner') == 0
