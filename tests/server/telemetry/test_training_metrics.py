# Copyright (c) ModelScope Contributors. All rights reserved.
"""Managed training metrics stay low-cardinality and survive registry resets."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

from twinkle.server.telemetry import metrics
from twinkle.server.model.pool.controller import TrainingController
from twinkle.server.telemetry.metrics import TrainingInstanceMetrics

_ALLOWED_ATTRIBUTES = {'backend', 'mode', 'state', 'status', 'reason', 'device_type', 'resident'}


@pytest.fixture(autouse=True)
def reset_registry():
    yield
    metrics.MetricsRegistry.reset()


class FakeInstrument:

    def __init__(self, name):
        self.name = name
        self.calls = []

    def add(self, value, attributes=None):
        self.calls.append((value, attributes or {}))

    def record(self, value, attributes=None):
        self.calls.append((value, attributes or {}))


class FakeMeter:

    def __init__(self):
        self.instruments = {}

    def _create(self, name, **kwargs):
        instrument = FakeInstrument(name)
        self.instruments[name] = instrument
        return instrument

    create_counter = _create
    create_histogram = _create
    create_up_down_counter = _create

    def create_observable_gauge(self, name, **kwargs):
        return self._create(name, **kwargs)


def build_registry(monkeypatch):
    meter = FakeMeter()
    monkeypatch.setattr(metrics, 'get_meter', lambda name: meter)
    metrics.MetricsRegistry.reset()
    return metrics.MetricsRegistry.get(), meter


def test_training_and_redis_instruments_are_registered(monkeypatch):
    registry, meter = build_registry(monkeypatch)
    training = registry.training_metrics()
    state = registry.state_backend_metrics()

    training.instances_active.set(2, tags={'backend': 'transformers', 'mode': 'lora', 'state': 'READY',
                                           'resident': 'false'})
    training.startup_seconds.observe(1.5, tags={'backend': 'transformers', 'mode': 'lora', 'status': 'success'})
    training.quarantined_total.inc(tags={'backend': 'megatron', 'mode': 'full', 'reason': 'timeout'})
    state.redis_health.set(1)
    state.update_atomic_conflicts.inc()
    state.update_atomic_retries.observe(2)

    expected = {
        'twinkle.training.instances.active',
        'twinkle.training.instance.startup.duration_seconds',
        'twinkle.training.instance.shutdown.duration_seconds',
        'twinkle.training.placement_group.allocation.duration_seconds',
        'twinkle.training.placement_group.removal.duration_seconds',
        'twinkle.training.quarantined.total',
        'twinkle.training.sweep.total',
        'twinkle.training.sweep.duration_seconds',
        'twinkle.training.devices.allocated',
        'twinkle.training.placement_groups.active',
        'twinkle.state.redis.health',
        'twinkle.state.redis.update_atomic.conflicts.total',
        'twinkle.state.redis.update_atomic.retries',
    }
    assert expected <= meter.instruments.keys()
    for instrument in meter.instruments.values():
        for _, attributes in instrument.calls:
            assert set(attributes) <= _ALLOWED_ATTRIBUTES
            assert not {'token', 'session_id', 'training_id', 'instance_id', 'request_id'} & set(attributes)


def test_training_metric_adapters_rebind_after_registry_reset(monkeypatch):
    first, _ = build_registry(monkeypatch)
    first_metrics = first.training_metrics()
    metrics.MetricsRegistry.reset()
    second_metrics = metrics.MetricsRegistry.get().training_metrics()
    assert first_metrics is not second_metrics
    assert first_metrics.startup_seconds._instrument is not second_metrics.startup_seconds._instrument


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
def test_controller_metrics_clear_old_states_and_released_resources(monkeypatch, backend):
    registry, meter = build_registry(monkeypatch)
    spec = SimpleNamespace(backend=backend, device_type='cuda')
    controller = TrainingController.__new__(TrainingController)
    controller.config = SimpleNamespace(get=lambda base_model, mode: spec)
    controller._metrics = registry.training_metrics()
    controller._instance_metrics = TrainingInstanceMetrics(controller._metrics)
    controller.instances = {
        str(index): SimpleNamespace(
            instance_id=f'instance_{index}', base_model='private-model', mode='lora',
            state='READY', resident=False, devices=[index * 2, index * 2 + 1],
            application_name=f'private-app-{index}', training_ids=[f'training_{index}'])
        for index in range(2)
    }

    def totals(name):
        values = defaultdict(float)
        for delta, attributes in meter.instruments[name].calls:
            assert set(attributes) <= _ALLOWED_ATTRIBUTES
            assert not {'instance_id', 'base_model', 'application_name', 'training_id'} & set(attributes)
            values[tuple(sorted(attributes.items()))] += delta
        return values

    instances_name = 'twinkle.training.instances.active'
    devices_name = 'twinkle.training.devices.allocated'
    groups_name = 'twinkle.training.placement_groups.active'
    controller._report_instance_metrics()
    first_calls = {name: len(meter.instruments[name].calls)
                   for name in (instances_name, devices_name, groups_name)}
    controller._report_instance_metrics()
    assert first_calls == {name: len(meter.instruments[name].calls) for name in first_calls}
    assert sum(totals(instances_name).values()) == 2
    assert sum(totals(devices_name).values()) == 4
    assert sum(totals(groups_name).values()) == 2

    controller.instances['0'].state = 'DRAINING'
    controller._report_instance_metrics()
    state_totals = {dict(key)['state']: value for key, value in totals(instances_name).items()}
    assert state_totals == {'READY': 1, 'DRAINING': 1}
    assert sum(totals(devices_name).values()) == 4
    assert sum(totals(groups_name).values()) == 2

    for instance in controller.instances.values():
        instance.state = 'STOPPED'
    controller._report_instance_metrics()
    for name in (instances_name, devices_name, groups_name):
        assert all(value == 0 for value in totals(name).values())


@pytest.mark.asyncio
async def test_repeated_failed_cleanup_is_one_quarantine_transition():
    from unittest.mock import MagicMock
    from tests.server.state.fake_backend import FakeBackend
    from tests.server.training.helpers import create, setup

    controller, state, driver, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'metric-retry')
    instruments = MagicMock()
    controller._metrics = instruments
    driver.fail_shutdown = True
    try:
        with pytest.raises(RuntimeError, match='quarantined'):
            await controller.release(identity, 'owner', 'session')
        for _ in range(2):
            await controller.sweep()
            await controller.wait_for_jobs()
        assert instruments.quarantined_total.inc.call_count == 1
        assert instruments.shutdown_seconds.observe.call_count == 3
    finally:
        driver.fail_shutdown = False
        await controller.shutdown()


@pytest.mark.parametrize('phase', ['worker_death', 'pg_removal'])
def test_quarantine_classification_uses_typed_phase(phase):
    from twinkle.server.exceptions import ResourceReleaseError
    assert TrainingInstanceMetrics.quarantine_reason(ResourceReleaseError('unconfirmed', phase=phase)) == phase
    assert TrainingInstanceMetrics.quarantine_reason(RuntimeError('user text: worker death')) == 'other'


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_start', [False, True])
async def test_startup_and_shutdown_histograms_follow_real_attempts(fail_start):
    from unittest.mock import MagicMock
    from tests.server.state.fake_backend import FakeBackend
    from tests.server.training.helpers import setup, request

    controller, state, driver, _ = await setup(backend=FakeBackend())
    controller._metrics = MagicMock()
    driver.fail_start = fail_start
    try:
        ref = await controller.create(request('timed-start'))
        await controller.wait_for_jobs()
        future = await state.get_future(ref['request_id'])
        assert future['status'] == ('failed' if fail_start else 'completed')
        call = controller._metrics.startup_seconds.observe.call_args
        assert call.args[0] >= 0
        assert call.kwargs['tags'] == {'backend': 'mock', 'mode': 'lora',
                                       'status': 'error' if fail_start else 'success'}
        if not fail_start:
            await controller.release(future['result']['training_id'], 'owner', 'session')
        call = controller._metrics.shutdown_seconds.observe.call_args
        assert call.args[0] >= 0 and call.kwargs['tags']['status'] == 'success'
    finally:
        await controller.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('failed', [False, True])
async def test_sweep_instruments_one_iteration_and_its_outcome(monkeypatch, failed):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from twinkle.server.model.pool.actor import TrainingControllerActor

    async def stop_after_iteration(delay):
        raise asyncio.CancelledError

    actor = SimpleNamespace(core=SimpleNamespace(sweep=AsyncMock(
        side_effect=RuntimeError('state unavailable') if failed else None)),
        config=SimpleNamespace(sweep_interval_s=5), _metrics=MagicMock())
    monkeypatch.setattr('twinkle.server.model.pool.actor.asyncio.sleep', stop_after_iteration)
    with pytest.raises(asyncio.CancelledError):
        await TrainingControllerActor.__ray_metadata__.modified_class._sweep(actor)
    tags = {'status': 'error' if failed else 'success'}
    actor._metrics.sweep_total.inc.assert_called_once_with(tags=tags)
    call = actor._metrics.sweep_seconds.observe.call_args
    assert call.args[0] >= 0 and call.kwargs['tags'] == tags


@pytest.mark.asyncio
@pytest.mark.parametrize('failed', [False, True])
async def test_driver_records_allocation_attempts(failed):
    from unittest.mock import AsyncMock, MagicMock
    from twinkle.server.config.training import TrainingConfig
    from twinkle.server.resources.allocation import DeviceAllocation
    from twinkle.server.model.pool.driver import RayInstanceDriver

    config = TrainingConfig(models=[{'name': 'tiny', 'model_id': 'tiny', 'backend': 'mock',
        'device_type': 'cpu', 'mode': 'full', 'config': {'num_devices': 1, 'device_mesh': {'device_type': 'cpu'}}}])
    store = SimpleNamespace(models=SimpleNamespace(unregister_replica_capacity=AsyncMock()))
    driver = RayInstanceDriver(config, store, 'test', 'controller')
    driver._metrics = MagicMock()
    allocation = DeviceAllocation('owned', 'pg', 'node', ('0',))
    driver.runtime.pool.allocate = AsyncMock(return_value=allocation,
        side_effect=RuntimeError('allocation failed') if failed else None)
    instance = SimpleNamespace(instance_id='instance', controller_epoch='epoch', application_name='owned',
                               base_model='tiny', mode='full')
    if failed:
        with pytest.raises(RuntimeError, match='allocation failed'):
            await driver.allocate(instance, config.models[0])
    else:
        assert (await driver.allocate(instance, config.models[0]))['devices'] == ['0']
    call = driver._metrics.allocation_seconds.observe.call_args
    assert call.args[0] >= 0
    assert call.kwargs['tags'] == {'backend': 'mock', 'mode': 'full', 'device_type': 'cpu',
                                   'status': 'error' if failed else 'success'}
