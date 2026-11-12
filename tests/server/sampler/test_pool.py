# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pool decisions against a controllable driver, without Ray or GPU side effects."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from twinkle.server.config.application_spec import SamplerArgs
from twinkle.server.sampler.pool.controller import SamplerController


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    yield


class Driver:

    def __init__(self):
        self.started = []
        self.stopped = []
        self.calls = []
        self.finish = asyncio.Event()
        self.start_gate = asyncio.Event()
        self.start_gate.set()
        self.fail_start = False
        self.fail_close = False

    async def start(self, instance):
        self.started.append(instance.identity)
        await self.start_gate.wait()
        if self.fail_start:
            raise RuntimeError('allocation failed')
        instance.devices = [str(len(self.started))]

    async def invoke(self, instance, ticket, payload, queue=None):
        self.calls.append((instance.identity, payload))
        return {'status': 200, 'headers': [], 'body': b'{"status":"pending"}'}

    async def wait_operation(self, instance, ticket):
        await self.finish.wait()

    async def load_weights(self, instance, path):
        self.calls.append((instance.identity, {'base_path': path}))

    async def shutdown(self, instance):
        if self.fail_close:
            raise RuntimeError('death unconfirmed')
        self.stopped.append(instance.identity)

    async def recover_orphans(self):
        pass


def setup(minimum=2, maximum=4, target=1):
    args = SamplerArgs(
        model_id='mock',
        sampler_type='mock',
        device_group={
            'name': 'sampler',
            'ranks': 1,
            'device_type': 'cpu'
        },
        device_mesh={'dp_size': 1},
        min_replicas=minimum,
        max_replicas=maximum,
        target_ongoing_requests=target,
        idle_timeout_s=10)
    driver = Driver()
    clock = SimpleNamespace(now=0)
    pool = SamplerController(
        args, driver, {
            'startup_timeout_s': 2,
            'shutdown_timeout_s': 1,
            'sweep_interval_s': 1
        }, clock=lambda: clock.now)
    return pool, driver, clock


def request(endpoint='twinkle/sample', body=None):
    return {
        'endpoint': endpoint,
        'body': json.dumps(body or {}).encode(),
        'method': 'POST',
        'headers': [(b'x-request-id', b'test')],
        'query_string': b''
    }


async def settle_starts(pool):
    await asyncio.gather(*tuple(pool.lifecycle.startups.values()), return_exceptions=True)


async def settle_jobs(pool):
    await asyncio.gather(*tuple(pool.lifecycle.jobs), return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_rejects_unconfirmed_release_and_allows_retry():
    pool, driver, _ = setup()
    await pool.initialize()
    driver.fail_close = True
    try:
        with pytest.raises(RuntimeError, match='shutdown.*unconfirmed'):
            await pool.shutdown()
        assert pool._owned_count() == 2 and not driver.stopped
        assert all(instance.state == 'QUARANTINED' for instance in pool.instances.values())
    finally:
        driver.fail_close = False
        await pool.shutdown()
    assert pool._owned_count() == 0 and len(driver.stopped) == 2


@pytest.mark.asyncio
async def test_minimum_expand_and_reclaim_only_idle_extras():
    pool, driver, clock = setup()
    await pool.initialize()
    assert len(driver.started) == 2
    try:
        for _ in range(8):
            await pool.request(request())
            await settle_starts(pool)
        assert len(driver.started) == 4 and pool._owned_count() == 4
        live = pool._ready_instances()
        assert all(instance.requests for instance in live)
        assert max(len(instance.requests) for instance in live) - min(len(instance.requests) for instance in live) <= 1
        clock.now = 30
        await pool.sweep()
        assert not driver.stopped
        driver.finish.set()
        await settle_jobs(pool)
        clock.now += 11
        await pool.sweep()
        await settle_jobs(pool)
        assert len(driver.stopped) == 2 and len(pool._ready_instances()) == 2
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_waiter_cancellation_preserves_generation_and_defers_control():
    pool, driver, _ = setup()
    await pool.initialize()
    await pool.request(request())
    control = asyncio.create_task(pool.request(request('twinkle/set_template', {'template_cls': 'Template'})))
    try:
        await asyncio.sleep(0)
        control.cancel()
        await asyncio.gather(control, return_exceptions=True)
        assert pool._guard._readers == 1
        assert sum(len(instance.requests) for instance in pool.instances.values()) == 1
        assert len(driver.calls) == 1
        driver.finish.set()
        await settle_jobs(pool)
        assert any(payload['endpoint'] == 'twinkle/set_template' for _, payload in driver.calls)
        assert pool._actions[0]['endpoint'] == 'twinkle/set_template'
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_new_instance_replays_controls_not_lora_weight_cache():
    pool, driver, _ = setup(minimum=1, target=1)
    driver.finish.set()
    await pool.initialize()
    try:
        await pool.request(request('twinkle/set_template', {'template_cls': 'Template'}))
        driver.finish.clear()
        await pool.request(request(body={'adapter_uri': 'checkpoint-A'}))
        await pool.request(request(body={'adapter_uri': 'checkpoint-B'}))
        driver.finish.set()
        await settle_starts(pool)
        await settle_jobs(pool)
        assert len(driver.started) == 2
        new = driver.started[-1]
        replay = [payload for identity, payload in driver.calls if identity == new]
        assert replay and all(payload['endpoint'] == 'twinkle/set_template' for payload in replay)
        assert all('adapter_uri' not in json.loads(payload['body']) for payload in pool._actions)
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_quarantine_counts_against_maximum_until_confirmed_death():
    pool, driver, _ = setup(minimum=1, maximum=1)
    await pool.initialize()
    instance = pool._ready_instances()[0]
    driver.fail_close = True
    await pool._close(instance)
    try:
        assert instance.state == 'QUARANTINED'
        for _ in range(3):
            await pool.sweep()
            await settle_jobs(pool)
        assert len(driver.started) == 1 and pool._owned_count() == 1
        driver.fail_close = False
        await pool.sweep()
        await settle_jobs(pool)
        await pool.sweep()
        await settle_starts(pool)
        assert len(driver.started) == 2 and len(pool._ready_instances()) == 1
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_startup_failure_rolls_back_loaded_minimum():
    pool, driver, _ = setup()
    original = driver.start

    async def fail_second(instance):
        if len(driver.started) == 1:
            driver.fail_start = True
        await original(instance)

    driver.start = fail_second
    with pytest.raises(RuntimeError, match='allocation'):
        await pool.initialize()
    assert not pool._ready and pool._owned_count() == 0


@pytest.mark.asyncio
async def test_lazy_first_requests_share_one_startup():
    pool, driver, _ = setup(minimum=0, maximum=1)
    await pool.initialize()
    driver.start_gate.clear()
    calls = [asyncio.create_task(pool.request(request())) for _ in range(8)]
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(driver.started) == 1
        driver.start_gate.set()
        await asyncio.wait_for(asyncio.gather(*calls), 2)
        assert len(driver.started) == 1
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['raise', 'http_error'])
async def test_partial_control_broadcast_fences_all_participants(fault):
    pool, driver, _ = setup()
    driver.finish.set()
    await pool.initialize()
    invoke = driver.invoke
    changed = []

    async def update(instance, ticket, payload, queue=None):
        changed.append(instance.identity)
        if len(changed) == 2:
            if fault == 'raise':
                raise ConnectionError('update ACK unknown')
            return {'status': 500, 'headers': [], 'body': b'error'}
        return await invoke(instance, ticket, payload, queue)

    driver.invoke = update
    try:
        if fault == 'raise':
            with pytest.raises(ConnectionError):
                await pool.request(request('twinkle/set_template'))
        else:
            reply = await pool.request(request('twinkle/set_template'))
            assert reply['status'] == 500
        await settle_jobs(pool)
        assert len(driver.stopped) == 2
        assert not pool._ready_instances() and not pool._actions
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_unknown_generation_exit_keeps_quarantined_capacity_reserved():
    pool, driver, clock = setup(minimum=1, maximum=1)
    await pool.initialize()
    driver.fail_close = True

    async def unknown(*args):
        raise TimeoutError('generation completion unknown')

    driver.wait_operation = unknown
    await pool.request(request())
    await settle_jobs(pool)
    instance = next(iter(pool.instances.values()))
    assert instance.state == 'QUARANTINED' and instance.requests
    clock.now = 100
    await pool.sweep()
    await settle_jobs(pool)
    assert len(driver.started) == 1 and instance.requests
    driver.fail_close = False
    await pool.sweep()
    await settle_jobs(pool)
    assert instance.state == 'STOPPED' and not instance.requests
    await pool.shutdown()


@pytest.mark.asyncio
async def test_replaying_config_has_the_same_startup_deadline():
    pool, driver, _ = setup(minimum=0, maximum=1)
    await pool.initialize()
    pool._actions.append(request('twinkle/set_template'))
    pool.budgets = pool.budgets.model_copy(update={'startup_timeout_s': 0.05})
    instance = pool._new_instance()
    with pytest.raises(TimeoutError):
        await pool.lifecycle.startups[instance.identity]
    assert instance.state == 'STOPPED' and driver.stopped == [instance.identity]
    await pool.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['control', 'weights'])
async def test_mutation_reserves_every_replica_against_idle_sweep(kind):
    pool, driver, clock = setup(minimum=1, maximum=2)
    driver.finish.set()
    await pool.initialize()
    pool._new_instance()
    await settle_starts(pool)
    clock.now = 20
    entered, resume = asyncio.Event(), asyncio.Event()
    invoke, load = driver.invoke, driver.load_weights

    async def update(instance, *args):
        entered.set()
        await resume.wait()
        assert instance.state == 'READY'
        return await (invoke(instance, *args) if kind == 'control' else load(instance, *args))

    if kind == 'control':
        driver.invoke = update
        task = asyncio.create_task(pool.request(request('twinkle/set_template')))
    else:
        driver.load_weights = update
        task = asyncio.create_task(pool.request(request(), full_path='B'))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await pool.sweep()
        # Only wait for the close jobs, never for the deliberately blocked mutation.
        await asyncio.gather(*tuple(pool.lifecycle.closing.values()))
        assert len(pool._ready_instances()) == 2
        assert not driver.stopped
    finally:
        resume.set()
        await asyncio.gather(task, return_exceptions=True)
        await pool.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['invoke', 'wait_operation', 'load_weights'])
async def test_control_deadline_fences_all_participants(stage):
    pool, driver, _ = setup()
    driver.finish.set()
    pool.args.queue_config.execution_timeout = 0.03
    entered, resume = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        await resume.wait()

    setattr(driver, stage, blocked)
    task = asyncio.create_task(pool.request(
        request() if stage == 'load_weights' else request('twinkle/set_template'),
        full_path='B' if stage == 'load_weights' else None))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), 0.3)
        assert task.done(), 'Configured execution deadline was not enforced'
        await settle_jobs(pool)
        assert not pool._ready_instances()
        assert len(driver.stopped) == 2
        assert not pool._actions
    finally:
        resume.set()
        await asyncio.gather(task, return_exceptions=True)
        await pool.shutdown()


@pytest.mark.asyncio
async def test_superseded_checkpoint_is_not_needed_to_start_new_replica():
    pool, driver, _ = setup(minimum=1)
    driver.finish.set()
    valid = {'A', 'B'}
    original = driver.load_weights

    async def load(instance, path):
        if path not in valid:
            raise FileNotFoundError(path)
        await original(instance, path)

    driver.load_weights = load
    try:
        for path in ('A', 'B'):
            await pool.request(request(), full_path=path)
            await settle_jobs(pool)
        valid.remove('A')
        instance = pool._new_instance()
        await pool.lifecycle.startups[instance.identity]
        assert instance.state == 'READY'
        assert [payload['base_path'] for identity, payload in driver.calls
                if identity == instance.identity and 'base_path' in payload] == ['B']
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_queued_control_expires_without_mutating_or_fencing_generation():
    pool, driver, _ = setup(minimum=1, maximum=1)
    pool.args.queue_config.queue_timeout = 0.02
    await pool.request(request())
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(pool.request(request('twinkle/set_template')), 1)
        assert len(driver.calls) == 1 and not driver.stopped
        driver.finish.set()
        await settle_jobs(pool)
        await pool.request(request('twinkle/set_template'))
        assert len(driver.calls) == 2
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_cancelled_control_waiter_keeps_replica_occupied_until_completion():
    pool, driver, clock = setup(minimum=1, maximum=2)
    await pool.initialize()
    pool._new_instance()
    await settle_starts(pool)
    entered = asyncio.Event()
    invoke = driver.invoke

    async def update(*args):
        entered.set()
        return await invoke(*args)

    driver.invoke = update
    task = asyncio.create_task(pool.request(request('twinkle/set_template')))
    try:
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        clock.now = 30
        await pool.sweep()
        assert len(pool._ready_instances()) == 2 and not driver.stopped
        driver.finish.set()
        await settle_jobs(pool)
        clock.now += 11
        await pool.sweep()
        await settle_jobs(pool)
        assert len(pool._ready_instances()) == 1
    finally:
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_control_timeout_retains_unknown_allocations_then_recovers():
    pool, driver, _ = setup(minimum=2, maximum=2)
    pool.args.queue_config.execution_timeout = 0.02
    driver.fail_close = True
    try:
        with pytest.raises(TimeoutError):
            await pool.request(request('twinkle/set_template'))
        await settle_jobs(pool)
        assert all(i.state == 'QUARANTINED' and i.requests for i in pool.instances.values())
        await pool.sweep()
        await settle_jobs(pool)
        assert len(driver.started) == 2
        driver.fail_close = False
        driver.finish.set()
        await pool.sweep()
        await settle_jobs(pool)
        await pool.sweep()
        await settle_starts(pool)
        await pool.request(request())
        assert len(driver.started) == 4
        assert len(pool._ready_instances()) == 2
    finally:
        driver.fail_close = False
        driver.finish.set()
        await pool.shutdown()


@pytest.mark.asyncio
async def test_replay_compacts_setters_per_owner_without_crossing_patch():
    pool, driver, _ = setup(minimum=1)
    driver.finish.set()
    try:
        for index in range(12):
            await pool.request(request('twinkle/set_template', {'template_cls': f'T{index}'}))
        for owner in ('one', 'two'):
            for rank in (4, 8):
                payload = request('twinkle/add_adapter_to_sampler', {'adapter_name': 'same', 'config': {'r': rank}})
                payload['headers'].append((b'twinkle-authorization', owner.encode()))
                await pool.request(payload)
        assert len(pool._actions) == 3
        await pool.request(request('twinkle/apply_patch', {'patch_cls': 'P'}))
        await pool.request(request('twinkle/set_template', {'template_cls': 'AfterPatch'}))
        await pool.request(request('twinkle/unload_adapter_paths', {'adapter_paths': ['unused']}))
        instance = pool._new_instance()
        await pool.lifecycle.startups[instance.identity]
        replay = [p for identity, p in driver.calls if identity == instance.identity]
        assert [p['endpoint'] for p in replay] == [
            'twinkle/set_template', 'twinkle/add_adapter_to_sampler', 'twinkle/add_adapter_to_sampler',
            'twinkle/apply_patch', 'twinkle/set_template']
        assert json.loads(replay[0]['body'])['template_cls'] == 'T11'
        assert all(json.loads(p['body'])['config']['r'] == 8 for p in replay[1:3])
        assert json.loads(replay[-1]['body'])['template_cls'] == 'AfterPatch'
    finally:
        await pool.shutdown()
