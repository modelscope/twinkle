import asyncio
from types import SimpleNamespace
import logging
import pytest
from twinkle.server.resources.instances import InstanceLifecycle


@pytest.mark.asyncio
async def test_spawn_logs_exception_with_traceback(caplog, monkeypatch):
    # Capture one path regardless of the parent logger's runtime configuration.
    target = logging.getLogger('twinkle.server.resources.instances')
    monkeypatch.setattr(target, 'handlers', [caplog.handler])
    monkeypatch.setattr(target, 'propagate', False)
    lifecycle = InstanceLifecycle('Training')

    async def broken():
        raise RuntimeError('background sentinel')

    with caplog.at_level(logging.ERROR, logger=target.name):
        lifecycle.spawn(broken())
        await lifecycle.join()
    records = [record for record in caplog.records if record.getMessage() == 'Training background job failed']
    assert records and records[0].exc_info[1].args == ('background sentinel',)
    assert records[0].exc_info[2] is not None
    assert not lifecycle.jobs


@pytest.mark.asyncio
async def test_concurrent_close_shares_one_stop_and_retries_quarantine(caplog, monkeypatch):
    import asyncio
    target = logging.getLogger('twinkle.server.resources.instances')
    monkeypatch.setattr(target, 'handlers', [caplog.handler])
    monkeypatch.setattr(target, 'propagate', False)
    caplog.set_level(logging.WARNING, logger=target.name)
    lifecycle = InstanceLifecycle('test')
    entered, resume = asyncio.Event(), asyncio.Event()
    calls, events = [], []

    async def stop():
        calls.append('stop')
        entered.set()
        await resume.wait()
        if len(calls) == 1:
            raise RuntimeError('death unconfirmed')

    async def stopped():
        events.append('stopped')

    async def quarantined(error):
        events.append(str(error))

    first = asyncio.create_task(lifecycle.close('one', stop, 1, stopped, quarantined))
    await entered.wait()
    second = asyncio.create_task(lifecycle.close('one', stop, 1, stopped, quarantined))
    await asyncio.sleep(0)
    resume.set()
    await asyncio.gather(first, second)
    assert calls == ['stop'] and events == ['death unconfirmed']
    records = [record for record in caplog.records if record.getMessage() == 'test instance one shutdown is unconfirmed']
    assert len(records) == 1 and records[0].exc_info[1].args == ('death unconfirmed',)
    assert records[0].exc_info[2] is not None
    await lifecycle.close('one', stop, 1, stopped, quarantined)
    assert calls == ['stop', 'stop'] and events[-1] == 'stopped'
    await lifecycle.join()
    assert not lifecycle.jobs and not lifecycle.closing

@pytest.mark.asyncio
async def test_shared_health_rule_tolerates_transient_probe_timeout():
    from twinkle.server.resources.instances import InstanceLifecycle
    clock = SimpleNamespace(now=0)
    lifecycle = InstanceLifecycle('health', clock=lambda: clock.now)

    async def unreachable():
        raise TimeoutError('control-plane probe timed out')

    async def healthy():
        return {'unhealthy': False}

    async def failed():
        return {'unhealthy': True}

    assert await lifecycle.health('instance', unreachable) is None
    clock.now = 29
    assert await lifecycle.health('instance', unreachable) is None
    clock.now = 30
    assert await lifecycle.health('instance', unreachable) is False
    assert await lifecycle.health('instance', healthy) is True
    assert not lifecycle.unreachable_since
    assert await lifecycle.health('instance', unreachable) is None
    assert await lifecycle.health('instance', failed) is False


def _patch_actor_cleanup_rpc(monkeypatch, rows):
    from unittest.mock import AsyncMock, Mock
    import ray
    from ray import serve
    from ray._private import gcs_utils, worker
    from ray.core.generated import gcs_service_pb2, gcs_service_pb2_grpc
    from ray.core.generated.gcs_pb2 import ActorTableData

    events = []

    async def lookup(request, **kwargs):
        events.append(('lookup', request.name, request.ray_namespace))
        return gcs_service_pb2.GetNamedActorInfoReply(actor_table_data=ActorTableData(
            actor_id=request.name.encode(), name=request.name, ray_namespace=request.ray_namespace))

    async def kill(request, **kwargs):
        events.append(('kill', request.actor_id, request.force_kill, request.no_restart))
        return gcs_service_pb2.KillActorViaGcsReply()

    async def state(request, **kwargs):
        events.append(('state', request.actor_id))
        return gcs_service_pb2.GetActorInfoReply(
            actor_table_data=ActorTableData(actor_id=request.actor_id, state=ActorTableData.DEAD))

    rpc = SimpleNamespace(GetNamedActorInfo=AsyncMock(side_effect=lookup),
                          KillActorViaGcs=AsyncMock(side_effect=kill), GetActorInfo=AsyncMock(side_effect=state))
    channel = AsyncMock()
    monkeypatch.setattr(gcs_utils, 'create_gcs_channel', lambda *args, **kwargs: channel)
    monkeypatch.setattr(gcs_service_pb2_grpc, 'ActorInfoGcsServiceStub', lambda unused: rpc)
    monkeypatch.setattr(worker.global_worker, 'gcs_client', SimpleNamespace(address='test-gcs'), raising=False)
    monkeypatch.setattr(ray.util, 'list_named_actors', lambda **kwargs: rows)
    monkeypatch.setattr(ray, 'get_actor', Mock(side_effect=AssertionError('Worker handles must not be deserialized')))
    monkeypatch.setattr(ray, 'kill', Mock(side_effect=AssertionError('Cleanup must use actor IDs')))
    monkeypatch.setattr(serve, 'delete', lambda name: events.append(('delete', name)))
    rpc.events, rpc.channel = events, channel
    return rpc


@pytest.mark.asyncio
async def test_shutdown_actor_lookup_cannot_block_the_close_deadline(monkeypatch):
    from unittest.mock import AsyncMock
    from twinkle.server.resources.runtime import RayRuntime

    rpc = _patch_actor_cleanup_rpc(monkeypatch, [{'namespace': 'test', 'name': 'owned-worker'}])
    entered = asyncio.Event()

    async def lookup(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    rpc.GetNamedActorInfo.side_effect = lookup
    runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    runtime.pool.remove = AsyncMock()
    stopped, quarantined = AsyncMock(), AsyncMock()
    lifecycle = InstanceLifecycle('test')
    await lifecycle.close('owned', lambda: runtime.shutdown('owned', 'app'), .02, stopped, quarantined)
    assert entered.is_set()
    quarantined.assert_awaited_once()
    stopped.assert_not_awaited()
    runtime.pool.remove.assert_not_awaited()
    rpc.KillActorViaGcs.assert_not_awaited()
    rpc.channel.__aexit__.assert_awaited_once()
    await lifecycle.join()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['sampler', 'training'])
async def test_serve_handle_lookup_cannot_block_driver_deadlines(monkeypatch, kind):
    import threading
    from types import SimpleNamespace
    from ray import serve
    from twinkle.server.resources.runtime import RayRuntime
    from twinkle.server.sampler.pool.driver import SamplerDriver
    from twinkle.server.model.pool.driver import RayInstanceDriver

    entered, resume, returned = threading.Event(), threading.Event(), threading.Event()

    def lookup(name):
        entered.set()
        try:
            if not resume.wait(2):
                raise TimeoutError('blocked Serve controller RPC')
            raise RuntimeError('lookup released after cancellation')
        finally:
            returned.set()

    monkeypatch.setattr(serve, 'get_app_handle', lookup)
    driver_type = SamplerDriver if kind == 'sampler' else RayInstanceDriver
    driver = driver_type.__new__(driver_type)
    driver.runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    instance = SimpleNamespace(application='app', application_name='app')
    operation = asyncio.create_task(driver.health(instance))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(operation, .02)
        assert not returned.is_set(), 'Serve handle lookup blocked the controller event loop'
    finally:
        resume.set()
        await asyncio.gather(operation, return_exceptions=True)
        await asyncio.to_thread(returned.wait, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['applications', 'placement_groups'])
async def test_orphan_inventory_cannot_block_recovery_deadlines(monkeypatch, kind):
    import threading
    from types import SimpleNamespace
    from ray import serve
    from twinkle.server.resources.runtime import RayRuntime

    # Import the module rather than ray.util's same-named public function.
    import importlib
    placement_module = importlib.import_module('ray.util.placement_group')
    entered, resume, returned = threading.Event(), threading.Event(), threading.Event()

    def inventory():
        entered.set()
        try:
            if not resume.wait(2):
                raise TimeoutError('blocked resource inventory RPC')
            return SimpleNamespace(applications={}) if kind == 'applications' else {}
        finally:
            returned.set()

    monkeypatch.setattr(
        serve, 'status', inventory if kind == 'applications' else lambda: SimpleNamespace(applications={}))
    monkeypatch.setattr(
        placement_module, 'placement_group_table', inventory if kind == 'placement_groups' else lambda: {})
    runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    operation = asyncio.create_task(runtime.recover_orphans(lambda name: None, None))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(operation, .02)
        assert not returned.is_set(), 'Resource inventory blocked the controller event loop'
    finally:
        resume.set()
        await asyncio.gather(operation, return_exceptions=True)
        await asyncio.to_thread(returned.wait, 3)


@pytest.mark.asyncio
async def test_shutdown_uses_actor_ids_and_keeps_namespace_and_instance_scope(monkeypatch):
    from unittest.mock import AsyncMock
    from twinkle.server.resources.runtime import RayRuntime

    rows = [{'namespace': 'test', 'name': 'owned-worker'},
            {'namespace': 'test', 'name': 'owned-registry'},
            {'namespace': 'other', 'name': 'owned-worker'},
            {'namespace': 'test', 'name': 'unrelated-worker'}]
    rpc = _patch_actor_cleanup_rpc(monkeypatch, rows)
    runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    runtime.pool.remove = AsyncMock(side_effect=lambda name: rpc.events.append(('remove', name)))
    await runtime.shutdown('owned', 'app')
    assert rpc.events == [
        ('lookup', 'owned-worker', 'test'), ('kill', b'owned-worker', True, True),
        ('lookup', 'owned-registry', 'test'), ('kill', b'owned-registry', True, True),
        ('delete', 'app'), ('state', b'owned-worker'), ('state', b'owned-registry'), ('remove', 'app')]
    rpc.channel.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['namespace', 'name', 'missing_id', 'lookup_status', 'kill_status', 'state_status'])
async def test_shutdown_requires_trusted_identity_and_successful_gcs_replies(monkeypatch, failure):
    from unittest.mock import AsyncMock
    from ray.core.generated.gcs_pb2 import ActorTableData
    from ray.core.generated.gcs_service_pb2 import GetNamedActorInfoReply, KillActorViaGcsReply, GetActorInfoReply
    from twinkle.server.exceptions import ResourceReleaseError
    from twinkle.server.resources.runtime import RayRuntime

    rpc = _patch_actor_cleanup_rpc(monkeypatch, [{'namespace': 'test', 'name': 'owned-worker'}])
    actor = ActorTableData(actor_id=b'owned-worker', name='owned-worker', ray_namespace='test')
    if failure == 'namespace':
        actor.ray_namespace = 'other'
    elif failure == 'name':
        actor.name = 'unrelated-worker'
    elif failure == 'missing_id':
        actor.actor_id = b''
    lookup = GetNamedActorInfoReply(actor_table_data=actor)
    if failure == 'lookup_status':
        lookup.status.code = 1
    rpc.GetNamedActorInfo.side_effect = None
    rpc.GetNamedActorInfo.return_value = lookup
    if failure == 'kill_status':
        reply = KillActorViaGcsReply()
        reply.status.code = 1
        rpc.KillActorViaGcs.side_effect = None
        rpc.KillActorViaGcs.return_value = reply
    if failure == 'state_status':
        reply = GetActorInfoReply(actor_table_data=ActorTableData(actor_id=b'owned-worker', state=ActorTableData.DEAD))
        reply.status.code = 1
        rpc.GetActorInfo.side_effect = None
        rpc.GetActorInfo.return_value = reply
    runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    runtime.pool.remove = AsyncMock()
    with pytest.raises(ResourceReleaseError):
        await runtime.shutdown('owned', 'app')
    runtime.pool.remove.assert_not_awaited()
    if failure not in ('kill_status', 'state_status'):
        rpc.KillActorViaGcs.assert_not_awaited()
    rpc.channel.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked_rpc', ['KillActorViaGcs', 'GetActorInfo'])
async def test_shutdown_remote_actor_rpc_remains_within_close_deadline(monkeypatch, blocked_rpc):
    from unittest.mock import AsyncMock
    from twinkle.server.resources.runtime import RayRuntime

    rpc = _patch_actor_cleanup_rpc(monkeypatch, [{'namespace': 'test', 'name': 'owned-worker'}])
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    getattr(rpc, blocked_rpc).side_effect = blocked
    runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    runtime.pool.remove = AsyncMock()
    stopped, quarantined = AsyncMock(), AsyncMock()
    lifecycle = InstanceLifecycle('test')
    await lifecycle.close('owned', lambda: runtime.shutdown('owned', 'app'), .02, stopped, quarantined)
    assert entered.is_set()
    quarantined.assert_awaited_once()
    stopped.assert_not_awaited()
    runtime.pool.remove.assert_not_awaited()
    await lifecycle.join()


@pytest.mark.asyncio
async def test_shutdown_waits_for_dead_state_not_only_kill_ack(monkeypatch):
    from unittest.mock import AsyncMock
    from ray.core.generated.gcs_pb2 import ActorTableData
    from ray.core.generated.gcs_service_pb2 import GetActorInfoReply
    from twinkle.server.resources.runtime import RayRuntime

    rpc = _patch_actor_cleanup_rpc(monkeypatch, [{'namespace': 'test', 'name': 'owned-worker'}])
    rpc.GetActorInfo.side_effect = [
        GetActorInfoReply(actor_table_data=ActorTableData(actor_id=b'owned-worker', state=state))
        for state in (ActorTableData.ALIVE, ActorTableData.DEAD)]
    runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    runtime.pool.remove = AsyncMock()
    await runtime.shutdown('owned', 'app')
    assert rpc.GetActorInfo.await_count == 2
    runtime.pool.remove.assert_awaited_once_with('app')


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', [
    'unconfirmed_worker', 'unavailable_worker', 'dead_worker', 'removed', 'removal_failed'
])
async def test_worker_death_precedes_group_release_and_removal_timing(monkeypatch, outcome):
    from unittest.mock import AsyncMock, Mock
    from ray.core.generated.gcs_service_pb2 import GetActorInfoReply
    from twinkle.server.exceptions import ResourceReleaseError
    from twinkle.server.resources.runtime import RayRuntime

    rows = ([{'namespace': 'test', 'name': 'owned-worker'}]
            if outcome in ('unconfirmed_worker', 'unavailable_worker', 'dead_worker') else [])
    rpc = _patch_actor_cleanup_rpc(monkeypatch, rows)
    if outcome == 'unavailable_worker':
        rpc.GetActorInfo.side_effect = TimeoutError('Control-plane connection lost')
    elif outcome == 'unconfirmed_worker':
        rpc.GetActorInfo.side_effect = None
        rpc.GetActorInfo.return_value = GetActorInfoReply()  # No identity or death witness.
    runtime = RayRuntime('test', allocation_timeout=1, shutdown_timeout=1)
    runtime.pool.remove = AsyncMock(side_effect=RuntimeError('PG removal failed')
                                    if outcome == 'removal_failed' else None)
    observe = Mock()
    if outcome in ('unconfirmed_worker', 'unavailable_worker'):
        with pytest.raises(ResourceReleaseError) as failure:
            await runtime.shutdown('owned', 'app', observe_removal=observe)
        assert failure.value.phase == 'worker_death'
        runtime.pool.remove.assert_not_awaited()
        observe.assert_not_called()
    else:
        if outcome == 'removal_failed':
            with pytest.raises(RuntimeError, match='PG removal failed'):
                await runtime.shutdown('owned', 'app', observe_removal=observe)
        else:
            await runtime.shutdown('owned', 'app', observe_removal=observe)
        runtime.pool.remove.assert_awaited_once_with('app')
        assert observe.call_count == 1
        duration, status = observe.call_args.args
        assert duration >= 0 and status == ('error' if outcome == 'removal_failed' else 'success')
