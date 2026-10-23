# Copyright (c) ModelScope Contributors. All rights reserved.
"""Initialization failures and backoff during automatic controller restarts."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from twinkle.server.resources.instances import InstanceLifecycle
from twinkle.server.model.pool.actor import TrainingControllerActor


@pytest.mark.asyncio
async def test_failed_initialization_restarts_only_after_backoff(monkeypatch):
    cls = TrainingControllerActor.__ray_metadata__.modified_class
    gate, sleeping = asyncio.Event(), asyncio.Event()
    delays = []
    handle = object()
    kill = Mock()
    get_actor = Mock(return_value=handle)
    original_sleep = asyncio.sleep

    async def sleep(delay):
        delays.append(delay)
        sleeping.set()
        await gate.wait()

    monkeypatch.setattr('twinkle.server.model.pool.actor.asyncio.sleep', sleep)
    monkeypatch.setattr('twinkle.server.model.pool.actor.ray.get_actor', get_actor)
    monkeypatch.setattr('twinkle.server.model.pool.actor.ray.kill', kill)
    actor = SimpleNamespace(core=SimpleNamespace(initialize=AsyncMock(side_effect=RuntimeError('allocation failed'))),
                            config=SimpleNamespace(sweep_interval_s=7), namespace='initialization-test', _monitors=[])
    actor._restart_after_failure = lambda: cls._restart_after_failure(actor)
    with pytest.raises(RuntimeError, match='allocation failed'):
        await cls._initialize(actor)
    await sleeping.wait()
    assert delays == [7] and not actor._monitors
    kill.assert_not_called()
    gate.set()
    await actor._restart_task
    await original_sleep(0)
    get_actor.assert_called_once_with('training-controller-initialization-test', namespace='initialization-test')
    kill.assert_called_once_with(handle, no_restart=False)


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_restart():
    cls = TrainingControllerActor.__ray_metadata__.modified_class
    wait = asyncio.Event()
    restart = asyncio.create_task(wait.wait())
    initialization = asyncio.get_running_loop().create_future()
    initialization.set_result(None)
    actor = SimpleNamespace(driver=SimpleNamespace(runtime=SimpleNamespace(wait_for_cleanup=AsyncMock())),
                                core=SimpleNamespace(shutdown=AsyncMock(), wait_for_jobs=AsyncMock()),
                            _monitors=[], _restart_task=restart, _initialization=initialization, _failures={})
    await cls.shutdown(actor)
    assert restart.cancelled()
    actor.core.shutdown.assert_awaited_once()
    actor.driver.runtime.wait_for_cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_heartbeat_keeps_healthy_peer_alive_while_probe_and_close_block():
    cls = TrainingControllerActor.__ray_metadata__.modified_class
    good, bad = [SimpleNamespace(instance_id=name, state='READY') for name in ('good', 'bad')]
    probe_gate, close_gate = asyncio.Event(), asyncio.Event()
    healthy_seen, close_started = asyncio.Event(), asyncio.Event()

    async def health(instance):
        if instance is bad:
            await probe_gate.wait()
        return {'unhealthy': instance is bad}

    async def fail(*args):
        close_started.set()
        await close_gate.wait()

    async def touch(identity):
        if identity == 'good':
            healthy_seen.set()

    actor = SimpleNamespace(
        store=SimpleNamespace(backend=SimpleNamespace(get=AsyncMock(return_value='active')),
                              key=lambda *args: 'active', models=SimpleNamespace(touch_replica_last_seen=touch)),
        core=SimpleNamespace(epoch='controller', instances={'bad': bad, 'good': good}, fail_instance=fail,
                             lifecycle=InstanceLifecycle('Training')),
        driver=SimpleNamespace(health=health), _unreachable_since={}, _failures={})
    actor._check_instance_health = lambda instance: cls._check_instance_health(actor, instance)
    actor._fail_unhealthy = lambda instance, message: cls._fail_unhealthy(actor, instance, message)
    monitor = asyncio.create_task(cls._heartbeat(actor))
    try:
        await asyncio.wait_for(healthy_seen.wait(), 1)
        assert not probe_gate.is_set()
        probe_gate.set()
        await asyncio.wait_for(close_started.wait(), 1)
        # Closing a bad peer does not occupy the heartbeat monitor.
        healthy_seen.clear()
        await cls._check_instance_health(actor, good)
        assert healthy_seen.is_set()
        cls._fail_unhealthy(actor, bad, 'duplicate')
        assert len(actor.core.lifecycle.named_jobs) == 1
    finally:
        probe_gate.set()
        close_gate.set()
        monitor.cancel()
        await asyncio.gather(monitor, *actor.core.lifecycle.named_jobs.values(), return_exceptions=True)
    assert not actor.core.lifecycle.named_jobs


@pytest.mark.asyncio
async def test_shutdown_cancels_and_joins_initialization_before_closing_core():
    cls = TrainingControllerActor.__ray_metadata__.modified_class
    started, exited = asyncio.Event(), asyncio.Event()

    async def initialize():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    initialization = asyncio.create_task(initialize())
    await started.wait()
    actor = SimpleNamespace(driver=SimpleNamespace(runtime=SimpleNamespace(wait_for_cleanup=AsyncMock())),
                                core=SimpleNamespace(shutdown=AsyncMock(), wait_for_jobs=AsyncMock()),
                            _initialization=initialization, _monitors=[], _failures={})
    await cls.shutdown(actor)
    assert initialization.cancelled() and exited.is_set()
    actor.core.shutdown.assert_awaited_once()
