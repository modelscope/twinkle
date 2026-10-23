# Copyright (c) ModelScope Contributors. All rights reserved.
"""One named controller actor per server namespace."""
from __future__ import annotations

import asyncio
import ray
import time

from twinkle.server.config.training import TrainingConfig
from twinkle.server.state import get_server_state
from twinkle.server.state.models import FutureFailureRecord
from twinkle.server.state.training_records import CreateTraining
from twinkle.server.telemetry.metrics import get_training_metrics
from twinkle.server.telemetry.worker_init import ensure_telemetry_initialized
from twinkle.utils.logger import get_logger
from .controller import TrainingController
from .driver import RayInstanceDriver

logger = get_logger()


def controller_name(namespace):
    return f'training-controller-{namespace}'


@ray.remote(num_cpus=0.1, max_restarts=-1, max_task_retries=0)
class TrainingControllerActor:

    def __init__(self, config, namespace):
        ensure_telemetry_initialized()
        self.config = TrainingConfig.model_validate(config)
        self.namespace = namespace
        self.store = get_server_state().create_training_store(namespace)
        self.driver = RayInstanceDriver(self.config, self.store, namespace, controller_name(namespace))
        self.core = TrainingController(self.config, self.store, self.driver)
        self._metrics = get_training_metrics()
        self._monitors = []
        self._initialization = asyncio.get_event_loop().create_task(self._initialize())

    async def initialize(self):
        await asyncio.shield(self._initialization)

    async def _initialize(self):
        try:
            await self.core.initialize()
        except Exception:
            logger.exception('Training controller initialization failed; restart is deferred')
            # First-time startup is rolled back by the launcher; auto-restart waits before retrying.
            self._restart_task = asyncio.create_task(self._restart_after_failure())
            raise
        self._monitors = [asyncio.create_task(self._heartbeat()), asyncio.create_task(self._sweep())]

    async def _restart_after_failure(self):
        await asyncio.sleep(max(5.0, self.config.sweep_interval_s))
        ray.kill(ray.get_actor(controller_name(self.namespace), namespace=self.namespace), no_restart=False)

    async def _heartbeat(self):
        while True:
            try:
                active = await self.store.backend.get(self.store.key('controller', 'active'))
                if active is None:
                    await self._recover_lost_state()
                    return
                await self.store.models.touch_replica_last_seen(self.core.epoch)
            except Exception:
                logger.exception('Training state heartbeat failed; admission remains fenced by the state backend')
                await asyncio.sleep(5)
                continue
            await asyncio.gather(*(self._check_instance_health(instance)
                                   for instance in list(self.core.instances.values())
                                   if instance.state in ('READY', 'RESETTING')))
            await asyncio.sleep(5)

    async def _check_instance_health(self, instance):

        async def probe():
            identity = await self.driver.health(instance)
            if not identity['unhealthy']:
                await self.store.models.touch_replica_last_seen(instance.instance_id)
            return identity

        healthy = await self.core.lifecycle.health(instance.instance_id, probe)
        if healthy is False:
            self._fail_unhealthy(instance, 'Training runtime became unavailable or reported a failed worker')

    def _fail_unhealthy(self, instance, message):
        self.core.lifecycle.spawn_once(('health', instance.instance_id), lambda: self.core.fail_instance(
            instance.instance_id,
            FutureFailureRecord(reason_code='orphaned_replica', attribution='server', message=message)))

    async def _sweep(self):
        while True:
            started, status = time.monotonic(), 'success'
            try:
                await self.core.sweep()
            except Exception:
                status = 'error'
                logger.exception('Training lifecycle sweep failed; resources remain owned')
            finally:
                self._metrics.sweep_total.inc(tags={'status': status})
                self._metrics.sweep_seconds.observe(time.monotonic() - started, tags={'status': status})
            await asyncio.sleep(self.config.sweep_interval_s)

    async def _recover_lost_state(self):
        # Reconstruct enough ownership for the next epoch to fail pending
        # futures and terminate compute. Sessions were lost, so old handles
        # cannot be reused, even while these rows are being restored.
        failure = FutureFailureRecord(
            reason_code='orphaned_replica',
            attribution='server',
            message='Training state was lost; create a new session and training handle.')
        for instance in self.core.instances.values():
            if instance.state != 'STOPPED':
                await self.store.save_instance(instance)
        for training in self.core.trainings.values():
            if training.state in ('PENDING', 'ACTIVE', 'CLOSING'):
                await self.store.save_training(training)
                for request_id in (training.create_request_id, *training.pending_operation_ids):
                    await self.store.future(
                        request_id, 'failed', training.training_id, failure=failure, owner_token=training.token)
        ray.kill(ray.get_actor(controller_name(self.namespace), namespace=self.namespace), no_restart=False)

    async def register_runtime(self, instance_id, epoch, incarnation):
        await self.core.register_runtime(instance_id, epoch, incarnation)

    async def create(self, request):
        await self.initialize()
        return await self.core.create(CreateTraining.model_validate(request))

    async def route(self, *args):
        await self.initialize()
        return await self.core.route(*args)

    async def admit(self, *args, **kwargs):
        await self.initialize()
        return await self.core.admit(*args, **kwargs)

    async def finish_operation(self, *args, **kwargs):
        return await self.core.finish_operation(*args, **kwargs)

    async def release(self, *args):
        return await self.core.release(*args)

    async def fail_instance(self, *args):
        return await self.core.fail_instance(*args)

    async def mark_dirty(self, instance_id, epoch):
        await self.core.mark_dirty(instance_id, epoch)

    async def diagnostics(self, base_model=None):
        await self.initialize()
        instances = [
            instance.model_dump() for instance in self.core.instances.values()
            if base_model is None or instance.base_model == base_model
        ]
        return {
            'ready': self.core._ready and self.core.residents_ready,
            'epoch': self.core.epoch,
            'instances': instances
        }

    async def shutdown(self):
        if not self._initialization.done():
            self._initialization.cancel()
        await asyncio.gather(self._initialization, return_exceptions=True)
        restart = getattr(self, '_restart_task', None)
        if restart is not None:
            restart.cancel()
            await asyncio.gather(restart, return_exceptions=True)
        for task in self._monitors:
            task.cancel()
        await asyncio.gather(*self._monitors, return_exceptions=True)
        await self.core.shutdown()
        await self.driver.runtime.wait_for_cleanup()
