# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray/Serve effects behind the controller's small runtime interface."""
from __future__ import annotations

import asyncio
import re
import time

from twinkle.server.resources.runtime import RayRuntime
from twinkle.server.state.training_records import InstanceRecord, TrainingRuntimeContext
from twinkle.server.telemetry.metrics import get_training_metrics
from twinkle.utils.logger import get_logger

logger = get_logger()


class RayInstanceDriver:

    def __init__(self, config, store, namespace, controller_name):
        self.config, self.store = config, store
        self.namespace, self.controller_name = namespace, controller_name
        self._metrics = get_training_metrics()
        self.runtime = RayRuntime(
            namespace, allocation_timeout=config.allocation_timeout_s, shutdown_timeout=config.shutdown_timeout_s)

    def application_name(self, instance):
        return self.runtime.pool.name(instance.controller_epoch, instance.instance_id)

    async def allocate(self, instance, spec):
        started, status = time.monotonic(), 'error'
        tags = {'backend': spec.backend, 'mode': spec.mode, 'device_type': spec.device_type}
        try:
            allocation = await self.runtime.pool.allocate(spec.config.num_devices, spec.device_type,
                                                          instance.controller_epoch, instance.instance_id)
            status = 'success'
            return {
                'application_name': allocation.name,
                'placement_group_id': allocation.placement_group_id,
                'node_id': allocation.node_id,
                'devices': list(allocation.devices)
            }
        finally:
            self._metrics.allocation_seconds.observe(time.monotonic() - started, tags={**tags, 'status': status})

    async def deploy(self, instance, spec):
        from ray.util.placement_group import get_placement_group

        from twinkle.server.model.app import build_model_app

        name = self.application_name(instance)
        context = TrainingRuntimeContext(
            instance_id=instance.instance_id,
            epoch=instance.controller_epoch,
            namespace=self.namespace,
            controller_name=self.controller_name,
            registry_name=name + '-registry')
        kwargs = spec.config.model_dump(
            exclude={
                'num_devices', 'device_mesh', 'min_replicas', 'max_replicas', 'idle_timeout_s', 'adapter_config',
                'queue_config'
            },
            exclude_none=True)
        if 'hybrid' in kwargs:
            kwargs['hybrid'] = spec.config.hybrid
        kwargs['train_mode'] = 'hybrid' if kwargs.get('hybrid') else spec.mode
        options = self.runtime.deployment_options(spec.deploy_options, name, spec.env_vars)
        app = build_model_app(
            model_id=spec.model_id,
            backend=spec.backend,
            nproc_per_node=spec.config.num_devices,
            device_group={
                'name': 'model',
                'ranks': spec.config.num_devices,
                'device_type': spec.device_type
            },
            device_mesh={
                **spec.config.device_mesh.model_dump(exclude_none=True), 'device_type': spec.device_type
            },
            deploy_options=options,
            queue_config=spec.config.queue_config,
            data_plane_url=spec.data_plane_url,
            training_context=context,
            placement_group=await asyncio.to_thread(get_placement_group, name),
            allocation_devices=list(instance.devices),
            **kwargs)
        await self.runtime.deploy(instance.instance_id, name, app, f'/api/v1/training-runtime/{instance.instance_id}',
                                  lambda: self.shutdown(instance))

    async def handle(self, instance):
        return await self.runtime.handle(instance.application_name or self.application_name(instance))

    async def create_training(self, instance, training, request):
        handle = await self.handle(instance)
        await handle.initialize_training.remote(training.model_dump(), request.model_dump())

    async def release_training(self, instance, training):
        handle = await self.handle(instance)
        await handle.release_training.remote(training.model_dump(), instance.resident and not instance.dirty)

    async def verify_operation(self, instance, request_id):
        handle = await self.handle(instance)
        return await handle.verify_operation.remote(request_id)

    async def shutdown(self, instance: InstanceRecord):
        name = instance.application_name or self.application_name(instance)
        try:
            spec = self.config.get(instance.base_model, instance.mode)
        except Exception:
            tags = {'backend': 'unknown', 'mode': instance.mode, 'device_type': 'unknown'}
        else:
            tags = {'backend': spec.backend, 'mode': spec.mode, 'device_type': spec.device_type}

        def observe_removal(duration, status):
            self._metrics.removal_seconds.observe(duration, tags={**tags, 'status': status})

        await self.runtime.shutdown(instance.instance_id, name, observe_removal=observe_removal)
        await self.store.models.unregister_replica_capacity(instance.instance_id)

    async def health(self, instance):
        async with asyncio.timeout(5):
            handle = await self.handle(instance)
            return await handle.runtime_identity.remote()

    async def recover_orphans(self):
        """Ownership names survive loss of the memory state actor."""
        pattern = re.compile(r'resources-' + re.escape(self.namespace)
                             + r'-(controller_[0-9a-f]{32})-(instance_[0-9a-f]{32})$')

        def decode(name):
            match = pattern.fullmatch(name)
            if match:
                return InstanceRecord(
                    controller_epoch=match[1],
                    instance_id=match[2],
                    application_name=name,
                    base_model='',
                    mode='full',
                    startup_deadline=0)

        await self.runtime.recover_orphans(decode, self.shutdown)
