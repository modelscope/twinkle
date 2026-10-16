# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray/Serve effects behind the controller's small runtime interface."""
from __future__ import annotations

import asyncio
import os
import re

from twinkle.server.launcher.env_propagation import build_propagated_env_vars
from .allocation import DevicePool
from .records import InstanceRecord


class RayInstanceDriver:

    def __init__(self, config, store, namespace, controller_name):
        self.config, self.store = config, store
        self.namespace, self.controller_name = namespace, controller_name
        self.pool = DevicePool(
            namespace, allocation_timeout=config.allocation_timeout_s, shutdown_timeout=config.shutdown_timeout_s)

    def application_name(self, instance):
        return self.pool.name(instance.controller_epoch, instance.instance_id)

    async def start(self, instance, spec):
        from ray import serve
        from ray.util.placement_group import get_placement_group

        from twinkle.server.model.app import build_model_app

        allocation = await self.pool.allocate(spec, instance.controller_epoch, instance.instance_id)
        name = self.application_name(instance)
        # Persist ownership before the first worker can be constructed.
        instance.application_name = name
        instance.placement_group_id = allocation.placement_group_id
        instance.node_id, instance.devices = allocation.node_id, list(allocation.devices)
        await self.store.save_instance(instance)
        env = build_propagated_env_vars()
        env.update(spec.model.env_vars)
        for key in ('PYTHONPATH', 'TWINKLE_DEFAULT_SAVE_DIR', 'TMPDIR', 'OMP_NUM_THREADS'):
            if key in os.environ:
                env.setdefault(key, os.environ[key])
        env.update({'TWINKLE_RAY_REGISTRY_NAME': name + '-registry', 'TWINKLE_RAY_NAMESPACE': self.namespace})
        context = {
            'instance_id': instance.instance_id,
            'epoch': instance.controller_epoch,
            'namespace': self.namespace,
            'controller_name': self.controller_name,
            'registry_name': name + '-registry'
        }
        kwargs = spec.config.model_dump(
            exclude={'num_devices', 'device_mesh', 'min_replicas', 'max_replicas', 'idle_timeout_s', 'queue_config'},
            exclude_none=True)
        if 'hybrid' in kwargs:
            kwargs['hybrid'] = spec.config.hybrid
        kwargs['train_mode'] = 'hybrid' if kwargs.get('hybrid') else spec.mode
        app = build_model_app(
            model_id=spec.model.model_id,
            backend=spec.model.backend,
            nproc_per_node=spec.config.num_devices,
            device_group={
                'name': 'model',
                'ranks': spec.config.num_devices,
                'device_type': spec.model.device_type
            },
            device_mesh={
                **spec.config.device_mesh.model_dump(exclude_none=True), 'device_type': spec.model.device_type
            },
            deploy_options={
                'num_replicas': 1,
                'ray_actor_options': {
                    'num_cpus': 0.1,
                    'runtime_env': {
                        'env_vars': env
                    }
                }
            },
            queue_config=spec.config.queue_config,
            data_plane_url=spec.model.data_plane_url,
            training_context=context,
            placement_group=get_placement_group(allocation.name),
            allocation_devices=list(allocation.devices),
            **kwargs)
        # Shield the actual thread so its late return can still be cleaned.
        deployment = asyncio.create_task(
            asyncio.to_thread(
                serve.run, app, name=name, route_prefix=f'/api/v1/training-runtime/{instance.instance_id}'))
        try:
            await asyncio.shield(deployment)
        except asyncio.CancelledError:

            async def clean_late():
                try:
                    await deployment
                except Exception:
                    pass
                await self.shutdown(instance)

            asyncio.create_task(clean_late())
            raise
        return {
            'application_name': name,
            'placement_group_id': allocation.placement_group_id,
            'node_id': allocation.node_id,
            'devices': list(allocation.devices)
        }

    def handle(self, instance):
        from ray import serve
        return serve.get_app_handle(instance.application_name or self.application_name(instance))

    async def create_training(self, instance, training, request):
        await self.handle(instance).initialize_training.remote(training.model_dump(), request.model_dump())

    async def release_training(self, instance, training):
        await self.handle(instance).release_training.remote(training.model_dump(), instance.resident
                                                            and not instance.dirty)

    async def verify_operation(self, instance, request_id):
        return await self.handle(instance).verify_operation.remote(request_id)

    async def shutdown(self, instance: InstanceRecord):
        import ray
        from ray import serve
        from ray.util import list_named_actors
        name = self.application_name(instance)
        owned = [
            actor for actor in list_named_actors(all_namespaces=True)
            if actor['namespace'] == self.namespace and instance.instance_id in actor['name']
        ]
        actors = []
        for actor in owned:
            try:
                handle = ray.get_actor(actor['name'], namespace=self.namespace)
            except ValueError:
                continue
            actors.append(handle)
            ray.kill(handle, no_restart=True)
        await asyncio.to_thread(serve.delete, instance.application_name or name)
        await self.pool.remove(name)
        # The PG kills even unnamed remaining children. Named workers must also
        # have failed readiness RPCs before the controller retires quota/tickets.
        for handle in actors:
            try:
                await asyncio.wait_for(handle.__ray_ready__.remote(), 5)
            except ray.exceptions.RayActorError:
                continue
            raise RuntimeError('Owned worker death could not be confirmed')
        await self.store.state.unregister_replica(instance.instance_id)

    async def health(self, instance):
        return await asyncio.wait_for(self.handle(instance).runtime_identity.remote(), 5)

    async def recover_orphans(self):
        """Ownership names survive loss of the memory state actor."""
        from ray import serve
        from ray.util.placement_group import placement_group_table
        pattern = re.compile(r'training-' + re.escape(self.namespace)
                             + r'-(controller_[0-9a-f]{32})-(instance_[0-9a-f]{32})$')
        names = set(serve.status().applications)
        names.update(row.get('name', '') for row in placement_group_table().values() if row.get('state') != 'REMOVED')
        for name in sorted(names):
            match = pattern.fullmatch(name)
            if match:
                orphan = InstanceRecord(
                    controller_epoch=match[1],
                    instance_id=match[2],
                    application_name=name,
                    base_model='',
                    mode='full',
                    startup_deadline=0)
                await asyncio.wait_for(self.shutdown(orphan), self.config.shutdown_timeout_s)
