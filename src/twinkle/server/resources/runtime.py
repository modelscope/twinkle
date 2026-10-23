# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray/Serve resource effects shared by training and sampling owners."""
from __future__ import annotations

import asyncio
import logging
import time
from copy import deepcopy

from twinkle.server.config.resources import build_worker_env_vars
from twinkle.server.exceptions import ResourceReleaseError
from .allocation import DevicePool
from .instances import InstanceLifecycle

logger = logging.getLogger(__name__)


class RayRuntime:
    """Own deployment threads and certify worker/PG death before releasing capacity."""

    def __init__(self, namespace, *, allocation_timeout, shutdown_timeout):
        self.namespace = namespace
        self.pool = DevicePool(namespace, allocation_timeout=allocation_timeout, shutdown_timeout=shutdown_timeout)
        self.deployments = {}
        self.lifecycle = InstanceLifecycle('Ray deployment cleanup')

    def deployment_options(self, options, name, env_vars=None):
        options = deepcopy(options)
        options.pop('autoscaling_config', None)
        options['num_replicas'] = 1
        actor = options.setdefault('ray_actor_options', {})
        actor.setdefault('num_cpus', 0.1)
        runtime_env = actor.setdefault('runtime_env', {})
        env = build_worker_env_vars()
        env.update(runtime_env.get('env_vars', {}))
        env.update(env_vars or {})
        env.update(TWINKLE_RAY_NAMESPACE=self.namespace, TWINKLE_RAY_REGISTRY_NAME=name + '-registry')
        runtime_env['env_vars'] = env
        return options

    async def deploy(self, identity, name, app, route_prefix, cleanup):
        from ray import serve
        deployment = asyncio.create_task(asyncio.to_thread(serve.run, app, name=name, route_prefix=route_prefix))
        self.deployments[identity] = deployment
        try:
            await asyncio.shield(deployment)
        except asyncio.CancelledError:

            async def clean_late():
                await asyncio.gather(deployment, return_exceptions=True)
                await cleanup()

            self.lifecycle.spawn(clean_late())
            raise
        finally:
            if deployment.done():
                self.deployments.pop(identity, None)

    async def wait_for_cleanup(self):
        await self.lifecycle.join()

    @staticmethod
    async def handle(name):

        def lookup():
            from ray import serve
            return serve.get_app_handle(name)

        # get_app_handle waits synchronously for the Serve controller. Keep
        # that RPC inside the caller's cancellable asynchronous deadline.
        return await asyncio.to_thread(lookup)

    async def shutdown(self, identity, name, *, observe_removal=None):
        import ray
        from ray import serve
        from ray.util import list_named_actors
        deployment = self.deployments.get(identity)
        if deployment is not None:
            await asyncio.gather(asyncio.shield(deployment), return_exceptions=True)
            self.deployments.pop(identity, None)
        # Both features give workers the controller identity, not a Serve replica ID.
        actors = []
        for row in await asyncio.to_thread(list_named_actors, all_namespaces=True):
            if row['namespace'] != self.namespace or identity not in row['name']:
                continue
            try:
                # Ray may import the worker class when resolving its handle.
                # Keep imports and GCS calls outside the controller event loop
                # so its shutdown deadline and other instances remain live.
                handle = await asyncio.to_thread(ray.get_actor, row['name'], namespace=self.namespace)
            except ValueError:
                continue
            actors.append(handle)
            await asyncio.to_thread(ray.kill, handle, no_restart=True)
        await asyncio.to_thread(serve.delete, name)
        for handle in actors:
            try:
                await asyncio.wait_for(handle.__ray_ready__.remote(), 5)
            except ray.exceptions.ActorDiedError:
                continue
            except ray.exceptions.RayActorError as exc:
                raise ResourceReleaseError('Owned worker death could not be confirmed', phase='worker_death') from exc
            raise ResourceReleaseError('Owned worker death could not be confirmed', phase='worker_death')

        started, status = time.monotonic(), 'error'
        try:
            await self.pool.remove(name)
            status = 'success'
        finally:
            if observe_removal is not None:
                observe_removal(time.monotonic() - started, status)

    async def recover_orphans(self, decode, shutdown):
        """Decode only this owner's names; never sweep another service's resources."""
        from ray import serve
        from ray.util.placement_group import placement_group_table
        applications, groups = await asyncio.gather(
            asyncio.to_thread(serve.status), asyncio.to_thread(placement_group_table))
        names = set(applications.applications)
        names.update(row.get('name', '') for row in groups.values() if row.get('state') != 'REMOVED')
        for name in sorted(names):
            instance = decode(name)
            if instance is not None:
                await asyncio.wait_for(shutdown(instance), self.pool.shutdown_timeout)
