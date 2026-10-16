# Copyright (c) ModelScope Contributors. All rights reserved.
"""Single-node device allocation. Only the owner removes placement groups."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from twinkle import Platform
from twinkle.infra._ray.resource_manager import _gpu_placement_group_cpus
from .catalog import TrainingSpec
from .errors import CapacityUnavailableError


@dataclass(frozen=True)
class DeviceAllocation:
    name: str
    placement_group_id: str
    node_id: str
    devices: tuple[str, ...]


def _probe_allocation(resource: str, device_type: str):
    import os
    import ray
    context = ray.get_runtime_context()
    if resource == 'CPU':
        devices = []
    else:
        env = os.environ.get(Platform.get_platform(device_type).visible_device_env())
        devices = env.split(',') if env else context.get_accelerator_ids().get(resource, [])
    return context.get_node_id(), [str(device) for device in devices]


class DevicePool:

    def __init__(self, namespace: str, *, allocation_timeout: float, shutdown_timeout: float):
        self.namespace = namespace
        self.allocation_timeout = allocation_timeout
        self.shutdown_timeout = shutdown_timeout

    def name(self, epoch: str, instance_id: str) -> str:
        return f'training-{self.namespace}-{epoch}-{instance_id}'

    async def allocate(self, spec: TrainingSpec, epoch: str, instance_id: str) -> DeviceAllocation:
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        resource = 'CPU' if spec.model.device_type == 'cpu' else Platform.get_platform(spec.model.device_type).__name__
        count = spec.config.num_devices
        candidates = [node for node in ray.nodes() if node.get('Alive') and node['Resources'].get(resource, 0) >= count]
        available = ray.available_resources()
        if not candidates or available.get(resource, 0) < count:
            raise CapacityUnavailableError(f'No single-node allocation available for {count} {resource} devices')
        cpus = min(_gpu_placement_group_cpus(int(node['Resources'].get('CPU', 0)), count) for node in candidates)
        bundle = {resource: count, 'CPU': count if resource == 'CPU' else cpus}
        name = self.name(epoch, instance_id)
        pg = placement_group([bundle], strategy='STRICT_PACK', name=name)
        try:
            await asyncio.wait_for(pg.ready(), self.allocation_timeout)
            options = {'num_cpus': 0.01, 'scheduling_strategy': PlacementGroupSchedulingStrategy(placement_group=pg)}
            if resource == 'GPU':
                options['num_gpus'] = count
            elif resource != 'CPU':
                options['resources'] = {resource: count}
            node, devices = await asyncio.wait_for(
                ray.remote(_probe_allocation).options(**options).remote(resource, spec.model.device_type),
                self.shutdown_timeout)
            if resource == 'CPU':
                devices = [str(index) for index in range(count)]
            if len(devices) != count or len(set(devices)) != count:
                raise RuntimeError('Ray allocation did not return the requested unique device IDs')
            return DeviceAllocation(name, pg.id.hex(), node, tuple(devices))
        except BaseException as exc:
            await self.remove(name)
            if isinstance(exc, TimeoutError):
                raise CapacityUnavailableError(
                    'Single-node allocation could not be confirmed; retry a new create later.') from exc
            raise

    async def remove(self, name: str) -> None:
        from ray.util.placement_group import get_placement_group, placement_group_table, remove_placement_group
        try:
            pg = get_placement_group(name)
        except ValueError:
            return
        remove_placement_group(pg)
        deadline = asyncio.get_running_loop().time() + self.shutdown_timeout
        while asyncio.get_running_loop().time() < deadline:
            if placement_group_table(pg).get('state') == 'REMOVED':
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f'Placement group removal was not confirmed: {name}')
