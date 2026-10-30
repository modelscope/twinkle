# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run in its own process: test production allocation against logical Ray GPUs."""
import asyncio
import json
import os
import ray
import uuid

from twinkle import DeviceGroup
from twinkle.infra._ray.resource_manager import ResourceManager
from twinkle.server.config.training import TrainingConfig
from twinkle.server.resources.allocation import CapacityUnavailableError, DevicePool


async def main():
    # These are synthetic Ray GPU reservations, not CUDA workloads.
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    namespace = 'alloc_' + uuid.uuid4().hex
    ray.init(
        address='local',
        namespace=namespace,
        num_cpus=4,
        num_gpus=8,
        _temp_dir='/tmp/twa-' + namespace[-8:],
        include_dashboard=False,
        dashboard_port=0,
        log_to_driver=False)
    pool = DevicePool(namespace, allocation_timeout=3, shutdown_timeout=10)
    allocations = []
    try:
        for index, count in enumerate((2, 2, 4)):
            config = TrainingConfig(models=[{'name': str(index), 'model_id': str(index), 'backend': 'mock', 'device_type': 'cuda', 'mode': 'full', 'config': {'num_devices': count, 'device_mesh': {'fsdp_size': count}}}])
            spec = config.get(str(index), 'full')
            allocation = await pool.allocate(spec.config.num_devices, spec.device_type, 'epoch', str(index))
            allocations.append(allocation)
        devices = [device for allocation in allocations for device in allocation.devices]
        assert len(set(devices)) == 8, devices
        assert len({allocation.node_id for allocation in allocations}) == 1
        try:
            await pool.allocate(spec.config.num_devices, spec.device_type, 'epoch', 'overflow')
            raise AssertionError('over-allocation was admitted')
        except CapacityUnavailableError:
            pass
        from ray.util.placement_group import get_placement_group
        group = DeviceGroup(name='owned', ranks=2, device_type='cuda')
        owned_pg = get_placement_group(allocations[0].name)
        view = ResourceManager.from_allocation(group, owned_pg, list(allocations[0].devices))
        assert view.get_config('owned').ranks == [0, 1]
        assert [entry['gpu_rank'][0] for entry in view.get_group('owned')] == list(allocations[0].devices)
        # A ResourceManager view cannot release a controller-owned allocation.
        view.destroy_placement_group()
        assert get_placement_group(allocations[0].name).id == owned_pg.id
        freed = allocations[0]
        await pool.remove(freed.name)
        two = TrainingConfig(models=[{'name': 'replacement', 'model_id': 'replacement', 'backend': 'mock', 'device_type': 'cuda', 'mode': 'full', 'config': {'num_devices': 2, 'device_mesh': {'fsdp_size': 2}}}]).get('replacement', 'full')
        replacement = await pool.allocate(two.config.num_devices, two.device_type, 'epoch', 'replacement')
        allocations.append(replacement)
        assert set(replacement.devices) == set(freed.devices)
        print(json.dumps({'passed': True, 'devices': devices, 'reused': list(replacement.devices)}))
    finally:
        try:
            for allocation in allocations:
                await pool.remove(allocation.name)
        finally:
            ray.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
