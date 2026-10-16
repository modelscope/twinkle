# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run in its own process: test production allocation against logical Ray GPUs."""
import asyncio
import json
import ray
import uuid

from twinkle import DeviceGroup
from twinkle.infra._ray.resource_manager import ResourceManager
from twinkle.server.config.training import TrainingConfig
from twinkle.server.training.allocation import DevicePool
from twinkle.server.training.catalog import TrainingCatalog
from twinkle.server.training.errors import CapacityUnavailableError


async def main():
    namespace = 'alloc_' + uuid.uuid4().hex
    ray.init(
        address='local', namespace=namespace, num_cpus=16, num_gpus=8, include_dashboard=False, log_to_driver=False)
    pool = DevicePool(namespace, allocation_timeout=3, shutdown_timeout=10)
    allocations = []
    try:
        for index, count in enumerate((2, 2, 4)):
            config = TrainingConfig(models=[{
                'name': str(index),
                'model_id': str(index),
                'backend': 'mock',
                'device_type': 'cuda',
                'modes': {
                    'full': {
                        'num_devices': count,
                        'device_mesh': {
                            'fsdp_size': count
                        }
                    }
                },
            }])
            spec = TrainingCatalog(config).get(str(index), 'full')
            allocation = await pool.allocate(spec, 'epoch', str(index))
            allocations.append(allocation)
        devices = [device for allocation in allocations for device in allocation.devices]
        assert len(set(devices)) == 8, devices
        assert len({allocation.node_id for allocation in allocations}) == 1
        assert ray.available_resources().get('GPU', 0) == 0
        try:
            await pool.allocate(spec, 'epoch', 'overflow')
            raise AssertionError('over-allocation was admitted')
        except CapacityUnavailableError:
            pass
        from ray.util.placement_group import get_placement_group
        group = DeviceGroup(name='owned', ranks=2, device_type='cuda')
        view = ResourceManager.from_allocation(group, get_placement_group(allocations[0].name),
                                               list(allocations[0].devices))
        assert view.get_config('owned').ranks == [0, 1]
        assert [entry['gpu_rank'][0] for entry in view.get_group('owned')] == list(allocations[0].devices)
        view.destroy_placement_group()
        assert ray.available_resources().get('GPU', 0) == 0
        freed = allocations[0]
        await pool.remove(freed.name)
        two = TrainingCatalog(
            TrainingConfig(models=[{
                'name': 'replacement',
                'model_id': 'replacement',
                'backend': 'mock',
                'device_type': 'cuda',
                'modes': {
                    'full': {
                        'num_devices': 2,
                        'device_mesh': {
                            'fsdp_size': 2
                        }
                    }
                },
            }])).get('replacement', 'full')
        replacement = await pool.allocate(two, 'epoch', 'replacement')
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
