# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise single-node admission with real Ray nodes and logical devices."""
import asyncio
import json
import ray
import uuid
from ray.cluster_utils import Cluster
from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group

from twinkle.server.config.training import TrainingConfig
from twinkle.server.training.allocation import DevicePool
from twinkle.server.training.catalog import TrainingCatalog
from twinkle.server.training.errors import CapacityUnavailableError


async def main():
    cluster = Cluster()
    occupied = []
    allocated = None
    pool = None
    try:
        cluster.add_node(num_cpus=4, num_gpus=4, resources={'left': 1}, include_dashboard=False)
        cluster.add_node(num_cpus=4, num_gpus=4, resources={'right': 1})
        ray.init(address=cluster.address, namespace='fragmentation_' + uuid.uuid4().hex, log_to_driver=False)
        pool = DevicePool(ray.get_runtime_context().namespace, allocation_timeout=1, shutdown_timeout=10)
        spec = TrainingCatalog(
            TrainingConfig(models=[{
                'name': 'two',
                'model_id': 'two',
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
            }])).get('two', 'full')
        for marker in ('left', 'right'):
            pg = placement_group([{'GPU': 3, marker: 0.01}], strategy='STRICT_PACK')
            occupied.append(pg)
            await asyncio.wait_for(pg.ready(), 20)
        assert ray.available_resources()['GPU'] == 2
        try:
            await pool.allocate(spec, 'epoch', 'fragmented')
        except CapacityUnavailableError:
            pass
        else:
            raise AssertionError('Two nodes with one free GPU each admitted a two-GPU instance')
        groups = placement_group_table().values()
        assert not any(group['name'] == pool.name('epoch', 'fragmented') and group['state'] != 'REMOVED'
                       for group in groups)
        remove_placement_group(occupied[0])
        for _ in range(200):
            if ray.available_resources().get('GPU', 0) == 5:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError('Owned GPU reservation was not released')
        allocated = await pool.allocate(spec, 'epoch', 'available')
        assert len(set(allocated.devices)) == 2
        print(
            json.dumps({
                'passed': True,
                'nodes': len(ray.nodes()),
                'fragmented_rejected': True,
                'pending_removed': True,
                'devices': allocated.devices
            }))
    finally:
        try:
            if allocated is not None:
                await pool.remove(allocated.name)
            for pg in occupied:
                remove_placement_group(pg)
        finally:
            ray.shutdown()
            cluster.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
