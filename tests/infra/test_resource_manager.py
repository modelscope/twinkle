# Copyright (c) Alibaba, Inc. and its affiliates.
from types import SimpleNamespace

import pytest

from twinkle import DeviceGroup, Platform
from twinkle.infra._ray.resource_manager import ResourceManager, _gpu_placement_group_cpus


def _stub_publish_addresses(self):
    # 58b2b979 added _publish_addresses, which probes each placement group's node address with a Ray
    # actor and keys a dict by the (hashable) Ray placement group. These tests fake placement groups as
    # SimpleNamespace/object and assert device ORDER, not address discovery, so attach a stub node_ip
    # directly -- preserving the placement['node_ip'] contract without the Ray probe.
    for placements in self.device_groups.values():
        for placement in placements:
            placement['node_ip'] = '127.0.0.1'
    for placement in self.rank_placements.values():
        placement['node_ip'] = '127.0.0.1'


def test_gpu_placement_group_cpus_uses_per_process_default(monkeypatch):
    monkeypatch.delenv('TWINKLE_GPU_PG_CPUS_PER_PROC', raising=False)

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=2) == 8


def test_gpu_placement_group_cpus_respects_node_cap(monkeypatch):
    monkeypatch.setenv('TWINKLE_GPU_PG_CPUS_PER_PROC', '16')

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=8) == 48


def test_gpu_placement_group_cpus_can_be_configured(monkeypatch):
    monkeypatch.setenv('TWINKLE_GPU_PG_CPUS_PER_PROC', '8')

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=2) == 16


@pytest.mark.parametrize('device_type', ['NPU', 'GPU'])
def test_worker_device_order_preserves_group_allocation(monkeypatch, device_type):
    import ray

    # Ray's accelerator IDs need not arrive in ascending order. This allocation reproduces
    # R06's sampler ASCEND_RT_VISIBLE_DEVICES=10,11,7,12 (device_count=0).
    ray_devices = '6,13,8,9,10,11,7,12'

    class Remote:

        def __init__(self, func):
            self.func = func

        def options(self, **kwargs):
            return self

        def remote(self):
            return ray_devices if self.func.__name__ == 'get_visible_devices' else 0

    monkeypatch.setattr(ray, 'remote', Remote)
    monkeypatch.setattr(ray, 'get', lambda value: value)
    monkeypatch.setattr(ray, 'get_runtime_context', lambda: SimpleNamespace(gcs_address='test-gcs'))
    monkeypatch.setattr(ray, 'nodes', lambda: [
        {'NodeID': 'test-node', 'Alive': True, 'Resources': {device_type: 8, 'CPU': 192}},
    ])
    monkeypatch.setattr(ray.util, 'placement_group', lambda bundles: SimpleNamespace(ready=lambda: True))
    monkeypatch.setattr(Platform, 'get_platform', lambda platform=None: type(device_type, (), {}))
    monkeypatch.setattr(ResourceManager, '_publish_addresses', _stub_publish_addresses)
    manager = ResourceManager(
        nproc_per_node=8, ncpu_proc_per_node=8,
        groups=[
            DeviceGroup(name='model', ranks=[0, 1, 2, 3], device_type=device_type),
            DeviceGroup(name='sampler', ranks=[4, 5, 6, 7], device_type=device_type, gpus_per_worker=4),
        ],
    )

    assert [placement['gpu_rank'][0] for placement in manager.get_group('model')] == [6, 13, 8, 9]
    assert manager.get_group('sampler')[0]['gpu_rank'] == [10, 11, 7, 12]
    assert [manager.rank_placements[rank]['gpu_rank'] for rank in range(8)] == [6, 13, 8, 9, 10, 11, 7, 12]


@pytest.mark.parametrize('device_type', ['NPU', 'GPU'])
def test_external_device_allocation_preserves_order(monkeypatch, device_type):
    import ray

    monkeypatch.setattr(ray, 'get_runtime_context', lambda: SimpleNamespace(gcs_address='test-gcs'))
    monkeypatch.setattr(ResourceManager, '_publish_addresses', _stub_publish_addresses)
    devices = ['6', '13', '8', '9', '10', '11', '7', '12']
    manager = ResourceManager.from_allocation(
        DeviceGroup(name='sampler', ranks=2, device_type=device_type, gpus_per_worker=4),
        placement_group=object(), devices=devices,
    )

    assert [placement['gpu_rank'] for placement in manager.get_group('sampler')] == [devices[:4], devices[4:]]
    assert devices == ['6', '13', '8', '9', '10', '11', '7', '12']


@pytest.mark.parametrize('device_type,visible_env,expected', [
    ('NPU', 'ASCEND_RT_VISIBLE_DEVICES', ['6,13', '7,12']),
    ('GPU', 'CUDA_VISIBLE_DEVICES', ['13,6', '12,7']),
])
@pytest.mark.parametrize('placement_source', ['group', 'lend', 'lend_registry'])
def test_worker_visibility_preserves_devices_and_orders_npu_ids(
        monkeypatch, device_type, visible_env, expected, placement_source):
    from unittest.mock import Mock

    import ray.util.scheduling_strategies

    from twinkle.infra._ray.ray_helper import RayHelper

    pg = object()
    devices = ['13', '6', '12', '7']
    # 58b2b979: create_workers awaits worker._twinkle_ready.remote() via ray.get; this test drives the
    # placement/visibility seam with Mock workers, so resolve refs by identity.
    monkeypatch.setattr(ray, 'get', lambda value, **kwargs: value)
    table = {rank: {'gpu_rank': device, 'placement_group': pg, 'node_ip': '127.0.0.1'}
             for rank, device in enumerate(devices)}
    manager = None if placement_source == 'lend_registry' else SimpleNamespace(rank_placements=table)
    monkeypatch.setattr(RayHelper, 'resource_manager', manager)
    monkeypatch.setattr(RayHelper, 'get_config', lambda key: table)
    info = {'device_type': device_type}
    monkeypatch.setattr(RayHelper, 'group_info', lambda group: info)
    monkeypatch.setattr(RayHelper, 'get_master_id_port', lambda placement_group: ('127.0.0.1', 12345))
    monkeypatch.setattr(RayHelper, '_created_actors', [])
    monkeypatch.setattr(Platform, 'get_platform', lambda platform: SimpleNamespace(
        visible_device_env=lambda: visible_env))
    monkeypatch.setattr(ray.util.scheduling_strategies, 'PlacementGroupSchedulingStrategy',
                        lambda **kwargs: SimpleNamespace(**kwargs))
    remote = Mock()
    remote.options.return_value = remote
    monkeypatch.setattr(RayHelper, '_get_remote_component', lambda worker_cls: remote)

    class Worker:
        pass

    placements = RayHelper.rank_placements([0, 1, 2, 3], width=2)
    info['placements'] = placements
    RayHelper.create_workers(
        Worker, 'sampler', 'all', instance_id='visibility',
        placements=None if placement_source == 'group' else placements)

    environments = [call.kwargs['runtime_env']['env_vars'] for call in remote.options.call_args_list]
    assert [env[visible_env] for env in environments] == expected
    assert all(env[f'RAY_EXPERIMENTAL_NOSET_{visible_env}'] == '1' for env in environments)
    assert [placement['ranks'] for placement in placements] == [[0, 1], [2, 3]]
    assert [placement['gpu_rank'] for placement in placements] == [['13', '6'], ['12', '7']]
    assert [table[rank]['gpu_rank'] for rank in range(4)] == devices
