# Copyright (c) Alibaba, Inc. and its affiliates.
from types import SimpleNamespace

import pytest

from twinkle import DeviceGroup, Platform
from twinkle.infra._ray.resource_manager import ResourceManager, _gpu_placement_group_cpus


def test_gpu_placement_group_cpus_uses_per_process_default(monkeypatch):
    monkeypatch.delenv('TWINKLE_GPU_PG_CPUS_PER_PROC', raising=False)

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=2) == 8


def test_gpu_placement_group_cpus_respects_node_cap(monkeypatch):
    monkeypatch.setenv('TWINKLE_GPU_PG_CPUS_PER_PROC', '16')

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=8) == 48


def test_gpu_placement_group_cpus_can_be_configured(monkeypatch):
    monkeypatch.setenv('TWINKLE_GPU_PG_CPUS_PER_PROC', '8')

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=2) == 16


@pytest.mark.parametrize('device_type,expected_model,expected_sampler', [
    ('NPU', [6, 13, 8, 9], [7, 10, 11, 12]),
    ('GPU', [6, 13, 8, 9], [10, 11, 7, 12]),
])
def test_worker_device_order_preserves_group_allocation(
        monkeypatch, device_type, expected_model, expected_sampler):
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
    manager = ResourceManager(
        nproc_per_node=8, ncpu_proc_per_node=8,
        groups=[
            DeviceGroup(name='model', ranks=[0, 1, 2, 3], device_type=device_type),
            DeviceGroup(name='sampler', ranks=[4, 5, 6, 7], device_type=device_type, gpus_per_worker=4),
        ],
    )

    assert [placement['gpu_rank'][0] for placement in manager.get_group('model')] == expected_model
    assert manager.get_group('sampler')[0]['gpu_rank'] == expected_sampler
    assert [manager.rank_placements[rank]['gpu_rank'] for rank in range(8)] == [6, 13, 8, 9, 10, 11, 7, 12]


@pytest.mark.parametrize('device_type,expected', [
    ('NPU', [['6', '8', '9', '13'], ['7', '10', '11', '12']]),
    ('GPU', [['6', '13', '8', '9'], ['10', '11', '7', '12']]),
])
def test_external_device_allocation_is_ordered_numerically_for_npu(monkeypatch, device_type, expected):
    import ray

    monkeypatch.setattr(ray, 'get_runtime_context', lambda: SimpleNamespace(gcs_address='test-gcs'))
    devices = ['6', '13', '8', '9', '10', '11', '7', '12']
    manager = ResourceManager.from_allocation(
        DeviceGroup(name='sampler', ranks=2, device_type=device_type, gpus_per_worker=4),
        placement_group=object(), devices=devices,
    )

    assert [placement['gpu_rank'] for placement in manager.get_group('sampler')] == expected
    assert devices == ['6', '13', '8', '9', '10', '11', '7', '12']
