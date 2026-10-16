# Copyright (c) Alibaba, Inc. and its affiliates.
from twinkle.infra._ray.resource_manager import _gpu_placement_group_cpus


def test_gpu_placement_group_cpus_uses_per_process_default(monkeypatch):
    monkeypatch.delenv('TWINKLE_GPU_PG_CPUS_PER_PROC', raising=False)

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=2) == 8


def test_gpu_placement_group_cpus_respects_node_cap(monkeypatch):
    monkeypatch.setenv('TWINKLE_GPU_PG_CPUS_PER_PROC', '16')

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=8) == 48


def test_gpu_placement_group_cpus_can_be_configured(monkeypatch):
    monkeypatch.setenv('TWINKLE_GPU_PG_CPUS_PER_PROC', '8')

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=2) == 16
