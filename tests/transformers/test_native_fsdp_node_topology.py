import torch.distributed as dist

from twinkle.model.transformers.strategy.native_fsdp import (
    NativeFSDPStrategy,
    _get_local_rank_info,
)


_TOPOLOGY_ENV_NAMES = (
    'TWINKLE_NODE_LOCAL_RANK',
    'TWINKLE_NODE_LOCAL_WORLD_SIZE',
    'TWINKLE_NODE_RANKS',
    'LOCAL_RANK',
    'LOCAL_WORLD_SIZE',
    'LOCAL_SIZE',
)


def _clear_topology_env(monkeypatch):
    for name in _TOPOLOGY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_native_fsdp_uses_ray_node_rank_instead_of_actor_device_index(monkeypatch):
    _clear_topology_env(monkeypatch)
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('TWINKLE_NODE_LOCAL_RANK', '1')
    monkeypatch.setenv('TWINKLE_NODE_LOCAL_WORLD_SIZE', '2')
    monkeypatch.setenv('TWINKLE_NODE_RANKS', '0,1')
    monkeypatch.setattr(dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 2)

    strategy = NativeFSDPStrategy(device_mesh=None, memory_efficient_init=True)

    assert strategy.is_node_local_source_rank() is False
    assert _get_local_rank_info() == (1, 2, 0, [0, 1])


def test_native_fsdp_accepts_explicit_noncontiguous_node_ranks(monkeypatch):
    _clear_topology_env(monkeypatch)
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('TWINKLE_NODE_LOCAL_RANK', '1')
    monkeypatch.setenv('TWINKLE_NODE_LOCAL_WORLD_SIZE', '2')
    monkeypatch.setenv('TWINKLE_NODE_RANKS', '0,2')
    monkeypatch.setattr(dist, 'get_rank', lambda: 2)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 3)

    assert _get_local_rank_info() == (2, 3, 0, [0, 2])


def test_native_fsdp_keeps_torchrun_local_rank_fallback(monkeypatch):
    _clear_topology_env(monkeypatch)
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '2')
    monkeypatch.setattr(dist, 'get_rank', lambda: 1)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 2)

    assert _get_local_rank_info() == (1, 2, 0, [0, 1])
