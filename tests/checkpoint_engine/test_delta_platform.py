# Copyright (c) ModelScope Contributors. All rights reserved.
"""Platform resolution of the delta transport and the entry points that use it."""
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from twinkle import DeviceMesh, Platform
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.checkpoint_engine.delta import DeltaSyncConfig, resolve_delta_engine_cls
from twinkle.checkpoint_engine.delta.trainer import mixin as delta_trainer
from twinkle.checkpoint_engine.delta.trainer.backends.fsdp import validate_fsdp2
from twinkle.checkpoint_engine.delta.transport.nccl_engine import DeltaShardedCheckpointEngine
from twinkle.checkpoint_engine.delta.trainer.mixin import DeltaTrainerMixin
from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy

HCCL_MODULE = 'twinkle.checkpoint_engine.delta.transport.hccl_engine'
NOT_IMPLEMENTED = 'not implemented yet'


def fake_platform(monkeypatch, name):
    cls = type(name, (), {})
    monkeypatch.setattr(Platform, 'get_platform', staticmethod(lambda platform=None: cls))
    return cls


# ---- resolver ----

def test_resolver_gpu():
    assert resolve_delta_engine_cls('GPU') is DeltaShardedCheckpointEngine


def test_resolver_npu_real_engine():
    from twinkle.checkpoint_engine.delta.transport.hccl_engine import DeltaHCCLCheckpointEngine
    assert resolve_delta_engine_cls('NPU') is DeltaHCCLCheckpointEngine


def test_resolver_npu_module_missing_not_implemented(monkeypatch):
    # sys.modules[name] = None makes the import raise ModuleNotFoundError(name=HCCL_MODULE)
    monkeypatch.setitem(sys.modules, HCCL_MODULE, None)
    with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
        resolve_delta_engine_cls('NPU')


def test_resolver_npu_returns_engine_class(monkeypatch):
    module = types.ModuleType(HCCL_MODULE)
    module.DeltaHCCLCheckpointEngine = type('DeltaHCCLCheckpointEngine', (), {})
    monkeypatch.setitem(sys.modules, HCCL_MODULE, module)
    assert resolve_delta_engine_cls('NPU') is module.DeltaHCCLCheckpointEngine


def test_resolver_does_not_mask_other_import_errors(monkeypatch):
    module = types.ModuleType(HCCL_MODULE)

    def boom(name):
        raise ImportError('torch_npu is missing')

    module.__getattr__ = boom
    monkeypatch.setitem(sys.modules, HCCL_MODULE, module)
    with pytest.raises(ImportError, match='torch_npu is missing') as info:
        resolve_delta_engine_cls('NPU')
    assert not isinstance(info.value, NotImplementedError)


def test_resolver_unknown_platform():
    with pytest.raises(ValueError, match=r"'CPU'.*GPU, NPU"):
        resolve_delta_engine_cls('CPU')


# ---- manager ----

def actors():
    return Mock(_actors=[1, 2]), Mock(_actors=[3])


def test_manager_gpu_still_constructs(monkeypatch):
    fake_platform(monkeypatch, 'GPU')
    manager = CheckpointEngineManager(*actors(), platform='GPU', mode='standalone', sync_backend='delta_sharded')
    assert manager.delta_engine_cls is DeltaShardedCheckpointEngine


def test_manager_full_sync_does_not_resolve_delta(monkeypatch):
    fake_platform(monkeypatch, 'GPU')
    manager = CheckpointEngineManager(*actors(), platform='GPU', mode='standalone')
    assert manager.delta_engine_cls is None


def test_manager_npu_constructs_with_hccl_engine(monkeypatch):
    fake_platform(monkeypatch, 'NPU')
    from twinkle.checkpoint_engine.delta.transport.hccl_engine import DeltaHCCLCheckpointEngine
    manager = CheckpointEngineManager(*actors(), platform='NPU', mode='standalone', sync_backend='delta_sharded')
    assert manager.delta_engine_cls is DeltaHCCLCheckpointEngine


def test_manager_npu_module_missing_fails_early(monkeypatch):
    fake_platform(monkeypatch, 'NPU')
    monkeypatch.setitem(sys.modules, HCCL_MODULE, None)
    with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
        CheckpointEngineManager(*actors(), platform='NPU', mode='standalone', sync_backend='delta_sharded')


def test_manager_npu_resolves_when_engine_available(monkeypatch):
    fake_platform(monkeypatch, 'NPU')
    module = types.ModuleType(HCCL_MODULE)
    module.DeltaHCCLCheckpointEngine = type('DeltaHCCLCheckpointEngine', (), {})
    monkeypatch.setitem(sys.modules, HCCL_MODULE, module)
    manager = CheckpointEngineManager(*actors(), platform='NPU', mode='standalone', sync_backend='delta_sharded')
    assert manager.delta_engine_cls is module.DeltaHCCLCheckpointEngine


def test_manager_non_standalone_still_rejected(monkeypatch):
    fake_platform(monkeypatch, 'NPU')
    with pytest.raises(ValueError, match='standalone'):
        CheckpointEngineManager(SimpleNamespace(), SimpleNamespace(), platform='NPU', sync_backend='delta_sharded')


def test_manager_unknown_platform(monkeypatch):
    fake_platform(monkeypatch, 'CPU')
    # standalone backend selection rejects it first; either way construction fails early
    with pytest.raises((ValueError, NotImplementedError)):
        CheckpointEngineManager(*actors(), platform='CPU', mode='standalone', sync_backend='delta_sharded')


# ---- trainer ----

class Trainer(DeltaTrainerMixin):

    def validate_delta_backend(self):
        pass


@pytest.mark.parametrize('name,device', [('GPU', 'cuda'), ('NPU', 'npu')])
def test_trainer_device_follows_platform(monkeypatch, name, device):
    fake_platform(monkeypatch, name)
    created = {}

    class FakeEngine:
        is_master = False

        def __init__(self, **kwargs):
            created.update(kwargs)

        def prepare(self):
            return 'metadata'

    monkeypatch.setattr(delta_trainer, 'resolve_delta_engine_cls', lambda platform: FakeEngine)
    monkeypatch.setattr(delta_trainer.dist, 'is_initialized', lambda: False)
    trainer = Trainer()
    trainer._lazy_wrap_model = Mock()
    result = trainer.prepare_delta_sync(DeltaSyncConfig())
    assert trainer._delta_device == device
    assert result == {'device': device, 'master_metadata': 'metadata'}
    assert isinstance(trainer._delta_engine, FakeEngine)


def test_trainer_npu_without_transport_fails_before_process_group(monkeypatch):
    fake_platform(monkeypatch, 'NPU')
    monkeypatch.setitem(sys.modules, HCCL_MODULE, None)
    new_group = Mock()
    monkeypatch.setattr(delta_trainer.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(delta_trainer.dist, 'new_group', new_group)
    with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
        Trainer()._initialize_delta_engine(DeltaSyncConfig())
    new_group.assert_not_called()


# ---- validate_fsdp2 ----

def fsdp_owner(monkeypatch, name, **config):
    fake_platform(monkeypatch, name)
    model = SimpleNamespace(config=SimpleNamespace(**config), state_dict=lambda: {'w': torch.ones(4)})
    strategy = Mock(spec=NativeFSDPStrategy)
    strategy.unwrap_model.return_value = model
    mesh = DeviceMesh.from_sizes(world_size=1, fsdp_size=1, device_type='cuda')
    return SimpleNamespace(strategy=strategy, model=model, device_mesh=mesh)


def test_validate_fsdp2_accepts_npu(monkeypatch):
    validate_fsdp2(fsdp_owner(monkeypatch, 'NPU'))


def test_validate_fsdp2_rejects_unknown_platform(monkeypatch):
    with pytest.raises(ValueError, match='GPU or NPU'):
        validate_fsdp2(fsdp_owner(monkeypatch, 'CPU'))


@pytest.mark.parametrize('name', ['GPU', 'NPU'])
def test_validate_fsdp2_rejects_moe(monkeypatch, name):
    with pytest.raises(ValueError, match='MoE'):
        validate_fsdp2(fsdp_owner(monkeypatch, name, num_experts=8))
