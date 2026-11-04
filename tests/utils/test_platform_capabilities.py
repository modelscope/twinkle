# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the ``Platform`` backend-capability queries.

These pin the contract that replaced the scattered
``Platform.get_platform().__name__ == 'XPU'`` string comparisons: the base
class provides the CUDA-like default and each platform overrides only what
actually differs.
"""
import pytest

from types import SimpleNamespace

from twinkle.checkpoint_engine import (CheckpointEngineManager, HCCLCheckpointEngine, NCCLCheckpointEngine,
                                       XCCLCheckpointEngine)
from twinkle.utils.platforms import GPU, Metax, MPS, NPU, Platform, XPU


def test_gpu_uses_base_defaults():
    # Ray autodetects CUDA devices and NCCL carries a bound device id fine,
    # so GPU must not override either capability.
    assert GPU.requires_unbound_default_pg() is False
    assert GPU.requires_unbound_pg_for_subgroups() is False
    assert GPU.ray_resources() == {}
    assert GPU.bootstrap() is None


def test_npu_overrides_device_capability_hooks():
    """The B-class NPU branches now live on the platform, not in components."""
    assert NPU.requires_gloo_subgroups() is True
    assert NPU.synthesizes_causal_mask() is True
    assert NPU.packs_padded_cu_seqlens() is True
    # GPU keeps CUDA defaults for the same hooks.
    assert GPU.requires_gloo_subgroups() is False
    assert GPU.synthesizes_causal_mask() is False
    assert GPU.packs_padded_cu_seqlens() is False


def test_npu_overrides_c_class_workarounds():
    """The C-class NPU workarounds (dtype walk, mcore flags, object-collective
    group) also route through platform hooks; base defaults keep GPU behaviour."""
    assert NPU.aligns_param_dtype_to_base() is True
    assert GPU.aligns_param_dtype_to_base() is False
    # resolve_object_collective_group must respect the #185 FSDP guard.
    fsdp_mesh = SimpleNamespace(has_dim=lambda dim: dim == 'fsdp')
    assert NPU.resolve_object_collective_group(fsdp_mesh) is None
    assert GPU.resolve_object_collective_group(SimpleNamespace()) is None


class _RecordingConfig:

    def __init__(self):
        self.use_flash_attn = False


def test_configure_model_config_default_noop(monkeypatch):
    """GPU must not touch a Megatron ModelConfig; NPU cannot be exercised without
    MindSpeed, so only the default side is asserted here."""
    config = _RecordingConfig()
    GPU.configure_model_config(config)
    assert config.use_flash_attn is False


def test_device_module_dispatch():
    """The torch.<x> namespace used by the Torch API chain comes from the platform."""
    import torch

    assert GPU.device_module() is torch.cuda
    assert GPU.supports_device_ipc() is True
    # NPU cannot be probed on this host (npu-smi absent); pin that the hooks are
    # explicit overrides so the delegating implementations cannot be lost silently.
    for capability in ('device_module', 'supports_device_ipc', 'ensure_ipc_reducer', 'apply_determinism'):
        assert capability in NPU.__dict__, f'NPU must override {capability} explicitly'
    # CPU-only fakes cannot be constructed here; None is the base default.
    assert MPS.device_module() is None


@pytest.mark.parametrize('platform_cls', [NPU, XPU])
def test_accelerators_require_unbound_default_pg(platform_cls):
    assert platform_cls.requires_unbound_default_pg() is True


def test_only_xpu_guards_subgroup_creation():
    """The pre-``new_group`` guard is XPU-only and must stay that way.

    NPU clears the binding once after ``init_process_group`` (to keep Gloo DP
    subgroups decoupled); it never needed the second guard before Megatron's
    ``new_group``, so answering ``True`` here would change NPU behaviour.
    """
    assert XPU.requires_unbound_pg_for_subgroups() is True
    assert NPU.requires_unbound_pg_for_subgroups() is False


def test_xpu_overrides_capabilities_explicitly():
    """XPU subclasses GPU for implementation reuse, not behavioural equality.

    Any capability that differs must appear in ``XPU.__dict__``; silently
    inheriting GPU's answer is the failure mode this guards against (e.g.
    inheriting the NCCL checkpoint engine deadlocks weight sync on BKCL).
    """
    assert issubclass(XPU, GPU), 'XPU is expected to reuse the cuda-alike GPU implementation'
    for capability in ('bootstrap', 'requires_unbound_default_pg', 'requires_unbound_pg_for_subgroups',
                       'ray_resources', 'checkpoint_engine'):
        assert capability in XPU.__dict__, f'XPU must override {capability} explicitly'


def test_metax_keeps_gpu_behaviour():
    """Metax (MACA) is CUDA-compatible at the torch level, so it inherits GPU
    wholesale -- including the NCCL checkpoint engine, which ms-swift validated
    on real Metax hardware. The only Metax-specific override is the vLLM patch
    hook."""
    assert issubclass(Metax, GPU)
    assert Metax.requires_unbound_default_pg() is False
    assert Metax.requires_unbound_pg_for_subgroups() is False
    assert Metax.ray_resources() == {}
    assert Metax.checkpoint_engine() == (NCCLCheckpointEngine, {})
    assert 'ensure_vllm_patch' in Metax.__dict__


def test_checkpoint_engine_per_platform():
    assert GPU.checkpoint_engine() == (NCCLCheckpointEngine, {})
    assert XPU.checkpoint_engine() == (XCCLCheckpointEngine, {})
    # NPU reuses its HCCL communicator across sync steps; the option lives
    # with the platform so driver and worker cannot disagree about it.
    assert NPU.checkpoint_engine() == (HCCLCheckpointEngine, {'rebuild_group': False})


def test_platform_without_checkpoint_engine_raises():
    with pytest.raises(NotImplementedError):
        MPS.checkpoint_engine()


def test_manager_and_worker_agree_on_engine():
    """Driver-side and worker-side engine selection must resolve to the same class.

    ``CheckpointEngineManager`` used to default to ``platform='GPU'``, which
    always resolves to the NVIDIA platform and therefore picked the NCCL
    engine even on XPU/NPU, while the workers built the XCCL/HCCL one.
    """
    assert CheckpointEngineManager.decide_backend_engine() is Platform.get_platform().checkpoint_engine()[0]
