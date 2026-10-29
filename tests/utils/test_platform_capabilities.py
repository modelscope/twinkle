# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the ``Platform`` backend-capability queries.

These pin the contract that replaced the scattered
``Platform.get_platform().__name__ == 'XPU'`` string comparisons: the base
class provides the CUDA-like default and each platform overrides only what
actually differs.
"""
import pytest

from twinkle.checkpoint_engine import (CheckpointEngineManager, HCCLCheckpointEngine, NCCLCheckpointEngine,
                                       XCCLCheckpointEngine)
from twinkle.utils.platforms import GPU, MPS, NPU, XPU, Platform


def test_gpu_uses_base_defaults():
    # Ray autodetects CUDA devices and NCCL carries a bound device id fine,
    # so GPU must not override either capability.
    assert GPU.requires_unbound_default_pg() is False
    assert GPU.requires_unbound_pg_for_subgroups() is False
    assert GPU.ray_resources() == {}
    assert GPU.bootstrap() is None


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
