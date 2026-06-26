import pytest
import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
    _NPU_OK = True
except ImportError:
    _NPU_OK = False


def test_imports():
    """NpuRMSNorm and npu_gated_rms_norm_forward import without torch_npu."""
    from twinkle.kernel.npu_impls.rms_norm import NpuRMSNorm, npu_gated_rms_norm_forward
    assert NpuRMSNorm is not None
    assert callable(npu_gated_rms_norm_forward)


def test_npu_rmsnorm_has_no_init():
    """Class-replacement contract: NpuRMSNorm must not define its own __init__."""
    from twinkle.kernel.npu_impls.rms_norm import NpuRMSNorm
    # If NpuRMSNorm defines __init__, it'd appear in NpuRMSNorm.__dict__
    assert '__init__' not in NpuRMSNorm.__dict__


@pytest.mark.skipif(not _NPU_OK, reason='torch_npu unavailable')
def test_npu_rmsnorm_forward_runs_on_npu():
    from twinkle.kernel.npu_impls.rms_norm import NpuRMSNorm

    class _Orig(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(8))
            self.variance_epsilon = 1e-6

    m = _Orig().to('npu')
    m.__class__ = NpuRMSNorm
    x = torch.randn(2, 8, device='npu')
    y = m(x)
    assert y.shape == (2, 8)