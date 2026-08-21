import pytest
import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
    _NPU_OK = True
except ImportError:
    _NPU_OK = False


def test_swiglu_imports():
    from twinkle.kernel.ops.swiglu.npu import npu_swiglu_forward
    assert callable(npu_swiglu_forward)


def test_swiglu_signature():
    import inspect

    from twinkle.kernel.ops.swiglu.npu import npu_swiglu_forward

    params = list(inspect.signature(npu_swiglu_forward).parameters)
    assert params == ['self', 'hidden_state']


@pytest.mark.skipif(not _NPU_OK, reason='torch_npu unavailable')
def test_npu_swiglu_matches_torch_reference():
    """Numerical parity: npu_swiglu(cat(gate, up)) ~= silu(gate(x)) * up(x), then down_proj."""
    import torch.nn.functional as F

    from twinkle.kernel.ops.swiglu.npu import npu_swiglu_forward

    class _Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(32, 64, bias=False)
            self.up_proj = nn.Linear(32, 64, bias=False)
            self.down_proj = nn.Linear(64, 32, bias=False)

    m = _Mlp().to('npu')
    x = torch.randn(2, 32, device='npu')
    ref = m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x))
    torch.testing.assert_close(npu_swiglu_forward(m, x), ref, rtol=1e-4, atol=1e-5)