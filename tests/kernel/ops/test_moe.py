import pytest
import torch
from torch import nn


def test_moe_imports():
    from twinkle.kernel.ops.moe.npu import (
        GmmFunction,
        npu_grouped_mm,
        npu_packed_moe_experts_forward,
        npu_qwen3_5_moe_sparse_block_forward,
    )
    assert issubclass(GmmFunction, torch.autograd.Function)
    assert callable(npu_grouped_mm)
    assert callable(npu_packed_moe_experts_forward)
    assert callable(npu_qwen3_5_moe_sparse_block_forward)


class _PackedExperts(nn.Module):

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_proj)
        self.down_proj = nn.Parameter(down_proj)


def test_normalize_packed_expert_weights_resolves_square_dsv4_gate_from_down_proj():
    from twinkle.kernel.ops.moe.npu import _normalize_packed_expert_weights

    # DeepSeek-V4 relation: hidden == 2 * intermediate. gate_up_proj is
    # therefore square even though it still uses F.linear [out, in] layout.
    experts, hidden, intermediate = 2, 8, 4
    gate_up = torch.arange(experts * hidden * hidden, dtype=torch.float32).reshape(experts, hidden, hidden)
    down = torch.arange(experts * hidden * intermediate,
                        dtype=torch.float32).reshape(experts, hidden, intermediate)
    module = _PackedExperts(gate_up, down)

    normalized_gate_up, normalized_down = _normalize_packed_expert_weights(module, torch.float32, hidden)

    assert torch.equal(normalized_gate_up, gate_up.transpose(1, 2))
    assert torch.equal(normalized_down, down.transpose(1, 2))


def test_normalize_packed_expert_weights_keeps_grouped_mm_layout():
    from twinkle.kernel.ops.moe.npu import _normalize_packed_expert_weights

    experts, hidden, intermediate = 2, 8, 4
    gate_up = torch.randn(experts, hidden, intermediate * 2)
    down = torch.randn(experts, intermediate, hidden)
    module = _PackedExperts(gate_up, down)

    normalized_gate_up, normalized_down = _normalize_packed_expert_weights(module, torch.float32, hidden)

    assert torch.equal(normalized_gate_up, gate_up)
    assert torch.equal(normalized_down, down)


def test_normalize_packed_expert_weights_rejects_inconsistent_layout():
    from twinkle.kernel.ops.moe.npu import _normalize_packed_expert_weights

    module = _PackedExperts(torch.randn(2, 7, 9), torch.randn(2, 5, 6))
    with pytest.raises(RuntimeError, match='Unable to determine packed expert weight layout'):
        _normalize_packed_expert_weights(module, torch.float32, hidden_dim=8)
