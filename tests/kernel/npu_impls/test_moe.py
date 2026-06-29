def test_moe_imports():
    from twinkle.kernel.npu_impls.moe import (
        GmmFunction,
        npu_grouped_mm,
        npu_packed_moe_experts_forward,
        npu_qwen3_5_moe_sparse_block_forward,
    )
    import torch
    assert issubclass(GmmFunction, torch.autograd.Function)
    assert callable(npu_grouped_mm)
    assert callable(npu_packed_moe_experts_forward)
    assert callable(npu_qwen3_5_moe_sparse_block_forward)