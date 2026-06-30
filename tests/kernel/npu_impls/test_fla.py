def test_fla_imports():
    from twinkle.kernel.npu_impls.fla import apply_qwen3_5_fla
    assert callable(apply_qwen3_5_fla)


def test_fla_disabled_by_env(monkeypatch):
    monkeypatch.setenv('TWINKLE_NPU_FLA', '0')
    from twinkle.kernel.npu_impls.fla import apply_qwen3_5_fla
    # With env=0, function returns 0 (no-op) without raising
    assert apply_qwen3_5_fla(None) == 0


def test_fla_skips_when_no_torch_npu(monkeypatch):
    import sys
    monkeypatch.setenv('TWINKLE_NPU_FLA', '1')
    monkeypatch.setitem(sys.modules, 'torch_npu', None)  # forces ImportError on import
    from twinkle.kernel.npu_impls import fla as fla_mod
    # Reload-tolerant: should return 0 when torch_npu is missing.
    assert fla_mod.apply_qwen3_5_fla(None) == 0


def test_fla_does_not_flip_flag_when_mindspeed_missing(monkeypatch):
    """On an NPU host where the MindSpeed FLA kernel cannot be imported,
    ``apply_qwen3_5_fla`` must NOT flip the global ``is_flash_linear_attention_available``
    flag — otherwise HF transformers would route Qwen3.5 onto a FLA fast path
    whose kernel is not installed (runtime failure)."""
    import sys
    import types

    import transformers.utils as tu

    monkeypatch.setenv('TWINKLE_NPU_FLA', '1')
    # Fake torch_npu as importable (with a real __spec__ so find_spec doesn't trip)
    import importlib.util
    spec = importlib.util.spec_from_loader('torch_npu', loader=None)
    fake_npu = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'torch_npu', fake_npu)
    # Stub causal_conv1d so the heavy real import chain doesn't run
    fake_conv = types.ModuleType('twinkle.kernel.causal_conv1d')
    fake_conv.npu_causal_conv1d_fn = object()
    monkeypatch.setitem(sys.modules, 'twinkle.kernel.causal_conv1d', fake_conv)
    # Force the MindSpeed-backed module import to fail
    monkeypatch.setitem(sys.modules, 'twinkle.kernel.chunk_gated_delta_rule', None)

    original_flag = tu.is_flash_linear_attention_available
    try:
        from twinkle.kernel.npu_impls.fla import apply_qwen3_5_fla
        assert apply_qwen3_5_fla(None) == 0
        assert tu.is_flash_linear_attention_available is original_flag, (
            'is_flash_linear_attention_available was flipped to True while the '
            'MindSpeed kernel is unavailable — this would break Qwen3.5 at runtime.'
        )
    finally:
        # Defensive cleanup in case the buggy path ran.
        tu.is_flash_linear_attention_available = original_flag