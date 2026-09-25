def test_fla_imports():
    from twinkle.kernel.ops.fla.npu import apply_qwen3_5_fla
    assert callable(apply_qwen3_5_fla)


def test_fla_disabled_by_env(monkeypatch):
    monkeypatch.setenv('TWINKLE_NPU_FLA', '0')
    from twinkle.kernel.ops.fla.npu import apply_qwen3_5_fla
    # With env=0, function returns 0 (no-op) without raising
    assert apply_qwen3_5_fla(None) == 0


def test_fla_skips_when_no_torch_npu(monkeypatch):
    import sys
    monkeypatch.setenv('TWINKLE_NPU_FLA', '1')
    monkeypatch.setitem(sys.modules, 'torch_npu', None)  # forces ImportError on import
    from twinkle.kernel.ops.fla import npu as fla_mod
    # Reload-tolerant: should return 0 when torch_npu is missing.
    assert fla_mod.apply_qwen3_5_fla(None) == 0


def test_fla_does_not_flip_flag_when_fla_missing(monkeypatch):
    """On an NPU host where the fla native operators cannot be imported,
    ``apply_qwen3_5_fla`` must NOT flip the global ``is_flash_linear_attention_available``
    flag — otherwise HF transformers would route Qwen3.5 onto a FLA fast path
    whose kernel is not installed (runtime failure)."""
    import sys

    import transformers.utils.import_utils as tui

    monkeypatch.setenv('TWINKLE_NPU_FLA', '1')
    # Fake torch_npu as importable (with a real __spec__ so find_spec doesn't trip)
    import importlib.util
    spec = importlib.util.spec_from_loader('torch_npu', loader=None)
    fake_npu = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'torch_npu', fake_npu)
    # Force the fla-backed operator import to fail. Stubbing the top-level ``fla``
    # package is not enough on a host where fla is installed: once a submodule is
    # in sys.modules, ``from fla.x.y import z`` resolves straight from the cache
    # without ever touching the parent, so the submodules are stubbed as None too.
    for fla_mod in ('fla', 'fla.modules', 'fla.modules.convolution', 'fla.ops', 'fla.ops.gated_delta_rule'):
        monkeypatch.setitem(sys.modules, fla_mod, None)

    original_flag = tui.is_flash_linear_attention_available
    try:
        from twinkle.kernel.ops.fla.npu import apply_qwen3_5_fla
        assert apply_qwen3_5_fla(None) == 0
        assert tui.is_flash_linear_attention_available is original_flag, (
            'is_flash_linear_attention_available was flipped to True while the '
            'fla native kernel is unavailable — this would break Qwen3.5 at runtime.'
        )
    finally:
        # Defensive cleanup in case the buggy path ran.
        tui.is_flash_linear_attention_available = original_flag
