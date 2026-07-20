import os

import torch
import torch.nn as nn
import pytest

import twinkle.kernel as k

try:
    import liger_kernel  # noqa: F401
    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False

requires_liger = pytest.mark.skipif(not _HAS_LIGER, reason='liger_kernel not installed')


@requires_liger
def test_liger_builtin_returns_dict():
    bundle = k.liger_builtin()
    assert isinstance(bundle, dict)
    assert len(bundle) > 0


@requires_liger
def test_liger_builtin_values_are_bare_impls():
    """unlike npu_builtin (which wraps in {'npu': impl}), liger_builtin emits
    *bare* impls because Liger self-dispatches across CUDA/NPU internally."""
    for key, value in k.liger_builtin().items():
        assert not isinstance(value, dict), f'value for {key!r} must be bare, got a device-dict'


@requires_liger
def test_liger_builtin_compose_with_user_override():
    sentinel = object()
    merged = {**k.liger_builtin(), 'fake.module.path.fn': sentinel}
    assert merged['fake.module.path.fn'] is sentinel


@requires_liger
def test_liger_builtin_skips_missing_modeling_modules(monkeypatch):
    """If every transformers modeling module is unimportable, the bundle must
    still return a dict (empty) rather than raising."""
    import twinkle.kernel.liger as liger_mod

    monkeypatch.setattr(liger_mod, '_import_optional', lambda name: None)
    bundle = liger_mod.liger_builtin()
    assert isinstance(bundle, dict)
    assert bundle == {}


@requires_liger
def test_liger_builtin_no_global_side_effects():
    """Liger's bundle must not mutate process-global state — contrast with
    npu_builtin which installs a global SDPA override. Concretely, the
    ``torch.nn.functional.cross_entropy`` callable (the thing a fused-linear-CE
    swap would touch) must be byte-identical before and after."""
    import torch.nn.functional as F

    before = F.cross_entropy
    k.liger_builtin()
    assert F.cross_entropy is before


def test_kernelize_class_replacement_applies():
    """kernelize must swap __class__ on a module whose type matches a bundle
    class key (independent of which families are installed)."""
    import torch.nn as nn

    class _FakeHFNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4))

        def forward(self, x):
            return x

    class _FakeLigerImpl(nn.Module):
        def forward(self, x):
            return x + 1

    parent = nn.Sequential(_FakeHFNorm())
    out = k.kernelize(parent, {_FakeHFNorm: _FakeLigerImpl})
    assert out is parent
    assert type(parent[0]) is _FakeLigerImpl


# ── CLI toggle (liger-independent: part of twinkle core) ──────────────────────


def test_enable_liger_defaults_false():
    from twinkle.cli import ModelArgs

    assert ModelArgs().enable_liger is False


def test_enable_liger_toggle_via_cli():
    from twinkle.cli import CLI

    on = CLI.from_args(['--enable-liger'])
    off = CLI.from_args(['--no-enable-liger'])
    assert on.model.enable_liger is True
    assert off.model.enable_liger is False


def test_enable_liger_env_var():
    from twinkle.cli import CLI

    old = os.environ.copy()
    try:
        os.environ.clear()
        os.environ['TWINKLE_ENABLE_LIGER'] = 'true'
        assert CLI.from_args(argv=[]).model.enable_liger is True
    finally:
        os.environ.clear()
        os.environ.update(old)
