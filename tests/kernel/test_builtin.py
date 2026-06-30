import importlib.machinery
import sys
import types

import torch
import torch.nn as nn

import pytest


def _fake_module(name: str):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


def test_npu_builtin_returns_dict():
    from twinkle.kernel.builtin import npu_builtin
    bundle = npu_builtin()
    assert isinstance(bundle, dict)
    assert len(bundle) > 0


def test_npu_builtin_values_are_npu_gated():
    """Every value in npu_builtin() must be wrapped in {'npu': ...} so it's
    safely no-op on CUDA/CPU."""
    from twinkle.kernel.builtin import npu_builtin
    for key, value in npu_builtin().items():
        assert isinstance(value, dict), f'value for {key!r} is not a device-dict'
        assert 'npu' in value, f'value for {key!r} is missing npu entry'


def test_npu_builtin_compose_with_user_override():
    """User-supplied keys override the builtin (via plain dict merge)."""
    from twinkle.kernel.builtin import npu_builtin
    sentinel = object()
    merged = {**npu_builtin(), 'fake.module.path.fn': sentinel}
    assert merged['fake.module.path.fn'] is sentinel


def test_npu_builtin_safe_on_cpu_model():
    """kernelize(cpu_model, npu_builtin()) must not raise and not modify."""
    from twinkle.kernel import kernelize
    from twinkle.kernel.builtin import npu_builtin

    m = nn.Sequential(nn.Linear(2, 2))
    pre_type = type(m[0])
    out = kernelize(m, npu_builtin())
    assert out is m
    assert type(m[0]) is pre_type  # no replacement happened (cpu device)


def test_npu_builtin_skips_missing_modeling_modules():
    """If transformers.models.qwen3_5 is not installed, the bundle must
    still produce a dict (with whatever subset is available)."""
    from twinkle.kernel.builtin import npu_builtin
    bundle = npu_builtin()  # must not raise
    assert isinstance(bundle, dict)


def test_npu_builtin_does_not_overwrite_global_sdpa_on_non_npu_host(monkeypatch):
    """Calling npu_builtin() on a CUDA/CPU host must not contaminate the
    global HF SDPA registry. The NPU impl inverts boolean masks, which is
    wrong for non-NPU execution."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from twinkle.kernel.builtin import npu_builtin
    from twinkle.utils.device_mesh import Platform

    monkeypatch.setattr(Platform, 'device_prefix', staticmethod(lambda platform=None: 'cuda'))
    original = ALL_ATTENTION_FUNCTIONS.get('sdpa')
    npu_builtin()
    assert ALL_ATTENTION_FUNCTIONS.get('sdpa') is original


def test_npu_builtin_skips_side_effects_on_non_npu_platform(monkeypatch):
    from twinkle.kernel import builtin
    from twinkle.kernel.npu_impls import fla
    from twinkle.utils.device_mesh import Platform

    installs = []
    fla_calls = []
    monkeypatch.setattr(Platform, 'device_prefix', staticmethod(lambda platform=None: 'cuda'))
    monkeypatch.setitem(sys.modules, 'torch_npu', _fake_module('torch_npu'))
    monkeypatch.setattr(builtin, '_install_sdpa', lambda impl: installs.append(impl))
    monkeypatch.setattr(fla, 'apply_qwen3_5_fla', lambda model: fla_calls.append(model))

    builtin.npu_builtin(nn.Linear(1, 1))

    assert installs == []
    assert fla_calls == []
