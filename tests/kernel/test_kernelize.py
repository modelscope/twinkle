import sys
import types

import pytest
import torch
import torch.nn as nn

from twinkle.kernel.core import HubRef, kernelize


class _SrcLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


class _DstLayer(nn.Module):
    def forward(self, x):
        return x + 100


def test_kernelize_class_to_class_replacement():
    parent = nn.Sequential(_SrcLayer(), _SrcLayer())
    out = kernelize(parent, {_SrcLayer: _DstLayer})
    assert out is parent
    assert type(parent[0]) is _DstLayer
    assert type(parent[1]) is _DstLayer


def test_kernelize_empty_mapping_returns_model():
    m = _SrcLayer()
    assert kernelize(m, {}) is m
    assert type(m) is _SrcLayer


def test_kernelize_string_key_calls_setattr():
    mod_name = 'tests.kernel._tmp_kernelize_str'
    mod = types.ModuleType(mod_name)
    mod.target_fn = lambda x: x
    sys.modules[mod_name] = mod
    try:
        new_fn = lambda x: x * 3  # noqa: E731
        kernelize(nn.Linear(1, 1), {f'{mod_name}.target_fn': new_fn})
        assert mod.target_fn is new_fn
    finally:
        sys.modules.pop(mod_name, None)


def test_kernelize_rejects_unknown_key_type():
    with pytest.raises(TypeError, match='Unsupported mapping target'):
        kernelize(nn.Linear(1, 1), {42: _DstLayer})


def test_kernelize_no_mapping_applies_default_config(monkeypatch, caplog):
    """kernelize(model) with no mapping applies DEFAULT_KERNEL_CONFIG.

    With every backend unavailable, the model stays unchanged and the default
    config path emits no WARNING-level noise. Backend resolution is mocked so
    the test is deterministic even when optional kernels are installed.
    """
    import logging

    from twinkle.kernel import core

    resolved = []

    def unavailable(op, backends, **kwargs):
        resolved.append((op, backends, kwargs))
        return None, None

    monkeypatch.setattr(core, 'resolve_impl', unavailable)

    parent = nn.Sequential(_SrcLayer())
    with caplog.at_level(logging.WARNING):
        out = kernelize(parent)
    assert out is parent
    assert type(parent[0]) is _SrcLayer
    assert resolved
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_kernelize_loads_hub_ref(monkeypatch):
    # Stand in for HF kernels: patch _load_hub_ref to return _DstLayer
    from twinkle.kernel import core as _core
    monkeypatch.setattr(_core, '_load_hub_ref', lambda ref: _DstLayer)

    parent = nn.Sequential(_SrcLayer())
    ref = HubRef('org/repo', 'X', revision='main')
    kernelize(parent, {_SrcLayer: ref})
    assert type(parent[0]) is _DstLayer
