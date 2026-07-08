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


def test_kernelize_device_dict_match(monkeypatch):
    from twinkle.utils.device_mesh import Platform

    parent = nn.Sequential(_SrcLayer())
    monkeypatch.setattr(Platform, 'device_prefix', staticmethod(lambda platform=None: 'cpu'))

    kernelize(parent, {_SrcLayer: {'cpu': _DstLayer, 'npu': nn.Identity}})

    assert type(parent[0]) is _DstLayer


def test_kernelize_uses_platform_device_prefix(monkeypatch):
    from twinkle.utils.device_mesh import Platform

    parent = nn.Sequential(_SrcLayer())  # params may still be CPU before FSDP placement
    monkeypatch.setattr(Platform, 'device_prefix', staticmethod(lambda platform=None: 'npu'))

    kernelize(parent, {_SrcLayer: {'npu': _DstLayer}})

    assert type(parent[0]) is _DstLayer


def test_kernelize_device_dict_miss_skips_silently(monkeypatch):
    from twinkle.utils.device_mesh import Platform

    parent = nn.Sequential(_SrcLayer())
    monkeypatch.setattr(Platform, 'device_prefix', staticmethod(lambda platform=None: 'cpu'))

    kernelize(parent, {_SrcLayer: {'npu': _DstLayer}})

    assert type(parent[0]) is _SrcLayer


def test_kernelize_rejects_unknown_key_type():
    with pytest.raises(TypeError, match='Unsupported mapping key'):
        kernelize(nn.Linear(1, 1), {42: _DstLayer})


def test_kernelize_no_mapping_on_npu_uses_npu_builtin(monkeypatch):
    """kernelize(model) with no mapping auto-detects NPU and applies npu_builtin."""
    from twinkle.utils.device_mesh import Platform
    import twinkle.kernel.builtin as builtin

    monkeypatch.setattr(Platform, 'device_prefix', staticmethod(lambda platform=None: 'npu'))
    monkeypatch.setattr(builtin, 'npu_builtin', lambda model=None: {_SrcLayer: _DstLayer})

    parent = nn.Sequential(_SrcLayer())
    out = kernelize(parent)
    assert out is parent
    assert type(parent[0]) is _DstLayer


def test_kernelize_no_mapping_on_non_npu_is_noop(monkeypatch):
    """kernelize(model) with no mapping on a non-NPU device must not touch the
    model and must not invoke npu_builtin (avoiding its side effects)."""
    from twinkle.utils.device_mesh import Platform
    import twinkle.kernel.builtin as builtin

    called = []
    monkeypatch.setattr(Platform, 'device_prefix', staticmethod(lambda platform=None: 'cuda'))
    monkeypatch.setattr(builtin, 'npu_builtin',
                        lambda model=None: called.append(model) or {_SrcLayer: _DstLayer})

    parent = nn.Sequential(_SrcLayer())
    out = kernelize(parent)
    assert out is parent
    assert type(parent[0]) is _SrcLayer
    assert called == []


def test_kernelize_loads_hub_ref(monkeypatch):
    # Stand in for HF kernels: patch _load_hub_ref to return _DstLayer
    from twinkle.kernel import core as _core
    monkeypatch.setattr(_core, '_load_hub_ref', lambda ref: _DstLayer)

    parent = nn.Sequential(_SrcLayer())
    ref = HubRef('org/repo', 'X', revision='main')
    kernelize(parent, {_SrcLayer: ref})
    assert type(parent[0]) is _DstLayer
