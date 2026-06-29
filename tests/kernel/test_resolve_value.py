import torch.nn as nn

from twinkle.kernel.core import HubRef, _resolve_value


class _ImplA(nn.Module):
    pass


class _ImplB(nn.Module):
    pass


def test_passthrough_class_value():
    assert _resolve_value(_ImplA, 'cuda') is _ImplA


def test_passthrough_callable_value():
    f = lambda x: x  # noqa: E731
    assert _resolve_value(f, 'npu') is f


def test_passthrough_hubref():
    ref = HubRef('org/repo', 'Layer', revision='main')
    assert _resolve_value(ref, 'cuda') is ref


def test_device_dict_match():
    val = {'npu': _ImplA, 'cuda': _ImplB}
    assert _resolve_value(val, 'npu') is _ImplA
    assert _resolve_value(val, 'cuda') is _ImplB


def test_device_dict_miss_returns_none():
    val = {'npu': _ImplA}
    assert _resolve_value(val, 'cuda') is None


def test_device_dict_nested():
    # nested dict -> recursive resolve
    val = {'npu': {'npu': _ImplA}}
    assert _resolve_value(val, 'npu') is _ImplA


def test_device_dict_miss_then_passthrough():
    # nested dict whose inner is also a dict that misses -> None
    val = {'npu': {'cuda': _ImplA}}
    assert _resolve_value(val, 'npu') is None