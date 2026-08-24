import torch.nn as nn

from twinkle.kernel.core import HubRef, resolve_direct_value


class _ImplA(nn.Module):
    pass


def test_passthrough_class_value():
    assert resolve_direct_value(_ImplA) is _ImplA


def test_passthrough_callable_value():
    f = lambda x: x  # noqa: E731
    assert resolve_direct_value(f) is f


def test_hubref_delegates_to_load(monkeypatch):
    """HubRef values are resolved via _load_hub_ref (lazy hub download)."""
    import twinkle.kernel.core as core

    sentinel = object()
    ref = HubRef('org/repo', 'Layer', revision='main')
    monkeypatch.setattr(core, '_load_hub_ref', lambda r: sentinel)
    assert resolve_direct_value(ref) is sentinel
