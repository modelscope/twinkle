import torch.nn as nn

from twinkle.kernel.core import _replace_class


class _Target(nn.Module):
    def forward(self, x):
        return x


class _Impl(nn.Module):
    def forward(self, x):
        return x + 1


class _SubTarget(_Target):
    pass


def test_replace_class_rewrites_exact_match():
    m = _Target()
    parent = nn.Sequential(_Target(), nn.Linear(1, 1))
    _replace_class(parent, _Target, _Impl)
    assert type(parent[0]) is _Impl


def test_replace_class_skips_subclass():
    parent = nn.Sequential(_SubTarget())
    _replace_class(parent, _Target, _Impl)
    # exact match only - _SubTarget should NOT be rewritten
    assert type(parent[0]) is _SubTarget


def test_replace_class_idempotent():
    m = nn.Sequential(_Target())
    _replace_class(m, _Target, _Impl)
    _replace_class(m, _Target, _Impl)  # second call must be safe
    assert type(m[0]) is _Impl