import sys
import types

import torch.nn as nn

from twinkle.kernel.core import _replace_attr, _replace_class


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


def test_replace_attr_sets_module_attribute():
    mod_name = 'tests.kernel._tmp_replace_attr'
    mod = types.ModuleType(mod_name)
    mod.target_fn = lambda x: x
    sys.modules[mod_name] = mod
    try:
        new_fn = lambda x: x * 2  # noqa: E731
        _replace_attr(f'{mod_name}.target_fn', new_fn)
        assert mod.target_fn is new_fn
    finally:
        sys.modules.pop(mod_name, None)


def test_replace_attr_supports_class_attribute():
    import sys
    import types

    mod_name = 'tests.kernel._tmp_class_attr'
    mod = types.ModuleType(mod_name)

    class Foo:
        def forward(self, x):
            return x
    mod.Foo = Foo
    sys.modules[mod_name] = mod
    try:
        new_forward = lambda self, x: x + 7  # noqa: E731
        _replace_attr(f'{mod_name}.Foo.forward', new_forward)
        assert Foo.forward is new_forward
    finally:
        sys.modules.pop(mod_name, None)