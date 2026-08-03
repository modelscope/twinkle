"""tests/kernel/test_registry.py -- registry/KernelChoice/installer behavior matrix.

Everything runs on CPU: fake ops / fake backends verify selection, fallback, log
levels and installer priority; no real NPU needed (VeOmni B2).
"""
import logging
import sys
import types

import pytest
import torch.nn as nn

from twinkle.kernel import core
from twinkle.kernel.core import default_installer, kernelize
from twinkle.kernel.registry import _OPS, KernelChoice, KernelImpl, get_op, register_op, resolve_impl


class _SrcLayer(nn.Module):
    def forward(self, x):
        return x


class _DstLayer(nn.Module):
    def forward(self, x):
        return x + 100


@pytest.fixture
def twinkle_log():
    """Capture records of the 'twinkle' logger (propagate=False, caplog cannot see it)."""
    logger = logging.getLogger('twinkle')
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _H(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@pytest.fixture
def fake_ops():
    """Snapshot and clear _OPS, register fake ops, restore after the test."""
    saved = dict(_OPS)
    _OPS.clear()
    yield
    _OPS.clear()
    _OPS.update(saved)


def _fake_impl(name='fake_impl'):
    fn = types.FunctionType(compile(f'def {name}(): pass', '<fake>', 'exec'), {})
    return fn


def _register_fake_op(name='fake_op', backends=('fake_a', 'fake_b'), installer=None, available_a=(True, None),
                      available_b=(True, None), calls=None):
    """Register one fake op; return (impl_a, impl_b, calls). calls records available/load invocations."""
    if calls is None:
        calls = []
    impl_a, impl_b = _fake_impl('impl_a'), _fake_impl('impl_b')

    def avail_a():
        calls.append('available_a')
        return available_a

    def avail_b():
        calls.append('available_b')
        return available_b

    def load_a(_target=None):
        calls.append('load_a')
        return impl_a

    def load_b(_target=None):
        calls.append('load_b')
        return impl_b

    register_op(
        name,
        implementations={
            'fake_a': KernelImpl(load=load_a, available=avail_a),
            'fake_b': KernelImpl(load=load_b, available=avail_b),
        },
        installer=installer,
    )
    return impl_a, impl_b, calls


# ── 1. direct impl pass-through ────────────────────────────────────────────


def test_direct_impl_passthrough(fake_ops):
    parent = nn.Sequential(_SrcLayer())
    kernelize(parent, {_SrcLayer: _DstLayer})
    assert type(parent[0]) is _DstLayer


# ── 2. HubRef pass-through ─────────────────────────────────────────────────


def test_hub_ref_passthrough(fake_ops, monkeypatch):
    monkeypatch.setattr(core, '_load_hub_ref', lambda ref: _DstLayer)
    parent = nn.Sequential(_SrcLayer())
    kernelize(parent, {_SrcLayer: core.HubRef('org/repo', 'X', revision='main')})
    assert type(parent[0]) is _DstLayer


# ── 3. KernelChoice first backend hits ─────────────────────────────────────


def test_choice_first_backend_wins(fake_ops):
    """First backend hits: default_installer receives impl_a; fake_b.available is never called."""
    impl_a, _, calls = _register_fake_op()
    mod_name = 'tests.kernel._tmp_choice_first'
    mod = types.ModuleType(mod_name)
    mod.slot = None
    sys.modules[mod_name] = mod
    try:
        kernelize(nn.Linear(1, 1), {f'{mod_name}.slot': KernelChoice(op='fake_op', backends=('fake_a', 'fake_b'))})
        assert mod.slot is impl_a
        assert 'available_b' not in calls
    finally:
        sys.modules.pop(mod_name, None)


# ── 4. available failure falls through ─────────────────────────────────────


def test_available_false_falls_back_with_reason(fake_ops, twinkle_log):
    impl_a, impl_b, _ = _register_fake_op(available_a=(False, 'fake_a needs CUDA 12.9'))
    mod_name = 'tests.kernel._tmp_avail_fb'
    mod = types.ModuleType(mod_name)
    mod.slot = None
    sys.modules[mod_name] = mod
    try:
        kernelize(nn.Linear(1, 1), {
            f'{mod_name}.slot': KernelChoice(op='fake_op', backends=('fake_a', 'fake_b'))})
        assert mod.slot is impl_b
        assert any('fake_a needs CUDA 12.9' in r.getMessage() and r.levelno >= logging.WARNING
                   for r in twinkle_log)
    finally:
        sys.modules.pop(mod_name, None)


# ── 5. load exception falls through ────────────────────────────────────────


def test_load_exception_falls_back(fake_ops):
    impl_b = _fake_impl('impl_b')
    register_op(
        'fake_op',
        implementations={
            'fake_a': KernelImpl(
                load=lambda _t=None: (_ for _ in ()).throw(ImportError('boom')),
                available=lambda: (True, None)),
            'fake_b': KernelImpl(load=lambda _t=None: impl_b, available=lambda: (True, None)),
        })
    mod_name = 'tests.kernel._tmp_load_fb'
    mod = types.ModuleType(mod_name)
    mod.slot = None
    sys.modules[mod_name] = mod
    try:
        kernelize(nn.Linear(1, 1), {f'{mod_name}.slot': KernelChoice(op='fake_op', backends=('fake_a', 'fake_b'))})
        assert mod.slot is impl_b
    finally:
        sys.modules.pop(mod_name, None)


# ── 6. all fail -> keep original + explicit mapping WARNING ────────────────


def test_all_backends_fail_keeps_original_and_warns(fake_ops, twinkle_log):
    _register_fake_op(available_a=(False, 'no a'), available_b=(False, 'no b'))
    called = []
    parent = nn.Sequential(_SrcLayer())
    kernelize(
        parent,
        {
            _SrcLayer:
                KernelChoice(
                    op='fake_op',
                    backends=('fake_a', 'fake_b'),
                    installer=lambda m, t, i: called.append(i)),
        })
    assert called == []  # installer never called
    assert type(parent[0]) is _SrcLayer  # original kept
    assert any('no available backend' in r.getMessage() and r.levelno >= logging.WARNING for r in twinkle_log)


# ── 7. default config path log levels (P1: fallbacks/failures all DEBUG) ───


def test_default_config_path_logs_debug_only(fake_ops, monkeypatch, twinkle_log):
    _register_fake_op(available_a=(False, 'no a'), available_b=(False, 'no b'))
    monkeypatch.setattr(
        'twinkle.kernel.config.DEFAULT_KERNEL_CONFIG',
        {_SrcLayer: KernelChoice(op='fake_op', backends=('fake_a', 'fake_b'))})
    parent = nn.Sequential(_SrcLayer())
    kernelize(parent)  # mapping=None -> default config path
    assert type(parent[0]) is _SrcLayer
    assert not [r for r in twinkle_log if r.levelno >= logging.WARNING]
    assert any('no available backend' in r.getMessage() for r in twinkle_log if r.levelno == logging.DEBUG)


# ── 8. unregistered backend -> warning, falls through ──────────────────────


def test_unregistered_backend_warns_and_falls_back(fake_ops, twinkle_log):
    _, impl_b, _ = _register_fake_op()
    mod_name = 'tests.kernel._tmp_unreg_be'
    mod = types.ModuleType(mod_name)
    mod.slot = None
    sys.modules[mod_name] = mod
    try:
        kernelize(nn.Linear(1, 1), {
            f'{mod_name}.slot': KernelChoice(op='fake_op', backends=('ghost', 'fake_b'))})
        assert mod.slot is impl_b
        assert any("'ghost' not registered" in r.getMessage() and r.levelno >= logging.WARNING
                   for r in twinkle_log)
    finally:
        sys.modules.pop(mod_name, None)


# ── 9. unregistered op -> ValueError ───────────────────────────────────────


def test_unregistered_op_raises(fake_ops):
    with pytest.raises(ValueError, match="'nope' is not registered"):
        kernelize(nn.Linear(1, 1), {_SrcLayer: KernelChoice(op='nope', backends=('fake_a', ))})


# ── 10. installer priority: choice > op > default ──────────────────────────


def test_installer_priority(fake_ops):
    calls = []
    choice_installer = lambda m, t, i: calls.append('choice')  # noqa: E731
    op_installer = lambda m, t, i: calls.append('op')  # noqa: E731

    _register_fake_op(installer=op_installer)
    parent = nn.Sequential(_SrcLayer())

    # choice.installer overrides op.installer
    kernelize(parent, {_SrcLayer: KernelChoice(op='fake_op', backends=('fake_a', ), installer=choice_installer)})
    assert calls == ['choice']

    # choice.installer empty -> op.installer
    calls.clear()
    kernelize(parent, {_SrcLayer: KernelChoice(op='fake_op', backends=('fake_a', ))})
    assert calls == ['op']

    # op.installer empty -> default_installer (class replacement really happens)
    register_op('fake_op2', implementations={
        'fake_a': KernelImpl(load=lambda _t=None: _DstLayer, available=lambda: (True, None)),
    })
    parent2 = nn.Sequential(_SrcLayer())
    kernelize(parent2, {_SrcLayer: KernelChoice(op='fake_op2', backends=('fake_a', ))})
    assert type(parent2[0]) is _DstLayer


# ── 11. custom installer signature & logical target pass-through ───────────


def test_custom_installer_receives_model_target_impl(fake_ops):
    received = []
    impl_a = _fake_impl('impl_a')
    register_op(
        'sdpa_like',
        implementations={'fake_a': KernelImpl(load=lambda _t=None: impl_a, available=lambda: (True, None))},
        installer=lambda m, t, i: received.append((m, t, i)))
    model = nn.Linear(1, 1)
    kernelize(model, {'sdpa': KernelChoice(op='sdpa_like', backends=('fake_a', ))})
    assert received == [(model, 'sdpa', impl_a)]  # logical target passed through unresolved


# ── 12. string target dispatch ─────────────────────────────────────────────


def test_dotted_target_dispatch(fake_ops, twinkle_log):
    mod_name = 'tests.kernel._tmp_dispatch'
    mod = types.ModuleType(mod_name)

    class Foo(nn.Module):
        def forward(self, x):
            return x

    mod.Foo = Foo
    mod.fn = lambda x: x
    sys.modules[mod_name] = mod
    try:
        # resolves to an nn.Module subclass -> __class__ swap
        parent = nn.Sequential(Foo())
        kernelize(parent, {f'{mod_name}.Foo': _DstLayer})
        assert type(parent[0]) is _DstLayer

        # resolves to a function -> setattr
        new_fn = lambda x: x * 3  # noqa: E731
        kernelize(nn.Linear(1, 1), {f'{mod_name}.fn': new_fn})
        assert mod.fn is new_fn

        # transformers family missing + explicit mapping -> WARNING skip (typo hint), no raise
        kernelize(nn.Linear(1, 1),
                  {'transformers.models.no_such_family.modeling_x.NoSuchRMSNorm': _DstLayer})
        assert any('family not installed' in r.getMessage() and 'typos' in r.getMessage()
                   for r in twinkle_log if r.levelno == logging.WARNING)
    finally:
        sys.modules.pop(mod_name, None)


def test_family_skip_default_path_stays_debug(twinkle_log):
    """Default config path (warn=False): a missing family stays DEBUG, not escalated to WARNING."""
    default_installer(nn.Linear(1, 1),
                      'transformers.models.no_such_family.modeling_x.NoSuchRMSNorm',
                      _DstLayer,
                      warn=False)
    matches = [r for r in twinkle_log if 'family not installed' in r.getMessage()]
    assert matches and all(r.levelno == logging.DEBUG for r in matches)


# ── 13. unresolvable non-family string + default installer -> explicit error ──


def test_unresolvable_non_family_string_raises(fake_ops):
    with pytest.raises(ValueError, match='Cannot resolve mapping target'):
        kernelize(nn.Linear(1, 1), {'no_such_pkg_zzz.mod.attr': _DstLayer})


# ── 14. installer exception propagates ─────────────────────────────────────


def test_installer_exception_propagates(fake_ops):
    def bad_installer(m, t, i):
        raise RuntimeError('half-installed state must be visible')

    register_op(
        'fake_op',
        implementations={'fake_a': KernelImpl(load=lambda _t=None: _DstLayer, available=lambda: (True, None))},
        installer=bad_installer)
    with pytest.raises(RuntimeError, match='half-installed'):
        kernelize(nn.Linear(1, 1), {_SrcLayer: KernelChoice(op='fake_op', backends=('fake_a', ))})


# ── 15. registration guards ────────────────────────────────────────────────


def test_register_op_defense(fake_ops):
    register_op('dup', implementations={'a': KernelImpl(load=lambda _t=None: 1, available=lambda: (True, None))})
    with pytest.raises(ValueError, match='already registered'):
        register_op('dup',
                    implementations={'a': KernelImpl(load=lambda _t=None: 1, available=lambda: (True, None))})
    with pytest.raises(ValueError, match='no implementations'):
        register_op('empty', implementations={})
    with pytest.raises(ValueError, match="'ghost' is not registered"):
        get_op('ghost')


# ── 16. logical target misused with default installer (single-segment string) -> explicit error ──


def test_logical_target_single_segment_raises(fake_ops):
    """A logical name like 'sdpa' paired with an op that has no dedicated installer ->
    default_installer fails to resolve it and raises ValueError mentioning a custom installer."""
    register_op(
        'fake_op',
        implementations={'fake_a': KernelImpl(load=lambda _t=None: _DstLayer, available=lambda: (True, None))})
    with pytest.raises(ValueError, match='logical targets require a custom installer'):
        kernelize(nn.Linear(1, 1), {'sdpa': KernelChoice(op='fake_op', backends=('fake_a', ))})


# ── 17. mixed mapping: KernelChoice + direct impl in one install ───────────


class _SrcLayer2(nn.Module):
    def forward(self, x):
        return x


def test_mixed_mapping_choice_and_direct(fake_ops):
    """One kernelize call: KernelChoice entries (via registry) and bare impl entries (pass-through) both take effect."""
    register_op(
        'fake_op',
        implementations={'fake_a': KernelImpl(load=lambda _t=None: _DstLayer, available=lambda: (True, None))})
    parent = nn.Sequential(_SrcLayer(), _SrcLayer2())
    kernelize(parent, {
        _SrcLayer: KernelChoice(op='fake_op', backends=('fake_a', )),
        _SrcLayer2: _DstLayer,
    })
    assert type(parent[0]) is _DstLayer
    assert type(parent[1]) is _DstLayer


# ── 18. family skip produces no success log ────────────────────────────────


def test_family_skip_produces_no_success_info(fake_ops, twinkle_log):
    """After a family-missing skip (installed=False), the main loop must not emit an INFO success log for that target."""
    kernelize(nn.Linear(1, 1), {'transformers.models.no_such_family.modeling_x.Foo': _DstLayer})
    assert not [r for r in twinkle_log if r.levelno == logging.INFO]


# ── 19. kernelize idempotent: safe to call repeatedly ──────────────────────


def test_kernelize_idempotent(fake_ops):
    """Second call: instances of the target class no longer exist (already swapped); zero replacements, no exception."""
    parent = nn.Sequential(_SrcLayer())
    kernelize(parent, {_SrcLayer: _DstLayer})
    kernelize(parent, {_SrcLayer: _DstLayer})
    kernelize(parent, {_SrcLayer: _DstLayer})
    assert type(parent[0]) is _DstLayer
