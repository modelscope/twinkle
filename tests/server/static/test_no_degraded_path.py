# Copyright (c) ModelScope Contributors. All rights reserved.
"""Static check: no silent-degradation symbols remain.

One wildcard search covering eight symbols; each must occur zero times in its scope.
The symbols are matched as identifiers (word boundaries) so that ``nccl_safe_megatron``
(the retained decorator), the ``twinkle.utils.nccl_safe`` module path, and unrelated
test names like ``test_zero_loss_...`` are not counted.
"""
from __future__ import annotations

import pathlib
import re

import twinkle

_TWINKLE_SRC = pathlib.Path(twinkle.__file__).resolve().parent
_REPO_ROOT = _TWINKLE_SRC.parent.parent  # .../src/twinkle -> repo root
_TESTS = _REPO_ROOT / 'tests'
_COOKBOOK = _REPO_ROOT / 'cookbook'
_SELF = pathlib.Path(__file__).resolve()

# symbol -> compiled identifier pattern.
_IDENT = {
    'safe_loss': re.compile(r'(?<![\w])safe_loss(?![\w])'),
    'SafeLossWrapper': re.compile(r'(?<![\w])SafeLossWrapper(?![\w])'),
    '_is_fail_fast': re.compile(r'(?<![\w])_is_fail_fast(?![\w])'),
    # bare nccl_safe: exclude nccl_safe_megatron (trailing _) and the module path (leading .)
    'nccl_safe': re.compile(r'(?<![\w.])nccl_safe(?![\w])'),
    '_force_zero_backward': re.compile(r'(?<![\w])_force_zero_backward(?![\w])'),
    '_iter_model_params': re.compile(r'(?<![\w])_iter_model_params(?![\w])'),
    '_zero_loss': re.compile(r'(?<![\w])_zero_loss(?![\w])'),
}
_FAIL_FAST = re.compile(r'TWINKLE_FAIL_FAST')


def _py_files(*roots: pathlib.Path):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*.py'):
            if path.resolve() == _SELF or '__pycache__' in path.parts:
                continue
            yield path


def _cookbook_files():
    if not _COOKBOOK.exists():
        return
    for pattern in ('*.yaml', '*.yml', '*.py'):
        yield from _COOKBOOK.rglob(pattern)


def test_no_degraded_symbols_remain():
    offenders: dict[str, list[str]] = {}
    for path in _py_files(_TWINKLE_SRC, _TESTS):
        text = path.read_text()
        for name, pat in _IDENT.items():
            if pat.search(text):
                offenders.setdefault(name, []).append(str(path))
    assert not offenders, f'silent-degradation symbols still present: {offenders}'


def test_no_fail_fast_switch_in_src_and_cookbook():
    offenders: list[str] = []
    for path in list(_py_files(_TWINKLE_SRC)) + list(_cookbook_files()):
        if _FAIL_FAST.search(path.read_text()):
            offenders.append(str(path))
    assert not offenders, f'TWINKLE_FAIL_FAST still present: {offenders}'
