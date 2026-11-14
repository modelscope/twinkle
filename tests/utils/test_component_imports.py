# Copyright (c) ModelScope Contributors. All rights reserved.
"""Explicit component imports preserve the base-class and trust contracts."""
import pytest

from twinkle.utils.loader import construct_class


def test_import_instantiates_the_requested_class(monkeypatch):
    from collections import Counter

    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '1')
    result = construct_class('collections:Counter', dict, [], a=2)
    assert type(result) is Counter and result == {'a': 2}


@pytest.mark.parametrize('name', ['collections:deque', 'collections:__name__'])
def test_import_rejects_other_component_types(monkeypatch, name):
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '1')
    with pytest.raises(TypeError, match='dict subclass'):
        construct_class(name, dict, [])


def test_safe_mode_rejects_import_before_executing_module(monkeypatch):
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '0')
    with pytest.raises(ValueError, match='TWINKLE_TRUST_REMOTE_CODE'):
        construct_class('does_not_exist:Component', dict, [])
