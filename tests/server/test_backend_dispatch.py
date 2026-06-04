# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property-based tests for the shared BackendSelector (R12).

# Feature: server-structural-refactor, Property 2: Backend dispatch constructs
exactly the matched backend.
# Feature: server-structural-refactor, Property 3: Backend dispatch rejects
non-members and constructs nothing.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from twinkle.server.exceptions import ConfigError
from twinkle.server.utils.backend_dispatch import BackendSelector

_VALUE = st.text(st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=6)


@settings(max_examples=100)
@given(
    members=st.lists(_VALUE, min_size=1, max_size=5, unique=True),
    idx=st.integers(min_value=0, max_value=4),
)
def test_property_2_dispatch_constructs_exactly_matched(members: list[str], idx: int) -> None:
    chosen = members[idx % len(members)]
    calls: dict[str, int] = {m: 0 for m in members}
    sentinels: dict[str, object] = {m: object() for m in members}

    def _make(m: str):
        def _ctor(kw: dict) -> object:
            calls[m] += 1
            return sentinels[m]

        return _ctor

    selector = BackendSelector('field', {m: _make(m) for m in members})
    result = selector.construct(chosen, {'k': 'v'})

    assert result is sentinels[chosen], 'returned the wrong backend'
    assert calls[chosen] == 1, 'matched ctor not invoked exactly once'
    for m in members:
        if m != chosen:
            assert calls[m] == 0, f'sibling ctor {m!r} was invoked'


@settings(max_examples=100)
@given(
    members=st.lists(_VALUE, min_size=1, max_size=5, unique=True),
    bad=st.one_of(
        st.none(),
        st.just(''),
        st.integers(),
        st.text(st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=6),  # uppercase
    ),
)
def test_property_3_reject_non_members_constructs_nothing(members: list[str], bad: object) -> None:
    # Ensure ``bad`` is genuinely not a member (uppercase text can't equal the
    # lowercase members, but guard anyway).
    if isinstance(bad, str) and bad in members:
        return
    calls: dict[str, int] = {m: 0 for m in members}

    def _make(m: str):
        def _ctor(kw: dict) -> object:
            calls[m] += 1
            return object()

        return _ctor

    selector = BackendSelector('field', {m: _make(m) for m in members})

    raised = False
    try:
        selector.construct(bad, {})  # type: ignore[arg-type]
    except ConfigError as exc:
        raised = True
        assert exc.field == 'field'
        assert exc.value == bad
        assert set(exc.allowed) == set(members)
    assert raised, 'ConfigError was not raised for a non-member value'
    assert all(c == 0 for c in calls.values()), 'a ctor was invoked for a rejected value'
