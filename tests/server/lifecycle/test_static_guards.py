# Copyright (c) ModelScope Contributors. All rights reserved.
"""Static / structural guards for the lifecycle refactor.

- the deleted symbols occur zero times under ``src/twinkle/**``.
- ``TaskEnvelope`` has exactly one construction site.
- The client-side invariant: the client HTTP timeout is <= 120 and strictly
  greater than the server Long_Poll_Window.
- The task status set has two independent declarations that must not drift.

These exist because the spec states most of its guarantees in prose. A prose claim that
nothing checks decays into a false claim -- as happened with "a consistency test asserts
the two sets are equal", which was written in a docstring while no such test existed.
"""
from __future__ import annotations

import pytest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / 'src' / 'twinkle'

# Symbols the refactor removed. A wildcard search (not a per-file list) must find
# each of them zero times across the whole server tree.
_FORBIDDEN_SYMBOLS = (
    'schedule_task_and_wait',
    'run_task',
    'persist_status',
    '_complete_result',
    '_complete_error',
)


@pytest.mark.parametrize('symbol', _FORBIDDEN_SYMBOLS)
def test_deleted_symbol_has_zero_occurrences(symbol):
    hits = []
    for path in _SRC.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        if symbol in text:
            hits.append(str(path.relative_to(_REPO_ROOT)))
    assert hits == [], f'{symbol!r} still occurs in: {hits}'


def test_client_http_timeout_bounds():
    from twinkle.server.lifecycle.poll_config import long_poll_window
    from twinkle_client.http.client import _HTTP_TIMEOUT

    assert _HTTP_TIMEOUT <= 120
    assert _HTTP_TIMEOUT > long_poll_window()


def test_task_envelope_has_exactly_one_construction_site():
    """'s structural precondition: one mapping point, mechanically enforced.

    ``envelope_from_record`` is the only place a FutureRecord becomes a TaskEnvelope, so
    ``failed`` always lands in ``error`` and never in ``result`` regardless of which
    endpoint answered. That was previously only a docstring claim -- a handler building a
    ``TaskEnvelope(...)`` itself would silently reintroduce the exact defect the single
    mapping point exists to prevent (a failure inside the Inline_Fast_Path window losing
    its payload), and every existing test would still pass.
    """
    sites = []
    for path in _SRC.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        for lineno, line in enumerate(text.splitlines(), start=1):
            if 'TaskEnvelope(' in line and 'class TaskEnvelope' not in line:
                sites.append(f'{path.relative_to(_REPO_ROOT)}:{lineno}')

    offenders = [site for site in sites if 'server/lifecycle/envelope.py' not in site]
    assert offenders == [], ('TaskEnvelope must only be constructed in lifecycle/envelope.py '
                             f'(via envelope_from_record); found: {offenders}')
    assert sites, 'expected to find the construction sites inside envelope.py'


def test_server_task_status_enum_matches_client_literal():
    """The two independent declarations of the task status set must not drift.

    ``twinkle_client.types.lifecycle.TaskStatus`` (a Literal on the wire model) and the
    server's ``TaskStatus`` enum are declared separately. ``envelope_from_record`` copies
    ``record['status']`` straight into ``TaskEnvelope.status``, so a value the server can
    write but the Literal does not list would fail pydantic validation *while serialising
    the response* -- i.e. a 500 from retrieve for a task that actually finished.

    The client module's comment claimed such a test existed; it did not. This is it.
    """
    from typing import get_args

    from twinkle.server.task_queue.types import TaskStatus as ServerTaskStatus
    from twinkle_client.types.lifecycle import TERMINAL_STATUSES
    from twinkle_client.types.lifecycle import TaskStatus as WireTaskStatus

    server_values = {member.value for member in ServerTaskStatus}
    wire_values = set(get_args(WireTaskStatus))
    assert server_values == wire_values, (f'task status sets drifted: server-only={server_values - wire_values}, '
                                          f'wire-only={wire_values - server_values}')
    assert TERMINAL_STATUSES <= wire_values, 'TERMINAL_STATUSES must be a subset of the declared statuses'


def test_client_future_layer_is_not_imported_by_the_server():
    """``_future.py`` carries an underscore because the dependency runs one way only.

    The server reverse-imports ``twinkle_client.types`` (the shared wire contract), but the
    client's polling layer is private to the client. An import in the other direction would
    make the server depend on client retry policy, which its own long-poll already owns.
    Another claim that lived only in a docstring.
    """
    offenders = []
    for path in _SRC.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        if 'twinkle_client._future' in text or 'from twinkle_client import _future' in text:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], f'server must not import the client future layer: {offenders}'
