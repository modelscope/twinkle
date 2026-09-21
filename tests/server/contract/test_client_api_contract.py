# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-API wire-surface guards.

Two guards with different strengths, kept apart on purpose:

1. :func:`test_route_inventory_matches_committed_snapshot` compares the live route
   inventory against ``client_api_routes.json``, which **is** committed. This is the
   real guard: an unintended route addition/removal, or a changed response/body model,
   fails here and shows up as a readable diff in the PR.
2. :func:`test_full_surface_extraction_is_self_consistent` only exercises the
   field-level extractor. ``client_api_baseline.json`` is a generated, gitignored
   artifact, so comparing against it cannot detect drift -- it would be comparing the
   code to itself. The test is therefore scoped to what it can honestly assert: that
   extraction runs, covers all five apps, and round-trips through JSON.

Plus the load-bearing structural invariants of the request-lifecycle refactor.
"""
from __future__ import annotations

import json

import pytest

from tests.server.contract.client_api_harness import (extract_full_surface, extract_route_inventory,
                                                      load_route_inventory)

_APPS = {'data_plane', 'gateway', 'model', 'processor', 'sampler'}


def test_route_inventory_matches_committed_snapshot():
    current = extract_route_inventory()
    committed = load_route_inventory()
    assert set(current) == _APPS

    diffs = []
    for app in sorted(set(current) | set(committed)):
        cur_routes, old_routes = current.get(app, {}), committed.get(app, {})
        for key in sorted(set(cur_routes) | set(old_routes)):
            if cur_routes.get(key) != old_routes.get(key):
                diffs.append(f'  [{app}] {key}: committed={old_routes.get(key)} current={cur_routes.get(key)}')
    assert not diffs, ('Client-facing route surface differs from the committed inventory.\n'
                       'If the change is intentional, regenerate and review the diff:\n'
                       '  python -m tests.server.contract.update_baseline\n' + '\n'.join(diffs))


def test_full_surface_extraction_is_self_consistent():
    # Scoped to what a self-generated snapshot can prove: the extractor works.
    surface = extract_full_surface()
    assert set(surface) == _APPS
    for app, contract in surface.items():
        assert contract['paths'], f'{app} exposed no routes'
    assert json.loads(json.dumps(surface, sort_keys=True)) == surface


def test_schedule_task_and_wait_removed():
    # Future records replace the in-process blocking wait.
    from twinkle.server.task_queue.mixin import TaskQueueMixin
    assert not hasattr(TaskQueueMixin, 'schedule_task_and_wait')
    assert hasattr(TaskQueueMixin, 'submit_and_peek')


def test_new_client_types_importable():
    # The only permitted client-side additions.
    import twinkle.protocol.types.base as base
    import twinkle.protocol.types.errors as errors

    for symbol in ('StrictRequest', 'ResponseModel', 'DataModel', 'backend_only'):
        assert hasattr(base, symbol)
    for symbol in ('ErrorPayload', 'ErrorCategory', 'QueueStateLiteral'):
        assert hasattr(errors, symbol)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
