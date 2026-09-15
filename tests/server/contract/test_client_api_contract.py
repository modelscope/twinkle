# Copyright (c) ModelScope Contributors. All rights reserved.
"""Zero-wire-change contract guard (T8.1 / R8 / Property 10).

Exports the request/response surface of all five apps and compares it field-by-field with
the canonical baseline. The diff must be empty. Also asserts the load-bearing
invariants: ``schedule_task_and_wait`` still
exists and the only client-side additions are ``types/base.py`` and ``types/errors.py``.
"""
from __future__ import annotations

import pytest

from tests.server.contract.client_api_harness import extract_full_surface, load_baseline


def test_wire_surface_matches_baseline():
    current = extract_full_surface()
    baseline = load_baseline()
    assert set(current) == {'data_plane', 'gateway', 'model', 'processor', 'sampler'}
    assert current == baseline, (
        'Client-facing wire surface changed vs the canonical baseline; '
        'this spec must be zero-wire-change. Diffing apps: '
        f'{[a for a in set(current) | set(baseline) if current.get(a) != baseline.get(a)]}')


def test_schedule_task_and_wait_not_removed():
    from twinkle.server.utils.task_queue.mixin import TaskQueueMixin
    assert hasattr(TaskQueueMixin, 'schedule_task_and_wait')


def test_new_client_types_importable():
    # The only permitted client-side additions.
    import twinkle_client.types.base as base
    import twinkle_client.types.errors as errors

    for symbol in ('StrictRequest', 'ResponseModel', 'DataModel', 'backend_only'):
        assert hasattr(base, symbol)
    for symbol in ('ErrorPayload', 'ErrorCategory', 'QueueStateLiteral'):
        assert hasattr(errors, symbol)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
