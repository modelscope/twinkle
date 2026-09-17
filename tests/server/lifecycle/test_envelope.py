# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the single FutureRecord -> TaskEnvelope mapping point (T1.2)."""
from __future__ import annotations

from twinkle.server.lifecycle.envelope import envelope_from_record


def test_completed_with_none_result_is_a_success_not_a_failure():
    """R1#4 / Property 4: `completed` + `result is None` is a valid success."""
    env = envelope_from_record('req-1', {'status': 'completed', 'result': None})
    assert env.status == 'completed'
    assert env.result is None
    assert env.error is None


def test_completed_carries_result_and_no_error():
    env = envelope_from_record('req-1', {'status': 'completed', 'result': {'loss': 0.5}})
    assert env.result == {'loss': 0.5}
    assert env.error is None


def test_legacy_two_field_failure_payload_is_backfilled_not_strict_validated():
    """R2#8 / Property 5: a pre-spec {error, category} payload maps to a legal envelope.

    It must go through error_payload_from_stored (missing error_code/request_id are
    backfilled), never a strict ErrorPayload.model_validate that would raise.
    """
    env = envelope_from_record('req-9', {'status': 'failed', 'result': {'error': 'boom', 'category': 'server'}})
    assert env.status == 'failed'
    assert env.result is None
    assert env.error is not None
    assert env.error.error == 'boom'
    assert env.error.category.value == 'server'
    assert env.error.error_code == 500          # backfilled
    assert env.error.request_id == 'req-9'       # backfilled from the argument


def test_non_terminal_record_carries_queue_state_and_no_payload():
    env = envelope_from_record(
        'req-3', {'status': 'running', 'queue_state': 'active', 'queue_state_reason': 'x'})
    assert env.status == 'running'
    assert env.result is None
    assert env.error is None
    assert env.queue_state == 'active'
    assert env.queue_state_reason == 'x'


def test_missing_record_falls_back_to_pending():
    env = envelope_from_record('req-4', None)
    assert env.status == 'pending'
    assert env.result is None
    assert env.error is None
