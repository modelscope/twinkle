# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the single FutureRecord -> TaskEnvelope mapping point."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from twinkle.server.lifecycle.envelope import envelope_from_record


def test_wire_maps_cover_exactly_the_canonical_reason_codes():
    """The Twinkle and Tinker failure maps must both stay in lockstep with the
    canonical reason-code set, so a newly added domain reason cannot silently
    fall back to 500 on one protocol face."""
    from twinkle.server.gateway.tinker_handlers import _tinker_error_from_failure
    from twinkle.server.lifecycle.envelope import error_payload_from_failure
    from twinkle.server.task_errors import FailureResponse

    for reason in FailureResponse.reasons():
        policy = FailureResponse.for_reason(reason)
        failure = {'reason_code': reason, 'message': 'failed', 'attribution': policy.category.value}
        native = error_payload_from_failure(failure, request_id='request')
        assert native.error_code == policy.error_code
        assert native.category is policy.category
        assert _tinker_error_from_failure(failure, request_id='request') == native.model_dump(
            mode='json', exclude_none=True)


def test_completed_with_none_result_is_a_success_not_a_failure():
    """`completed` + `result is None` is a valid success."""
    env = envelope_from_record('req-1', {'status': 'completed', 'result': None})
    assert env.status == 'completed'
    assert env.result is None
    assert env.error is None


def test_completed_carries_result_and_no_error():
    env = envelope_from_record('req-1', {'status': 'completed', 'result': {'loss': 0.5}})
    assert env.result == {'loss': 0.5}
    assert env.error is None


def test_domain_failure_maps_to_twinkle_error_payload():
    env = envelope_from_record(
        'req-9', {
            'status': 'failed',
            'failure': {
                'reason_code': 'execution_timeout',
                'message': 'backend timed out',
                'attribution': 'server',
                'diagnostic': 'full traceback',
            },
        })
    assert env.status == 'failed'
    assert env.result is None
    assert env.error is not None
    assert env.error.error == 'backend timed out'
    assert env.error.category.value == 'server'
    assert env.error.error_code == 504
    assert env.error.request_id == 'req-9'
    assert env.error.traceback == 'full traceback'


def test_legacy_failure_in_result_is_not_accepted():
    with pytest.raises(ValidationError):
        envelope_from_record(
            'req-old', {'status': 'failed', 'result': {'error': 'boom', 'category': 'server'}})


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
