# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for direct/streaming ErrorPayload construction and Tinker parsing."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from twinkle.server.task_errors import build_error_payload, task_error_payload
from twinkle.protocol.types.errors import ErrorCategory, ErrorPayload


def test_overlong_traceback_is_trimmed_tail_kept_with_marker():
    long_tb = 'X' * 10 + ('line\n' * 40000)  # well over 65536 chars
    assert len(long_tb) > 65536

    payload = task_error_payload(
        'RuntimeError: boom', request_id='req_1', error_code=500, traceback_text=long_tb)

    tb = payload['traceback']
    assert tb is not None
    assert len(tb) <= 65536
    assert 'truncated' in tb  # truncation marker present
    assert tb.endswith('line\n')  # tail preserved


def test_task_error_payload_shapes_and_sanitizes_errors():
    payload = task_error_payload(
        'RuntimeError: boom\n  File "/server/path.py", line 1', request_id='req_1', error_code=500)

    assert payload == {
        'error': 'RuntimeError: boom',
        'category': ErrorCategory.Server.value,
        'error_code': 500,
        'request_id': 'req_1',
    }


def test_build_error_payload_bounds_text_and_preserves_metadata():
    details = [{'field': 'adapter_name'}]
    payload = build_error_payload(
        f'bad input {"X" * 2048}\nignored second line',
        request_id='req_build',
        error_code=422,
        category='user',
        traceback_text='server stack',
        details=details,
    )

    assert isinstance(payload, ErrorPayload)
    assert len(payload.error) == 1024
    assert '\n' not in payload.error
    assert payload.category is ErrorCategory.User
    assert payload.error_code == 422
    assert payload.request_id == 'req_build'
    assert payload.traceback is None
    assert payload.details == details
    assert task_error_payload(
        f'bad input {"X" * 2048}\nignored second line',
        request_id='req_build',
        error_code=422,
        category='user',
        traceback_text='server stack',
        details=details,
    ) == payload.model_dump(mode='json', exclude_none=True)


def test_user_category_carries_no_traceback():
    payload = task_error_payload(
        'invalid field', request_id='req_2', error_code=422,
        category=ErrorCategory.User, traceback_text='Traceback (most recent call last): ...')

    assert payload['category'] == ErrorCategory.User.value
    assert 'traceback' not in payload


def test_error_category_matches_tinker_wire_values():
    from tinker.types import RequestErrorCategory

    assert {item.value for item in RequestErrorCategory} == {item.value for item in ErrorCategory}


def test_tinker_sdk_parses_six_field_like_two_field():
    """Tinker's RequestFailedResponse ignores extra fields, so a six-field
    payload parses equal to a two-field one on the declared fields.

    tinker's RequestErrorCategory values are lowercase ('server'), so the payloads
    here use that value; the point under test is that the four extra fields are
    ignored, not the category spelling."""
    from tinker.types import RequestFailedResponse

    two = {'error': 'boom', 'category': 'server'}
    six = task_error_payload('boom', request_id='req_9', error_code=504)

    parsed_six = RequestFailedResponse.model_validate(six)
    parsed_two = RequestFailedResponse.model_validate(two)

    assert parsed_six.error == parsed_two.error
    assert parsed_six.category == parsed_two.category


@pytest.mark.parametrize('category', [ErrorCategory.User, ErrorCategory.Unknown])
def test_non_server_traceback_is_rejected(category):
    with pytest.raises(ValidationError):
        ErrorPayload(
            error='bad input',
            category=category,
            error_code=400,
            request_id='req_11',
            traceback='server stack',
        )
