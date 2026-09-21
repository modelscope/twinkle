# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client error-response parsing ( / Requirement 3 #7-#11)."""
from __future__ import annotations

import pytest
import requests

from twinkle_client.exceptions import TwinkleHTTPError
from twinkle_client.http.client import _handle_response


class _Resp:
    """Minimal stand-in for requests.Response for _handle_response."""

    def __init__(self, status_code, *, body=None, text='', url='http://x'):
        self.status_code = status_code
        self.ok = status_code < 400
        self._body = body
        self.text = text
        self.url = url

    def json(self):
        if self._body is None:
            raise ValueError('no json')
        return self._body


def test_structured_error_reads_top_level_fields():
    """Top-level category/error_code/request_id are preferred."""
    resp = _Resp(422, body={'error': 'bad input', 'category': 'user', 'error_code': 422, 'request_id': 'req-7'})
    with pytest.raises(TwinkleHTTPError) as exc:
        _handle_response(resp)
    assert isinstance(exc.value, requests.HTTPError) # existing except clauses keep working
    assert exc.value.status_code == 422
    assert exc.value.error_code == 422
    assert exc.value.category == 'user'
    assert exc.value.request_id == 'req-7'
    assert exc.value.details is None
    assert exc.value.traceback is None
    assert 'bad input' in str(exc.value)


def test_detail_only_error_falls_back_to_unknown_category():
    """A non-ErrorPayload JSON body uses the lowercase unknown category."""
    resp = _Resp(404, body={'detail': 'Not Found'})
    with pytest.raises(TwinkleHTTPError) as exc:
        _handle_response(resp)
    assert exc.value.status_code == 404
    assert exc.value.category == 'unknown'
    assert exc.value.error_code is None
    assert 'Not Found' in str(exc.value)


def test_non_json_body_falls_back_to_text():
    resp = _Resp(500, body=None, text='raw traceback text')
    with pytest.raises(TwinkleHTTPError) as exc:
        _handle_response(resp)
    assert exc.value.category == 'unknown'
    assert 'raw traceback text' in str(exc.value)


def test_validation_details_are_preserved():
    details = [{'loc': ['body', 'items', 0], 'msg': 'invalid', 'type': 'value_error'}]
    resp = _Resp(
        422,
        body={
            'error': 'request validation failed',
            'category': 'user',
            'error_code': 422,
            'request_id': 'req-details',
            'details': details,
        },
    )
    with pytest.raises(TwinkleHTTPError) as exc:
        _handle_response(resp)
    assert exc.value.details == details
    assert exc.value.traceback is None


def test_server_traceback_is_preserved():
    traceback_text = 'Traceback (most recent call last):\n  File "/srv/app.py", line 1\nRuntimeError: boom'
    resp = _Resp(
        500,
        body={
            'error': 'RuntimeError: boom',
            'category': 'server',
            'error_code': 500,
            'request_id': 'req-trace',
            'traceback': traceback_text,
        },
    )
    with pytest.raises(TwinkleHTTPError) as exc:
        _handle_response(resp)
    assert exc.value.traceback == traceback_text
    assert exc.value.details is None


def test_410_raises_stop_iteration_not_http_error():
    """410 keeps raising StopIteration, not an HTTP error."""
    resp = _Resp(410, body={'detail': 'exhausted'})
    with pytest.raises(StopIteration):
        _handle_response(resp)


def test_ok_response_passes_through():
    resp = _Resp(200, body={'status': 'ok'})
    assert _handle_response(resp) is resp
