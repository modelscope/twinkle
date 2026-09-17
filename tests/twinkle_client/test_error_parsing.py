# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client error-response parsing (T3.4 / Requirement 3 #7-#11)."""
from __future__ import annotations

import pytest
import requests

from twinkle_client.exceptions import TwinkleHTTPError
from twinkle_client.http.http_utils import _handle_response


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
    """R3#7/#8: top-level category/error_code/request_id are preferred."""
    resp = _Resp(422, body={'error': 'bad input', 'category': 'user', 'error_code': 422, 'request_id': 'req-7'})
    with pytest.raises(TwinkleHTTPError) as exc:
        _handle_response(resp)
    assert isinstance(exc.value, requests.HTTPError)   # R3#8: existing except clauses keep working
    assert exc.value.status_code == 422
    assert exc.value.error_code == 422
    assert exc.value.category == 'user'
    assert exc.value.request_id == 'req-7'
    assert 'bad input' in str(exc.value)


def test_detail_only_error_falls_back_to_unknown_category():
    """R3#7: FastAPI's built-in {detail: ...} maps to category='Unknown'."""
    resp = _Resp(404, body={'detail': 'Not Found'})
    with pytest.raises(TwinkleHTTPError) as exc:
        _handle_response(resp)
    assert exc.value.status_code == 404
    assert exc.value.category == 'Unknown'
    assert exc.value.error_code is None
    assert 'Not Found' in str(exc.value)


def test_non_json_body_falls_back_to_text():
    resp = _Resp(500, body=None, text='raw traceback text')
    with pytest.raises(TwinkleHTTPError) as exc:
        _handle_response(resp)
    assert exc.value.category == 'Unknown'
    assert 'raw traceback text' in str(exc.value)


def test_410_raises_stop_iteration_not_http_error():
    """R3#9: 410 keeps raising StopIteration, not an HTTP error."""
    resp = _Resp(410, body={'detail': 'exhausted'})
    with pytest.raises(StopIteration):
        _handle_response(resp)


def test_ok_response_passes_through():
    resp = _Resp(200, body={'status': 'ok'})
    assert _handle_response(resp) is resp
