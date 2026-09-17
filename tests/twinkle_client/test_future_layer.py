# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client_Future_Layer unit tests (T2.2 / Requirement 4).

``resolve`` is exercised against fabricated envelopes and a monkeypatched
``_post_retrieve``; no server or network is involved.
"""
from __future__ import annotations

import pytest
import requests

from twinkle_client import _future
from twinkle_client.exceptions import TaskFailedError, TaskRecordLostError, TaskWaitTimeoutError
from twinkle_client.types.errors import ErrorPayload
from twinkle_client.types.lifecycle import TaskEnvelope


class _Model:
    """A model_cls that records what it deserialized."""

    def __init__(self, result):
        self.result = result

    @classmethod
    def model_validate(cls, value):
        return _Model(value)


def _completed(result):
    return TaskEnvelope(request_id='r', status='completed', result=result)


def _failed(**kw):
    payload = ErrorPayload(error='boom', category='server', error_code=500, request_id='r', **kw)
    return TaskEnvelope(request_id='r', status='failed', error=payload)


def _running():
    return TaskEnvelope(request_id='r', status='running', queue_state='active')


def test_terminal_submit_issues_no_retrieve(monkeypatch):
    """R8#1: a task terminal in the submit envelope makes zero retrieve calls."""
    def _boom(_request_id):
        raise AssertionError('retrieve must not be called for a terminal submit')

    monkeypatch.setattr(_future, '_post_retrieve', _boom)
    out = _future.resolve(_completed({'loss': 1.0}), model_cls=_Model)
    assert out.result == {'loss': 1.0}


def test_terminal_submit_failure_raises_taskfailed_with_payload(monkeypatch):
    """Property 0: a failure in the submit envelope raises TaskFailedError, payload intact."""
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r: pytest.fail('no retrieve'))
    with pytest.raises(TaskFailedError) as exc:
        _future.resolve(_failed(), model_cls=_Model)
    assert exc.value.error == 'boom'
    assert exc.value.category == 'server'
    assert exc.value.request_id == 'r'
    assert exc.value.error_code == 500
    assert not isinstance(exc.value, requests.HTTPError)   # R3#10


def test_model_cls_none_returns_none_result(monkeypatch):
    """R4#9: a method that returned None before still returns None (not swallowed)."""
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r: pytest.fail('no retrieve'))
    assert _future.resolve(_completed(None), model_cls=None) is None


def test_non_terminal_submit_polls_until_terminal(monkeypatch):
    replies = [_running(), _running(), _completed({'ok': 1})]
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r: replies.pop(0))
    out = _future.resolve(_running(), model_cls=_Model)
    assert out.result == {'ok': 1}
    assert replies == []


def test_404_is_bounded_then_raises_record_lost(monkeypatch):
    def _always_404(_request_id):
        e = requests.HTTPError('404')
        e.status_code = 404
        raise e

    monkeypatch.setattr(_future, '_post_retrieve', _always_404)
    with pytest.raises(TaskRecordLostError):
        _future.resolve(_running(), model_cls=_Model)


def test_transport_5xx_is_bounded_then_reraises(monkeypatch):
    monkeypatch.setattr(_future.time, 'sleep', lambda _s: None)   # no real backoff sleeps

    def _always_503(_request_id):
        e = requests.HTTPError('503')
        e.status_code = 503
        raise e

    monkeypatch.setattr(_future, '_post_retrieve', _always_503)
    with pytest.raises(requests.HTTPError):
        _future.resolve(_running(), model_cls=_Model)


def test_non_retryable_4xx_reraises_immediately(monkeypatch):
    def _400(_request_id):
        e = requests.HTTPError('400')
        e.status_code = 400
        raise e

    monkeypatch.setattr(_future, '_post_retrieve', _400)
    with pytest.raises(requests.HTTPError):
        _future.resolve(_running(), model_cls=_Model)


def test_total_timeout_raises_wait_timeout(monkeypatch):
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r: _running())
    with pytest.raises(TaskWaitTimeoutError) as exc:
        _future.resolve(_running(), model_cls=_Model, total_timeout=0.0)
    assert exc.value.request_id == 'r'


def test_success_resets_both_retry_counters(monkeypatch):
    """R4#8: a successful reply zeroes both counters, so intermittent 404s never sum up."""
    seq = []

    def _mixed(_request_id):
        seq.append(1)
        n = len(seq)
        if n in (1, 2, 4, 5):          # 404s interleaved with a success at n==3
            e = requests.HTTPError('404')
            e.status_code = 404
            raise e
        if n == 3:
            return _running()          # success resets not_found_count
        return _completed({'done': True})

    monkeypatch.setattr(_future, '_post_retrieve', _mixed)
    out = _future.resolve(_running(), model_cls=_Model)
    assert out.result == {'done': True}
