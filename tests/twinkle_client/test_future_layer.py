# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client future-layer unit tests.

``resolve`` is exercised against fabricated envelopes and a monkeypatched
``_post_retrieve``; no server or network is involved.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from twinkle_client import _future
from twinkle_client.exceptions import TaskFailedError, TaskRecordLostError, TaskWaitTimeoutError
from twinkle.protocol.types.errors import ErrorPayload
from twinkle.protocol.types.lifecycle import TaskEnvelope, UploadProgress


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
    """A task terminal in the submit envelope makes zero retrieve calls."""

    def _boom(_request_id, _transport):
        raise AssertionError('retrieve must not be called for a terminal submit')

    monkeypatch.setattr(_future, '_post_retrieve', _boom)
    out = _future.resolve(_completed({'loss': 1.0}), model_cls=_Model)
    assert out.result == {'loss': 1.0}


def test_terminal_submit_failure_raises_taskfailed_with_payload(monkeypatch):
    """A failure in the submit envelope raises TaskFailedError, payload intact."""
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r, _transport: pytest.fail('no retrieve'))
    with pytest.raises(TaskFailedError) as exc:
        _future.resolve(_failed(), model_cls=_Model)
    assert exc.value.error == 'boom'
    assert exc.value.category == 'server'
    assert exc.value.request_id == 'r'
    assert exc.value.error_code == 500
    assert not isinstance(exc.value, requests.HTTPError)


def test_terminal_failure_keeps_remote_traceback(monkeypatch):
    """The client exception must carry the server diagnostic, not only the summary."""
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r, _transport: pytest.fail('no retrieve'))
    diagnostic = 'Traceback (most recent call last):\nAssertionError: data/mask length mismatch at sample 0'
    with pytest.raises(TaskFailedError) as exc:
        _future.resolve(_failed(traceback=diagnostic), model_cls=_Model)
    assert exc.value.traceback == diagnostic


def test_model_cls_none_returns_none_result(monkeypatch):
    """A method that returned None before still returns None (not swallowed)."""
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r, _transport: pytest.fail('no retrieve'))
    assert _future.resolve(_completed(None), model_cls=None) is None


def test_non_terminal_submit_polls_until_terminal(monkeypatch):
    replies = [_running(), _running(), _completed({'ok': 1})]
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r, _transport, **_: replies.pop(0))
    out = _future.resolve(_running(), model_cls=_Model)
    assert out.result == {'ok': 1}
    assert replies == []


@pytest.mark.parametrize('legacy', [False, True])
def test_upload_poll_cursor_and_legacy_response(monkeypatch, legacy):
    seen = []
    running = _running()
    if legacy:
        payload = running.model_dump(exclude={'progress'})
        replies = [(_completed(None), {})]
    else:
        payload = running.model_dump()
        uploading = running.model_copy(update={
            'progress': UploadProgress(sequence=2, stage='uploading', uploaded_bytes=5, total_bytes=10)})
        completed = TaskEnvelope(
            request_id='r', status='completed', result={'repo_id': 'owner/model', 'revision': 'sha'},
            progress=UploadProgress(sequence=4, stage='batch_committed', uploaded_bytes=10, total_bytes=10,
                                    committed_bytes=10))
        replies = [(uploading, {'after_progress': 0}), (uploading, {'after_progress': 2}),
                   (completed, {'after_progress': 2})]

    def retrieve(_request_id, _transport, *, http_request_id, **kwargs):
        reply, expected = replies.pop(0)
        assert kwargs == expected
        return reply

    monkeypatch.setattr(_future, '_post_retrieve', retrieve)
    receipt = _future.resolve_response(SimpleNamespace(json=lambda: payload), None, on_progress=seen.append)
    assert [p.stage for p in seen] == ([] if legacy else ['uploading', 'batch_committed'])
    assert receipt == (None if legacy else {'repo_id': 'owner/model', 'revision': 'sha'})
    assert replies == []


def test_404_is_bounded_then_raises_record_lost(monkeypatch):

    def _always_404(_request_id, _transport, **_):
        e = requests.HTTPError('404')
        e.status_code = 404
        raise e

    monkeypatch.setattr(_future, '_post_retrieve', _always_404)
    with pytest.raises(TaskRecordLostError):
        _future.resolve(_running(), model_cls=_Model)


def test_transport_5xx_is_bounded_then_reraises(monkeypatch):
    monkeypatch.setattr(_future.time, 'sleep', lambda _s: None)  # no real backoff sleeps

    def _always_503(_request_id, _transport, **_):
        e = requests.HTTPError('503')
        e.status_code = 503
        raise e

    monkeypatch.setattr(_future, '_post_retrieve', _always_503)
    with pytest.raises(requests.HTTPError):
        _future.resolve(_running(), model_cls=_Model)


def test_connection_error_is_retried_then_succeeds(monkeypatch):
    from twinkle_client.http import ClientContext, ClientTransport

    monkeypatch.setattr(_future.time, 'sleep', lambda _s: None)
    replies = [requests.ConnectionError('reset'), _running(), _completed({'ok': True})]
    seen = []

    def post(url, *, headers, json, **_):
        seen.append(headers['x-request-id'])
        assert json == {'request_id': 'r'}
        value = replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(status_code=200, ok=True, json=value.model_dump)

    transport = ClientTransport(ClientContext('http://server', 'key'), session=SimpleNamespace(post=post))
    out = _future.resolve(_running(), model_cls=_Model, transport=transport)
    assert out.result == {'ok': True}
    assert len(seen) == 3
    assert seen[0] == seen[1] != seen[2]


def test_connection_error_is_bounded_then_reraised(monkeypatch):
    monkeypatch.setattr(_future.time, 'sleep', lambda _s: None)
    calls = 0

    def _always_fails(_request_id, _transport, **_):
        nonlocal calls
        calls += 1
        raise requests.ConnectionError('reset')

    monkeypatch.setattr(_future, '_post_retrieve', _always_fails)
    with pytest.raises(requests.ConnectionError, match='reset'):
        _future.resolve(_running(), model_cls=_Model)
    assert calls == _future._TRANSPORT_RETRY_MAX + 1


def test_non_retryable_4xx_reraises_immediately(monkeypatch):

    def _400(_request_id, _transport, **_):
        e = requests.HTTPError('400')
        e.status_code = 400
        raise e

    monkeypatch.setattr(_future, '_post_retrieve', _400)
    with pytest.raises(requests.HTTPError):
        _future.resolve(_running(), model_cls=_Model)


def test_total_timeout_raises_wait_timeout(monkeypatch):
    monkeypatch.setattr(_future, '_post_retrieve', lambda _r, _transport: _running())
    with pytest.raises(TaskWaitTimeoutError) as exc:
        _future.resolve(_running(), model_cls=_Model, total_timeout=0.0)
    assert exc.value.request_id == 'r'


def test_success_resets_both_retry_counters(monkeypatch):
    """A successful reply zeroes both counters, so intermittent 404s never sum up."""
    seq = []

    def _mixed(_request_id, _transport, **_):
        seq.append(1)
        n = len(seq)
        if n in (1, 2, 4, 5):  # 404s interleaved with a success at n==3
            e = requests.HTTPError('404')
            e.status_code = 404
            raise e
        if n == 3:
            return _running()  # success resets not_found_count
        return _completed({'done': True})

    monkeypatch.setattr(_future, '_post_retrieve', _mixed)
    out = _future.resolve(_running(), model_cls=_Model)
    assert out.result == {'done': True}
