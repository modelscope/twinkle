# Copyright (c) ModelScope Contributors. All rights reserved.
"""Wire tests for the twinkle Retrieve_Endpoint (T1.4, Property 4/5, R8#4/#5).

These use a fake state and FastAPI's TestClient; no Ray runtime is needed, so they
live outside the state-actor fixtures.
"""
from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from twinkle.server.gateway.twinkle_handlers import _register_twinkle_routes


class _State:
    """A fake ServerState whose get_future returns a fixed record (or None)."""

    def __init__(self, record):
        self._record = record

    async def get_future(self, request_id: str):
        return self._record


class _Gateway:

    def __init__(self, record):
        self.state = _State(record)


def _client(record) -> TestClient:
    app = FastAPI()
    _register_twinkle_routes(app, lambda: _Gateway(record))
    return TestClient(app)


def test_completed_with_null_result_returns_200_and_null(monkeypatch):
    """Property 4 / R8#4: completed + result=None is 200 with result null, not 500."""
    client = _client({'status': 'completed', 'result': None})
    resp = client.post('/twinkle/retrieve_future', json={'request_id': 'req-1'})
    assert resp.status_code == 200
    body = resp.json()
    assert body['status'] == 'completed'
    assert body['result'] is None
    assert body['error'] is None


def test_legacy_two_field_failure_returns_200_and_valid_envelope():
    """Property 5 / R8#5: a {error, category} record is 200 with a legal envelope."""
    client = _client({'status': 'failed', 'result': {'error': 'boom', 'category': 'server'}})
    resp = client.post('/twinkle/retrieve_future', json={'request_id': 'req-2'})
    assert resp.status_code == 200
    body = resp.json()
    assert body['status'] == 'failed'
    assert body['result'] is None
    assert body['error']['error'] == 'boom'
    assert body['error']['category'] == 'server'
    assert body['error']['error_code'] == 500
    assert body['error']['request_id'] == 'req-2'


def test_always_missing_record_404s_only_after_the_full_window(monkeypatch):
    """R2#5: a request_id that never appears returns 404, and only after waiting a window."""
    monkeypatch.setenv('TWINKLE_LONG_POLL_TIMEOUT', '0.3')
    client = _client(None)
    start = time.monotonic()
    resp = client.post('/twinkle/retrieve_future', json={'request_id': 'ghost'})
    waited = time.monotonic() - start
    assert resp.status_code == 404
    assert 'ghost' in resp.json()['detail']
    # It must fold the missing record into the wait loop, not short-circuit.
    assert waited >= 0.3


def test_terminal_record_returns_immediately(monkeypatch):
    """A record already terminal must not wait out the window."""
    monkeypatch.setenv('TWINKLE_LONG_POLL_TIMEOUT', '30')
    client = _client({'status': 'completed', 'result': {'ok': True}})
    start = time.monotonic()
    resp = client.post('/twinkle/retrieve_future', json={'request_id': 'req-5'})
    waited = time.monotonic() - start
    assert resp.status_code == 200
    assert resp.json()['result'] == {'ok': True}
    assert waited < 5.0
