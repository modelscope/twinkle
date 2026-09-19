# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tinker /retrieve_future wire regression (T6.3 / Property 6 / R8#8).

The tinker endpoint's response shape and status-code semantics must be unchanged by
this spec, across all three shapes: ``try_again`` / ``{error, category}`` / bare
result. It shares the poll_config window but keeps its own wire contract.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from twinkle.server.gateway.tinker_handlers import _register_tinker_routes


class _State:

    def __init__(self, record):
        self._record = record

    async def get_future(self, request_id):
        return self._record


class _Gateway:

    def __init__(self, record):
        self.state = _State(record)


def _client(record):
    app = FastAPI()
    _register_tinker_routes(app, lambda: _Gateway(record))
    return TestClient(app)


def test_try_again_shape_for_non_terminal(monkeypatch):
    monkeypatch.setenv('TWINKLE_LONG_POLL_TIMEOUT', '0.2')
    monkeypatch.setenv('TWINKLE_POLL_INTERVAL', '0.05')
    resp = _client({'status': 'running', 'queue_state': 'active'}).post(
        '/retrieve_future', json={'request_id': 'r'})
    assert resp.status_code == 200
    assert resp.json()['type'] == 'try_again'


def test_error_category_shape_for_failed():
    resp = _client({'status': 'failed', 'result': {'error': 'boom', 'category': 'server'}}).post(
        '/retrieve_future', json={'request_id': 'r'})
    assert resp.status_code == 200
    body = resp.json()
    assert body['error'] == 'boom'
    assert body['category'] == 'server'
    assert 'type' not in body


def test_bare_result_shape_for_completed():
    resp = _client({'status': 'completed', 'result': {'foo': 'bar'}}).post(
        '/retrieve_future', json={'request_id': 'r'})
    assert resp.status_code == 200
    assert resp.json() == {'foo': 'bar'}
