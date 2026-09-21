# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import ast
import asyncio
from pathlib import Path

from twinkle.server.gateway import use_cases as services


class _State:

    def __init__(self, records):
        self.records = list(records)

    async def get_future(self, request_id):
        return self.records.pop(0)


async def _no_sleep(_seconds):
    return None


def test_poll_future_returns_canonical_terminal_record(monkeypatch):
    monkeypatch.setattr(services, 'long_poll_window', lambda: 10)
    monkeypatch.setattr(services, 'retrieve_poll_interval', lambda: 0)
    monkeypatch.setattr(services.asyncio, 'sleep', _no_sleep)
    state = _State([None, {'status': 'running'}, {'status': 'completed', 'result': None}])

    outcome = asyncio.run(services.poll_future(state, 'request-1'))

    assert outcome.timed_out is False
    assert outcome.record == {'status': 'completed', 'result': None}


def test_gateway_services_do_not_import_protocol_models():
    path = Path(services.__file__)
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not any(module == 'tinker.types' or module.startswith('twinkle_client.types') for module in imports)


class _FakeGateway:
    """Minimal ``GatewayServer`` stand-in; ``poll_future`` is patched so state is unused."""

    state = None


def _parity_client(monkeypatch, canonical_record):
    """Register both real wire adapters on one app, feeding both the same canonical record."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from twinkle.server.gateway import tinker_handlers, twinkle_handlers

    async def _fixed_poll(_state, _request_id):
        return services.FuturePollResult(record=canonical_record, timed_out=False)

    app = FastAPI()
    monkeypatch.setattr(tinker_handlers, 'poll_future', _fixed_poll)
    monkeypatch.setattr(twinkle_handlers, 'poll_future', _fixed_poll)
    tinker_handlers._register_gateway_tinker_routes(app, lambda: _FakeGateway())
    twinkle_handlers._register_gateway_twinkle_routes(app, lambda: _FakeGateway())
    return TestClient(app)


def test_same_completed_record_diverges_into_protocol_specific_wire_shapes(monkeypatch):
    """One canonical ``completed`` record -> Tinker raw result vs Twinkle TaskEnvelope."""
    client = _parity_client(monkeypatch, {'status': 'completed', 'result': {'loss': 1.0}})

    tinker_resp = client.post('/retrieve_future', json={'request_id': 'r1'})
    twinkle_resp = client.post('/twinkle/retrieve_future', json={'request_id': 'r1'})

    assert tinker_resp.status_code == 200
    assert twinkle_resp.status_code == 200
    # Tinker returns the raw result payload; Twinkle wraps it in a canonical envelope.
    assert tinker_resp.json() == {'loss': 1.0}
    twinkle_body = twinkle_resp.json()
    assert twinkle_body['request_id'] == 'r1'
    assert twinkle_body['status'] == 'completed'
    assert twinkle_body['result'] == {'loss': 1.0}


def test_completed_null_result_keeps_the_two_protocols_divergent(monkeypatch):
    """The load-bearing difference: null result is a 500 for Tinker but a valid 200 envelope for Twinkle."""
    client = _parity_client(monkeypatch, {'status': 'completed', 'result': None})

    tinker_resp = client.post('/retrieve_future', json={'request_id': 'r1'})
    twinkle_resp = client.post('/twinkle/retrieve_future', json={'request_id': 'r1'})

    assert tinker_resp.status_code == 500
    assert twinkle_resp.status_code == 200
    assert twinkle_resp.json()['status'] == 'completed'
