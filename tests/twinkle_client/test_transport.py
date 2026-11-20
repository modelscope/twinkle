# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from twinkle.protocol.headers import H_CLIENT_ID, H_REQUEST_ID
from twinkle.protocol.types.session import CreateSessionRequest
from twinkle_client.http import ClientContext, ClientTransport
from twinkle_client.http.context import capture_transport, clear_default_transport, set_default_transport
from twinkle_client.manager import TwinkleClient


class _Response:
    ok = True
    status_code = 200
    url = 'http://server'
    text = ''

    def json(self):
        return {}


class _Session:

    def __init__(self):
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append(('post', url, kwargs))
        return _Response()

    def get(self, url, **kwargs):
        self.calls.append(('get', url, kwargs))
        return _Response()

    def delete(self, url, **kwargs):
        self.calls.append(('delete', url, kwargs))
        return _Response()

    def close(self):
        self.closed = True


def _transport(name: str) -> ClientTransport:
    return ClientTransport(
        ClientContext(
            base_url=f'http://{name}',
            api_key=f'{name}-key',
            session_id=f'{name}-session',
            client_id=f'{name}-client',
        ),
        session=_Session(),
    )


def test_transport_assigns_request_ids_and_preserves_client_identity():
    transport = _transport('alpha')
    transport.get('/resource')
    transport.post('/resource')
    transport.post_model('/resource', CreateSessionRequest())
    transport.delete('/resource')
    request_ids = set()
    for _, url, kwargs in transport._session.calls:
        assert url == 'http://alpha/api/v1/resource'
        headers = kwargs['headers']
        assert headers['Authorization'] == 'Bearer alpha-key'
        assert headers['X-Twinkle-Session-Id'] == 'alpha-session'
        assert headers[H_CLIENT_ID] == 'alpha-client'
        assert 'x-ray-serve-request-id' not in {name.lower() for name in headers}
        request_ids.add(uuid.UUID(headers[H_REQUEST_ID]))
    assert len(request_ids) == 4


def test_session_attachment_preserves_client_identity():
    context = ClientContext('http://server', 'key')
    assert uuid.UUID(context.client_id)
    assert replace(context, session_id='session').client_id == context.client_id


def test_transport_status_retry_reuses_request_id():
    seen = []

    class Handler(BaseHTTPRequestHandler):

        def do_GET(self):
            seen.append((self.headers[H_REQUEST_ID], self.headers[H_CLIENT_ID]))
            self.send_response(503 if len(seen) == 1 else 200)
            self.end_headers()

    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        transport = ClientTransport(ClientContext(f'http://127.0.0.1:{server.server_port}', 'key'), timeout=3)
        try:
            assert transport.get('/resource').status_code == 200
            assert len(seen) == 2
            assert seen[0] == seen[1]
            assert uuid.UUID(seen[0][0])
            assert seen[0][1] == transport.context.client_id
        finally:
            transport.close()
            server.shutdown()
            worker.join(timeout=3)


def test_tinker_defaults_carry_the_client_identity(monkeypatch):
    from twinkle_client.http import context
    from twinkle_client.utils.patch_tinker import _make_patched_service_client_init

    monkeypatch.setattr(context, '_default_transport', _transport('tinker'))
    captured = {}

    def original(_, **kwargs):
        captured.update(kwargs['default_headers'])

    _make_patched_service_client_init(original)(object(), api_key='explicit')
    assert captured[H_CLIENT_ID] == captured[H_REQUEST_ID] == 'tinker-client'
    assert captured['Twinkle-Authorization'] == 'Bearer explicit'
    assert 'x-ray-serve-request-id' not in {name.lower() for name in captured}


def test_wrapper_captures_default_once_and_factory_is_explicit():
    from twinkle_client.model import MultiLoraTransformersModel

    transport_a = _transport('alpha')
    transport_b = _transport('beta')
    set_default_transport(transport_a)
    legacy_model = MultiLoraTransformersModel('model')
    set_default_transport(transport_b)

    client_a = TwinkleClient(transport=transport_a)
    factory_model = client_a.model('factory-model')

    assert legacy_model._transport is transport_a
    assert factory_model._transport is transport_a
    assert legacy_model.server_url.startswith('http://alpha/api/v1/')
    assert factory_model.server_url.startswith('http://alpha/api/v1/')


def test_close_is_idempotent_and_does_not_clear_another_default():
    transport_a = _transport('alpha')
    transport_b = _transport('beta')
    client_a = TwinkleClient(transport=transport_a)
    set_default_transport(transport_b)

    client_a.close()
    client_a.close()

    assert transport_a.closed
    assert transport_a._session.closed
    assert not transport_b.closed


def test_http_public_api_has_no_legacy_context_getters_or_setters():
    import twinkle_client.http as http

    assert not ({
        'get_base_url',
        'get_api_key',
        'get_session_id',
        'get_request_id',
        'set_base_url',
        'set_api_key',
        'set_session_id',
        'set_request_id',
    } & set(http.__all__))


def test_default_transport_fallback_logs_info_once(monkeypatch, caplog):
    import twinkle_client.http.context as context

    monkeypatch.setattr(context, '_default_transport', None)
    with caplog.at_level(logging.INFO, logger='twinkle_client'):
        transport = capture_transport()
        assert capture_transport() is transport
    records = [record for record in caplog.records if 'No explicit Twinkle client configured' in record.message]
    assert len(records) == 1
    assert transport.context.base_url in records[0].message
    transport.close()
    clear_default_transport(transport)


def test_publish_blocks_rebind_and_replacement_does_not_close_old(caplog):
    first = _transport('first')
    first.bind_context(ClientContext(base_url='http://bound', api_key='key'))
    set_default_transport(first)

    with pytest.raises(RuntimeError, match='published'):
        first.bind_context(ClientContext(base_url='http://too-late', api_key='key'))

    second = _transport('second')
    with caplog.at_level(logging.WARNING, logger='twinkle_client'):
        set_default_transport(second)
    assert any('Replacing default Twinkle transport' in record.message for record in caplog.records)
    assert not first.closed
    clear_default_transport(second)
    first.close()
    second.close()


def test_transport_adapter_configuration_and_post_retry_boundary():
    transport = ClientTransport(ClientContext(base_url='http://pool', api_key='key'))
    adapter = transport._session.get_adapter('http://')
    assert adapter._pool_maxsize == 32
    assert adapter._pool_block is True
    assert adapter.max_retries.allowed_methods == frozenset({'GET', 'DELETE'})
    assert 'POST' not in adapter.max_retries.allowed_methods
    transport.close()


def test_client_closes_distinct_heartbeat_transport():
    main = _transport('main')
    heartbeat = _transport('heartbeat')
    client = TwinkleClient(transport=main, heartbeat_transport=heartbeat)
    client.close()
    assert main.closed and heartbeat.closed


class _CapabilitiesResponse(_Response):

    def json(self):
        return {'supported_models': [{'model_name': 'm'}]}


class _CapabilitiesSession(_Session):

    def get(self, url, **kwargs):
        self.calls.append(('get', url, kwargs))
        return _CapabilitiesResponse()


def _capability_client(name: str) -> TwinkleClient:
    transport = ClientTransport(
        ClientContext(base_url=f'http://{name}', api_key=f'{name}-key', session_id=f'{name}-session'),
        session=_CapabilitiesSession(),
    )
    return TwinkleClient(transport=transport)


def test_capability_cache_is_per_transport_and_not_process_global():
    from twinkle.protocol.types.server import GetServerCapabilitiesResponse

    client_a = _capability_client('alpha')
    client_b = _capability_client('beta')

    first = client_a.get_server_capabilities()
    again = client_a.get_server_capabilities()

    # Cached on the transport instance: the second query reuses it, no second GET.
    assert isinstance(first, GetServerCapabilitiesResponse)
    assert again is first
    assert sum(1 for call in client_a.transport._session.calls if call[0] == 'get') == 1

    # Isolation: caching on A must never populate another transport's cache.
    assert client_b.transport.cached_capabilities is None
    client_b.get_server_capabilities()
    assert client_b.transport.cached_capabilities is not client_a.transport.cached_capabilities
