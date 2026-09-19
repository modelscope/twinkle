# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from twinkle_client.http import ClientContext, ClientTransport
from twinkle_client.http.context import set_default_transport
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
            routing_id=f'{name}-routing',
        ),
        session=_Session(),
    )


def test_context_normalizes_url_and_transport_builds_stable_headers():
    transport = _transport('alpha')
    transport.post('/resource')
    _, url, kwargs = transport._session.calls[-1]

    assert url == 'http://alpha/api/v1/resource'
    assert kwargs['headers']['Authorization'] == 'Bearer alpha-key'
    assert kwargs['headers']['X-Twinkle-Session-Id'] == 'alpha-session'
    assert kwargs['headers']['x-request-id'] == 'alpha-routing'


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
    from twinkle_client.types.server import GetServerCapabilitiesResponse

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
