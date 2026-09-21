# Copyright (c) ModelScope Contributors. All rights reserved.
"""Directed tests for ``gateway/proxy.py``.

Covers the route-URL construction now sourced from ``gateway.routes``, the
``H_MULTIPLEX`` header compatibility in ``_prepare_headers``, and the 502 ``ErrorPayload``
fallback when the upstream is unreachable.
"""
from __future__ import annotations

import json

import pytest
from starlette.requests import Request
from unittest.mock import AsyncMock

from twinkle.server.gateway.proxy import ServiceProxy
from twinkle.protocol.headers import H_MULTIPLEX, H_MULTIPLEX_LEGACY, H_REQUEST_ID


def _make_request(headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    scope = {
        'type': 'http',
        'method': 'POST',
        'headers': headers or [],
        'query_string': b'',
        'path': '/',
    }
    return Request(scope)


@pytest.mark.parametrize(
    'route_prefix,host,expected',
    [
        ('/api/v1', 'localhost', 'http://localhost:8000/api/v1/model/Qwen/tinker/forward'),
        ('/api/v1/', 'localhost', 'http://localhost:8000/api/v1/model/Qwen/tinker/forward'),
        ('', 'localhost', 'http://localhost:8000/model/Qwen/tinker/forward'),
        ('/api/v1', '0.0.0.0', 'http://localhost:8000/api/v1/model/Qwen/tinker/forward'),
    ],
)
def test_build_target_url(route_prefix, host, expected):
    proxy = ServiceProxy(http_options={'host': host, 'port': 8000}, route_prefix=route_prefix)
    assert proxy._build_target_url('model', 'Qwen', 'tinker/forward') == expected


def test_prepare_headers_sets_multiplex_from_request_id():
    proxy = ServiceProxy(http_options={}, route_prefix='/api/v1')
    headers = proxy._prepare_headers({H_REQUEST_ID: 'req-1'})
    assert headers.get(H_MULTIPLEX) == 'req-1'
    assert headers.get(H_MULTIPLEX_LEGACY) == 'req-1'
    # ``host`` / ``content-length`` are stripped before forwarding.
    assert 'host' not in {k.lower() for k in headers}


@pytest.mark.asyncio
async def test_proxy_request_502_fallback_returns_error_payload():
    proxy = ServiceProxy(http_options={'host': 'localhost', 'port': 8000}, route_prefix='/api/v1')
    proxy.client.request = AsyncMock(side_effect=RuntimeError('upstream down'))
    request = _make_request(headers=[(H_REQUEST_ID.encode(), b'req-9')])

    response = await proxy.proxy_request(request, 'tinker/forward', 'Qwen', 'model', body_override=b'{}')

    assert response.status_code == 502
    payload = json.loads(response.body)
    assert payload['error_code'] == 502
    assert payload['category'] == 'server'
    assert payload['request_id'] == 'req-9'
