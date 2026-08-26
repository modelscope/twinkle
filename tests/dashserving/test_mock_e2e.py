# Copyright (c) ModelScope Contributors. All rights reserved.
"""Local HTTP tunnel mock using the production DashServing adapter code.

This HTTP-level test lives outside ``tests/server`` because it does not
need that suite's session-scoped Ray cluster.
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from twinkle.server.dashserving import create_dashserving_app
from twinkle.server.dashserving.proxy import RuntimeProxy
from twinkle.server.common.tunnel import TunnelResponse


def _build_runtime_mock() -> FastAPI:
    """Represent the existing server exposing Twinkle and Tinker APIs."""
    app = FastAPI()

    @app.api_route('/api/v1/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def twinkle_endpoint(path: str, request: Request) -> JSONResponse:
        if path == 'twinkle/healthz':
            return JSONResponse({'status': 'ok'})
        if path.endswith('/expired'):
            return JSONResponse({'detail': 'checkpoint expired'}, status_code=410)

        body = await request.json() if request.method == 'POST' else None
        observed_headers = {
            name: request.headers.get(name)
            for name in (
                'x-request-id',
                'x-ray-serve-request-id',
                'serve_multiplexed_model_id',
                'serve-multiplexed-model-id',
                'authorization',
                'twinkle-authorization',
                'x-twinkle-session-id',
            )
        }
        return JSONResponse({
            'method': request.method,
            'path': request.url.path,
            'query': dict(request.query_params),
            'headers': observed_headers,
            'body': body,
        })

    return app


def _build_dashserving_mock(adapter_client: httpx.AsyncClient) -> FastAPI:
    """Represent DS forwarding a Native HTTP request to runtime `/api`."""
    app = FastAPI()

    @app.post('/invoke')
    async def invoke(request: Request) -> Response:
        ds_request_id = request.headers.get('x-mock-ds-request-id') or str(uuid.uuid4())
        adapter_response = await adapter_client.post(
            '/api',
            content=await request.body(),
            headers={
                'Content-Type': 'application/json',
                'X-DashServing-Request-Id': ds_request_id,
            },
        )
        response_headers = {
            name: value
            for name, value in adapter_response.headers.items()
            if name.lower().startswith('x-dashserving-')
        }
        return Response(
            content=adapter_response.content,
            status_code=adapter_response.status_code,
            headers=response_headers,
            media_type='application/json',
        )

    return app


def _build_modelscope_mock(ds_client: httpx.AsyncClient) -> FastAPI:
    """Represent the ModelScope public route and tunnel codec."""
    app = FastAPI()

    @app.api_route('/tinker/api/v1/{path:path}', methods=['GET', 'POST', 'DELETE'])
    @app.api_route('/twinkle/api/v1/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def runtime_route(path: str, request: Request) -> Response:
        authorization = request.headers.get('authorization')
        twinkle_request_id = request.headers.get('x-request-id')
        if not authorization:
            return JSONResponse({'detail': 'missing authorization'}, status_code=401)
        if not twinkle_request_id:
            return JSONResponse({'detail': 'missing x-request-id'}, status_code=400)

        forwarded_headers = {
            'authorization',
            'serve-multiplexed-model-id',
            'serve_multiplexed_model_id',
            'twinkle-authorization',
            'x-ray-serve-request-id',
            'x-request-id',
            'x-twinkle-session-id',
        }
        tunnel_request = {
            'method': request.method,
            # The mounted route already consumed the public prefix.
            'path': f'/api/v1/{path}',
            'query': dict(request.query_params),
            'headers': {
                name: value
                for name, value in request.headers.items()
                if name.lower() in forwarded_headers
            },
            'body': await request.json() if request.method == 'POST' else None,
        }
        ds_request_id = f'ds-{uuid.uuid4()}'
        ds_response = await ds_client.post(
            '/invoke',
            json=tunnel_request,
            headers={'X-Mock-DS-Request-Id': ds_request_id},
        )
        if ds_response.status_code == 504:
            return JSONResponse({'detail': 'DashServing timeout'}, status_code=504)
        if ds_response.status_code != 200:
            return JSONResponse({'detail': 'DashServing invocation failed'}, status_code=502)

        required_ds_headers = {
            'x-dashserving-request-id',
            'x-dashserving-attributes',
            'x-dashserving-usage',
            'x-dashserving-status-code',
            'x-dashserving-status-name',
            'x-dashserving-status-message',
        }
        if not required_ds_headers.issubset(ds_response.headers):
            return JSONResponse({'detail': 'DashServing response error'}, status_code=502)
        if ds_response.headers['x-dashserving-request-id'] != ds_request_id:
            return JSONResponse({'detail': 'DashServing request ID mismatch'}, status_code=502)
        if ds_response.headers['x-dashserving-status-code'] != '200':
            return JSONResponse({'detail': 'DashServing status error'}, status_code=502)
        attributes = json.loads(ds_response.headers['x-dashserving-attributes'])
        if attributes != {}:
            return JSONResponse({'detail': 'Unexpected DashServing attributes'}, status_code=502)

        tunnel_response = TunnelResponse.model_validate(ds_response.json())
        response_headers = {}
        content_type = tunnel_response.headers.get('content-type')
        if content_type:
            response_headers['Content-Type'] = content_type
        return JSONResponse(
            tunnel_response.body,
            status_code=tunnel_response.status_code,
            headers=response_headers,
        )

    return app


@asynccontextmanager
async def _mock_chain() -> AsyncIterator[tuple[httpx.AsyncClient, httpx.AsyncClient]]:
    """Build Client -> ModelScope -> DS -> real Adapter -> Twinkle mock."""
    runtime_app = _build_runtime_mock()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime_app),
        base_url='http://runtime-internal',
    ) as runtime_client:
        proxy = RuntimeProxy('http://runtime-internal', client=runtime_client)
        adapter_app = create_dashserving_app(proxy=proxy)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=adapter_app),
            base_url='http://adapter',
        ) as adapter_client:
            ds_app = _build_dashserving_mock(adapter_client)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=ds_app),
                base_url='http://dashserving',
            ) as ds_client:
                modelscope_app = _build_modelscope_mock(ds_client)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=modelscope_app),
                    base_url='http://modelscope',
                ) as user_client:
                    yield user_client, adapter_client


def _client_headers(*, session_id: str | None = None) -> dict[str, str]:
    headers = {
        'Authorization': 'Bearer user-token',
        'Twinkle-Authorization': 'Bearer user-token',
        'x-request-id': 'client-sticky-01',
        'X-Ray-Serve-Request-Id': 'client-sticky-01',
        'serve_multiplexed_model_id': 'client-sticky-01',
        'Serve-Multiplexed-Model-Id': 'client-sticky-01',
    }
    if session_id:
        headers['X-Twinkle-Session-Id'] = session_id
    return headers


@pytest.mark.asyncio
async def test_post_crosses_the_complete_mock_chain() -> None:
    async with _mock_chain() as (client, _adapter):
        response = await client.post(
            '/twinkle/api/v1/model/Qwen/Qwen3.6-27B/twinkle/forward_backward?timeout=20',
            headers=_client_headers(session_id='session-123'),
            json={'inputs': [], 'adapter_name': 'default'},
        )

    assert response.status_code == 200
    assert 'x-dashserving-request-id' not in response.headers
    observed = response.json()
    assert observed['method'] == 'POST'
    assert observed['path'] == '/api/v1/model/Qwen/Qwen3.6-27B/twinkle/forward_backward'
    assert observed['query'] == {'timeout': '20'}
    assert observed['body'] == {'inputs': [], 'adapter_name': 'default'}
    assert observed['headers'] == {
        'x-request-id': 'client-sticky-01',
        'x-ray-serve-request-id': 'client-sticky-01',
        'serve_multiplexed_model_id': 'client-sticky-01',
        'serve-multiplexed-model-id': 'client-sticky-01',
        'authorization': 'Bearer user-token',
        'twinkle-authorization': 'Bearer user-token',
        'x-twinkle-session-id': 'session-123',
    }


@pytest.mark.asyncio
async def test_get_query_and_inner_status_survive_the_chain() -> None:
    async with _mock_chain() as (client, _adapter):
        get_response = await client.get(
            '/twinkle/api/v1/twinkle/training_runs?limit=20',
            headers=_client_headers(),
        )
        gone_response = await client.delete(
            '/twinkle/api/v1/twinkle/checkpoints/expired',
            headers=_client_headers(),
        )

    assert get_response.status_code == 200
    assert get_response.json()['method'] == 'GET'
    assert get_response.json()['query'] == {'limit': '20'}
    assert get_response.json()['body'] is None
    assert gone_response.status_code == 410
    assert gone_response.json() == {'detail': 'checkpoint expired'}


@pytest.mark.asyncio
async def test_tinker_request_crosses_the_same_adapter() -> None:
    async with _mock_chain() as (client, _adapter):
        response = await client.post(
            '/tinker/api/v1/create_model',
            headers=_client_headers(session_id='session-123'),
            json={'base_model': 'Qwen/Qwen3.6-27B'},
        )

    assert response.status_code == 200
    observed = response.json()
    assert observed['path'] == '/api/v1/create_model'
    assert observed['body'] == {'base_model': 'Qwen/Qwen3.6-27B'}
    assert observed['headers']['authorization'] == 'Bearer user-token'
    assert observed['headers']['x-request-id'] == 'client-sticky-01'


@pytest.mark.asyncio
async def test_adapter_native_http_contract() -> None:
    async with _mock_chain() as (_client, adapter):
        tunnel_request = {
            'method': 'POST',
            'path': '/api/v1/create_model',
            'headers': _client_headers(),
            'body': {'base_model': 'Qwen/Qwen3.6-27B'},
        }
        response = await adapter.post('/api', json=tunnel_request)

    assert response.status_code == 200
    assert response.headers['x-dashserving-request-id']
    assert response.headers['x-dashserving-status-code'] == '200'
    assert response.headers['x-dashserving-status-name'] == 'Success'
    assert response.headers['x-dashserving-attributes'] == '{}'
    assert response.headers['x-dashserving-usage'] == '{}'


@pytest.mark.asyncio
async def test_adapter_health_checks_runtime_upstream() -> None:
    async with _mock_chain() as (_client, adapter):
        response = await adapter.get('/health')

    assert response.status_code == 200
    assert response.json() == {
        'status': 'healthy',
        'runtime_upstream': 'healthy',
    }
