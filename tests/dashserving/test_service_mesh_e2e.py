# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU Gateway -> Service Mesh -> GPU Adapter integration mock."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from twinkle.server.dashserving import create_dashserving_app
from twinkle.server.dashserving.proxy import RuntimeProxy
from twinkle.server.mesh_gateway import create_mesh_gateway_app
from twinkle.server.mesh_gateway.proxy import ServiceMeshProxy

_GPU_SERVICE_ID = 'gpu-training-service'


def _build_runtime_mock() -> FastAPI:
    app = FastAPI()

    @app.api_route('/api/v1/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def endpoint(path: str, request: Request) -> JSONResponse:
        if path == 'twinkle/healthz':
            return JSONResponse({'status': 'ok'})
        if path.endswith('/expired'):
            return JSONResponse({'detail': 'checkpoint expired'}, status_code=410)

        return JSONResponse({
            'method': request.method,
            'path': request.url.path,
            'query': dict(request.query_params),
            'headers': {
                'authorization': request.headers.get('authorization'),
                'x-request-id': request.headers.get('x-request-id'),
            },
            'body': await request.json() if request.method == 'POST' else None,
        })

    return app


def _build_service_mesh_mock(adapter_client: httpx.AsyncClient) -> FastAPI:
    """Mock the platform routing by service ID and Native HTTP conversion."""
    app = FastAPI()
    app.state.requests = []

    @app.post('/api/v2/inference')
    async def inference(request: Request) -> JSONResponse:
        mesh_request = await request.json()
        app.state.requests.append(mesh_request)
        request_id = mesh_request['header']['request_id']
        if mesh_request['header'].get('service_id') != _GPU_SERVICE_ID:
            return JSONResponse({
                'header': {
                    'request_id': request_id,
                    'status_code': 404,
                    'status_name': 'ServiceNotFound',
                    'status_message': 'Unknown service ID.',
                },
                'payload': {'output': {}},
            })

        adapter_response = await adapter_client.post(
            '/api',
            json=mesh_request['payload']['input'],
            headers={'X-DashServing-Request-Id': request_id},
        )
        status_code = int(adapter_response.headers['x-dashserving-status-code'])
        return JSONResponse({
            'header': {
                'request_id': request_id,
                'status_code': status_code,
                'status_name': adapter_response.headers['x-dashserving-status-name'],
                'status_message': adapter_response.headers['x-dashserving-status-message'],
            },
            'payload': {'output': adapter_response.json()},
        })

    return app


@asynccontextmanager
async def _mock_chain() -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    """Build Client -> CPU Gateway -> Mesh Mock -> GPU Adapter -> Twinkle Mock."""
    runtime_app = _build_runtime_mock()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime_app),
        base_url='http://runtime',
    ) as runtime_client:
        adapter_app = create_dashserving_app(
            proxy=RuntimeProxy('http://runtime', client=runtime_client))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=adapter_app),
            base_url='http://gpu-adapter',
        ) as adapter_client:
            mesh_app = _build_service_mesh_mock(adapter_client)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=mesh_app),
                base_url='http://service-mesh',
            ) as mesh_client:
                gateway_app = create_mesh_gateway_app(
                    gpu_service_id=_GPU_SERVICE_ID,
                    proxy=ServiceMeshProxy(
                        _GPU_SERVICE_ID,
                        mesh_url='http://service-mesh/api/v2/inference',
                        client=mesh_client,
                    ),
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=gateway_app),
                    base_url='http://cpu-gateway',
                ) as user_client:
                    yield user_client, mesh_app


def _client_headers() -> dict[str, str]:
    return {
        'Authorization': 'Bearer user-token',
        'X-Request-Id': 'client-request-01',
    }


@pytest.mark.asyncio
async def test_twinkle_request_crosses_cpu_mesh_and_gpu() -> None:
    async with _mock_chain() as (client, mesh_app):
        response = await client.post(
            '/api/v1/model/Qwen/Qwen3.6-27B/twinkle/forward_backward?timeout=20',
            headers=_client_headers(),
            json={'inputs': [], 'adapter_name': 'default'},
        )

    assert response.status_code == 200
    assert response.json() == {
        'method': 'POST',
        'path': '/api/v1/model/Qwen/Qwen3.6-27B/twinkle/forward_backward',
        'query': {'timeout': '20'},
        'headers': {
            'authorization': 'Bearer user-token',
            'x-request-id': 'client-request-01',
        },
        'body': {'inputs': [], 'adapter_name': 'default'},
    }
    mesh_request = mesh_app.state.requests[0]
    assert mesh_request['header'] == {
        'request_id': 'client-request-01',
        'service_id': _GPU_SERVICE_ID,
    }
    assert mesh_request['payload']['parameters'] == {}


@pytest.mark.asyncio
async def test_tinker_and_inner_error_status_are_restored_for_the_client() -> None:
    async with _mock_chain() as (client, _mesh_app):
        tinker_response = await client.post(
            '/api/v1/create_model',
            headers=_client_headers(),
            json={'base_model': 'Qwen/Qwen3.6-27B'},
        )
        expired_response = await client.delete(
            '/api/v1/twinkle/checkpoints/expired',
            headers=_client_headers(),
        )

    assert tinker_response.status_code == 200
    assert tinker_response.json()['path'] == '/api/v1/create_model'
    assert expired_response.status_code == 410
    assert expired_response.json() == {'detail': 'checkpoint expired'}


@pytest.mark.asyncio
async def test_cpu_gateway_health_is_local() -> None:
    async with _mock_chain() as (client, _mesh_app):
        response = await client.get('/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'healthy'}
