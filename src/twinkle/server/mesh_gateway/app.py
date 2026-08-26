# Copyright (c) ModelScope Contributors. All rights reserved.
"""Public Twinkle HTTP gateway backed by a configured Service Mesh target."""
from __future__ import annotations

import httpx
import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import AsyncIterator

from twinkle.server.common.tunnel import TunnelRequest
from .proxy import ServiceMeshError, ServiceMeshProxy


def create_mesh_gateway_app(
    *,
    gpu_service_id: str,
    mesh_url: str = 'http://127.0.0.1:8880/api/v2/inference',
    timeout_seconds: float = 620.0,
    proxy: ServiceMeshProxy | None = None,
) -> FastAPI:
    """Create the CPU-side public gateway for one GPU training service."""
    owns_proxy = proxy is None
    mesh_proxy = proxy or ServiceMeshProxy(
        gpu_service_id,
        mesh_url=mesh_url,
        timeout_seconds=timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_proxy:
            await mesh_proxy.close()

    app = FastAPI(
        title='Twinkle Service Mesh Gateway',
        description='Public Twinkle and Tinker gateway for one GPU service',
        lifespan=lifespan,
    )

    @app.get('/health')
    async def health() -> JSONResponse:
        return JSONResponse({'status': 'healthy'})

    @app.api_route('/api/v1/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def forward(path: str, request: Request) -> JSONResponse:
        request_id = request.headers.get('x-request-id') or str(uuid.uuid4())
        try:
            body = await _request_body(request)
            tunnel_request = TunnelRequest(
                method=request.method,
                path=f'/api/v1/{path}',
                query=dict(request.query_params),
                headers=dict(request.headers),
                body=body,
            )
        except (json.JSONDecodeError, ValueError):
            return JSONResponse({'detail': 'Request body must be JSON.'}, status_code=400)

        try:
            tunnel_response = await mesh_proxy.forward(request_id, tunnel_request)
        except httpx.TimeoutException:
            return JSONResponse({'detail': 'GPU service timed out.'}, status_code=504)
        except (httpx.HTTPError, ServiceMeshError):
            return JSONResponse({'detail': 'GPU service is unavailable.'}, status_code=502)

        return JSONResponse(
            tunnel_response.body,
            status_code=tunnel_response.status_code,
            headers=tunnel_response.headers,
        )

    return app


async def _request_body(request: Request):
    if not await request.body():
        return None
    return await request.json()
