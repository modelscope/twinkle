# Copyright (c) ModelScope Contributors. All rights reserved.
"""FastAPI application implementing DashServing Native HTTP for Twinkle."""
from __future__ import annotations

import httpx
import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from typing import AsyncIterator

from twinkle.server.common.tunnel import TunnelRequest
from twinkle.utils.logger import get_logger
from .proxy import RuntimeProxy

logger = get_logger()

_DS_REQUEST_ID = 'X-DashServing-Request-Id'
_DS_ATTRIBUTES = 'X-DashServing-Attributes'
_DS_USAGE = 'X-DashServing-Usage'
_DS_STATUS_CODE = 'X-DashServing-Status-Code'
_DS_STATUS_NAME = 'X-DashServing-Status-Name'
_DS_STATUS_MESSAGE = 'X-DashServing-Status-Message'


def create_dashserving_app(
    *,
    upstream_url: str = 'http://127.0.0.1:8000',
    timeout_seconds: float = 600.0,
    proxy: RuntimeProxy | None = None,
) -> FastAPI:
    """Create the standalone DashServing adapter application.

    A caller-provided proxy is useful for tests and is owned by the caller.
    Otherwise this application creates and closes its own proxy client.
    """
    owns_proxy = proxy is None
    tunnel_proxy = proxy or RuntimeProxy(
        upstream_url=upstream_url,
        timeout_seconds=timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_proxy:
            await tunnel_proxy.close()

    app = FastAPI(
        title='Twinkle DashServing Adapter',
        description='DashServing Native HTTP adapter for the Twinkle server',
        lifespan=lifespan,
    )

    @app.get('/health')
    async def health() -> JSONResponse:
        if await tunnel_proxy.health():
            return JSONResponse({
                'status': 'healthy',
                'runtime_upstream': 'healthy',
            })
        return JSONResponse(
            {
                'status': 'unhealthy',
                'runtime_upstream': 'unhealthy',
            },
            status_code=503,
        )

    @app.post('/api')
    async def native_http(request: Request) -> JSONResponse:
        request_id = request.headers.get(_DS_REQUEST_ID) or str(uuid.uuid4())

        try:
            body = await request.json()
            tunnel_request = TunnelRequest.model_validate(body)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return _error_response(
                request_id=request_id,
                status_code=400,
                status_name='InvalidRequest',
                message='Invalid tunnel request body.',
            )

        try:
            tunnel_response = await tunnel_proxy.forward(tunnel_request)
        except httpx.TimeoutException:
            return _error_response(
                request_id=request_id,
                status_code=504,
                status_name='UpstreamTimeout',
                message='Runtime upstream timed out.',
            )
        except httpx.HTTPError:
            return _error_response(
                request_id=request_id,
                status_code=502,
                status_name='UpstreamError',
                message='Runtime upstream request failed.',
            )
        except Exception:
            logger.exception('Unhandled DashServing adapter error request_id=%s', request_id)
            return _error_response(
                request_id=request_id,
                status_code=500,
                status_name='InternalError',
                message='DashServing adapter failed.',
            )

        return JSONResponse(
            tunnel_response.model_dump(mode='json'),
            status_code=200,
            headers=_dashserving_headers(
                request_id=request_id,
                status_code=200,
                status_name='Success',
                message='Success.',
            ),
        )

    return app


def _error_response(
    *,
    request_id: str,
    status_code: int,
    status_name: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        {'error': {
            'code': status_name,
            'message': message,
        }},
        status_code=status_code,
        headers=_dashserving_headers(
            request_id=request_id,
            status_code=status_code,
            status_name=status_name,
            message=message,
        ),
    )


def _dashserving_headers(
    *,
    request_id: str,
    status_code: int,
    status_name: str,
    message: str,
) -> dict[str, str]:
    return {
        _DS_REQUEST_ID: request_id,
        _DS_ATTRIBUTES: '{}',
        _DS_USAGE: '{}',
        _DS_STATUS_CODE: str(status_code),
        _DS_STATUS_NAME: status_name,
        _DS_STATUS_MESSAGE: message,
    }
