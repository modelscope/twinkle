# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Proxy utilities for forwarding requests to internal model/sampler services.

Moved from tinker/proxy.py. Updated proxy_to_model and proxy_to_sampler
to prepend the 'tinker/' prefix to endpoints so they route to /tinker/* paths
on the unified model/sampler deployments.
"""
from __future__ import annotations

import httpx
from fastapi import Request, Response
from typing import Any

from twinkle.utils.logger import get_logger

logger = get_logger()


class ServiceProxy:
    """HTTP proxy for routing requests to internal model and sampler services.

    Handles:
    1. URL construction using localhost to avoid external routing loops
    2. Header forwarding with appropriate cleanup
    3. Debug logging for troubleshooting
    4. Error handling and response forwarding

    Tinker endpoints are routed to /tinker/<endpoint> on the unified deployments.
    """

    def __init__(
        self,
        http_options: dict[str, Any] | None = None,
        route_prefix: str = '/api/v1',
    ):
        self.http_options = http_options or {}
        self.route_prefix = route_prefix
        # Disable proxy env vars to avoid external routing
        self.client = httpx.AsyncClient(timeout=None, trust_env=False)

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient to release connections."""
        await self.client.aclose()

    def _build_target_url(self, service_type: str, base_model: str, endpoint: str) -> str:
        """Build the target URL for internal service routing.

        Args:
            service_type: Either 'model' or 'sampler'
            base_model: The base model name for routing
            endpoint: The target endpoint name (already includes tinker/ or twinkle/ prefix)

        Returns:
            Complete target URL for the internal service
        """
        prefix = self.route_prefix.rstrip('/') if self.route_prefix else ''
        host = self.http_options.get('host', 'localhost')
        port = self.http_options.get('port', 8000)

        if host == '0.0.0.0':
            host = 'localhost'

        base_url = f'http://{host}:{port}'
        return f'{base_url}{prefix}/{service_type}/{base_model}/{endpoint}'

    def _prepare_headers(self, request_headers) -> dict[str, str]:
        """Prepare headers for proxying by removing problematic headers."""
        logger.debug('prepare_headers request_headers=%s', request_headers)
        headers = dict(request_headers)
        headers.pop('host', None)
        headers.pop('content-length', None)
        request_id = request_headers.get('X-Ray-Serve-Request-Id')
        if request_id is not None and not request_headers.get('Serve-Multiplexed-Model-Id'):
            headers['Serve-Multiplexed-Model-Id'] = request_id
        return headers

    async def proxy_request(
        self,
        request: Request,
        endpoint: str,
        base_model: str,
        service_type: str,
    ) -> Response:
        """Generic proxy method to forward requests to model or sampler services.

        Args:
            request: The incoming FastAPI request
            endpoint: The target endpoint path (e.g., 'tinker/create_model')
            base_model: The base model name for routing
            service_type: Either 'model' or 'sampler'

        Returns:
            Proxied response from the target service
        """
        body_bytes = await request.body()
        target_url = self._build_target_url(service_type, base_model, endpoint)
        headers = self._prepare_headers(request.headers)

        try:
            logger.debug(
                'proxy_request service=%s endpoint=%s target_url=%s request_id=%s',
                service_type,
                endpoint,
                target_url,
                headers.get('Serve-Multiplexed-Model-Id'),
            )

            response = await self.client.request(
                method=request.method,
                url=target_url,
                content=body_bytes,
                headers=headers,
                params=request.query_params,
            )

            logger.debug(
                'proxy_response status=%s body_preview=%s',
                response.status_code,
                response.text[:200],
            )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get('content-type'),
            )
        except Exception as e:
            logger.error('Proxy error: %s', str(e), exc_info=True)
            return Response(content=f'Proxy Error: {str(e)}', status_code=502)

    async def proxy_to_model(self, request: Request, endpoint: str, base_model: str) -> Response:
        """Proxy request to model's tinker endpoint (/tinker/<endpoint>).

        Args:
            request: The incoming FastAPI request
            endpoint: The tinker endpoint name (e.g., 'create_model', 'forward')
            base_model: The base model name for routing
        """
        return await self.proxy_request(request, f'tinker/{endpoint}', base_model, 'model')

    async def proxy_to_sampler(self, request: Request, endpoint: str, base_model: str) -> Response:
        """Proxy request to sampler's tinker endpoint (/tinker/<endpoint>).

        Args:
            request: The incoming FastAPI request
            endpoint: The tinker endpoint name (e.g., 'asample')
            base_model: The base model name for routing
        """
        return await self.proxy_request(request, f'tinker/{endpoint}', base_model, 'sampler')
