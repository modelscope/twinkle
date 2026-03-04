# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Proxy utilities for forwarding requests to internal services.

This module provides HTTP proxy functionality to route requests from the Tinker server
to appropriate model or sampler services based on base_model routing.
"""

from __future__ import annotations

import httpx
import os
from fastapi import Request, Response
from typing import Any

from twinkle.utils.logger import get_logger

logger = get_logger()


class ServiceProxy:
    """HTTP proxy for routing requests to internal model and sampler services.

    This proxy handles:
    1. URL construction using localhost to avoid external routing loops
    2. Header forwarding with appropriate cleanup
    3. Debug logging for troubleshooting
    4. Error handling and response forwarding
    """

    def __init__(
        self,
        http_options: dict[str, Any] | None = None,
        route_prefix: str = '/api/v1',
    ):
        """Initialize the service proxy.

        Args:
            http_options: HTTP server options (host, port) for internal routing
            route_prefix: URL prefix for routing (default: '/api/v1')
        """
        self.http_options = http_options or {}
        self.route_prefix = route_prefix
        # Disable proxy for internal requests to avoid routing through external proxies
        self.client = httpx.AsyncClient(timeout=None, trust_env=False)

    def _build_target_url(self, service_type: str, base_model: str, endpoint: str) -> str:
        """Build the target URL for internal service routing.

        Constructs URLs using localhost to avoid extra external hops.
        When requests come from www.modelscope.com/twinkle, we proxy to
        localhost:port directly instead of back to modelscope.com.

        Args:
            service_type: Either 'model' or 'sampler'
            base_model: The base model name for routing
            endpoint: The target endpoint name

        Returns:
            Complete target URL for the internal service
        """
        prefix = self.route_prefix.rstrip('/') if self.route_prefix else ''
        host = self.http_options.get('host', 'localhost')
        port = self.http_options.get('port', 8000)

        # Use localhost for internal routing
        if host == '0.0.0.0':
            host = 'localhost'

        base_url = f'http://{host}:{port}'
        return f'{base_url}{prefix}/{service_type}/{base_model}/{endpoint}'

    def _prepare_headers(self, request_headers) -> dict[str, str]:
        """Prepare headers for proxying by removing problematic headers.

        Args:
            request_headers: Original request headers (case-insensitive from FastAPI)

        Returns:
            Cleaned headers safe for proxying
        """
        logger.debug('prepare_headers request_headers=%s', request_headers)
        # Convert to dict while preserving case-insensitive lookups for special headers
        headers = dict(request_headers)
        # Remove headers that should not be forwarded
        headers.pop('host', None)
        headers.pop('content-length', None)
        # Add serve_multiplexed_model_id for sticky sessions if present
        # Use case-insensitive lookup from original request_headers
        request_id = request_headers.get('X-Ray-Serve-Request-Id')
        if request_id is not None:
            headers['serve_multiplexed_model_id'] = request_id
        return headers

    async def proxy_request(
        self,
        request: Request,
        endpoint: str,
        base_model: str,
        service_type: str,
    ) -> Response:
        """Generic proxy method to forward requests to model or sampler services.

        This method consolidates the common proxy logic for both model and sampler endpoints.

        Args:
            request: The incoming FastAPI request
            endpoint: The target endpoint name (e.g., 'create_model', 'asample')
            base_model: The base model name for routing
            service_type: Either 'model' or 'sampler' to determine the target service

        Returns:
            Proxied response from the target service
        """
        body_bytes = await request.body()
        target_url = self._build_target_url(service_type, base_model, endpoint)
        # Pass original request.headers (case-insensitive) instead of dict conversion
        headers = self._prepare_headers(request.headers)

        try:
            # Debug logging for troubleshooting proxy issues
            logger.debug(
                'proxy_request service=%s endpoint=%s target_url=%s request_id=%s',
                service_type,
                endpoint,
                target_url,
                headers.get('serve_multiplexed_model_id'),
            )

            # Forward the request to the target service
            response = await self.client.request(
                method=request.method,
                url=target_url,
                content=body_bytes,
                headers=headers,
                params=request.query_params,
            )

            # Debug logging for response
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
        """Proxy request to model endpoint.

        Routes the request to the appropriate model deployment based on base_model.

        Args:
            request: The incoming FastAPI request
            endpoint: The target endpoint name (e.g., 'create_model', 'forward')
            base_model: The base model name for routing

        Returns:
            Proxied response from the model service
        """
        return await self.proxy_request(request, endpoint, base_model, 'model')

    async def proxy_to_sampler(self, request: Request, endpoint: str, base_model: str) -> Response:
        """Proxy request to sampler endpoint.

        Routes the request to the appropriate sampler deployment based on base_model.

        Args:
            request: The incoming FastAPI request
            endpoint: The target endpoint name (e.g., 'asample')
            base_model: The base model name for routing

        Returns:
            Proxied response from the sampler service
        """
        return await self.proxy_request(request, endpoint, base_model, 'sampler')
