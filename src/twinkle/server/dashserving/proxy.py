# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fixed-upstream proxy used by the DashServing adapter."""
from __future__ import annotations

import httpx
from typing import Any

from .schemas import TunnelRequest, TunnelResponse

_HOP_BY_HOP_HEADERS = {
    'connection',
    'content-length',
    'host',
    'transfer-encoding',
}


class RuntimeProxy:
    """Proxy tunnel requests to one configured Twinkle server.

    The target origin is constructor configuration, never request data. This is
    the security boundary that prevents the adapter from becoming an open proxy.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        timeout_seconds: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._upstream_url = upstream_url.rstrip('/')
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get(
                f'{self._upstream_url}/api/v1/twinkle/healthz',
                timeout=5.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def forward(self, tunnel_request: TunnelRequest) -> TunnelResponse:
        headers = self._build_headers(tunnel_request)
        request_kwargs: dict[str, Any] = {
            'method': tunnel_request.method,
            'url': f'{self._upstream_url}{tunnel_request.path}',
            'params': tunnel_request.query,
            'headers': headers,
        }
        if tunnel_request.body is not None:
            request_kwargs['json'] = tunnel_request.body

        response = await self._client.request(**request_kwargs)
        if not response.content:
            response_body = None
        else:
            try:
                response_body = response.json()
            except ValueError:
                response_body = response.text

        response_headers = {
            'content-type': response.headers.get('content-type', 'application/json'),
        }
        replica_id = response.headers.get('x-twinkle-replica-id')
        if replica_id:
            response_headers['x-twinkle-replica-id'] = replica_id

        return TunnelResponse(
            status_code=response.status_code,
            headers=response_headers,
            body=response_body,
        )

    @staticmethod
    def _build_headers(tunnel_request: TunnelRequest) -> dict[str, str]:
        return {
            name: value
            for name, value in tunnel_request.headers.items() if name.lower() not in _HOP_BY_HOP_HEADERS
        }
