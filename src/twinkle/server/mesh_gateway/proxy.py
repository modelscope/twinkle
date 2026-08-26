# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for the platform-provided Service Mesh endpoint."""
from __future__ import annotations

import httpx

from twinkle.server.common.tunnel import TunnelRequest, TunnelResponse


class ServiceMeshError(Exception):
    """The mesh returned a response that cannot be restored for the client."""


class ServiceMeshProxy:
    """Send Twinkle requests to one configured GPU service through Service Mesh."""

    def __init__(
        self,
        service_id: str,
        *,
        mesh_url: str = 'http://127.0.0.1:8880/api/v2/inference',
        timeout_seconds: float = 620.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._service_id = service_id
        self._mesh_url = mesh_url
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def forward(self, request_id: str, tunnel_request: TunnelRequest) -> TunnelResponse:
        response = await self._client.post(
            self._mesh_url,
            json={
                'header': {
                    'request_id': request_id,
                    'service_id': self._service_id,
                },
                'payload': {
                    'input': tunnel_request.model_dump(mode='json'),
                    'parameters': {},
                },
            },
        )
        response.raise_for_status()

        try:
            payload = response.json()
            header = payload['header']
            if header.get('status_code') != 200:
                raise ServiceMeshError(header.get('status_message', 'Service Mesh request failed.'))
            return TunnelResponse.model_validate(payload['payload']['output'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ServiceMeshError('Invalid Service Mesh response.') from exc
