# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared HTTP forwarding between services; independent of gateway and training controller."""

from __future__ import annotations
from typing import Any, TypedDict

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from twinkle.protocol.headers import H_MULTIPLEX, H_MULTIPLEX_LEGACY, H_REQUEST_ID, H_REQUEST_ID_LEGACY
from twinkle.protocol.types.errors import ErrorCategory, ErrorPayload
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.telemetry.tracing import inject_context
from twinkle.utils.logger import get_logger

logger = get_logger()


class ForwardedRequest(TypedDict):
    """HTTP request fields carried intact across an internal Ray RPC."""

    endpoint: str
    body: bytes
    method: str
    headers: list[tuple[bytes, bytes]]
    query_string: bytes


class ForwardedReply(TypedDict):
    status: int
    headers: list[tuple[bytes, bytes]]
    body: bytes


class ServiceProxy:
    def __init__(self, http_options: dict[str, Any] | None = None, route_prefix: str = "/api/v1"):
        self.http_options = http_options or {}
        self.route_prefix = route_prefix
        queue = TaskQueueConfig()
        budget = queue.queue_timeout + queue.effective_execution_timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(budget, connect=10, pool=10),
            trust_env=False,
            verify=not self.http_options.get("ssl_certfile"),
        )

    async def close(self) -> None:
        await self.client.aclose()

    def _build_target_url(self, service_type: str, base_model: str, endpoint: str) -> str:
        host = self.http_options.get("host", "localhost")
        port = self.http_options.get("port", 8000)
        if host == "0.0.0.0":
            host = "localhost"
        scheme = "https" if self.http_options.get("ssl_certfile") else "http"
        return f"{scheme}://{host}:{port}{self.route_prefix.rstrip('/')}/{service_type}/{base_model}/{endpoint}"

    def _prepare_headers(self, request_headers) -> dict[str, str]:
        headers = dict(request_headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        request_id = request_headers.get(H_REQUEST_ID) or request_headers.get(H_REQUEST_ID_LEGACY)
        if request_id is not None and not request_headers.get(H_MULTIPLEX_LEGACY):
            headers[H_MULTIPLEX_LEGACY] = request_id
        if request_id is not None:
            headers.setdefault(H_REQUEST_ID, request_id)
            headers.setdefault(H_MULTIPLEX, request_id)
        return headers

    async def proxy_request(
        self,
        request: Request,
        endpoint: str,
        base_model: str,
        service_type: str,
        *,
        body_override: bytes | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Response:
        body_bytes = body_override if body_override is not None else await request.body()
        target_url = self._build_target_url(service_type, base_model, endpoint)
        headers = self._prepare_headers(request.headers)
        if extra_headers:
            headers.update(extra_headers)
        inject_context(headers)
        try:
            response = await self.client.request(
                method=request.method, url=target_url, content=body_bytes, headers=headers, params=request.query_params
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type"),
            )
        except Exception as exc:
            logger.error("Proxy error: %s", str(exc), exc_info=True)
            request_id = request.headers.get(H_REQUEST_ID) or request.headers.get(H_REQUEST_ID_LEGACY) or ""
            payload = ErrorPayload(
                error=f"Proxy Error: {exc}", category=ErrorCategory.Server, error_code=502, request_id=request_id
            )
            return JSONResponse(status_code=502, content=payload.model_dump(mode="json", exclude_none=True))

    async def proxy_request_stream(
        self,
        request: Request,
        endpoint: str,
        base_model: str,
        service_type: str,
        *,
        body_override: bytes | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        body_bytes = body_override if body_override is not None else await request.body()
        target_url = self._build_target_url(service_type, base_model, endpoint)
        headers = self._prepare_headers(request.headers)
        if extra_headers:
            headers.update(extra_headers)
        inject_context(headers)
        async with self.client.stream(method="POST", url=target_url, content=body_bytes, headers=headers) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise httpx.HTTPStatusError(
                    f"Upstream returned {response.status_code}",
                    request=response.request,
                    response=httpx.Response(status_code=response.status_code, content=body),
                )
            async for line in response.aiter_lines():
                if line:
                    yield line

    async def proxy_to_model(self, request: Request, endpoint: str, base_model: str) -> Response:
        return await self.proxy_request(request, f"tinker/{endpoint}", base_model, "model")

    async def proxy_to_sampler(self, request: Request, endpoint: str, base_model: str) -> Response:
        return await self.proxy_request(request, f"tinker/{endpoint}", base_model, "sampler")
