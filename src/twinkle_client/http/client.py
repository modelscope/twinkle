# Copyright (c) ModelScope Contributors. All rights reserved.
"""Instance-owned HTTP transport."""
from __future__ import annotations

import requests
from collections.abc import Mapping
from requests.adapters import HTTPAdapter
from typing import Any
from urllib3.util.retry import Retry

from twinkle.protocol.headers import build_routing_headers
from twinkle.protocol.types.errors import ErrorCategory, ErrorPayload
from twinkle_client._request_builder import to_wire_value
from twinkle_client.exceptions import TwinkleClientValidationError, TwinkleHTTPError
from .context import ClientContext, capture_transport

# Must be greater than the server long-poll window and below common gateway idle limits.
_HTTP_TIMEOUT = 90
DEFAULT_TIMEOUT = object()


def _handle_response(response: requests.Response) -> requests.Response:
    if response.status_code == 410:
        try:
            detail = response.json().get('detail', 'Iterator exhausted')
        except Exception:
            detail = response.text or 'Iterator exhausted'
        raise StopIteration(detail)

    if response.ok:
        return response

    try:
        body = response.json()
    except Exception:
        body = None
    payload: ErrorPayload | None = None
    if isinstance(body, dict):
        try:
            payload = ErrorPayload.model_validate(body)
        except Exception:
            payload = None

    if payload is not None:
        summary = payload.error or response.text
        category = payload.category.value
        error_code = payload.error_code
        request_id = payload.request_id
        details = payload.details
        traceback_text = payload.traceback
    elif isinstance(body, dict):
        summary = body.get('detail') or response.text
        category = ErrorCategory.Unknown.value
        error_code = request_id = details = traceback_text = None
    else:
        summary = response.text
        category = ErrorCategory.Unknown.value
        error_code = request_id = details = traceback_text = None

    message = f'{response.status_code} Error for url: {response.url}\nServer detail:\n{summary}'
    raise TwinkleHTTPError(
        message,
        response=response,
        status_code=response.status_code,
        error_code=error_code,
        category=category,
        request_id=request_id,
        details=details,
        traceback=traceback_text,
    )


class ClientTransport:
    """The sole request-time owner of URL, identity, headers, and HTTP resources.

    ``post`` recursively converts arbitrary JSON-like values through
    :func:`to_wire_value`; ``post_model`` serializes a validated Pydantic model
    directly and excludes ``None`` fields. The two entry points intentionally
    remain distinct.

    The adapter retries only idempotent GET/DELETE requests. POST is never
    transparently replayed because control-plane calls such as ``create_session``
    do not carry a deduplication key; read-only future retrieval handles retries
    explicitly in the future layer.
    """

    def __init__(
        self,
        context: ClientContext,
        *,
        session: requests.Session | None = None,
        timeout: float = _HTTP_TIMEOUT,
        pool_maxsize: int = 32,
    ) -> None:
        if pool_maxsize < 2:
            raise ValueError('pool_maxsize must be at least 2')
        self._context = context
        self._session = session or requests.Session()
        if session is None:
            retry = Retry(
                total=3,
                connect=3,
                read=0,
                status=3,
                backoff_factor=0.25,
                status_forcelist=(408, 429, 500, 502, 503, 504),
                allowed_methods=frozenset({'GET', 'DELETE'}),
                respect_retry_after_header=True,
                raise_on_status=False,
            )
            adapter = HTTPAdapter(
                max_retries=retry,
                pool_connections=pool_maxsize,
                pool_maxsize=pool_maxsize,
                pool_block=True,
            )
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
        self._timeout = timeout
        self._closed = False
        self._published = False
        self._capabilities: object | None = None

    @property
    def context(self) -> ClientContext:
        return self._context

    @property
    def closed(self) -> bool:
        return self._closed

    def bind_context(self, context: ClientContext) -> None:
        """Replace provisional identity before the transport is published to wrappers."""
        self._ensure_open()
        if self._published:
            raise RuntimeError('Cannot rebind a published ClientTransport')
        self._context = context

    def _mark_published(self) -> None:
        self._published = True

    @property
    def cached_capabilities(self) -> object | None:
        return self._capabilities

    @cached_capabilities.setter
    def cached_capabilities(self, value: object) -> None:
        self._capabilities = value

    def url(self, path_or_url: str = '') -> str:
        if path_or_url.startswith(('http://', 'https://')):
            return path_or_url
        if not path_or_url:
            return self._context.base_url
        return f'{self._context.base_url}/{path_or_url.lstrip("/")}'

    def _headers(self, additional_headers: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = build_routing_headers(self._context.routing_id, f'Bearer {self._context.api_key}')
        if self._context.session_id:
            headers['X-Twinkle-Session-Id'] = self._context.session_id
        if additional_headers:
            headers.update(additional_headers)
        return headers

    def _request_timeout(self, timeout: object) -> float | None:
        return self._timeout if timeout is DEFAULT_TIMEOUT else timeout  # type: ignore[return-value]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError('ClientTransport is closed')

    def get(self,
            path_or_url: str = '',
            *,
            params: Mapping[str, Any] | None = None,
            headers: Mapping[str, str] | None = None,
            timeout: float | None | object = DEFAULT_TIMEOUT) -> requests.Response:
        self._ensure_open()
        response = self._session.get(
            self.url(path_or_url),
            headers=self._headers(headers),
            params=to_wire_value(params or {}),
            timeout=self._request_timeout(timeout),
        )
        return _handle_response(response)

    def post(self,
             path_or_url: str = '',
             *,
             json_data: Mapping[str, Any] | None = None,
             data: Any = None,
             headers: Mapping[str, str] | None = None,
             timeout: float | None | object = DEFAULT_TIMEOUT) -> requests.Response:
        self._ensure_open()
        if isinstance(data, (bytes, bytearray, memoryview)):
            raise TwinkleClientValidationError('Binary request bodies are not supported by this transport')
        response = self._session.post(
            self.url(path_or_url),
            headers=self._headers(headers),
            json=to_wire_value(json_data or {}),
            data=data,
            timeout=self._request_timeout(timeout),
        )
        return _handle_response(response)

    def post_model(self,
                   path_or_url: str,
                   body: Any,
                   *,
                   headers: Mapping[str, str] | None = None,
                   timeout: float | None | object = DEFAULT_TIMEOUT) -> requests.Response:
        from twinkle_client._request_builder import request_json
        self._ensure_open()
        request_headers = {'content-type': 'application/json', **dict(headers or {})}
        response = self._session.post(
            self.url(path_or_url),
            headers=self._headers(request_headers),
            data=request_json(body),
            timeout=self._request_timeout(timeout),
        )
        return _handle_response(response)

    def delete(self,
               path_or_url: str = '',
               *,
               params: Mapping[str, Any] | None = None,
               headers: Mapping[str, str] | None = None,
               timeout: float | None | object = DEFAULT_TIMEOUT) -> requests.Response:
        self._ensure_open()
        response = self._session.delete(
            self.url(path_or_url),
            headers=self._headers(headers),
            params=to_wire_value(params or {}),
            timeout=self._request_timeout(timeout),
        )
        return _handle_response(response)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.close()
