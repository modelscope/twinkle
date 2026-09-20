# Copyright (c) ModelScope Contributors. All rights reserved.
"""Instance-owned HTTP transport and legacy module-level compatibility facade."""
from __future__ import annotations

import requests
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from twinkle_client.exceptions import TwinkleClientValidationError, TwinkleHTTPError
from twinkle_client.types.errors import ErrorCategory, ErrorPayload
from .context import ClientContext, capture_transport
from .headers import build_routing_headers

# Must be greater than the server long-poll window and below common gateway idle limits.
_HTTP_TIMEOUT = 90
_UNSET = object()
_JSON_PRIMITIVES = (str, int, float, bool, type(None))


def _serialize_value(value: Any) -> Any:
    if isinstance(value, _JSON_PRIMITIVES):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raise TwinkleClientValidationError('Binary values are not supported by the JSON transport')
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_value(asdict(value))
    # Single source of truth for leaf/domain objects (pydantic models, remote
    # component handles, DatasetMeta / LoraConfig, numpy / torch): reuse the
    # request builder's converter so a value serializes identically whether it
    # goes out via ``post(json_data=...)`` or via ``post_model(body=...)``.
    from twinkle_client._request_builder import to_wire_value
    return to_wire_value(value)


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
    """The sole request-time owner of URL, identity, headers, and HTTP resources."""

    def __init__(
        self,
        context: ClientContext,
        *,
        session: requests.Session | None = None,
        timeout: float = _HTTP_TIMEOUT,
    ) -> None:
        self._context = context
        self._session = session or requests.Session()
        self._timeout = timeout
        self._closed = False
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
        self._context = context

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
        return self._timeout if timeout is _UNSET else timeout  # type: ignore[return-value]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError('ClientTransport is closed')

    def get(
        self,
        path_or_url: str = '',
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None | object = _UNSET,
    ) -> requests.Response:
        self._ensure_open()
        response = self._session.get(
            self.url(path_or_url),
            headers=self._headers(headers),
            params=_serialize_value(params or {}),
            timeout=self._request_timeout(timeout),
        )
        return _handle_response(response)

    def post(
        self,
        path_or_url: str = '',
        *,
        json_data: Mapping[str, Any] | None = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None | object = _UNSET,
    ) -> requests.Response:
        self._ensure_open()
        if isinstance(data, (bytes, bytearray, memoryview)):
            raise TwinkleClientValidationError('Binary request bodies are not supported by this transport')
        response = self._session.post(
            self.url(path_or_url),
            headers=self._headers(headers),
            json=_serialize_value(json_data or {}),
            data=data,
            timeout=self._request_timeout(timeout),
        )
        return _handle_response(response)

    def post_model(
        self,
        path_or_url: str,
        body: Any,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None | object = _UNSET,
    ) -> requests.Response:
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

    def delete(
        self,
        path_or_url: str = '',
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None | object = _UNSET,
    ) -> requests.Response:
        self._ensure_open()
        response = self._session.delete(
            self.url(path_or_url),
            headers=self._headers(headers),
            params=_serialize_value(params or {}),
            timeout=self._request_timeout(timeout),
        )
        return _handle_response(response)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.close()


# Compatibility facade. Core wrappers always pass their captured transport explicitly;
# only legacy external callers may omit it and resolve the current default here.
def http_get(
    url: str | None = None,
    params: Mapping[str, Any] | None = None,
    additional_headers: Mapping[str, str] | None = None,
    timeout: float | None = _HTTP_TIMEOUT,
    *,
    transport: ClientTransport | None = None,
) -> requests.Response:
    return capture_transport(transport).get(url or '', params=params, headers=additional_headers, timeout=timeout)


def http_post(
    url: str | None = None,
    json_data: Mapping[str, Any] | None = None,
    data: Any = None,
    additional_headers: Mapping[str, str] | None = None,
    timeout: float | None = _HTTP_TIMEOUT,
    *,
    transport: ClientTransport | None = None,
) -> requests.Response:
    return capture_transport(transport).post(
        url or '', json_data=json_data, data=data, headers=additional_headers, timeout=timeout)


def http_post_model(
    url: str,
    body: Any,
    additional_headers: Mapping[str, str] | None = None,
    timeout: float | None = _HTTP_TIMEOUT,
    *,
    transport: ClientTransport | None = None,
) -> requests.Response:
    return capture_transport(transport).post_model(url, body, headers=additional_headers, timeout=timeout)


def http_delete(
    url: str | None = None,
    params: Mapping[str, Any] | None = None,
    additional_headers: Mapping[str, str] | None = None,
    timeout: float | None = _HTTP_TIMEOUT,
    *,
    transport: ClientTransport | None = None,
) -> requests.Response:
    return capture_transport(transport).delete(url or '', params=params, headers=additional_headers, timeout=timeout)
