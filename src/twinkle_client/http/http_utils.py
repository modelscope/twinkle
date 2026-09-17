import requests
from typing import Any, Dict, Optional

from twinkle_client.exceptions import TwinkleHTTPError
from .headers import build_routing_headers
from .utils import get_api_key, get_base_url, get_request_id, get_session_id

# Single shared HTTP timeout for every client request (was three separate 600s).
# Must be <= 120 and strictly greater than the server Long_Poll_Window (default 30),
# so a retrieve that waits a full window still completes within the timeout and, being
# < a typical 60s gateway idle limit, survives the gateway.
_HTTP_TIMEOUT = 90


def _build_headers(additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build HTTP headers with request ID and authorization.

    Args:
        additional_headers: Additional headers to include

    Returns:
        Dictionary of headers
    """
    headers = build_routing_headers(get_request_id(), 'Bearer ' + get_api_key())

    if session_id := get_session_id():
        headers['X-Twinkle-Session-Id'] = session_id

    if additional_headers:
        headers.update(additional_headers)

    return headers


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize parameters, handling special objects like processors.

    Args:
        params: Parameters to serialize

    Returns:
        Serialized parameters dictionary
    """
    serialized = {}
    for key, value in params.items():
        if hasattr(value, 'processor_id'):
            serialized[key] = value.processor_id
        elif hasattr(value, '__dict__'):
            from twinkle_client.common.serialize import serialize_object
            serialized[key] = serialize_object(value)
        else:
            serialized[key] = value
    return serialized


def _handle_response(response: requests.Response) -> requests.Response:
    """Handle common response processing.

    Raises:
        StopIteration: When server returns HTTP 410 (iterator exhausted).
        TwinkleHTTPError: When the server returns a 4xx/5xx (other than 410). It
            inherits ``requests.HTTPError`` so existing ``except`` clauses keep
            working, and carries the server's top-level ``error_code`` / ``category``
            / ``request_id`` when present. When those fields are absent (FastAPI's
            built-in 404/405, or a gateway passthrough), it falls back to ``detail``
            with ``category='Unknown'``.
    """
    # Convert HTTP 410 Gone to StopIteration (an iterator has been exhausted).
    if response.status_code == 410:
        raise StopIteration(response.json().get('detail', 'Iterator exhausted'))

    if not response.ok:
        try:
            body = response.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            category = body.get('category', 'Unknown')
            error_code = body.get('error_code')
            request_id = body.get('request_id')
            summary = body.get('error') or body.get('detail') or response.text
        else:
            category, error_code, request_id, summary = 'Unknown', None, None, response.text
        http_error_msg = (
            f'{response.status_code} Error for url: {response.url}\n'
            f'Server detail:\n{summary}'
        )
        raise TwinkleHTTPError(
            http_error_msg,
            response=response,
            status_code=response.status_code,
            error_code=error_code,
            category=category,
            request_id=request_id,
        )

    return response


def http_get(
    url: Optional[str] = None,
    params: Optional[Dict[str, Any]] = {},
    additional_headers: Optional[Dict[str, str]] = {},
    timeout: int = _HTTP_TIMEOUT,
) -> requests.Response:
    """
    Send HTTP GET request with required headers.

    Args:
        url: The target URL
        params: Query parameters
        additional_headers: Additional headers to include
        timeout: Request timeout in seconds

    Returns:
        requests.Response object
    """
    url = url or get_base_url()
    headers = _build_headers(additional_headers)
    serialized_params = _serialize_params(params)

    response = requests.get(
        url,
        headers=headers,
        params=serialized_params,
        timeout=timeout,
    )

    return _handle_response(response)


def http_post(
    url: Optional[str] = None,
    json_data: Optional[Dict[str, Any]] = {},
    data: Optional[Any] = {},
    additional_headers: Optional[Dict[str, str]] = {},
    timeout: Optional[int] = _HTTP_TIMEOUT,
) -> requests.Response:
    """
    Send HTTP POST request with required headers.

    Args:
        url: The target URL
        json_data: JSON data to send in request body
        data: Form data or raw data to send in request body
        additional_headers: Additional headers to include
        timeout: Request timeout in seconds; None disables the timeout.

    Returns:
        requests.Response object

    Raises:
        StopIteration: When server returns HTTP 410 (iterator exhausted)
    """
    url = url or get_base_url()
    headers = _build_headers(additional_headers)
    serialized_json = _serialize_params(json_data)

    response = requests.post(
        url,
        headers=headers,
        json=serialized_json,
        data=data,
        timeout=timeout,
    )

    return _handle_response(response)


def http_delete(
    url: Optional[str] = None,
    params: Optional[Dict[str, Any]] = {},
    additional_headers: Optional[Dict[str, str]] = {},
    timeout: int = _HTTP_TIMEOUT,
) -> requests.Response:
    """
    Send HTTP DELETE request with required headers.

    Args:
        url: The target URL
        params: Query parameters
        additional_headers: Additional headers to include
        timeout: Request timeout in seconds

    Returns:
        requests.Response object
    """
    url = url or get_base_url()
    headers = _build_headers(additional_headers)
    serialized_params = _serialize_params(params)

    response = requests.delete(
        url,
        headers=headers,
        params=serialized_params,
        timeout=timeout,
    )

    return _handle_response(response)
