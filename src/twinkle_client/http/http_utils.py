import requests
from numbers import Number
from typing import Any, Callable, Dict, Mapping, Optional

from .utils import get_api_key, get_base_url, get_request_id


def _build_headers(additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build HTTP headers with request ID and authorization.

    Args:
        additional_headers: Additional headers to include

    Returns:
        Dictionary of headers
    """
    headers = {
        'serve_multiplexed_model_id': get_request_id(),
        'Authorization': 'Bearer ' + get_api_key(),
        'Twinkle-Authorization': 'Bearer ' + get_api_key(),  # For server compatibility
    }

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
            from twinkle.server.twinkle.common.serialize import serialize_object
            serialized[key] = serialize_object(value)
        else:
            serialized[key] = value
    return serialized


def _handle_response(response: requests.Response) -> requests.Response:
    """
    Handle common response processing.

    Args:
        response: Response object

    Returns:
        Response object

    Raises:
        StopIteration: When server returns HTTP 410 (iterator exhausted)
    """
    # Convert HTTP 410 Gone to StopIteration
    # This indicates an iterator has been exhausted
    if response.status_code == 410:
        raise StopIteration(response.json().get('detail', 'Iterator exhausted'))

    return response


def http_get(
    url: Optional[str] = None,
    params: Optional[Dict[str, Any]] = {},
    additional_headers: Optional[Dict[str, str]] = {},
    timeout: int = 300,
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
    timeout: int = 300,
) -> requests.Response:
    """
    Send HTTP POST request with required headers.

    Args:
        url: The target URL
        json_data: JSON data to send in request body
        data: Form data or raw data to send in request body
        additional_headers: Additional headers to include
        timeout: Request timeout in seconds

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
    timeout: int = 300,
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
