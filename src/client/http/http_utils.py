import requests
from typing import Optional, Dict, Any


def http_get(
    url: str,
    request_id: str,
    authorization: str,
    params: Optional[Dict[str, Any]] = None,
    additional_headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> requests.Response:
    """
    Send HTTP GET request with required headers.

    Args:
        url: The target URL
        request_id: Request ID for X-Ray-Serve-Request-Id header
        authorization: Authorization token for Authorization header
        params: Query parameters
        additional_headers: Additional headers to include
        timeout: Request timeout in seconds

    Returns:
        requests.Response object
    """
    headers = {
        "X-Ray-Serve-Request-Id": request_id,
        "Authorization": authorization,
    }
    
    if additional_headers:
        headers.update(additional_headers)
    
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=timeout,
    )
    
    return response


def http_post(
    url: str,
    request_id: str,
    authorization: str,
    json_data: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    additional_headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> requests.Response:
    """
    Send HTTP POST request with required headers.

    Args:
        url: The target URL
        request_id: Request ID for X-Ray-Serve-Request-Id header
        authorization: Authorization token for Authorization header
        json_data: JSON data to send in request body
        data: Form data or raw data to send in request body
        additional_headers: Additional headers to include
        timeout: Request timeout in seconds

    Returns:
        requests.Response object
    """
    headers = {
        "X-Ray-Serve-Request-Id": request_id,
        "Authorization": authorization,
    }
    
    if additional_headers:
        headers.update(additional_headers)
    
    response = requests.post(
        url,
        headers=headers,
        json=json_data,
        data=data,
        timeout=timeout,
    )
    
    return response
