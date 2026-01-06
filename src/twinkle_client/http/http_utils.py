from typing import Optional, Dict, Any
from .utils import TWINKLE_REQUEST_ID, TWINKLE_SERVER_TOKEN
import requests


def http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    additional_headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
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
    headers = {
        "X-Ray-Serve-Request-Id": TWINKLE_REQUEST_ID,
        "Authorization": 'Bearer ' + TWINKLE_SERVER_TOKEN,
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
    json_data: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    additional_headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
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
    """
    headers = {
        "X-Ray-Serve-Request-Id": TWINKLE_REQUEST_ID,
        "Authorization": 'Bearer ' + TWINKLE_SERVER_TOKEN,
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
