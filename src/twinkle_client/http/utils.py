import os
import uuid
from datetime import datetime
from typing import Optional

TWINKLE_SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000')
TWINKLE_SERVER_TOKEN = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')

# Global variables for configuration
_base_url: Optional[str] = None
_api_key: Optional[str] = None
_session_id: Optional[str] = None
_request_id: Optional[str] = None


def set_base_url(url: str):
    """Set the base URL for HTTP requests."""
    global _base_url
    _base_url = url.rstrip('/')


def get_base_url() -> str:
    """Get the current base URL."""
    base_url = _base_url or TWINKLE_SERVER_URL
    if not base_url.endswith('/api/v1'):
        base_url += '/api/v1'
    return base_url


def set_api_key(api_key: str):
    """Set the API key for HTTP requests."""
    global _api_key
    _api_key = api_key


def get_api_key() -> str:
    """Get the current API key."""
    return _api_key or TWINKLE_SERVER_TOKEN


def set_session_id(session_id: str):
    """Set the session ID."""
    global _session_id
    _session_id = session_id


def get_session_id() -> Optional[str]:
    """Get the current session ID."""
    return _session_id


def set_request_id(request_id: str):
    """Set the global request ID for HTTP requests (shared across all threads)."""
    global _request_id
    _request_id = request_id


def get_request_id() -> str:
    """Get the global request ID or generate and cache a new one."""
    global _request_id
    if _request_id is not None:
        return _request_id
    _request_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '-' + str(uuid.uuid4().hex)[0:8]
    return _request_id
