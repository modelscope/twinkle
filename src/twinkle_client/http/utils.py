import os
import uuid
from contextvars import ContextVar
from typing import Optional
from datetime import datetime

TWINKLE_SERVER_URL = os.environ.get("TWINKLE_SERVER_URL", "http://127.0.0.1:8000")
TWINKLE_SERVER_TOKEN = os.environ.get("TWINKLE_SERVER_TOKEN", "EMPTY_TOKEN")

# Context variables for flexible configuration
_base_url_context: ContextVar[Optional[str]] = ContextVar('base_url', default=None)
_api_key_context: ContextVar[Optional[str]] = ContextVar('api_key', default=None)

# Global static request ID shared across all threads
# This ensures heartbeat threads use the same request ID as the main training thread
_global_request_id: Optional[str] = None


def set_base_url(url: str):
    """Set the base URL for HTTP requests in the current context."""
    _base_url_context.set(url.rstrip('/'))


def get_base_url() -> Optional[str]:
    """Get the current base URL from context or environment variable."""
    return _base_url_context.get() or TWINKLE_SERVER_URL


def clear_base_url():
    """Clear the base URL context, falling back to environment variable."""
    _base_url_context.set(None)


def set_api_key(api_key: str):
    """Set the API key for HTTP requests in the current context."""
    _api_key_context.set(api_key)


def get_api_key() -> str:
    """Get the current API key from context or environment variable."""
    return _api_key_context.get() or TWINKLE_SERVER_TOKEN


def clear_api_key():
    """Clear the API key context, falling back to environment variable."""
    _api_key_context.set(None)


def set_request_id(request_id: str):
    """Set the global request ID for HTTP requests (shared across all threads)."""
    global _global_request_id
    _global_request_id = request_id


def get_request_id() -> str:
    """Get the global request ID or generate and cache a new one."""
    global _global_request_id
    if _global_request_id is not None:
        return _global_request_id
    # Generate a new request ID and cache it globally for consistency across threads
    _global_request_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '-' + str(uuid.uuid4().hex)[0:8]
    return _global_request_id


def clear_request_id():
    """Clear the global request ID."""
    global _global_request_id
    _global_request_id = None
