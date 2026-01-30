import os
import uuid
from contextvars import ContextVar
from typing import Optional
from datetime import datetime

TWINKLE_SERVER_URL = os.environ.get("TWINKLE_SERVER_URL", "http://127.0.0.1:8000")
TWINKLE_SERVER_TOKEN = os.environ.get("TWINKLE_SERVER_TOKEN", "tml-EMPTY_TOKEN") # Must start with tml-
TWINKLE_REQUEST_ID = datetime.now().strftime('%Y%m%d_%H%M%S') + str(uuid.uuid4().hex)[0:8]

# Context variables for flexible configuration
_base_url_context: ContextVar[Optional[str]] = ContextVar('base_url', default=None)
_api_key_context: ContextVar[Optional[str]] = ContextVar('api_key', default=None)


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
