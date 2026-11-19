# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared request ID and authentication headers."""

# -- request id --
H_REQUEST_ID = 'x-request-id'
H_REQUEST_ID_LEGACY = 'X-Ray-Serve-Request-Id'

# -- auth --
H_AUTH = 'Authorization'
H_AUTH_TWINKLE = 'Twinkle-Authorization'

def build_routing_headers(request_id: str, auth: str = '') -> dict[str, str]:
    """Build headers carrying the request ID and Twinkle authentication."""
    return {
        H_REQUEST_ID: request_id,
        H_REQUEST_ID_LEGACY: request_id,
        H_AUTH: auth,
        H_AUTH_TWINKLE: auth,
    }
