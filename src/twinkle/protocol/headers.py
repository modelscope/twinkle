# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared request identity and authentication headers."""

# -- request id --
H_REQUEST_ID = 'x-request-id'
H_CLIENT_ID = 'X-Twinkle-Client-Id'
H_SESSION_ID = 'X-Twinkle-Session-Id'
H_SAMPLE_IDEMPOTENCY_KEY = 'Idempotency-Key'
H_SAMPLE_IDEMPOTENCY_SUPPORTED = 'X-Twinkle-Sample-Idempotency'

# -- auth --
H_AUTH = 'Authorization'
H_AUTH_TWINKLE = 'Twinkle-Authorization'


def build_request_headers(request_id: str, auth: str = '', *, client_id: str = '') -> dict[str, str]:
    """Build headers carrying the request ID, optional client ID and authentication."""
    headers = {
        H_REQUEST_ID: request_id,
        H_AUTH: auth,
        H_AUTH_TWINKLE: auth,
    }
    if client_id:
        headers[H_CLIENT_ID] = client_id
    return headers
