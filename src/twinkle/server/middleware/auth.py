# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from twinkle.protocol.headers import H_AUTH, H_AUTH_TWINKLE, H_CLIENT_ID, H_REQUEST_ID, H_SESSION_ID
from twinkle.server.telemetry.identity import current_user_id, user_fingerprint

_OPENAI_COMPAT_SUFFIXES = ('/chat/completions', '/models')


async def verify_request_token(request: Request, call_next):
    """
    Middleware to verify request token and extract request metadata.

    This middleware:
    1. Extracts the Bearer token from Authorization header
    2. Validates the token
    3. Extracts x-request-id (skipped for healthz
       and OpenAI-compat endpoints)
    4. Stores token, session ID, client ID and request ID in request.state

    Args:
        request: The FastAPI Request object
        call_next: The next middleware/handler in the chain

    Returns:
        JSONResponse with error if validation fails, otherwise the response from call_next
    """
    authorization = (request.headers.get(H_AUTH_TWINKLE) or request.headers.get(H_AUTH))
    token = authorization[7:] if authorization and authorization.startswith('Bearer ') else authorization
    if not is_token_valid(token):
        return JSONResponse(status_code=403, content={'detail': 'Invalid token'})

    path = request.url.path
    skip_request_id = '/healthz' in path or any(path.endswith(s) for s in _OPENAI_COMPAT_SUFFIXES)

    request_id = request.headers.get(H_REQUEST_ID, '')
    if not skip_request_id and not request_id:
        return JSONResponse(status_code=400, content={'detail': f'Missing {H_REQUEST_ID} header'})

    request.state.request_id = request_id
    client_id = request.headers.get(H_CLIENT_ID)
    request.state.client_id = client_id or request_id
    request.state.token = token
    request.state.session_id = request.headers.get(H_SESSION_ID) or ''
    request.state.user_id = user_fingerprint(token, client_id)
    binding = current_user_id.set(request.state.user_id)
    try:
        return await call_next(request)
    finally:
        current_user_id.reset(binding)


def is_token_valid(token: str) -> bool:
    """
    Validate user authentication token.

    Currently accepts all tokens. Override this function to implement
    actual token validation logic (e.g., JWT verification, API key lookup).

    Args:
        token: The authentication token to validate

    Returns:
        True if token is valid, False otherwise
    """
    return True


def get_token_from_request(request: Request) -> str:
    """
    Extract authentication token from request.

    Args:
        request: The FastAPI Request object

    Returns:
        The extracted token or empty string if not found
    """
    return getattr(request.state, 'token', '') or ''


def get_session_id_from_request(request: Request) -> str:
    """
    Extract session ID from request.

    Args:
        request: The FastAPI Request object

    Returns:
        The extracted session ID or empty string if not found
    """
    return getattr(request.state, 'session_id', '') or ''
