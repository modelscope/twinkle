# Copyright (c) ModelScope Contributors. All rights reserved.
from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Any


async def verify_request_token(request: Request, call_next):
    """
    Middleware to verify request token and extract request metadata.

    This middleware:
    1. Extracts the Bearer token from Authorization header
    2. Validates the token
    3. Extracts serve_multiplexed_model_id for sticky sessions
    4. Stores token and request_id in request.state for later use

    Args:
        request: The FastAPI Request object
        call_next: The next middleware/handler in the chain

    Returns:
        JSONResponse with error if validation fails, otherwise the response from call_next
    """
    authorization = request.headers.get('Twinkle-Authorization')
    token = authorization[7:] if authorization and authorization.startswith('Bearer ') else authorization
    if not is_token_valid(token):
        return JSONResponse(status_code=403, content={'detail': 'Invalid token'})

    request_id = request.headers.get('serve_multiplexed_model_id')
    if not request_id:
        return JSONResponse(
            status_code=400,
            content={'detail': 'Missing serve_multiplexed_model_id header, required for sticky session'})
    request.state.request_id = request_id
    request.state.token = token
    response = await call_next(request)
    return response


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
