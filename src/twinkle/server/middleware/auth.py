# Copyright (c) ModelScope Contributors. All rights reserved.
import secrets
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from twinkle.protocol.headers import H_AUTH, H_AUTH_TWINKLE, H_CLIENT_ID, H_REQUEST_ID, H_SESSION_ID
from twinkle.server.identity import current_tenant_fingerprint, tenant_fingerprint

_OPENAI_COMPAT_SUFFIXES = ('/chat/completions', '/models')


def deny_without_api_key(request: Request, api_key: str | None) -> JSONResponse | None:
    """A 401 when ``api_key`` is configured and the request lacks a matching Bearer token.

    The deployment-level credential check shared by every public HTTP ingress (gateway and
    sampler alike): ``verify_request_token`` only extracts the tenant identity -- its
    ``is_token_valid`` hook accepts every token by design -- so an ingress that must be
    authenticated installs this gate on top. ``None`` means the operator opted out of auth.
    Accepts the same two headers the gateway forwards to the sampler (``Twinkle-Authorization``
    first, then ``Authorization``), and compares in constant time.
    """
    if api_key is None:
        return None
    header = request.headers.get(H_AUTH_TWINKLE) or request.headers.get(H_AUTH) or ''
    supplied = header[len('Bearer '):] if header.startswith('Bearer ') else ''
    if not supplied or not secrets.compare_digest(supplied.encode(), api_key.encode()):
        return JSONResponse(
            status_code=401,
            content={'detail': 'Missing or invalid API key; send "Authorization: Bearer <key>".'},
        )
    return None


def install_api_key_gate(app, api_key: str | None, *, open_suffixes: tuple[str, ...] = ('/healthz',)) -> None:
    """Register the Bearer gate as the OUTERMOST middleware of a deployment app.

    Registered after ``build_deployment_app``'s stack so a rejected request short-circuits
    before identity extraction or any handler runs. ``open_suffixes`` keeps liveness probes
    reachable without credentials (matched on the path suffix because the app is mounted
    under a deployment-specific route prefix it does not itself know).
    """
    if api_key is None:
        return

    @app.middleware('http')
    async def require_api_key(request: Request, call_next):
        if not any(request.url.path.endswith(suffix) for suffix in open_suffixes):
            denied = deny_without_api_key(request, api_key)
            if denied is not None:
                return denied
        return await call_next(request)


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
    request.state.tenant_fingerprint = tenant_fingerprint(token)
    binding = current_tenant_fingerprint.set(request.state.tenant_fingerprint)
    try:
        return await call_next(request)
    finally:
        current_tenant_fingerprint.reset(binding)


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
