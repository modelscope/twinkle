# Copyright (c) ModelScope Contributors. All rights reserved.
"""
OpenAI-compatible gateway handlers.

Endpoints /chat/completions and /models registered via _register_openai_routes(app, self_fn).
Translates OpenAI request/response shapes and proxies to the existing sampler
/twinkle/sample (non-streaming) or /twinkle/sample_stream (streaming) routes.
"""
from __future__ import annotations
import asyncio
import logging
import os
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import json
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from twinkle.protocol.headers import (
    H_AUTH,
    H_AUTH_TWINKLE,
    H_CLIENT_ID,
    H_REQUEST_ID,
    H_SESSION_ID,
    build_request_headers,
)
from twinkle_server.lifecycle.envelope import envelope_from_record

if TYPE_CHECKING:
    from .app import GatewayServer

import httpx

import twinkle.protocol.types as types
from twinkle_server.utils import get_template_for_model
from .openai_bridge import (
    InferModelConflictError,
    make_error,
    translate_chat_request,
    translate_completion_request,
    translate_completion_response,
    translate_completion_stream_chunk,
    translate_embedding_request,
    translate_embedding_response,
    translate_infer_request,
    translate_infer_response,
    translate_response,
    translate_stream_chunk,
)
from .use_cases import poll_future

logger = logging.getLogger(__name__.replace('twinkle_server', 'twinkle.server', 1))


_OPENAI_SAMPLE_TIMEOUT_ENV = 'TWINKLE_OPENAI_SAMPLE_TIMEOUT'
_DEFAULT_OPENAI_SAMPLE_TIMEOUT = 1800.0

_OPENAI_ROUTE_WAIT_ENV = 'TWINKLE_OPENAI_ROUTE_WAIT_TIMEOUT'
_DEFAULT_OPENAI_ROUTE_WAIT_TIMEOUT = 300.0
_ROUTE_MOUNT_POLL_INTERVAL = 1.0


def _route_wait_timeout() -> float:
    return float(os.environ.get(_OPENAI_ROUTE_WAIT_ENV, str(_DEFAULT_OPENAI_ROUTE_WAIT_TIMEOUT)))


def _can_wait_for_sampler(gateway: GatewayServer, base_model: str, response: Response) -> bool:
    """A missing route can still mount while its pool controller exists."""
    import ray

    if not ray.is_initialized():
        return False
    try:
        error = json.loads(response.body)
    except ValueError:
        error = None
    if isinstance(error, dict) and 'error_code' in error:
        return False

    from twinkle_server.sampler.pool.controller import sampler_controller_name, sampling_pool_id

    namespace = os.environ.get('TWINKLE_STATE_RAY_NAMESPACE', 'twinkle_cluster')
    pool_id = sampling_pool_id(f"{gateway.route_prefix.rstrip('/')}/sampler/{base_model}")
    try:
        ray.get_actor(sampler_controller_name(namespace, pool_id), namespace=namespace)
    except ValueError:
        return False
    return True


class SamplerWaitTimeoutError(TimeoutError):
    """The sampler task did not reach a terminal state inside the gateway's budget."""


def _sample_wait_timeout() -> float:
    return float(os.environ.get(_OPENAI_SAMPLE_TIMEOUT_ENV, str(_DEFAULT_OPENAI_SAMPLE_TIMEOUT)))


async def _resolve_sampler_envelope(gateway: GatewayServer, sampler_data: dict[str, Any],
                                    request: Request) -> types.TaskEnvelope | None:
    """Resolve the sampler's TaskEnvelope to a terminal state.

    ``/twinkle/sample`` always answers with a TaskEnvelope; a slow sample is
    still ``queued``/``running`` when the proxy returns. Polling the shared
    state backend is the only way to obtain its result without changing the
    upstream wire contract. Returns ``None`` when the future record was lost.

    ``TWINKLE_OPENAI_SAMPLE_TIMEOUT`` bounds the entire terminal-state wait,
    including state reads. Client disconnection is checked between long polls.
    """
    envelope = types.TaskEnvelope.model_validate(sampler_data)
    if envelope.status in types.TERMINAL_STATUSES:
        return envelope

    async def wait_for_terminal(envelope: types.TaskEnvelope) -> types.TaskEnvelope | None:
        while envelope.status not in types.TERMINAL_STATUSES:
            if await request.is_disconnected():
                raise SamplerWaitTimeoutError('Client disconnected while waiting for the sampling task.')
            # ``poll_future`` waits out its whole window before reporting a missing
            # record, so check once here: a lost record is a terminal answer, not a
            # reason to keep the coroutine alive.
            if await gateway.state.get_future(envelope.request_id) is None:
                return None
            outcome = await poll_future(gateway.state, envelope.request_id)
            if outcome.record is None:
                return None
            envelope = envelope_from_record(envelope.request_id, outcome.record)
        return envelope

    timeout = _sample_wait_timeout()
    try:
        return await asyncio.wait_for(wait_for_terminal(envelope), timeout=timeout)
    except SamplerWaitTimeoutError:
        raise
    except TimeoutError:
        raise SamplerWaitTimeoutError(
            f'Sampling task {envelope.request_id} did not finish within {timeout:g}s.') from None


def _register_openai_routes(app: FastAPI, self_fn: Callable[[], GatewayServer]) -> None:
    """Register OpenAI-compatible routes on the gateway FastAPI app."""

    @app.post('/chat/completions')
    async def chat_completions(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ):
        """OpenAI-compatible chat completions endpoint.

        Translates OpenAI request to SampleRequest and proxies to the sampler.
        """
        body = await request.json()

        # Validate and translate
        try:
            sample_request, model = translate_chat_request(body)
        except ValueError as e:
            return _bad_request(e)

        if not body.get('stream', False):
            request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
            return await _proxy_sample(
                self,
                request,
                sample_request,
                model,
                endpoint='twinkle/sample',
                translate=lambda data, m: translate_response(data, m, request_id),
            )

        request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
        is_first = True

        def _chat_chunk(delta_text: str, finish_reason: str | None, response_model: str) -> dict[str, Any]:
            nonlocal is_first
            chunk = translate_stream_chunk(
                delta_text=delta_text,
                model=response_model,
                finish_reason=finish_reason,
                request_id=request_id,
                is_first=is_first,
            )
            is_first = False
            return chunk

        return await _streaming_response(self, request, sample_request, model, _chat_chunk)

    @app.post('/completions')
    async def completions(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ):
        """OpenAI-compatible text completion endpoint.

        The prompt is wrapped as single-user-turn trajectories (see
        :func:`translate_completion_request`) and served by the same sampler path as chat,
        streaming and non-streaming alike.
        """
        body = await request.json()
        try:
            sample_request, model = translate_completion_request(body)
        except ValueError as e:
            return _bad_request(e)

        request_id = f'cmpl-{uuid.uuid4().hex[:24]}'
        if body.get('stream', False):
            return await _streaming_response(
                self, request, sample_request, model,
                lambda delta, finish, response_model: translate_completion_stream_chunk(
                    delta_text=delta, model=response_model, finish_reason=finish, request_id=request_id),
            )

        return await _proxy_sample(
            self,
            request,
            sample_request,
            model,
            endpoint='twinkle/sample',
            translate=lambda data, m: translate_completion_response(data, m, request_id),
        )

    @app.post('/embeddings')
    async def embeddings(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ):
        """OpenAI-compatible embeddings endpoint.

        Proxies to the sampler's ``/twinkle/encode`` pooling forward. Only a sampler built with a
        pooling head can serve it; a generation-only sampler returns an error, which is surfaced as-is.
        """
        body = await request.json()
        try:
            encode_request, model = translate_embedding_request(body)
        except ValueError as e:
            return _bad_request(e)
        return await _proxy_sample(
            self,
            request,
            encode_request,
            model,
            endpoint='twinkle/encode',
            translate=translate_embedding_response,
        )

    @app.post('/infer')
    async def infer(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ):
        """Rollout inference: batched token ids + logprobs for RL training.

        Accepts a list of infer requests (or ``{'infer_requests': [...], 'request_config': {...}}``),
        fans them out to a single batched sampler call, and returns per-prompt token ids, prompt/sequence
        logprobs and each sequence's ``new_input_feature`` -- the token-in-token-out surface a trainer
        recomputes ratios against, without re-tokenizing the served text.
        """
        body = await request.json()
        try:
            sample_request, model = translate_infer_request(body)
        except InferModelConflictError as e:
            return JSONResponse(status_code=400, content=make_error(message=str(e), param='model'))
        except ValueError as e:
            return _bad_request(e)
        return await _proxy_sample(
            self,
            request,
            sample_request,
            model,
            endpoint='twinkle/sample',
            translate=lambda data, m: translate_infer_response(data),
        )

    @app.get('/health')
    @app.get('/ping')
    @app.post('/ping')
    async def health():
        """Liveness probe. The gateway answering at all means the deployment is up."""
        return JSONResponse(status_code=200, content={})

    @app.get('/models')
    async def list_models(
            request: Request,
            self: GatewayServer = Depends(self_fn),
    ):
        """OpenAI-compatible model listing endpoint."""
        models = []
        for m in self.supported_models:
            models.append({
                'id': m.model_name,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': self.owned_by,
            })
        return JSONResponse(content={
            'object': 'list',
            'data': models,
        })


@dataclass(frozen=True)
class _ModelRoute:
    base_model: str
    response_model: str
    deployed_adapter: bool = False


async def _resolve_model_route(
    gateway: GatewayServer, model: str, sample_request: dict[str, Any],
) -> _ModelRoute | None:
    """Resolve the sampler, response identity and adapter from explicit bindings."""
    if model in gateway.supported_model_names:
        return _ModelRoute(model, model)

    for entry in gateway.supported_models:
        if entry.base_model and entry.model_name == model:
            return _ModelRoute(entry.base_model, model, deployed_adapter=True)

    target = gateway.openai_model_aliases.get(model)
    if target in gateway.supported_model_names:
        return _ModelRoute(target, target)

    # Training metadata locates the base sampler but has no inference weights.
    if sample_request.get('adapter_uri'):
        metadata = await gateway.state.get_model_metadata(model)
        if metadata and metadata.get('base_model') in gateway.supported_model_names:
            return _ModelRoute(metadata['base_model'], model)
    return None


def _sampler_body(sample_request: dict[str, Any], route: _ModelRoute) -> bytes:
    body = {**sample_request, 'adapter_name': ''}
    if route.deployed_adapter or sample_request.get('adapter_uri'):
        body['adapter_name'] = route.response_model
    if route.deployed_adapter:
        body['require_deployed_adapter'] = True
    return json.dumps(body).encode()


def _sampler_headers(request: Request) -> dict[str, str]:
    """Carry the caller's identity and assign a request ID when absent."""
    request_id = request.headers.get(H_REQUEST_ID) or uuid.uuid4().hex
    auth = request.headers.get(H_AUTH_TWINKLE) or request.headers.get(H_AUTH) or ''
    headers = build_request_headers(request_id, auth, client_id=request.headers.get(H_CLIENT_ID, ''))
    if session_id := request.headers.get(H_SESSION_ID):
        headers[H_SESSION_ID] = session_id
    return headers


def _bad_request(exc: ValueError) -> JSONResponse:
    """A 400 for a translator's ``ValueError``; the exception text names the offending field."""
    return JSONResponse(
        status_code=400,
        content=make_error(message=f'Missing or invalid field: {exc}', param=str(exc) or None),
    )


def _model_not_found(model: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=make_error(
            message=f"Model '{model}' not found. Use a model listed by GET /models or a configured alias.",
            error_type='model_not_found',
            param='model',
            code='model_not_found',
        ),
    )


def _sampler_http_error(model: str, status: int, message: str) -> JSONResponse:
    if status == 404:
        return _model_not_found(model)
    return JSONResponse(status_code=status, content=make_error(message=message, error_type='server_error'))


def _deny_without_key(request: Request, api_key: str | None) -> JSONResponse | None:
    """A 401 when ``api_key`` is configured and the request lacks a matching Bearer token, else None.

    When the gateway was deployed without an ``api_key`` the port is unauthenticated and this is a
    no-op -- the operator opted out, matching a plain OpenAI server started without ``--api-key``.
    The comparison is constant-time: a Bearer check on a public port is measurable by anyone who
    can reach it.
    """
    if api_key is None:
        return None
    header = request.headers.get('authorization') or ''
    supplied = header[len('Bearer '):] if header.startswith('Bearer ') else ''
    if not supplied or not secrets.compare_digest(supplied.encode(), api_key.encode()):
        return JSONResponse(
            status_code=401,
            content=make_error(
                message='Missing or invalid API key; send "Authorization: Bearer <key>".',
                error_type='authentication_error',
                code='invalid_api_key',
            ),
        )
    return None


async def _proxy_sample(
    gateway: GatewayServer,
    request: Request,
    sample_request: dict[str, Any],
    model: str,
    *,
    endpoint: str,
    translate: Callable[[dict[str, Any], str], Any],
) -> JSONResponse:
    """Resolve routing, ensure the sampler template, drive one queued task to a terminal state,
    translate the result.

    The shared body of every non-streaming OpenAI endpoint (chat/completions/infer/embeddings): they
    differ only in ``endpoint`` and in how the sampler's dict is turned back into an OpenAI shape, which
    the caller passes as ``translate(data, model)``. Keeping the resolve → template → proxy →
    wait → error sequence in one place means a fix to routing applies to all of them at once.

    The sampler replies with a TaskEnvelope, and the payload of a completed task rides in
    ``result`` -- never at the top level. A task that does not settle inside the sampler's inline
    fast path is long-polled here until it does (see ``_await_terminal_envelope``), so a generation
    of realistic length still completes within one HTTP call instead of coming back as a pending
    envelope that would translate into an empty-but-200 response.
    """
    route = await _resolve_model_route(gateway, model, sample_request)
    if route is None:
        return _model_not_found(model)

    sampler_headers = _sampler_headers(request)
    body_bytes = _sampler_body(sample_request, route)
    # The catalog can advertise a model while its pool is still loading. Only
    # retry a missing route with a live pool, and re-attempt template setup then.
    route_deadline = time.monotonic() + _route_wait_timeout()
    while True:
        await _ensure_template(gateway, route.base_model, sampler_headers, request)
        response = await gateway.proxy.proxy_request(
            request,
            endpoint=endpoint,
            base_model=route.base_model,
            service_type='sampler',
            body_override=body_bytes,
            extra_headers=sampler_headers,
        )
        if response.status_code != 404 or time.monotonic() >= route_deadline:
            break
        if await request.is_disconnected():
            break
        if not await asyncio.to_thread(_can_wait_for_sampler, gateway, route.base_model, response):
            break
        await asyncio.sleep(_ROUTE_MOUNT_POLL_INTERVAL)
    if response.status_code != 200:
        return _sampler_http_error(model, response.status_code, f'Sampler error: {response.body.decode()[:500]}')

    sampler_data = json.loads(response.body)
    try:
        envelope = await _resolve_sampler_envelope(gateway, sampler_data, request)
    except SamplerWaitTimeoutError as exc:
        return JSONResponse(status_code=504, content=make_error(message=str(exc), error_type='timeout'))
    if envelope is None:
        return JSONResponse(
            status_code=500,
            content=make_error(
                message=f"Sampling task {sampler_data.get('request_id')} was lost by the server",
                error_type='server_error'))
    if envelope.status != 'completed':
        error = envelope.error
        return JSONResponse(
            status_code=error.error_code if error is not None else 500,
            content=make_error(
                message=error.error if error is not None else f'Sampling task {envelope.status}',
                error_type=('invalid_request_error'
                            if error is not None and error.category.value == 'user' else 'server_error')))
    payload = envelope.result or {}
    expected_key = 'data' if endpoint == 'twinkle/encode' else 'samples'
    if not payload.get(expected_key):
        return JSONResponse(status_code=502, content=make_error(
            message=f'Sampler completed without returning any {expected_key}', error_type='server_error'))
    result = translate(payload, route.response_model)
    resp_headers = {}
    replica_id = response.headers.get('x-twinkle-replica-id')
    if replica_id:
        resp_headers['X-Twinkle-Replica-Id'] = replica_id
    return JSONResponse(content=result, headers=resp_headers)


async def _streaming_response(
    gateway: GatewayServer,
    request: Request,
    sample_request: dict[str, Any],
    model: str,
    chunk_translator: Callable[[str, str | None, str], dict[str, Any]],
) -> Response:
    route = await _resolve_model_route(gateway, model, sample_request)
    if route is None:
        return _model_not_found(model)
    sampler_headers = _sampler_headers(request)
    await _ensure_template(gateway, route.base_model, sampler_headers, request)
    upstream = gateway.proxy.proxy_request_stream(
        request,
        endpoint='twinkle/sample_stream',
        base_model=route.base_model,
        service_type='sampler',
        body_override=_sampler_body(sample_request, route),
        extra_headers=sampler_headers,
    )
    return _SamplerStreamingResponse(upstream, route.response_model, chunk_translator)


class _SamplerStreamingResponse(Response):
    """Keep the upstream stream open for the downstream response's ASGI lifetime."""

    def __init__(
        self,
        upstream: AbstractAsyncContextManager[httpx.Response],
        model: str,
        translate: Callable[[str, str | None, str], dict[str, Any]],
    ):
        super().__init__(media_type='text/event-stream')
        self.upstream = upstream
        self.model = model
        self.translate = translate

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with AsyncExitStack() as stack:
            # Check the upstream status before committing HTTP 200 to the client.
            try:
                upstream = await stack.enter_async_context(self.upstream)
                if upstream.status_code >= 400:
                    await upstream.aread()
                    response = _sampler_http_error(
                        self.model, upstream.status_code, f'Sampler error: {upstream.text[:500]}')
                else:
                    response = StreamingResponse(
                        self._events(upstream.aiter_lines()),
                        media_type=self.media_type,
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
                    )
            except httpx.RequestError as error:
                response = _sampler_http_error(self.model, 502, f'Sampler connection failed: {type(error).__name__}')
            await response(scope, receive, send)

    async def _events(self, lines: AsyncIterator[str]) -> AsyncIterator[str]:
        async for line in lines:
            if not line:
                continue
            chunk = json.loads(line)
            if isinstance(chunk.get('error'), str):
                error = make_error(message=chunk['error'], error_type='server_error')
                yield f'data: {json.dumps(error)}\n\n'
                return
            event = self.translate(chunk.get('delta', ''), chunk.get('finish_reason'), self.model)
            yield f'data: {json.dumps(event)}\n\n'
        yield 'data: [DONE]\n\n'


def _register_api_key_gate(app: FastAPI, self_fn: Callable[[], GatewayServer]) -> None:
    """Register the app-level Bearer gate for ``api_key``.

    The gate covers every route, not just the OpenAI endpoints: the tinker training endpoints and
    the ``/twinkle/*`` task surface are strictly more sensitive than generation, and protecting chat
    while leaving weight-mutating calls anonymous would be security theater. Liveness probes and
    the model listing stay open so load balancers can do their job. Registered after the routes
    (and therefore wrapping the cleanup middleware ``build_deployment_app`` installs) so a rejected
    request short-circuits before any handler runs.
    """
    open_paths = frozenset({'/health', '/ping', '/models', '/twinkle/healthz', '/twinkle/healthz/deep'})

    @app.middleware('http')
    async def require_api_key(request: Request, call_next):
        gateway = self_fn()
        if gateway.api_key is not None:
            # ``request.url.path`` carries the mount prefix (e.g. ``/api/v1/models``) while
            # ``open_paths`` are app-relative, so strip the gateway's own route_prefix before the
            # membership test -- otherwise a non-empty prefix (the default) gates the public
            # discovery/liveness endpoints too, breaking the OpenAI "discovery is open" convention
            # and load-balancer health probes.
            path = request.url.path
            prefix = (gateway.route_prefix or '').rstrip('/')
            if prefix and path.startswith(prefix):
                path = path[len(prefix):] or '/'
            if path not in open_paths:
                denied = _deny_without_key(request, gateway.api_key)
                if denied is not None:
                    return denied
        return await call_next(request)


async def _ensure_template(
    gateway: GatewayServer,
    base_model: str,
    sampler_headers: dict[str, str],
    request: Request,
) -> None:
    """Ensure the sampler has a chat template set for encoding Trajectory inputs.

    Called once per base_model (cached on the ``GatewayServer`` instance). On failure,
    logs a warning but doesn't block -- the sampler will return its own error if needed.
    """
    if base_model in gateway._template_initialized:
        return

    template_cls = get_template_for_model(base_model)
    set_template_body = json.dumps({
        'template_cls': template_cls,
        'adapter_name': '',
    }).encode()

    try:
        resp = await gateway.proxy.proxy_request(
            request,
            endpoint='twinkle/set_template',
            base_model=base_model,
            service_type='sampler',
            body_override=set_template_body,
            extra_headers=sampler_headers,
        )
        if resp.status_code == 200:
            gateway._template_initialized.add(base_model)
        else:
            logger.warning('set_template failed: %s', resp.body.decode()[:200])
    except Exception as e:
        logger.warning('set_template call failed: %s', e)
