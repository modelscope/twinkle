# Copyright (c) ModelScope Contributors. All rights reserved.
"""
OpenAI-compatible gateway handlers.

Endpoints /chat/completions and /models registered via _register_openai_routes(app, self_fn).
Translates OpenAI request/response shapes and proxies to the existing sampler
/twinkle/sample (non-streaming) or /twinkle/sample_stream (streaming) routes.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import TYPE_CHECKING, Any

from twinkle.protocol.headers import H_AUTH, H_AUTH_TWINKLE, build_routing_headers

if TYPE_CHECKING:
    from .app import GatewayServer

import httpx

import twinkle.protocol.types as types
from twinkle.server.lifecycle.envelope import envelope_from_record
from twinkle.server.utils import get_template_for_model
from twinkle.utils.logger import get_logger
from .openai_bridge import make_error, translate_chat_request, translate_response, translate_stream_chunk
from .use_cases import poll_future

logger = get_logger()

TWINKLE_SAMPLE = 'twinkle/sample'
TWINKLE_SAMPLE_STREAM = 'twinkle/sample_stream'
TWINKLE_SET_TEMPLATE = 'twinkle/set_template'

_OPENAI_SAMPLE_TIMEOUT_ENV = 'TWINKLE_OPENAI_SAMPLE_TIMEOUT'
_DEFAULT_OPENAI_SAMPLE_TIMEOUT = 1800.0


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
    except asyncio.TimeoutError:
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

        Translates OpenAI request to SampleRequest, injects sticky routing
        headers using the model field, and proxies to the sampler.
        """
        body = await request.json()

        # Validate and translate
        try:
            sample_request, sticky_key = translate_chat_request(body)
        except ValueError as e:
            param = str(e) if str(e) in ('model', 'messages') else None
            return JSONResponse(
                status_code=400,
                content=make_error(
                    message=f'Missing or invalid field: {e}',
                    param=param,
                ),
            )

        # Resolve base_model: check adapter metadata first, then check if model
        # is itself a supported base model name
        model = body['model']
        base_model = await _resolve_base_model(self, model)
        if base_model is None:
            return JSONResponse(
                status_code=404,
                content=make_error(
                    message=f"Model '{model}' not found. Register it as an adapter or use a supported base model.",
                    error_type='model_not_found',
                    param='model',
                ),
            )

        # Build sticky routing headers — use token (or session_id) so
        # each client gets its own sampler adapter slot isolation, rather
        # than the model name which would share slots across all callers.
        sticky_key = getattr(request.state, 'session_id', '') or getattr(request.state, 'token', '') or sticky_key
        sticky_headers = _build_sticky_headers(sticky_key, request)
        body_bytes = json.dumps(sample_request).encode()

        # Ensure sampler has a chat template set (required for Trajectory inputs)
        await _ensure_template(self, base_model, sticky_headers, request)

        stream = body.get('stream', False)

        if not stream:
            # Non-streaming: proxy to /twinkle/sample, translate response
            response = await self.proxy.proxy_request(
                request,
                endpoint=TWINKLE_SAMPLE,
                base_model=base_model,
                service_type='sampler',
                body_override=body_bytes,
                extra_headers=sticky_headers,
            )

            if response.status_code != 200:
                return JSONResponse(
                    status_code=response.status_code,
                    content=make_error(
                        message=f'Sampler error: {response.body.decode()[:500]}',
                        error_type='server_error',
                    ),
                )

            sampler_data = json.loads(response.body)
            try:
                envelope = await _resolve_sampler_envelope(self, sampler_data, request)
            except SamplerWaitTimeoutError as e:
                return JSONResponse(
                    status_code=504,
                    content=make_error(message=str(e), error_type='timeout'),
                )
            if envelope is None:
                return JSONResponse(
                    status_code=500,
                    content=make_error(
                        message=f'Sampling task {sampler_data.get("request_id")} was lost by the server',
                        error_type='server_error',
                    ),
                )
            if envelope.status != 'completed':
                error = envelope.error
                message = error.error if error is not None else f'Sampling task {envelope.status}'
                error_type = ('invalid_request_error'
                              if error is not None and error.category.value == 'user' else 'server_error')
                return JSONResponse(
                    status_code=error.error_code if error is not None else 500,
                    content=make_error(message=message, error_type=error_type),
                )

            sampler_result = envelope.result or {}
            if not sampler_result.get('samples'):
                return JSONResponse(
                    status_code=502,
                    content=make_error(
                        message='Sampler completed without returning any samples',
                        error_type='server_error',
                    ),
                )

            request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
            openai_response = translate_response(sampler_result, model, request_id)
            resp_headers = {}
            replica_id = response.headers.get('x-twinkle-replica-id')
            if replica_id:
                resp_headers['X-Twinkle-Replica-Id'] = replica_id
            return JSONResponse(content=openai_response, headers=resp_headers)

        else:
            # Streaming: proxy to /twinkle/sample_stream, translate to SSE
            request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'

            async def _sse_generator():
                is_first = True
                try:
                    async for line in self.proxy.proxy_request_stream(
                            request,
                            endpoint=TWINKLE_SAMPLE_STREAM,
                            base_model=base_model,
                            service_type='sampler',
                            body_override=body_bytes,
                            extra_headers=sticky_headers,
                    ):
                        chunk_data = json.loads(line)
                        delta_text = chunk_data.get('delta', '')
                        finish_reason = chunk_data.get('finish_reason')

                        openai_chunk = translate_stream_chunk(
                            delta_text=delta_text,
                            model=model,
                            finish_reason=finish_reason,
                            request_id=request_id,
                            is_first=is_first,
                        )
                        is_first = False
                        yield f'data: {json.dumps(openai_chunk)}\n\n'

                    yield 'data: [DONE]\n\n'
                except httpx.HTTPStatusError as e:
                    error_body = e.response.content.decode()[:500]
                    error_chunk = make_error(
                        message=f'Streaming error: {error_body}',
                        error_type='server_error',
                    )
                    yield f'data: {json.dumps(error_chunk)}\n\n'
                    yield 'data: [DONE]\n\n'

            return StreamingResponse(
                _sse_generator(),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                },
            )

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
                'owned_by': 'twinkle',
            })
        return JSONResponse(content={
            'object': 'list',
            'data': models,
        })


async def _resolve_base_model(gateway: GatewayServer, model: str) -> str | None:
    """Resolve the base_model for routing given an adapter/model name.

    Checks:
    1. Model metadata in state (adapter → base_model mapping)
    2. Whether the model name itself is a supported base model
    """
    # Check if it's a registered adapter with metadata
    try:
        metadata = await gateway.state.get_model_metadata(model)
        if metadata and metadata.get('base_model'):
            return metadata['base_model']
    except Exception:
        pass

    # Check if it's directly a supported base model
    if model in gateway.supported_model_names:
        return model

    # Fallback: if there's exactly one supported model, use it
    if len(gateway.supported_model_names) == 1:
        return next(iter(gateway.supported_model_names))

    return None


def _build_sticky_headers(sticky_key: str, request: Request) -> dict[str, str]:
    """Build the headers needed for sticky session routing."""
    auth = (request.headers.get(H_AUTH_TWINKLE) or request.headers.get(H_AUTH) or '')
    return build_routing_headers(sticky_key, auth)


async def _ensure_template(
    gateway: GatewayServer,
    base_model: str,
    sticky_headers: dict[str, str],
    request: Request,
) -> None:
    """Ensure the sampler has a chat template set for encoding Trajectory inputs.

    Called once per base_model (cached on the ``GatewayServer`` instance). On failure,
    logs a warning but doesn't block -- the sampler will return its own error if needed.
    """
    if base_model in gateway._template_initialized:
        return

    template_cls = get_template_for_model(base_model)
    # The sampler's strict schema accepts only template_cls / adapter_name /
    # init_kwargs. It already knows its own model_id and injects it when the
    # template constructor needs one, so sending model_id here is rejected 422.
    set_template_body = json.dumps({
        'template_cls': template_cls,
        'adapter_name': '',
    }).encode()

    try:
        resp = await gateway.proxy.proxy_request(
            request,
            endpoint=TWINKLE_SET_TEMPLATE,
            base_model=base_model,
            service_type='sampler',
            body_override=set_template_body,
            extra_headers=sticky_headers,
        )
        if resp.status_code == 200:
            gateway._template_initialized.add(base_model)
        else:
            logger.warning('set_template failed: %s', resp.body.decode()[:200])
    except Exception as e:
        logger.warning('set_template call failed: %s', e)
