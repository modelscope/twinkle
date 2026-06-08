# Copyright (c) ModelScope Contributors. All rights reserved.
"""
OpenAI-compatible gateway handlers.

Endpoints /chat/completions and /models registered via _register_openai_routes(app, self_fn).
Translates OpenAI request/response shapes and proxies to the existing sampler
/twinkle/sample (non-streaming) or /twinkle/sample_stream (streaming) routes.
"""
from __future__ import annotations

import json
import uuid
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .app import GatewayServer

import httpx

from twinkle.utils.logger import get_logger
from .openai_bridge import make_error, translate_chat_request, translate_response, translate_stream_chunk

logger = get_logger()


def _register_openai_routes(app: FastAPI, self_fn: Callable[[], 'GatewayServer']) -> None:
    """Register OpenAI-compatible routes on the gateway FastAPI app."""

    @app.post('/chat/completions')
    async def chat_completions(
        request: Request,
        self: 'GatewayServer' = Depends(self_fn),
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
                    type='model_not_found',
                    param='model',
                ),
            )

        # Build sticky routing headers
        sticky_headers = _build_sticky_headers(sticky_key, request)
        body_bytes = json.dumps(sample_request).encode()

        stream = body.get('stream', False)

        if not stream:
            # Non-streaming: proxy to /twinkle/sample, translate response
            response = await self.proxy.proxy_request(
                request,
                endpoint='twinkle/sample',
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
                        type='server_error',
                    ),
                )

            sampler_data = json.loads(response.body)
            request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'
            openai_response = translate_response(sampler_data, model, request_id)
            return JSONResponse(content=openai_response)

        else:
            # Streaming: proxy to /twinkle/sample_stream, translate to SSE
            request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'

            async def _sse_generator():
                is_first = True
                try:
                    async for line in self.proxy.proxy_request_stream(
                        request,
                        endpoint='twinkle/sample_stream',
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
                        type='server_error',
                    )
                    yield f'data: {json.dumps(error_chunk)}\n\n'
                    yield 'data: [DONE]\n\n'

            return StreamingResponse(
                _sse_generator(),
                media_type='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
            )

    @app.get('/models')
    async def list_models(
        request: Request,
        self: 'GatewayServer' = Depends(self_fn),
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


async def _resolve_base_model(gateway: 'GatewayServer', model: str) -> str | None:
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
    supported_names = [m.model_name for m in gateway.supported_models]
    if model in supported_names:
        return model

    # Fallback: if there's exactly one supported model, use it
    if len(supported_names) == 1:
        return supported_names[0]

    return None


def _build_sticky_headers(sticky_key: str, request: Request) -> dict[str, str]:
    """Build the headers needed for sticky session routing."""
    auth = (
        request.headers.get('Twinkle-Authorization')
        or request.headers.get('Authorization')
        or ''
    )
    return {
        'X-Ray-Serve-Request-Id': sticky_key,
        'Serve-Multiplexed-Model-Id': sticky_key,
        'serve_multiplexed_model_id': sticky_key,
        'x-request-id': sticky_key,
        'Twinkle-Authorization': auth,
    }
