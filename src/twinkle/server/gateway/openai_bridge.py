# Copyright (c) ModelScope Contributors. All rights reserved.
"""
OpenAI-compatible translation bridge.

Pure functions that translate between OpenAI API shapes and Twinkle's
internal SampleRequest/SampleResponseModelList types. No FastAPI or
server dependency — fully unit-testable in isolation.
"""
from __future__ import annotations

import time
import uuid
from typing import Any


def translate_chat_request(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Translate an OpenAI chat completion request body to a SampleRequest dict.

    Returns:
        (sample_request_dict, sticky_key) where sticky_key is the model field
        used for Ray Serve multiplex routing.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    model = body.get('model')
    if not model or not isinstance(model, str):
        raise ValueError('model')

    messages = body.get('messages')
    if not messages:
        raise ValueError('messages')

    # Build Trajectory input (OpenAI messages are already in the right shape)
    trajectory: dict[str, Any] = {'messages': messages}
    if body.get('tools'):
        trajectory['tools'] = body['tools']

    # Build sampling_params dict
    sampling_params: dict[str, Any] = {}

    if 'temperature' in body and body['temperature'] is not None:
        sampling_params['temperature'] = body['temperature']
    if 'top_p' in body and body['top_p'] is not None:
        sampling_params['top_p'] = body['top_p']
    if 'max_tokens' in body and body['max_tokens'] is not None:
        sampling_params['max_tokens'] = body['max_tokens']
    if 'max_completion_tokens' in body and body['max_completion_tokens'] is not None:
        sampling_params['max_tokens'] = body['max_completion_tokens']
    if 'seed' in body and body['seed'] is not None:
        sampling_params['seed'] = body['seed']
    if 'stop' in body and body['stop'] is not None:
        sampling_params['stop'] = body['stop']
    if 'n' in body and body['n'] is not None:
        sampling_params['num_samples'] = body['n']
    if 'frequency_penalty' in body and body['frequency_penalty'] is not None:
        sampling_params['repetition_penalty'] = 1.0 + body['frequency_penalty']

    # logprobs: OpenAI uses (logprobs: bool, top_logprobs: int)
    if body.get('logprobs') and body.get('top_logprobs') is not None:
        sampling_params['logprobs'] = body['top_logprobs']

    sample_request = {
        'inputs': trajectory,
        'sampling_params': sampling_params or None,
        'adapter_name': model,
    }

    adapter_uri = body.get('adapter_uri')
    if adapter_uri:
        sample_request['adapter_uri'] = adapter_uri

    return sample_request, model


def translate_response(
    sampler_response: dict[str, Any],
    model: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Translate a SampleResponseModelList dict to an OpenAI ChatCompletion dict."""
    if request_id is None:
        request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'

    samples = sampler_response.get('samples', [])
    choices = []
    total_tokens = 0

    for sample in samples:
        sequences = sample.get('sequences', [])
        for seq in sequences:
            decoded = seq.get('decoded') or ''
            finish_reason = _map_stop_reason(seq.get('stop_reason'))
            tokens = seq.get('tokens', [])
            total_tokens += len(tokens)

            choices.append({
                'index': len(choices),
                'message': {
                    'role': 'assistant',
                    'content': decoded,
                },
                'finish_reason': finish_reason,
            })

    return {
        'id': request_id,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': choices,
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': total_tokens,
            'total_tokens': total_tokens,
        },
    }


def translate_stream_chunk(
    delta_text: str,
    model: str,
    index: int = 0,
    finish_reason: str | None = None,
    request_id: str | None = None,
    is_first: bool = False,
) -> dict[str, Any]:
    """Build one OpenAI ChatCompletionChunk for SSE streaming."""
    if request_id is None:
        request_id = f'chatcmpl-{uuid.uuid4().hex[:24]}'

    delta: dict[str, Any] = {}
    if is_first:
        delta['role'] = 'assistant'
    if delta_text:
        delta['content'] = delta_text

    return {
        'id': request_id,
        'object': 'chat.completion.chunk',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'index': index,
            'delta': delta,
            'finish_reason': finish_reason,
        }],
    }


def make_error(
    message: str,
    error_type: str = 'invalid_request_error',
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-shaped error response body."""
    error: dict[str, Any] = {
        'message': message,
        'type': error_type,
    }
    if param is not None:
        error['param'] = param
    if code is not None:
        error['code'] = code
    return {'error': error}


def _map_stop_reason(stop_reason: str | None) -> str:
    """Map Twinkle stop_reason to OpenAI finish_reason."""
    if stop_reason in ('stop', 'abort', 'error'):
        return 'stop'
    return 'length'
