# Copyright (c) ModelScope Contributors. All rights reserved.
"""
OpenAI-compatible translation bridge.

Pure functions that translate between OpenAI API shapes and Twinkle's
internal SampleRequest/SampleResponseModelList types. No FastAPI or
server dependency — fully unit-testable in isolation.

The response is OpenAI's shape and nothing else. Training on what an endpoint
served needs the token ids the sampler emitted, not the text re-tokenized -- but
that is not this gateway's job: a rollout that trains on an agent's own requests
serves them itself (``twinkle_agentic.rollout.PolicyEndpoint``), in the trainer's
process, and reports each round through a callback. An extra field here would be a
second, weaker way to do the same thing, over a hop that has already lost the
sampler's own objects.
"""
from __future__ import annotations

import time
import uuid
from typing import Any


def _extract_sampling_params(body: dict[str, Any]) -> dict[str, Any]:
    """Pull the sampling knobs out of an OpenAI request body into a twinkle ``sampling_params`` dict.

    Shared by the chat and completion translators so both read the same fields with the same OpenAI →
    twinkle name mapping (``n`` → ``num_samples``, ``frequency_penalty`` → ``repetition_penalty``, the
    ``logprobs`` bool + ``top_logprobs`` int pair → twinkle's single int). Only fields the body actually
    sets are carried, so twinkle's own defaults stand for the rest.
    """
    sampling_params: dict[str, Any] = {}
    if body.get('temperature') is not None:
        sampling_params['temperature'] = body['temperature']
    if body.get('top_p') is not None:
        sampling_params['top_p'] = body['top_p']
    if body.get('max_tokens') is not None:
        sampling_params['max_tokens'] = body['max_tokens']
    if body.get('max_completion_tokens') is not None:
        sampling_params['max_tokens'] = body['max_completion_tokens']
    if body.get('seed') is not None:
        sampling_params['seed'] = body['seed']
    if body.get('stop') is not None:
        sampling_params['stop'] = body['stop']
    if body.get('n') is not None:
        sampling_params['num_samples'] = body['n']
    if body.get('frequency_penalty') is not None:
        sampling_params['repetition_penalty'] = 1.0 + body['frequency_penalty']
    if body.get('logprobs') and body.get('top_logprobs') is not None:
        sampling_params['logprobs'] = body['top_logprobs']
    return sampling_params


def _require_model(body: dict[str, Any]) -> str:
    model = body.get('model')
    if not model or not isinstance(model, str):
        raise ValueError('model')
    return model


def _is_token_ids(value: Any) -> bool:
    """True for a non-empty list of plain ints -- OpenAI's token-id prompt/input form."""
    return (isinstance(value, list) and bool(value)
            and all(isinstance(t, int) and not isinstance(t, bool) for t in value))


def _text_trajectory(text: str) -> dict[str, Any]:
    """One bare text as a single-user-turn Trajectory.

    The gateway has no tokenizer, so text prompts run through the model's own chat template;
    token-id prompts skip it entirely by arriving as a pre-encoded InputFeature.
    """
    return {'messages': [{'role': 'user', 'content': text}]}


def translate_chat_request(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Translate an OpenAI chat completion request body to a SampleRequest dict.

    Returns:
        (sample_request_dict, sticky_key) where sticky_key is the model field
        used for Ray Serve multiplex routing.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    model = _require_model(body)

    messages = body.get('messages')
    if not messages:
        raise ValueError('messages')

    # Build Trajectory input (OpenAI messages are already in the right shape)
    trajectory: dict[str, Any] = {'messages': messages}
    if body.get('tools'):
        trajectory['tools'] = body['tools']

    sampling_params = _extract_sampling_params(body)

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
    prompt_tokens = 0

    for sample in samples:
        sequences = sample.get('sequences', [])
        prompt_token_ids = sample.get('prompt_token_ids')
        # Counted once per sample, not once per sequence: n>1 shares one prompt.
        prompt_tokens += len(prompt_token_ids or [])
        for seq in sequences:
            decoded = seq.get('decoded') or ''
            finish_reason = _map_stop_reason(seq.get('stop_reason'))
            tokens = seq.get('tokens', [])
            total_tokens += len(tokens)

            choice: dict[str, Any] = {
                'index': len(choices),
                'message': {
                    'role': 'assistant',
                    'content': decoded,
                },
                'finish_reason': finish_reason,
            }
            choices.append(choice)

    return {
        'id': request_id,
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': choices,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': total_tokens,
            'total_tokens': prompt_tokens + total_tokens,
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


def translate_completion_request(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Translate an OpenAI text-completion request body to a SampleRequest dict.

    ``prompt`` takes any of OpenAI's four forms -- a string, a list of strings, a list of token
    ids, or a list of such lists. Text prompts are wrapped as single-user-turn ``Trajectory``
    entries (the gateway has no tokenizer, and running the prompt through the model's own chat
    template is what a served chat model expects); token-id lists arrive as pre-encoded
    ``InputFeature`` entries instead. ``logprobs`` here is OpenAI's completion-style bare int (not
    the chat bool + ``top_logprobs`` pair), so it is mapped on its own after the shared extractor.

    Returns:
        ``(sample_request_dict, sticky_key)`` -- same contract as :func:`translate_chat_request`.

    Raises:
        ValueError: If ``model`` or ``prompt`` is missing or invalid, or ``echo`` / ``suffix`` is
            set (this gateway does not echo the prompt or generate mid-prompt, and silently
            dropping either would return a response that is not what was asked for).
    """
    model = _require_model(body)

    for unsupported in ('echo', 'suffix'):
        if body.get(unsupported):
            raise ValueError(unsupported)

    prompt = body.get('prompt')
    if isinstance(prompt, str):
        inputs: list[dict[str, Any]] = [_text_trajectory(prompt)]
    elif isinstance(prompt, list) and prompt and all(isinstance(t, str) for t in prompt):
        inputs = [_text_trajectory(t) for t in prompt]
    elif _is_token_ids(prompt):
        inputs = [{'input_ids': prompt}]
    elif isinstance(prompt, list) and prompt and all(_is_token_ids(t) for t in prompt):
        inputs = [{'input_ids': t} for t in prompt]
    else:
        raise ValueError('prompt')

    sampling_params = _extract_sampling_params(body)
    if isinstance(body.get('logprobs'), int) and not isinstance(body['logprobs'], bool):
        sampling_params['logprobs'] = body['logprobs']

    sample_request = {
        'inputs': inputs,
        'sampling_params': sampling_params or None,
        'adapter_name': model,
    }

    adapter_uri = body.get('adapter_uri')
    if adapter_uri:
        sample_request['adapter_uri'] = adapter_uri

    return sample_request, model


def translate_completion_response(
    sampler_response: dict[str, Any],
    model: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Translate a SampleResponseModelList dict to an OpenAI ``text_completion`` dict.

    The completion counterpart of :func:`translate_response`: one choice per sampled sequence, carrying
    the decoded text under ``text`` instead of a chat ``message``. ``logprobs`` is left null -- OpenAI's
    completion logprobs shape (token/bytes/top_logprobs arrays) is not what a rollout needs; ``/infer``
    returns the raw per-token logprobs instead.
    """
    if request_id is None:
        request_id = f'cmpl-{uuid.uuid4().hex[:24]}'

    samples = sampler_response.get('samples', [])
    choices: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0

    for sample in samples:
        prompt_tokens += len(sample.get('prompt_token_ids') or [])
        for seq in sample.get('sequences', []):
            tokens = seq.get('tokens', [])
            completion_tokens += len(tokens)
            choices.append({
                'index': len(choices),
                'text': seq.get('decoded') or '',
                'finish_reason': _map_stop_reason(seq.get('stop_reason')),
                'logprobs': None,
            })

    return {
        'id': request_id,
        'object': 'text_completion',
        'created': int(time.time()),
        'model': model,
        'choices': choices,
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    }


def translate_completion_stream_chunk(
    delta_text: str,
    model: str,
    finish_reason: str | None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Translate one sampler stream delta to an OpenAI ``text_completion.chunk``.

    The streaming counterpart of :func:`translate_completion_response`. The sampler's NDJSON lines
    carry ``{"delta", "finish_reason"}``; each becomes one chunk whose ``text`` is the delta.
    """
    if request_id is None:
        request_id = f'cmpl-{uuid.uuid4().hex[:24]}'
    return {
        'id': request_id,
        'object': 'text_completion.chunk',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'index': 0,
            'text': delta_text or '',
            'finish_reason': finish_reason,
        }],
    }


def translate_embedding_request(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Translate an OpenAI embeddings request body to a ``/twinkle/encode`` request dict.

    ``input`` is a string or a list of strings; each becomes a single-user-turn ``Trajectory`` so the
    pooling model applies its own chat/pool template. ``dimensions`` (Matryoshka truncation) is carried
    into ``pooling_params`` when set; ``task`` is fixed to ``embed``.

    Returns:
        ``(encode_request_dict, sticky_key)``.

    Raises:
        ValueError: If ``model`` or ``input`` is missing or invalid.
    """
    model = _require_model(body)

    raw_input = body.get('input')
    if isinstance(raw_input, str):
        inputs = [_text_trajectory(raw_input)]
    elif isinstance(raw_input, list) and raw_input and all(isinstance(t, str) for t in raw_input):
        inputs = [_text_trajectory(t) for t in raw_input]
    elif _is_token_ids(raw_input):
        inputs = [{'input_ids': raw_input}]
    elif isinstance(raw_input, list) and raw_input and all(_is_token_ids(t) for t in raw_input):
        inputs = [{'input_ids': t} for t in raw_input]
    else:
        raise ValueError('input')

    pooling_params: dict[str, Any] = {'task': 'embed'}
    if isinstance(body.get('dimensions'), int) and not isinstance(body['dimensions'], bool):
        pooling_params['dimensions'] = body['dimensions']

    encode_request = {
        'inputs': inputs,
        'pooling_params': pooling_params,
        'adapter_name': model,
    }
    return encode_request, model


def translate_embedding_response(encode_data: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a ``/twinkle/encode`` result (``{'data': [[float, ...], ...]}``) to OpenAI's embedding list."""
    vectors = encode_data.get('data', [])
    return {
        'object': 'list',
        'model': model,
        'data': [{
            'object': 'embedding',
            'index': index,
            'embedding': vector
        } for index, vector in enumerate(vectors)],
        # usage is a zero placeholder: the pooling forward has no tokenizer to count tokens, and
        # OpenAI makes the field mandatory. Clients that bill on token counts must count their side.
        'usage': {
            'prompt_tokens': 0,
            'total_tokens': 0
        },
    }


def translate_infer_request(body: Any) -> tuple[dict[str, Any], str]:
    """Translate a rollout ``/infer`` body to a batched SampleRequest dict.

    Accepts either a bare list of infer requests or ``{'infer_requests': [...], 'request_config': {...}}``.
    Each entry contributes one ``Trajectory`` (its ``messages``); ``request_config`` supplies the shared
    sampling knobs. Logprobs are forced on: rollout recomputes importance ratios against the served
    per-token logprobs, so a request that forgot to ask for them would still need them.

    Returns:
        ``(sample_request_dict, sticky_key)`` where sticky_key is the resolved model name.

    Raises:
        ValueError: If no infer requests are present or an entry lacks ``messages``.
    """
    if isinstance(body, list):
        infer_requests, request_config = body, {}
    elif isinstance(body, dict):
        infer_requests = body.get('infer_requests') or ([body] if body.get('messages') else [])
        request_config = body.get('request_config') or {}
    else:
        raise ValueError('infer_requests')

    if not infer_requests:
        raise ValueError('infer_requests')

    inputs = []
    for item in infer_requests:
        messages = item.get('messages') if isinstance(item, dict) else None
        if not messages:
            raise ValueError('messages')
        inputs.append({'messages': messages})

    model = request_config.get('model') or next(
        (item.get('model') for item in infer_requests if isinstance(item, dict) and item.get('model')), None)
    if not model or not isinstance(model, str):
        raise ValueError('model')

    sampling_params = _extract_sampling_params(request_config)
    # Rollout recomputes importance ratios against the served per-token logprobs, so they are
    # forced on: at least the minimum count even when the request did not ask for them.
    sampling_params['logprobs'] = max(int(sampling_params.get('logprobs') or 0), 1)
    for name in ('top_k', 'prompt_logprobs'):
        if request_config.get(name) is not None:
            sampling_params[name] = request_config[name]

    sample_request = {
        'inputs': inputs,
        'sampling_params': sampling_params,
        'adapter_name': model,
    }
    return sample_request, model


def translate_infer_response(sampler_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate a SampleResponseModelList dict to rollout's token-level records.

    One record per sampled prompt, carrying the exact ids the model ran on (``prompt_token_ids``), the
    prompt's own logprobs, and per-sequence ``tokens`` / ``logprobs`` / ``new_input_feature``. This is
    the token-in-token-out surface: a trainer gets trainable features back without re-tokenizing text.
    """
    records = []
    for sample in sampler_response.get('samples', []):
        records.append({
            'prompt_token_ids': sample.get('prompt_token_ids'),
            'prompt_logprobs': sample.get('prompt_logprobs'),
            'topk_prompt_logprobs': sample.get('topk_prompt_logprobs'),
            'sequences': [{
                'tokens': seq.get('tokens'),
                'logprobs': seq.get('logprobs'),
                'decoded': seq.get('decoded'),
                'finish_reason': _map_stop_reason(seq.get('stop_reason')),
                'new_input_feature': seq.get('new_input_feature'),
            } for seq in sample.get('sequences', [])],
        })
    return records
