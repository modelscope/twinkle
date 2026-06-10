# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the OpenAI translation bridge (pure functions)."""
from __future__ import annotations

import pytest

from twinkle.server.gateway.openai_bridge import (make_error, translate_chat_request, translate_response,
                                                  translate_stream_chunk)

# ---------- translate_chat_request ----------------------------------------- #


class TestTranslateChatRequest:

    def test_basic_chat(self):
        body = {
            'model': 'lora-abc',
            'messages': [{
                'role': 'user',
                'content': 'Hello'
            }],
        }
        req, sticky = translate_chat_request(body)
        assert sticky == 'lora-abc'
        assert req['adapter_name'] == 'lora-abc'
        assert req['inputs'] == {'messages': [{'role': 'user', 'content': 'Hello'}]}
        assert req['sampling_params'] is None

    def test_with_tools(self):
        body = {
            'model': 'my-model',
            'messages': [{
                'role': 'user',
                'content': 'weather?'
            }],
            'tools': [{
                'type': 'function',
                'function': {
                    'name': 'get_weather'
                }
            }],
        }
        req, _ = translate_chat_request(body)
        assert req['inputs']['tools'] == body['tools']

    def test_sampling_params_mapping(self):
        body = {
            'model': 'x',
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'temperature': 0.7,
            'top_p': 0.9,
            'max_tokens': 100,
            'seed': 42,
            'stop': ['END'],
            'n': 3,
        }
        req, _ = translate_chat_request(body)
        sp = req['sampling_params']
        assert sp['temperature'] == 0.7
        assert sp['top_p'] == 0.9
        assert sp['max_tokens'] == 100
        assert sp['seed'] == 42
        assert sp['stop'] == ['END']
        assert sp['num_samples'] == 3

    def test_max_completion_tokens_fallback(self):
        body = {
            'model': 'x',
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'max_completion_tokens': 50,
        }
        req, _ = translate_chat_request(body)
        assert req['sampling_params']['max_tokens'] == 50

    def test_frequency_penalty_to_repetition_penalty(self):
        body = {
            'model': 'x',
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'frequency_penalty': 0.5,
        }
        req, _ = translate_chat_request(body)
        assert req['sampling_params']['repetition_penalty'] == 1.5

    def test_logprobs_mapping(self):
        body = {
            'model': 'x',
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'logprobs': True,
            'top_logprobs': 5,
        }
        req, _ = translate_chat_request(body)
        assert req['sampling_params']['logprobs'] == 5

    def test_logprobs_false_not_mapped(self):
        body = {
            'model': 'x',
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'logprobs': False,
            'top_logprobs': 5,
        }
        req, _ = translate_chat_request(body)
        assert req['sampling_params'] is None

    def test_missing_model_raises(self):
        with pytest.raises(ValueError, match='model'):
            translate_chat_request({'messages': [{'role': 'user', 'content': 'hi'}]})

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match='model'):
            translate_chat_request({'model': '', 'messages': [{'role': 'user', 'content': 'hi'}]})

    def test_missing_messages_raises(self):
        with pytest.raises(ValueError, match='messages'):
            translate_chat_request({'model': 'x'})

    def test_none_values_ignored(self):
        body = {
            'model': 'x',
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'temperature': None,
            'top_p': None,
            'max_tokens': None,
        }
        req, _ = translate_chat_request(body)
        assert req['sampling_params'] is None


# ---------- translate_response --------------------------------------------- #


class TestTranslateResponse:

    def test_single_choice(self):
        sampler_resp = {
            'samples': [{
                'sequences': [{
                    'stop_reason': 'stop',
                    'tokens': [1, 2, 3],
                    'decoded': 'Hello world',
                }]
            }]
        }
        result = translate_response(sampler_resp, 'my-model', 'chatcmpl-123')
        assert result['id'] == 'chatcmpl-123'
        assert result['model'] == 'my-model'
        assert result['object'] == 'chat.completion'
        assert len(result['choices']) == 1
        assert result['choices'][0]['message']['content'] == 'Hello world'
        assert result['choices'][0]['finish_reason'] == 'stop'
        assert result['usage']['completion_tokens'] == 3

    def test_multiple_choices_from_num_samples(self):
        sampler_resp = {
            'samples': [{
                'sequences': [
                    {
                        'stop_reason': 'stop',
                        'tokens': [1],
                        'decoded': 'A'
                    },
                    {
                        'stop_reason': 'length',
                        'tokens': [2, 3],
                        'decoded': 'BC'
                    },
                ]
            }]
        }
        result = translate_response(sampler_resp, 'x')
        assert len(result['choices']) == 2
        assert result['choices'][0]['index'] == 0
        assert result['choices'][0]['message']['content'] == 'A'
        assert result['choices'][1]['index'] == 1
        assert result['choices'][1]['finish_reason'] == 'length'

    def test_length_finish_reason(self):
        sampler_resp = {'samples': [{'sequences': [{'stop_reason': 'length', 'tokens': [1], 'decoded': 'x'}]}]}
        result = translate_response(sampler_resp, 'x')
        assert result['choices'][0]['finish_reason'] == 'length'

    def test_empty_decoded(self):
        sampler_resp = {'samples': [{'sequences': [{'stop_reason': 'stop', 'tokens': [], 'decoded': None}]}]}
        result = translate_response(sampler_resp, 'x')
        assert result['choices'][0]['message']['content'] == ''


# ---------- translate_stream_chunk ----------------------------------------- #


class TestTranslateStreamChunk:

    def test_first_chunk_has_role(self):
        chunk = translate_stream_chunk('Hello', 'model-a', is_first=True, request_id='id-1')
        assert chunk['object'] == 'chat.completion.chunk'
        assert chunk['choices'][0]['delta']['role'] == 'assistant'
        assert chunk['choices'][0]['delta']['content'] == 'Hello'
        assert chunk['choices'][0]['finish_reason'] is None

    def test_middle_chunk(self):
        chunk = translate_stream_chunk(' world', 'model-a', request_id='id-1')
        assert 'role' not in chunk['choices'][0]['delta']
        assert chunk['choices'][0]['delta']['content'] == ' world'

    def test_final_chunk_with_finish_reason(self):
        chunk = translate_stream_chunk('', 'model-a', finish_reason='stop', request_id='id-1')
        assert chunk['choices'][0]['finish_reason'] == 'stop'
        assert 'content' not in chunk['choices'][0]['delta']


# ---------- make_error ----------------------------------------------------- #


class TestMakeError:

    def test_basic_error(self):
        err = make_error('something broke', error_type='server_error')
        assert err == {'error': {'message': 'something broke', 'type': 'server_error'}}

    def test_with_param(self):
        err = make_error('bad model', param='model')
        assert err['error']['param'] == 'model'

    def test_with_code(self):
        err = make_error('not found', error_type='model_not_found', code='model_not_found')
        assert err['error']['code'] == 'model_not_found'
