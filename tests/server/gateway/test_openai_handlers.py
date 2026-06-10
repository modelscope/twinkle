# Copyright (c) ModelScope Contributors. All rights reserved.
"""Integration tests for OpenAI-compatible gateway handlers.

Uses a minimal FastAPI app with the OpenAI routes directly registered
(bypassing the full app_scaffold middleware stack) to test handler logic.
"""
from __future__ import annotations

import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from twinkle.server.gateway.openai_bridge import translate_response

# ---------- Fixtures ------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_template_cache():
    from twinkle.server.gateway.openai_handlers import _template_initialized
    _template_initialized.clear()
    yield
    _template_initialized.clear()


@pytest.fixture
def mock_gateway():
    """Build a minimal FastAPI app with OpenAI routes and a mock GatewayServer."""
    import twinkle_client.types as types
    from twinkle.server.gateway.openai_handlers import _register_openai_routes

    mock_state = AsyncMock()
    mock_state.get_model_metadata = AsyncMock(return_value=None)

    mock_proxy = MagicMock()
    mock_proxy.proxy_request = AsyncMock()
    mock_proxy.proxy_request_stream = MagicMock()

    mock_self = MagicMock()
    mock_self.state = mock_state
    mock_self.proxy = mock_proxy
    mock_self.supported_models = [types.SupportedModel(model_name='Qwen/Qwen3.5-4B')]
    mock_self._supported_model_names = frozenset(['Qwen/Qwen3.5-4B'])

    app = FastAPI()
    _register_openai_routes(app, lambda: mock_self)

    return mock_self, app


# ---------- /chat/completions tests ---------------------------------------- #


class TestChatCompletions:

    def _make_sample_response(self, text='Hello!', stop_reason='stop'):
        """Build a mock Response matching what proxy_request returns."""
        from fastapi import Response
        body = json.dumps({
            'samples': [{
                'sequences': [{
                    'stop_reason': stop_reason,
                    'tokens': list(range(len(text))),
                    'decoded': text,
                }]
            }]
        }).encode()
        return Response(content=body, status_code=200, media_type='application/json')

    def test_basic_chat_completion(self, mock_gateway):
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response('Hi there!')

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
                'max_tokens': 20,
            },
            headers={'Authorization': 'Bearer test-key'},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data['object'] == 'chat.completion'
        assert data['model'] == 'Qwen/Qwen3.5-4B'
        assert len(data['choices']) == 1
        assert data['choices'][0]['message']['role'] == 'assistant'
        assert data['choices'][0]['message']['content'] == 'Hi there!'
        assert data['choices'][0]['finish_reason'] == 'stop'
        assert 'usage' in data

    def test_missing_model_returns_400(self, mock_gateway):
        _, app = mock_gateway
        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={'messages': [{
                'role': 'user',
                'content': 'Hello'
            }]},
            headers={'Authorization': 'Bearer test-key'},
        )
        assert resp.status_code == 400
        assert 'model' in resp.json()['error']['message']

    def test_missing_messages_returns_400(self, mock_gateway):
        _, app = mock_gateway
        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={'model': 'x'},
            headers={'Authorization': 'Bearer test-key'},
        )
        assert resp.status_code == 400
        assert 'messages' in resp.json()['error']['message']

    def test_model_not_found_returns_404(self, mock_gateway):
        mock_self, app = mock_gateway
        mock_self.supported_models = []  # No supported models
        mock_self._supported_model_names = frozenset()

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'nonexistent-model',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer test-key'},
        )
        assert resp.status_code == 404
        assert resp.json()['error']['type'] == 'model_not_found'

    def test_sticky_headers_injected(self, mock_gateway):
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response()

        client = TestClient(app)
        client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer my-key'},
        )

        # Check that proxy_request was called with sticky headers
        call_kwargs = mock_self.proxy.proxy_request.call_args
        extra_headers = call_kwargs.kwargs.get('extra_headers', {})
        assert extra_headers['x-request-id'] == 'Qwen/Qwen3.5-4B'
        assert extra_headers['serve_multiplexed_model_id'] == 'Qwen/Qwen3.5-4B'
        assert extra_headers['Serve-Multiplexed-Model-Id'] == 'Qwen/Qwen3.5-4B'
        assert extra_headers['Twinkle-Authorization'] == 'Bearer my-key'

    def test_body_override_is_sample_request(self, mock_gateway):
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response()

        client = TestClient(app)
        client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
                'temperature': 0.5,
            },
            headers={'Authorization': 'Bearer key'},
        )

        call_kwargs = mock_self.proxy.proxy_request.call_args
        body_bytes = call_kwargs.kwargs.get('body_override')
        assert body_bytes is not None
        body = json.loads(body_bytes)
        assert body['adapter_name'] == 'Qwen/Qwen3.5-4B'
        assert body['inputs'] == {'messages': [{'role': 'user', 'content': 'Hello'}]}
        assert body['sampling_params']['temperature'] == 0.5

    def test_adapter_resolves_via_metadata(self, mock_gateway):
        mock_self, app = mock_gateway
        mock_self.state.get_model_metadata.return_value = {'base_model': 'Qwen/Qwen3.5-4B'}
        mock_self.proxy.proxy_request.return_value = self._make_sample_response()

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'my-fine-tuned-adapter',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer key'},
        )
        assert resp.status_code == 200

        # Verify proxy was called with the correct base_model
        call_kwargs = mock_self.proxy.proxy_request.call_args
        assert call_kwargs.kwargs.get('base_model') or call_kwargs.args[2] == 'Qwen/Qwen3.5-4B'


# ---------- /chat/completions streaming tests ------------------------------ #


class TestChatCompletionsStreaming:

    def test_stream_request_uses_sample_stream_endpoint(self, mock_gateway):
        mock_self, app = mock_gateway

        async def fake_stream(*args, **kwargs):
            yield json.dumps({'delta': 'Hello', 'finish_reason': None})
            yield json.dumps({'delta': ' world', 'finish_reason': 'stop'})

        mock_self.proxy.proxy_request_stream = fake_stream

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hi'
                }],
                'stream': True,
            },
            headers={'Authorization': 'Bearer key'},
        )
        assert resp.status_code == 200
        assert resp.headers['content-type'].startswith('text/event-stream')

        # Parse SSE events
        lines = [line for line in resp.text.split('\n') if line.startswith('data: ')]
        assert len(lines) == 3  # 2 chunks + [DONE]
        assert lines[-1] == 'data: [DONE]'

        # First chunk should have role
        chunk1 = json.loads(lines[0].removeprefix('data: '))
        assert chunk1['object'] == 'chat.completion.chunk'
        assert chunk1['choices'][0]['delta']['role'] == 'assistant'
        assert chunk1['choices'][0]['delta']['content'] == 'Hello'

        # Second chunk
        chunk2 = json.loads(lines[1].removeprefix('data: '))
        assert chunk2['choices'][0]['delta']['content'] == ' world'
        assert chunk2['choices'][0]['finish_reason'] == 'stop'


# ---------- /models tests -------------------------------------------------- #


class TestListModels:

    def test_list_models(self, mock_gateway):
        _, app = mock_gateway
        client = TestClient(app)
        resp = client.get('/models', headers={'Authorization': 'Bearer key'})
        assert resp.status_code == 200
        data = resp.json()
        assert data['object'] == 'list'
        assert len(data['data']) == 1
        assert data['data'][0]['id'] == 'Qwen/Qwen3.5-4B'
        assert data['data'][0]['object'] == 'model'
