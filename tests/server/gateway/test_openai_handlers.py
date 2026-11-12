# Copyright (c) ModelScope Contributors. All rights reserved.
"""Integration tests for OpenAI-compatible gateway handlers.

Uses a minimal FastAPI app with the OpenAI routes directly registered
(bypassing the full app_scaffold middleware stack) to test handler logic.
"""
from __future__ import annotations

import asyncio
import json
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from twinkle.server.gateway.openai_bridge import translate_response

# ---------- Fixtures ------------------------------------------------------- #


@pytest.fixture
def mock_gateway():
    """Build a minimal FastAPI app with OpenAI routes and a mock GatewayServer."""
    import twinkle.protocol.types as types
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
    mock_self.supported_model_names = frozenset(['Qwen/Qwen3.5-4B'])
    # Per-instance template cache; a fresh mock per test isolates it, so the
    # former module-global clear fixture is no longer needed.
    mock_self._template_initialized = set()

    app = FastAPI()
    _register_openai_routes(app, lambda: mock_self)

    return mock_self, app


# ---------- /chat/completions tests ---------------------------------------- #


class TestChatCompletions:

    def _make_sample_response(self, text='Hello!', stop_reason='stop', status='completed'):
        """Build a mock Response matching what proxy_request returns.

        ``/twinkle/sample`` always answers with a TaskEnvelope whose completed
        result carries the samples; the raw samples dict is not the wire shape.
        """
        from fastapi import Response
        result = {
            'samples': [{
                'sequences': [{
                    'stop_reason': stop_reason,
                    'tokens': list(range(len(text))),
                    'decoded': text,
                }]
            }]
        }
        payload = {'request_id': 'req-mock', 'status': status}
        if status == 'completed':
            payload['result'] = result
        body = json.dumps(payload).encode()
        return Response(content=body, status_code=200, media_type='application/json')

    def _samples_result(self, text='Hello!'):
        return {
            'samples': [{
                'sequences': [{
                    'stop_reason': 'stop',
                    'tokens': list(range(len(text))),
                    'decoded': text,
                }]
            }]
        }

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
        mock_self.supported_model_names = frozenset()

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

    def test_non_stream_polls_queued_task_to_completion(self, mock_gateway):
        """A queued sampler envelope is polled until completed, then translated."""
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response(status='queued')
        mock_self.state.get_future = AsyncMock(return_value={
            'status': 'completed',
            'result': self._samples_result('Polled!'),
        })

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer key'},
        )

        assert resp.status_code == 200
        assert resp.json()['choices'][0]['message']['content'] == 'Polled!'
        mock_self.state.get_future.assert_awaited()

    def test_non_stream_failed_task_returns_error(self, mock_gateway):
        """A failed sampler envelope becomes an OpenAI error, never an empty 200."""
        from fastapi import Response
        mock_self, app = mock_gateway
        envelope = {
            'request_id': 'req-failed',
            'status': 'failed',
            'result': None,
            'error': {
                'error': 'data/mask length mismatch at sample 0',
                'category': 'server',
                'error_code': 500,
                'request_id': 'req-failed',
            },
        }
        mock_self.proxy.proxy_request.return_value = Response(
            content=json.dumps(envelope).encode(), status_code=200, media_type='application/json')

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer key'},
        )

        assert resp.status_code == 500
        assert 'mismatch' in resp.json()['error']['message']
        assert resp.json()['error']['type'] == 'server_error'

    def test_non_stream_wait_is_bounded(self, mock_gateway, monkeypatch):
        """A task that never terminates returns 504 instead of polling forever."""
        monkeypatch.setenv('TWINKLE_OPENAI_SAMPLE_TIMEOUT', '0.01')
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response(status='running')
        mock_self.state.get_future = AsyncMock(return_value={'status': 'running'})

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer key'},
        )

        assert resp.status_code == 504
        assert resp.json()['error']['type'] == 'timeout'
        assert 'did not finish' in resp.json()['error']['message']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('blocked_step', ['disconnect_check', 'initial_read', 'long_poll'])
    async def test_non_stream_timeout_covers_each_wait_step(self, mock_gateway, monkeypatch, blocked_step):
        from twinkle.server.gateway import openai_handlers

        monkeypatch.setenv('TWINKLE_OPENAI_SAMPLE_TIMEOUT', '0.01')
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response(status='running')
        mock_self.state.get_future = AsyncMock(return_value={'status': 'running'})
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        if blocked_step == 'disconnect_check':
            monkeypatch.setattr(Request, 'is_disconnected', blocked)
        elif blocked_step == 'initial_read':
            mock_self.state.get_future.side_effect = blocked
        else:
            monkeypatch.setattr(openai_handlers, 'poll_future', blocked)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            resp = await asyncio.wait_for(
                client.post('/chat/completions', json={
                    'model': 'Qwen/Qwen3.5-4B',
                    'messages': [{'role': 'user', 'content': 'Hello'}],
                }), timeout=1.0)

        assert started.is_set()
        assert cancelled.is_set()
        assert resp.status_code == 504
        assert resp.json()['error']['type'] == 'timeout'
        assert 'did not finish' in resp.json()['error']['message']

    @pytest.mark.asyncio
    async def test_non_stream_polls_share_one_total_budget(self, mock_gateway, monkeypatch):
        from twinkle.server.gateway import openai_handlers
        from twinkle.server.gateway.use_cases import FuturePollResult

        monkeypatch.setenv('TWINKLE_OPENAI_SAMPLE_TIMEOUT', '0.05')
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response(status='running')
        mock_self.state.get_future = AsyncMock(return_value={'status': 'running'})
        calls = 0

        async def slow_poll(*args, **kwargs):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.02)
            status = 'completed' if calls == 10 else 'running'
            return FuturePollResult(record={
                'status': status,
                'result': self._samples_result('Late!') if status == 'completed' else None,
            }, timed_out=status != 'completed')

        monkeypatch.setattr(openai_handlers, 'poll_future', slow_poll)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            resp = await asyncio.wait_for(
                client.post('/chat/completions', json={
                    'model': 'Qwen/Qwen3.5-4B',
                    'messages': [{'role': 'user', 'content': 'Hello'}],
                }), timeout=1.0)

        assert resp.status_code == 504
        assert resp.json()['error']['type'] == 'timeout'
        assert calls < 10

    @pytest.mark.asyncio
    async def test_non_stream_disconnect_stops_before_state_reads(self, mock_gateway, monkeypatch):
        monkeypatch.setattr(Request, 'is_disconnected', AsyncMock(return_value=True))
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response(status='running')

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            resp = await client.post('/chat/completions', json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{'role': 'user', 'content': 'Hello'}],
            })

        assert resp.status_code == 504
        assert 'disconnected' in resp.json()['error']['message']
        mock_self.state.get_future.assert_not_awaited()

    def test_non_stream_completed_task_needs_no_wait_budget(self, mock_gateway, monkeypatch):
        monkeypatch.setenv('TWINKLE_OPENAI_SAMPLE_TIMEOUT', '0')
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response('Done!')

        with TestClient(app) as client:
            resp = client.post('/chat/completions', json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{'role': 'user', 'content': 'Hello'}],
            })

        assert resp.status_code == 200
        assert resp.json()['choices'][0]['message']['content'] == 'Done!'
        mock_self.state.get_future.assert_not_awaited()

    def test_non_stream_lost_record_is_a_server_error(self, mock_gateway):
        """A vanished future record is the server's loss, not a client 404."""
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response(status='queued')
        mock_self.state.get_future = AsyncMock(return_value=None)

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer key'},
        )

        assert resp.status_code == 500
        assert 'lost' in resp.json()['error']['message']
        assert resp.json()['error']['type'] == 'server_error'

    def test_template_request_uses_only_declared_sampler_fields(self, mock_gateway):
        """The sampler's strict schema rejects top-level model_id; do not send it."""
        from fastapi import Response
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.side_effect = [
            Response(content=b'{"status":"ok"}', status_code=200, media_type='application/json'),
            self._make_sample_response('Hi'),
        ]

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer key'},
        )

        assert resp.status_code == 200
        template_call = mock_self.proxy.proxy_request.call_args_list[0]
        template_body = json.loads(template_call.kwargs['body_override'])
        assert set(template_body) == {'template_cls', 'adapter_name'}
        assert 'Qwen/Qwen3.5-4B' in mock_self._template_initialized

    def test_failed_template_call_is_not_cached(self, mock_gateway):
        """A rejected set_template must not be marked initialized."""
        from fastapi import Response
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.side_effect = [
            Response(content=b'{}', status_code=422, media_type='application/json'),
            self._make_sample_response('Hi'),
        ]

        client = TestClient(app)
        resp = client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers={'Authorization': 'Bearer key'},
        )

        assert resp.status_code == 200
        assert 'Qwen/Qwen3.5-4B' not in mock_self._template_initialized


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
