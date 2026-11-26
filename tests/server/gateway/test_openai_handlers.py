# Copyright (c) ModelScope Contributors. All rights reserved.
"""Integration tests for OpenAI-compatible gateway handlers.

Uses a minimal FastAPI app with the OpenAI routes directly registered
(bypassing the full app_scaffold middleware stack) to test handler logic.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import json
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

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
    mock_self.openai_model_aliases = {}
    # Per-instance template cache; a fresh mock per test isolates it, so the
    # former module-global clear fixture is no longer needed.
    mock_self._template_initialized = set()
    mock_self.owned_by = 'twinkle'

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

    @pytest.mark.parametrize('adapter_missing', [False, True])
    def test_only_missing_routes_wait_for_a_starting_pool(self, mock_gateway, monkeypatch, adapter_missing):
        import ray

        from twinkle.protocol.types import SupportedModel
        from twinkle.server.gateway import openai_handlers
        from twinkle.server.sampler.pool.controller import sampler_controller_name, sampling_pool_id

        gateway, app = mock_gateway
        base = next(iter(gateway.supported_model_names))
        gateway.route_prefix = '/v1'
        gateway._template_initialized.add(base)
        gateway.supported_models.append(SupportedModel(model_name='adapter', base_model=base))
        get_actor = MagicMock(return_value=object())
        monkeypatch.setattr(ray, 'is_initialized', lambda: True)
        monkeypatch.setattr(ray, 'get_actor', get_actor)
        monkeypatch.setenv('TWINKLE_STATE_RAY_NAMESPACE', 'test')
        monkeypatch.setattr(openai_handlers, '_ROUTE_MOUNT_POLL_INTERVAL', 0)

        missing = JSONResponse({'error_code': 404} if adapter_missing else {'detail': 'Not Found'}, status_code=404)
        gateway.proxy.proxy_request.side_effect = [missing, self._make_sample_response()]
        response = TestClient(app).post('/chat/completions', json={
            'model': 'adapter', 'messages': [{'role': 'user', 'content': 'Hello'}],
        })
        if adapter_missing:
            assert response.status_code == 404
            get_actor.assert_not_called()
            gateway.proxy.proxy_request.assert_awaited_once()
        else:
            assert response.status_code == 200
            assert response.json()['model'] == 'adapter'
            get_actor.assert_called_once_with(
                sampler_controller_name('test', sampling_pool_id(f'/v1/sampler/{base}')), namespace='test')
            assert gateway.proxy.proxy_request.await_count == 2

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

    @pytest.mark.parametrize(('endpoint', 'body'), [
        ('/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}]}),
        ('/chat/completions', {'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True}),
        ('/completions', {'prompt': 'Hello'}),
        ('/completions', {'prompt': 'Hello', 'stream': True}),
        ('/embeddings', {'input': 'Hello'}),
        ('/infer', {'infer_requests': [{'model': 'stopped-model', 'messages': [{'role': 'user', 'content': 'Hi'}]}]}),
    ])
    def test_only_remaining_model_cannot_take_another_models_requests(self, mock_gateway, endpoint, body):
        gateway, app = mock_gateway
        response = TestClient(app).post(endpoint, json={'model': 'stopped-model', **body})
        assert response.status_code == 404
        assert response.json()['error']['code'] == 'model_not_found'
        gateway.proxy.proxy_request.assert_not_called()
        gateway.proxy.proxy_request_stream.assert_not_called()

    def test_alias_to_removed_model_is_not_rebound(self, mock_gateway):
        gateway, app = mock_gateway
        gateway.openai_model_aliases = {'default': 'stopped-model'}
        response = TestClient(app).post(
            '/chat/completions', json={'model': 'default', 'messages': [{'role': 'user', 'content': 'Hello'}]})
        assert response.status_code == 404
        gateway.proxy.proxy_request.assert_not_called()

    @pytest.mark.parametrize('has_request_id', [False, True])
    def test_sampler_forwards_request_and_client_identity(self, mock_gateway, has_request_id):
        mock_self, app = mock_gateway
        mock_self.proxy.proxy_request.return_value = self._make_sample_response()

        client = TestClient(app)
        headers = {'Authorization': 'Bearer my-key', 'X-Twinkle-Client-Id': 'client-id',
                   'X-Twinkle-Session-Id': 'session-id'}
        if has_request_id:
            headers['x-request-id'] = 'request-id'
        client.post(
            '/chat/completions',
            json={
                'model': 'Qwen/Qwen3.5-4B',
                'messages': [{
                    'role': 'user',
                    'content': 'Hello'
                }],
            },
            headers=headers,
        )

        forwarded = mock_self.proxy.proxy_request.call_args.kwargs['extra_headers']
        assert forwarded['x-request-id']
        if has_request_id:
            assert forwarded['x-request-id'] == 'request-id'
        assert forwarded['X-Twinkle-Client-Id'] == 'client-id'
        assert forwarded['X-Twinkle-Session-Id'] == 'session-id'
        assert forwarded['Twinkle-Authorization'] == 'Bearer my-key'

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
        assert body['adapter_name'] == ''
        assert body['inputs'] == {'messages': [{'role': 'user', 'content': 'Hello'}]}
        assert body['sampling_params']['temperature'] == 0.5

    def test_training_metadata_requires_explicit_checkpoint(self, mock_gateway):
        mock_self, app = mock_gateway
        mock_self.state.get_model_metadata.return_value = {'base_model': 'Qwen/Qwen3.5-4B'}
        mock_self.proxy.proxy_request.return_value = self._make_sample_response()

        client = TestClient(app)
        body = {
            'model': 'my-fine-tuned-adapter',
            'messages': [{'role': 'user', 'content': 'Hello'}],
        }
        assert client.post('/chat/completions', json=body).status_code == 404
        mock_self.proxy.proxy_request.assert_not_called()
        resp = client.post('/chat/completions', json={**body, 'adapter_uri': 'twinkle://checkpoint'})
        assert resp.status_code == 200
        forwarded = mock_self.proxy.proxy_request.call_args.kwargs
        assert forwarded['base_model'] == 'Qwen/Qwen3.5-4B'
        assert json.loads(forwarded['body_override'])['adapter_uri'] == 'twinkle://checkpoint'

    @pytest.mark.parametrize(('name', 'response_model', 'adapter_name'), [
        ('default', 'Qwen/Qwen3.5-4B', ''),
        ('served-lora', 'served-lora', 'served-lora'),
    ])
    def test_resolved_name_controls_response_and_adapter(self, mock_gateway, name, response_model, adapter_name):
        from twinkle.protocol.types import SupportedModel

        gateway, app = mock_gateway
        gateway.openai_model_aliases = {'default': 'Qwen/Qwen3.5-4B'}
        gateway.supported_models.append(SupportedModel(model_name='served-lora', base_model='Qwen/Qwen3.5-4B'))
        gateway.proxy.proxy_request.return_value = self._make_sample_response()
        response = TestClient(app).post(
            '/chat/completions', json={'model': name, 'messages': [{'role': 'user', 'content': 'Hello'}]})

        assert response.status_code == 200
        assert response.json()['model'] == response_model
        forwarded = gateway.proxy.proxy_request.call_args.kwargs
        assert forwarded['base_model'] == 'Qwen/Qwen3.5-4B'
        sampler_body = json.loads(forwarded['body_override'])
        assert sampler_body['adapter_name'] == adapter_name
        assert sampler_body.get('require_deployed_adapter', False) == bool(adapter_name)

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

    @pytest.mark.parametrize('endpoint, input_body', [
        ('/chat/completions', {'messages': [{'role': 'user', 'content': 'Hi'}]}),
        ('/completions', {'prompt': 'Hi'}),
    ])
    def test_stream_request_uses_sample_stream_endpoint(self, mock_gateway, endpoint, input_body):
        mock_self, app = mock_gateway
        mock_self.openai_model_aliases = {'default': 'Qwen/Qwen3.5-4B'}
        mock_self.proxy.proxy_request.return_value = Response(status_code=200)

        @asynccontextmanager
        async def fake_stream(*args, **kwargs):
            assert kwargs['endpoint'] == 'twinkle/sample_stream'
            assert kwargs['base_model'] == 'Qwen/Qwen3.5-4B'
            assert json.loads(kwargs['body_override'])['adapter_name'] == ''
            yield httpx.Response(200, content='\n'.join([
                json.dumps({'delta': 'Hello', 'finish_reason': None}),
                json.dumps({'delta': ' world', 'finish_reason': 'stop'}),
            ]).encode())

        mock_self.proxy.proxy_request_stream = fake_stream

        client = TestClient(app)
        resp = client.post(
            endpoint,
            json={**input_body, 'model': 'default', 'stream': True},
            headers={'Authorization': 'Bearer key'},
        )
        assert resp.status_code == 200
        assert resp.headers['content-type'].startswith('text/event-stream')

        # Parse SSE events
        lines = [line for line in resp.text.split('\n') if line.startswith('data: ')]
        assert len(lines) == 3  # 2 chunks + [DONE]
        assert lines[-1] == 'data: [DONE]'

        chunk1 = json.loads(lines[0].removeprefix('data: '))
        chunk2 = json.loads(lines[1].removeprefix('data: '))
        assert chunk1['model'] == chunk2['model'] == 'Qwen/Qwen3.5-4B'
        if endpoint == '/chat/completions':
            assert chunk1['object'] == 'chat.completion.chunk'
            assert chunk1['choices'][0]['delta'] == {'role': 'assistant', 'content': 'Hello'}
            assert chunk2['choices'][0]['delta']['content'] == ' world'
        else:
            assert chunk1['object'] == 'text_completion.chunk'
            assert chunk1['choices'][0]['text'] == 'Hello'
            assert chunk2['choices'][0]['text'] == ' world'
        assert chunk2['choices'][0]['finish_reason'] == 'stop'

    @pytest.mark.parametrize('stream', [False, True])
    def test_missing_sampler_returns_http_404(self, mock_gateway, stream):
        gateway, app = mock_gateway
        gateway.proxy.proxy_request.return_value = Response(status_code=404)

        @asynccontextmanager
        async def missing_stream(*args, **kwargs):
            yield httpx.Response(404)

        gateway.proxy.proxy_request_stream = missing_stream
        response = TestClient(app).post('/chat/completions', json={
            'model': 'Qwen/Qwen3.5-4B', 'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': stream,
        })
        assert response.status_code == 404
        assert response.json()['error']['code'] == 'model_not_found'

class TestInfer:

    @pytest.fixture
    def infer_gateway(self, mock_gateway):
        gateway, app = mock_gateway
        model = next(iter(gateway.supported_model_names))
        gateway._template_initialized.add(model)
        gateway.proxy.proxy_request.return_value = JSONResponse({
            'request_id': 'infer-request',
            'status': 'completed',
            'result': {'samples': [
                {'sequences': [{'tokens': [token], 'stop_reason': 'stop'}]} for token in (11, 22)
            ]},
        })
        return gateway, TestClient(app), model

    @pytest.mark.parametrize('batch_model', [False, True])
    def test_conflicting_model_names_return_400_before_proxying(self, infer_gateway, batch_model):
        gateway, client, model = infer_gateway
        entries = [
            {'model': model, 'messages': [{'role': 'user', 'content': 'First'}]},
            {'model': 'another-model', 'messages': [{'role': 'user', 'content': 'Second'}]},
        ]
        if batch_model:
            body = {'request_config': {'model': model}, 'infer_requests': entries[1:]}
        else:
            body = entries
        response = client.post('/infer', json=body)

        assert response.status_code == 400
        error = response.json()['error']
        assert error['param'] == 'model'
        assert 'conflicting model names' in error['message'].lower()
        gateway.proxy.proxy_request.assert_not_called()

    @pytest.mark.parametrize('mode', ['entries', 'batch', 'both', 'first_entry'])
    def test_same_model_batch_reaches_one_sampler(self, infer_gateway, mode):
        gateway, client, model = infer_gateway
        entries = [
            {'messages': [{'role': 'user', 'content': 'First'}]},
            {'messages': [{'role': 'user', 'content': 'Second'}]},
        ]
        if mode in ('entries', 'both', 'first_entry'):
            entries[0]['model'] = model
        if mode in ('entries', 'both'):
            entries[1]['model'] = model
        body = {'request_config': {'model': model}, 'infer_requests': entries} if mode in ('batch', 'both') else entries
        response = client.post('/infer', json=body)

        assert response.status_code == 200
        assert [row['sequences'][0]['tokens'] for row in response.json()] == [[11], [22]]
        gateway.proxy.proxy_request.assert_awaited_once()
        forwarded = gateway.proxy.proxy_request.call_args.kwargs
        assert forwarded['base_model'] == model
        assert forwarded['endpoint'] == 'twinkle/sample'
        assert len(json.loads(forwarded['body_override'])['inputs']) == 2


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

    def test_alias_cannot_reuse_a_registered_name(self, monkeypatch):
        from twinkle.server.gateway import app as gateway_app

        monkeypatch.setattr(gateway_app, 'get_server_state', MagicMock())
        with pytest.raises(ValueError, match='conflict with registered model names: model-a'):
            gateway_app.GatewayServer(
                supported_models=['model-a', 'model-b'], openai_model_aliases={'model-a': 'model-b'})

    def test_reconfigure_keeps_adapters_and_updates_aliases(self, monkeypatch):
        from twinkle.server.gateway import app as gateway_app

        monkeypatch.setattr(gateway_app, 'get_server_state', MagicMock())
        gateway = gateway_app.GatewayServer(
            supported_models=['model-a', 'model-b'], openai_model_aliases={'alias-a': 'model-a'})
        gateway._template_initialized.update(['model-a', 'model-b'])
        adapter = {'model_name': 'adapter-b', 'base_model': 'model-b'}
        gateway.reconfigure({
            'supported_models': ['model-b', adapter],
            'openai_model_aliases': {'alias-b': 'model-b'},
        })
        assert gateway.supported_model_names == {'model-b'}
        assert gateway._template_initialized == {'model-b'}
        assert gateway.get_openai_model_catalog() == {
            'models': [{'model_name': 'model-b'}, adapter],
            'aliases': {'alias-b': 'model-b'},
        }
