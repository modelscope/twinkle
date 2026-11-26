# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import json
import pytest
import pytest_asyncio

from tests.server.state.fake_backend import FakeBackend
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.deployment import build_deployment_app
from twinkle.server.gateway.tinker_handlers import _register_gateway_tinker_routes
from twinkle.server.sampler.concurrency import SamplerConcurrency
from twinkle.server.sampler.tinker_handlers import _register_tinker_sampler_routes
from twinkle.server.sampler.twinkle_handlers import _register_twinkle_sampler_routes
from twinkle.server.state.server_state import ServerState
from twinkle.server.task_queue.mixin import TaskQueueMixin


@pytest.fixture(scope="session")
def _ray_runtime():
    yield


@pytest_asyncio.fixture
async def sampler_queue():
    queue = TaskQueueMixin()
    queue.state = ServerState(backend=FakeBackend())
    queue.replica_id = 'sampler-test'
    queue.deploy_adapters = {}
    queue.sampling_guard = SamplerConcurrency()
    queue._init_task_queue(TaskQueueConfig(), deployment_name='Sampler', telemetry_base_model='base')
    calls = []

    async def on_request_start(request):
        return request.state.token

    async def compute(_, inputs, *args, **kwargs):
        calls.extend(item['input_ids'] for item in inputs)
        return [SimpleNamespace(
            sequences=[], prompt_token_ids=inputs[0]['input_ids'], prompt_logprobs=None, topk_prompt_logprobs=None)]

    queue._on_request_start = on_request_start
    queue.call_backend = compute
    queue.sampler = SimpleNamespace(sample=object())
    try:
        yield queue, calls
    finally:
        await queue.shutdown_task_queue()


@pytest.mark.asyncio
async def test_concurrent_sample_key_conflict_checks_the_winning_receipt(sampler_queue, monkeypatch):
    queue, calls = sampler_queue
    first_reads = 0
    both_missing = asyncio.Event()
    get_submission = queue.state.futures.get_submission

    async def read_submission(*args):
        nonlocal first_reads
        record = await get_submission(*args)
        first_reads += 1
        if first_reads <= 2:
            assert record is None
            if first_reads == 2:
                both_missing.set()
            await both_missing.wait()
        return record

    # Both route handlers must pass their early check before either can enqueue.
    monkeypatch.setattr(queue.state.futures, 'get_submission', read_submission)
    app = build_deployment_app('Sampler', lambda app, _: _register_twinkle_sampler_routes(app, lambda: queue))
    headers = {'x-request-id': 'call', 'Authorization': 'Bearer owner', 'Idempotency-Key': 'same-key'}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://sampler', headers=headers,
    ) as client:
        async with asyncio.timeout(3):
            replies = await asyncio.gather(*(
                client.post('/twinkle/sample', json={'inputs': [{'input_ids': [value]}]}) for value in (1, 2)
            ))
        accepted, conflict = sorted(replies, key=lambda r: r.status_code)
        assert (accepted.status_code, conflict.status_code) == (200, 400)
        assert 'different submission' in conflict.json()['error']
        prompt, = calls
        replay = await client.post('/twinkle/sample', json={'inputs': [{'input_ids': prompt}]})
        assert replay.json()['request_id'] == accepted.json()['request_id']
        assert calls == [prompt]


@pytest.mark.asyncio
async def test_tinker_retries_submission_and_polling_with_the_same_sequence(sampler_queue):
    from tinker import types
    from tinker._client import AsyncTinker
    from tinker.lib.internal_client_holder import InternalClientHolder
    from tinker.lib.public_interfaces.sampling_client import SamplingClient
    from tinker.lib.retry_handler import RetryConfig

    from twinkle_client import init_tinker_client

    init_tinker_client()
    queue, calls = sampler_queue
    await queue.state.create_sampling_session({'base_model': 'base'}, sampling_session_id='test-session')

    def routes(app, _):
        _register_tinker_sampler_routes(app, lambda: queue)
        _register_gateway_tinker_routes(app, lambda: queue)

    app = build_deployment_app('Sampler', routes)
    asgi = httpx.ASGITransport(app=app)
    sequences = []
    lost_poll = False

    async def send(request):
        nonlocal lost_poll
        if request.url.path == '/api/v1/asample':
            request.url = request.url.copy_with(path='/tinker/asample')
            sequences.append(json.loads(request.content)['seq_id'])
            response = await asgi.handle_async_request(request)
            if len(sequences) == 1:
                assert response.status_code == 200
                raise httpx.ReadTimeout('lost accepted reply', request=request)
            return response
        assert request.url.path == '/api/v1/retrieve_future'
        if not lost_poll:
            lost_poll = True
            return httpx.Response(410, request=request)
        request.url = request.url.copy_with(path='/retrieve_future')
        return await asgi.handle_async_request(request)

    class Holder:
        _sample_backoff_until = None
        _is_retryable_exception = staticmethod(InternalClientHolder._is_retryable_exception)
        execute_with_retries = InternalClientHolder.execute_with_retries
        run_coroutine_threadsafe = InternalClientHolder.run_coroutine_threadsafe
        get_loop = staticmethod(asyncio.get_running_loop)

        def get_telemetry(self):
            return None

        def estimate_bytes_count_in_model_input(self, prompt):
            return 4

        def sample_dispatch_rate_limit(self, _):
            return nullcontext()

        def aclient(self, _):
            return nullcontext(client)

    transport = httpx.MockTransport(send)
    async with httpx.AsyncClient(transport=transport) as http:
        client = AsyncTinker(
            base_url='http://sampler', api_key='owner', http_client=http,
            default_headers={'x-request-id': 'call', 'Authorization': 'Bearer owner'},
        )
        sampler = SamplingClient(
            Holder(), sampling_session_id='test-session', subprocess_sampling=False,
            retry_config=RetryConfig(retry_delay_base=0, jitter_factor=0),
        )
        prompt = types.ModelInput.from_ints([1])
        params = types.SamplingParams(max_tokens=1)
        async with asyncio.timeout(8):
            await sampler.sample_async(prompt, 1, params)
            await sampler.sample_async(prompt, 1, params)

    assert lost_poll
    assert sequences[:3] == [0, 0, 0]
    assert len(sequences) == 4 and sequences[3] != 0
    assert calls == [[1], [1]]
