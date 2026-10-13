"""Shared route fixtures for sampler concurrency and stream lifecycle tests."""
from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from ray.exceptions import GetTimeoutError
from starlette.requests import Request
from tinker import types as tinker_types

from twinkle.data_format import SampledSequence, SampleResponse
from twinkle.protocol.types import SampleRequest
from twinkle.server.sampler.app import SamplerManagement
from twinkle.server.sampler.concurrency import SamplerConcurrency
from twinkle.server.sampler.tinker_handlers import _register_tinker_sampler_routes
from twinkle.server.sampler.twinkle_handlers import _register_twinkle_sampler_routes
from twinkle.server.task_queue.backend_gate import BackendGate
from twinkle.server.task_queue.config import TaskQueueConfig
from twinkle.server.task_queue.types import BackendBusyError
from twinkle.server.task_queue.worker import ComputeWorker


async def _until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), 3)



class _State:
    def __init__(self):
        self.records = {}

    async def store_future_status(self, request_id, status, model_id=None, **kwargs):
        self.records[request_id] = {'status': status, **kwargs}

    async def get_future(self, request_id):
        return self.records.get(request_id)



class _Limiter:
    async def check_and_record(self, *args):
        return True, None



class _Sampler:
    def __init__(self):
        self.weight = 0
        self.loads = []
        self.active = 0
        self.peak = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self._lock = threading.Lock()
        self.template_calls = 0

    def set_template(self, *args, **kwargs):
        self.template_calls += 1

    def load_full_weights_from_path(self, path):
        self.weight = int(path.rsplit('/', 1)[-1])
        self.loads.append(self.weight)

    def sample(self, inputs, sampling_params=None, **kwargs):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.entered.set()
        try:
            if not self.release.wait(3):
                raise TimeoutError('test did not release generation')
            token = self.weight or inputs[0]['input_ids'][0]
            return [SampleResponse(
                prompt_token_ids=list(item['input_ids']),
                sequences=[SampledSequence(stop_reason='length', tokens=[token], logprobs=[[(token, -0.1)]])],
            ) for item in inputs]
        finally:
            with self._lock:
                self.active -= 1



class _Service(SamplerManagement):
    def __init__(self, *, timeout=2, limit=128):
        self.model_id = 'mock-model'
        self.replica_id = 'sampler-test'
        self.state = _State()
        self.sampler = _Sampler()
        self.sampling_guard = SamplerConcurrency(limit)
        self.tinker_setup_lock = asyncio.Lock()
        self.tinker_template_set = True
        self._task_queue_config = TaskQueueConfig(execution_timeout=timeout, concurrent_task_limit=limit)
        self._rate_limiter = _Limiter()
        self._task_metrics = None
        self._deployment_name = None
        self._event_loop = None
        self._backend_gate = BackendGate()
        self._compute_worker = ComputeWorker(
            state=self.state, config=self._task_queue_config, task_metrics=None, deployment_name=None)
        app = FastAPI()
        _register_tinker_sampler_routes(app, lambda: self)
        _register_twinkle_sampler_routes(app, lambda: self)
        self.endpoints = {route.path: route.endpoint for route in app.routes}

    async def _on_request_start(self, request):
        return 'token'

    async def submit(self, dialect, token=7, path=None):
        request = Request({'type': 'http', 'headers': []})
        if dialect == 'tinker':
            body = tinker_types.SampleRequest(
                prompt=tinker_types.ModelInput.from_ints([token]), model_path=path,
                sampling_params=tinker_types.SamplingParams(max_tokens=1, temperature=0))
            ref = await self.endpoints['/tinker/asample'](request, body, self)
            return ref['request_id']
        body = SampleRequest(inputs=[{'input_ids': [token]}], adapter_uri=path,
                             sampling_params={'max_tokens': 1, 'temperature': 0})
        envelope = await self.endpoints['/twinkle/sample'](request, body, self)
        return envelope.request_id

    async def result(self, request_id):
        await _until(lambda: self.state.records[request_id]['status'] in ('completed', 'failed'))
        return self.state.records[request_id]

    async def close(self):
        self.sampler.release.set()
        self.sampling_guard.close()
        await self._compute_worker.stop()
        await _until(lambda: not self.sampling_guard._jobs)
        self._backend_gate.shutdown()



def _tokens(record, dialect):
    assert record['status'] == 'completed', record
    result = record['result']
    if dialect == 'tinker':
        return result.sequences[0].tokens
    return result['samples'][0]['sequences'][0]['tokens']
