# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real SDK boundaries, plus subprocess SIGINT tests over a local HTTP server."""
import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import json
import pytest
import requests

from twinkle_client.exceptions import TaskFailedError, TwinkleHTTPError
from twinkle_client.http import ClientContext, ClientTransport
from twinkle_client.manager import TwinkleClient
from twinkle_client.resources import RolloutResources


class Response:
    url = 'http://unit-test'
    text = 'rejected'

    def __init__(self, body, status=200):
        self.body, self.status_code = body, status
        self.ok = status < 400

    def json(self):
        return self.body


class Session:
    def __init__(self):
        self.calls, self.closed, self.failure = [], False, None
        self.step_started, self.step_release = threading.Event(), threading.Event()
        self.sample_barrier = None
        self.block_step = False

    def post(self, url, **kwargs):
        endpoint = url.rsplit('/', 1)[-1]
        self.calls.append(endpoint)
        if endpoint == 'create':
            return Response({'training_handles': True})
        result = None
        if endpoint == 'add_adapter_to_model':
            result = {'training_id': 'training-1', 'adapter_name': 'slot'}
        if endpoint == 'sample':
            self.sample_barrier.wait(timeout=5)
            result = {'samples': []}
        if endpoint in ('forward_backward', 'calculate_metric'):
            result = {'result': {'loss': 0.1}}
        if endpoint == 'save':
            result = {'twinkle_path': 'twinkle://saved-policy'}
        if endpoint == 'retrieve_future' and self.failure == 'poll_rejected':
            return Response({'detail': 'polling rejected'}, 401)
        if endpoint == 'clip_grad_and_step':
            self.step_started.set()
            if self.block_step:
                assert self.step_release.wait(timeout=5)
            if self.failure == 'timeout':
                raise requests.Timeout('reply lost')
            if self.failure == 'rejected':
                return Response({'detail': 'request rejected'}, 400)
            if self.failure == 'request_timeout':
                return Response({'detail': 'upstream timed out'}, 408)
            if self.failure == 'failed_task':
                return Response({'request_id': 'step-1', 'status': 'failed',
                    'error': {'error': 'backend failed', 'category': 'server', 'error_code': 500,
                              'request_id': 'step-1'}})
            if self.failure == 'poll_rejected':
                return Response({'request_id': 'step-1', 'status': 'running'})
        return Response({'request_id': endpoint, 'status': 'completed', 'result': result})

    def close(self):
        assert not self.block_step or self.step_release.is_set()
        self.closed = True


def client_for(session):
    transport = ClientTransport(ClientContext(base_url='http://unit-test', api_key='unit-test-key'), session=session)
    return TwinkleClient(transport=transport)


@pytest.mark.parametrize('failure,error_type', [
    ('timeout', requests.Timeout), ('failed_task', TaskFailedError), ('poll_rejected', TwinkleHTTPError),
    ('request_timeout', TwinkleHTTPError),
])
def test_unconfirmed_remote_operation_retains_resources_and_is_not_retried(failure, error_type):
    session = Session()
    client = client_for(session)
    resources = RolloutResources(lambda: client)
    closed = []
    with pytest.raises(error_type) as caught:
        with resources:
            resources.own_environment(SimpleNamespace(ensure_idle=lambda: None, close=lambda: closed.append('env')))
            model = client.model('model')
            model.add_adapter_to_model('slot')
            session.failure = failure
            model.clip_grad_and_step()
    assert not session.closed and not closed
    assert resources.client is client and resources._environments
    assert 'unconfirmed' in '\n'.join(caught.value.__notes__)
    assert model.completed_step_calls == 0
    assert session.calls.count('clip_grad_and_step') == 1
    with pytest.raises(RuntimeError, match='stopped'):
        model.clip_grad_and_step()
    with pytest.raises(BaseExceptionGroup, match='unconfirmed'):
        client.close()
    assert session.calls.count('clip_grad_and_step') == 1


@pytest.mark.parametrize('reject_locally', [False, True])
def test_known_rejection_does_not_quarantine_settled_client(reject_locally):
    session = Session()
    with pytest.raises(TypeError if reject_locally else TwinkleHTTPError):
        with RolloutResources(lambda: client_for(session)) as resources:
            model = resources.client.model('model')
            model.add_adapter_to_model('slot')
            session.failure = 'rejected'
            model.clip_grad_and_step(**({'seq_id': 8} if reject_locally else {}))
    assert session.closed
    assert not resources._operation_scope.uncertain_errors
    assert session.calls.count('clip_grad_and_step') == (0 if reject_locally else 1)


def test_sampling_remains_concurrent_while_a_guarded_training_call_is_blocked():
    session = Session()
    session.sample_barrier = threading.Barrier(2)
    with RolloutResources(lambda: client_for(session)) as resources:
        model = resources.client.model('model')
        model.add_adapter_to_model('slot')
        sampler = resources.client.sampler('model')
        session.block_step = True
        with ThreadPoolExecutor(3) as pool:
            train = pool.submit(model.clip_grad_and_step)
            try:
                assert session.step_started.wait(timeout=5)
                samples = [pool.submit(sampler.sample, [{'input_ids': [1]}]) for _ in range(2)]
                assert [future.result(timeout=5) for future in samples] == [[], []]
                assert not train.done()
            finally:
                session.step_release.set()
            train.result(timeout=5)
        assert model.completed_step_calls == 1
    assert session.closed


def test_scopes_do_not_stop_another_clients_operations():
    a, b = Session(), Session()
    with RolloutResources(lambda: client_for(b)) as other:
        model_b = other.client.model('model')
        model_b.add_adapter_to_model('slot')
        with pytest.raises(requests.Timeout):
            with RolloutResources(lambda: client_for(a)) as cancelled:
                model_a = cancelled.client.model('model')
                model_a.add_adapter_to_model('slot')
                a.failure = 'timeout'
                model_a.clip_grad_and_step()
        model_b.clip_grad_and_step()
        assert model_b.completed_step_calls == 1 and model_a.completed_step_calls == 0
        assert not b.closed
    assert b.closed and not a.closed


def test_direct_client_close_waits_for_the_active_operation():
    session = Session()
    with RolloutResources(lambda: client_for(session)) as resources:
        client = resources.client
        model = client.model('model')
        model.add_adapter_to_model('slot')
        session.block_step = True
        closing = threading.Event()

        def close():
            closing.set()
            client.close()

        with ThreadPoolExecutor(2) as pool:
            train = pool.submit(model.clip_grad_and_step)
            try:
                assert session.step_started.wait(timeout=5)
                stopped = pool.submit(close)
                assert closing.wait(timeout=5)
                assert not stopped.done() and not session.closed
            finally:
                session.step_release.set()
            train.result(timeout=5)
            stopped.result(timeout=5)
        assert model.completed_step_calls == 1 and session.closed


def test_interrupt_after_enqueue_before_future_return_keeps_result_registration():
    session = Session()
    client = client_for(session)
    with pytest.raises(KeyboardInterrupt, match='after enqueue'):
        with RolloutResources(lambda: client) as resources:
            model = client.model('model')
            model.add_adapter_to_model('slot')
            submit = resources.operations._executor.submit

            def interrupted_submit(*args, **kwargs):
                submit(*args, **kwargs)
                raise KeyboardInterrupt('after enqueue')

            resources.operations._executor.submit = interrupted_submit
            model.clip_grad_and_step()
    assert model.completed_step_calls == 1
    assert session.calls.count('clip_grad_and_step') == 1
    assert session.closed


def test_a_second_scope_cannot_close_or_stop_an_already_owned_client():
    session = Session()
    client = client_for(session)
    with RolloutResources(lambda: client):
        with pytest.raises(ValueError, match='already belongs'):
            with RolloutResources(lambda: client):
                pytest.fail('client ownership cannot be transferred')
        assert not session.closed
        model = client.model('model')
        model.add_adapter_to_model('slot')
        model.clip_grad_and_step()
        assert model.completed_step_calls == 1
    assert session.closed


def test_recipe_trains_and_publishes_through_real_sdk_methods(monkeypatch):
    from twinkle_agentic.rollout.multi_turn import RolloutSample

    path = Path(__file__).parents[2] / 'cookbook/client/async_rl/streaming_multi_turn_grpo.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('scoped_training_recipe', path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    samples = [RolloutSample(
        encoded={'input_ids': [1, 10], 'labels': [10, -100]}, response_token_ids=[10],
        rollout_logprobs=[-0.1], response_loss_mask=[1], orm_reward=[reward], group_key='group',
        extra={'trajectory_idx': index, 'adapter_uri': 'policy-0'},
    ) for index, reward in enumerate((1, -1, 1, -1))]

    class Driver:
        def __init__(self):
            self.stats = SimpleNamespace(publishes=0, optimizer_steps=0)
            self.paths = []

        def __iter__(self):
            return iter([samples])

        def request_publish(self, push):
            self.paths.append(push(None))
            self.stats.publishes += 1

        def commit(self, steps):
            self.stats.optimizer_steps += steps

    session, driver = Session(), Driver()
    with RolloutResources(lambda: client_for(session)) as resources:
        model = resources.client.model('model')
        model.add_adapter_to_model('slot')
        recipe._train_loop(driver, model, 'test-policy')
        assert model.completed_step_calls == driver.stats.optimizer_steps == 1
        assert driver.paths == ['twinkle://saved-policy']
        assert model.last_checkpoint.twinkle_path == driver.paths[0]
    assert session.calls == ['create', 'add_adapter_to_model', 'forward_backward',
                             'clip_grad_and_step', 'calculate_metric', 'save']
    assert session.closed


@pytest.mark.parametrize('endpoint', [
    'add_adapter_to_model', 'clip_grad_and_step', 'save', 'poll', 'sampler_create', 'sampler_template', 'legacy',
])
def test_real_http_and_repeated_sigint_record_outcome_before_cleanup(endpoint):
    # Signals go to a child, so pytest and the user's terminal are never interrupted.
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), endpoint],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed == {'endpoint': endpoint, 'calls': 1, 'registered': True, 'closed': True}


def signal_scenario(endpoint):
    started, release = threading.Event(), threading.Event()
    target = {'poll': 'retrieve_future', 'legacy': 'clip_grad_and_step',
              'sampler_template': 'set_template'}.get(endpoint, endpoint)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            action = self.path.rsplit('/', 1)[-1]
            if action == 'create' and '/sampler/' in self.path:
                action = 'sampler_create'
            calls.append(action)
            if action == target and endpoint != 'legacy':
                started.set()
                assert release.wait(timeout=5)
            result = None
            if action in ('create', 'sampler_create'):
                body = {'training_handles': True}
            elif action == 'set_template':
                body = {'status': 'ok'}
            else:
                if action == 'add_adapter_to_model':
                    result = {'training_id': 'training-1', 'adapter_name': 'slot'}
                elif action == 'save':
                    result = {'twinkle_path': 'twinkle://saved-policy'}
                body = {'request_id': action, 'status': 'completed', 'result': result}
                if endpoint == 'poll' and action == 'clip_grad_and_step':
                    body = {'request_id': 'step-1', 'status': 'running'}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    client = TwinkleClient(transport=ClientTransport(
        ClientContext(base_url=f'http://127.0.0.1:{server.server_port}', api_key='unit-test-key')))

    def interrupt():
        assert started.wait(timeout=5)
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        release.set()

    signal_thread = None
    try:
        try:
            with RolloutResources(lambda: client) as resources:
                model = client.model('model')
                if endpoint != 'add_adapter_to_model':
                    model.add_adapter_to_model('slot')
                if endpoint == 'sampler_template':
                    sampler = client.sampler('model')
                if endpoint != 'legacy':
                    signal_thread = threading.Thread(target=interrupt, daemon=True)
                    signal_thread.start()
                if endpoint == 'add_adapter_to_model':
                    model.add_adapter_to_model('slot')
                elif endpoint == 'save':
                    model.save('policy-1')
                elif endpoint == 'legacy':
                    resources.operations.run(model.clip_grad_and_step)
                elif endpoint == 'sampler_create':
                    client.sampler('model')
                elif endpoint == 'sampler_template':
                    sampler.set_template('unit-template')
                else:
                    model.clip_grad_and_step()
                assert endpoint == 'legacy', 'expected KeyboardInterrupt after remote success'
        except KeyboardInterrupt:
            assert endpoint != 'legacy'
        assert model._training_ids == {'slot': 'training-1'}
        assert model.completed_step_calls == (1 if endpoint in ('clip_grad_and_step', 'legacy', 'poll') else 0)
        if endpoint == 'save':
            assert model.last_checkpoint.twinkle_path == 'twinkle://saved-policy'
        assert client.transport.closed
        assert 'cancel' not in calls
        print(json.dumps({  # noqa: T201 - subprocess result
            'endpoint': endpoint, 'calls': calls.count(target), 'registered': True, 'closed': True}))
    finally:
        release.set()
        if signal_thread is not None:
            signal_thread.join(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


if __name__ == '__main__':
    signal_scenario(sys.argv[1])
