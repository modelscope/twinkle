# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def recipe():
    path = Path(__file__).parents[2] / 'cookbook/client/async_rl/multi_turn_grpo.py'
    spec = importlib.util.spec_from_file_location('multi_turn_grpo_recipe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cancel_and_release(task, started, release):
    assert started.wait(timeout=5)
    task.cancel()
    task.cancel()
    release.set()


def test_protected_wait_registers_result_before_repeated_cancellation(recipe):
    started, release = threading.Event(), threading.Event()

    def save():
        started.set()
        assert release.wait(timeout=5)
        return SimpleNamespace(twinkle_path='twinkle://policy-1')

    async def run():
        stop_event = asyncio.Event()
        snapshots = []
        task = asyncio.create_task(
            recipe._await_operation(save, stop_event=stop_event, on_result=snapshots.append))
        await asyncio.to_thread(_cancel_and_release, task, started, release)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stop_event.is_set()
        return snapshots

    snapshots = asyncio.run(run())
    assert [item.twinkle_path for item in snapshots] == ['twinkle://policy-1']


@pytest.mark.parametrize('stage', ['backward', 'step'])
def test_train_group_cancellation_stops_later_operations(recipe, stage):
    started, release = threading.Event(), threading.Event()

    class Model:
        def __init__(self):
            self.calls = []

        def forward_backward(self, **kwargs):
            self.calls.append('backward')
            if stage == 'backward':
                started.set()
                assert release.wait(timeout=5)

        def clip_grad_and_step(self, **kwargs):
            self.calls.append('step')
            if stage == 'step':
                started.set()
                assert release.wait(timeout=5)

        def calculate_metric(self, **kwargs):
            self.calls.append('metric')
            return {'result': {}}

    class Advantages:
        @staticmethod
        def tolist():
            return [1.0, -1.0, 0.5, -0.5]

    model = Model()
    rolled = [{
        'response_token_ids': [10],
        'rollout_logprobs': [-0.1],
        'stop_reason': 'stop',
        'truncated': False,
    } for _ in range(4)]

    async def run():
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            recipe._train_group(model, rolled, [1.0, 0.0, 1.0, 0.0], lambda *args, **kwargs: Advantages(),
                                stop_event))
        await asyncio.to_thread(_cancel_and_release, task, started, release)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert model.calls == ([stage] if stage == 'backward' else ['backward', 'step'])


def test_environment_creation_cancellation_preserves_handle_and_skips_rollout(recipe, monkeypatch):
    started, release = threading.Event(), threading.Event()
    env = SimpleNamespace(sandbox_id='sandbox-1')
    attempt = SimpleNamespace(env=env)

    def create(on_created):
        on_created(env)
        started.set()
        assert release.wait(timeout=5)
        return attempt

    monkeypatch.setattr(recipe, '_create_attempt', create)

    class Rollout:
        calls = 0

        async def acall(self, *args, **kwargs):
            self.calls += 1
            return []

    rollout = Rollout()
    environments = []

    async def run():
        stop_event = asyncio.Event()
        task = asyncio.create_task(recipe._rollout_group(rollout, 'policy-0', environments, stop_event))
        await asyncio.to_thread(_cancel_and_release, task, started, release)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert environments == [env]
    assert rollout.calls == 0


def test_verification_cancellation_stops_later_verify_calls(recipe, monkeypatch):
    started, release = threading.Event(), threading.Event()

    class Env:
        def __init__(self):
            self.verify_calls = 0

        def verify(self):
            self.verify_calls += 1
            started.set()
            assert release.wait(timeout=5)
            return 0.0

    attempts = [SimpleNamespace(
        env=Env(), manager=object(), trajectory={'messages': []}) for _ in range(recipe.NUM_GENERATIONS)]
    created = []

    def create(on_created):
        attempt = attempts.pop(0)
        created.append(attempt)
        return attempt

    monkeypatch.setattr(recipe, '_create_attempt', create)

    class Rollout:
        async def acall(self, trajectories, **kwargs):
            return [{
                'response_token_ids': [10],
                'rollout_logprobs': [-0.1],
                'stop_reason': 'stop',
                'truncated': False,
            } for _ in trajectories]

    async def run():
        stop_event = asyncio.Event()
        task = asyncio.create_task(recipe._rollout_group(Rollout(), 'policy-0', [], stop_event))
        await asyncio.to_thread(_cancel_and_release, task, started, release)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert [attempt.env.verify_calls for attempt in created] == [1, 0, 0, 0]


def test_cleanup_repeated_cancellation_closes_environment_before_client(recipe):
    started, release = threading.Event(), threading.Event()
    events = []

    class Env:
        sandbox_id = 'sandbox-cleanup'

        @staticmethod
        def ensure_idle():
            return None

        @staticmethod
        def close():
            started.set()
            assert release.wait(timeout=5)
            events.append('env-closed')

    class Client:
        @staticmethod
        def close():
            events.append('client-closed')

    async def run():
        stop_event = asyncio.Event()
        task = asyncio.create_task(recipe._drain_cleanup(Client(), [Env()], set(), stop_event))
        await asyncio.to_thread(_cancel_and_release, task, started, release)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert events == ['env-closed', 'client-closed']


def test_cleanup_cancellation_does_not_replace_primary_error(recipe):
    started, release = threading.Event(), threading.Event()

    class Env:
        sandbox_id = 'sandbox-primary'

        @staticmethod
        def ensure_idle():
            return None

        @staticmethod
        def close():
            started.set()
            assert release.wait(timeout=5)

    class Client:
        closed = False

        def close(self):
            self.closed = True

    client = Client()

    async def run():
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            recipe._drain_cleanup(client, [Env()], set(), stop_event, propagate_cancel=False))
        await asyncio.to_thread(_cancel_and_release, task, started, release)
        return await task

    assert asyncio.run(run()) == []
    assert client.closed


def test_complete_recipe_train_runs_updates_and_closes_client(recipe, monkeypatch):
    calls = []

    class Model:
        def __getattr__(self, name):
            if name == 'save':
                def save(policy, **kwargs):
                    calls.append(('save', policy))
                    return SimpleNamespace(twinkle_path=f'twinkle://{policy}')
                return save
            if name == 'forward_backward':
                return lambda **kwargs: calls.append(('backward', len(kwargs['inputs'])))
            if name == 'clip_grad_and_step':
                return lambda **kwargs: calls.append(('step', 1))
            if name == 'calculate_metric':
                return lambda **kwargs: {'result': {}}
            return lambda *args, **kwargs: None

    class Sampler:
        @staticmethod
        def set_template(*args, **kwargs):
            return None

    class Client:
        def __init__(self):
            self.model_instance = Model()
            self.sampler_instance = Sampler()
            self.closed = False

        def model(self, model_id):
            return self.model_instance

        def sampler(self, model_id):
            return self.sampler_instance

        def close(self):
            self.closed = True

    class Advantages:
        @staticmethod
        def tolist():
            return [1.0, -1.0, 0.5, -0.5]

    client = Client()
    monkeypatch.setattr(recipe, 'init_twinkle_client', lambda **kwargs: client)
    monkeypatch.setattr(recipe, 'Qwen3_5Template', lambda **kwargs: SimpleNamespace(truncation_strategy='delete'))
    monkeypatch.setattr(recipe, 'ClientMultiTurnRollout', lambda *args, **kwargs: object())
    monkeypatch.setattr(recipe, 'GRPOAdvantage', lambda: lambda *args, **kwargs: Advantages())
    monkeypatch.setattr(recipe, 'MAX_GROUPS', 2)

    async def fake_rollout(rollout, uri, environments, stop_event):
        rolled = [{
            'response_token_ids': [10],
            'rollout_logprobs': [-0.1],
            'stop_reason': 'stop',
            'truncated': False,
        } for _ in range(recipe.NUM_GENERATIONS)]
        return rolled, [1.0, 0.0, 1.0, 0.0]

    monkeypatch.setattr(recipe, '_rollout_group', fake_rollout)
    asyncio.run(recipe.train())

    assert calls.count(('step', 1)) == 2
    assert [call for call in calls if call[0] == 'save'] == [
        ('save', 'policy-0'), ('save', 'policy-1'), ('save', 'policy-2')]
    assert client.closed


class _Files:

    def __init__(self):
        self.data = {}
        self.write_error = None
        self.read_error = None
        self.write_calls = 0

    def write(self, path, content):
        self.write_calls += 1
        self.data[path] = content
        if self.write_error is not None:
            raise self.write_error

    def read(self, path):
        if self.read_error is not None:
            raise self.read_error
        return self.data[path]


def _make_env(recipe):
    env = object.__new__(recipe._TaskAgentEnv)
    env._workspace = '/workspace'
    env._sandbox = SimpleNamespace(sandbox_id='sandbox-test', files=_Files(), set_timeout=lambda timeout: None)
    env._task_infrastructure_error = None
    env._execution_lock = threading.RLock()
    env._active_commands = 0
    env._uncertain_execution = None
    env._command_timeout = 30
    env._refresh_timeout = False
    env._pre_tool_call = None
    env._custom_handlers = {}
    env._custom_tools = []
    env._include_default_tools = True
    return env


def test_file_side_effect_response_loss_is_unknown_and_not_retried(recipe):
    env = _make_env(recipe)
    env.sandbox.files.write_error = ConnectionError('response lost')

    result = env.step('write_file', {'path': 'solution.py', 'content': 'new content'})

    assert result.info['error']
    assert env.sandbox.files.write_calls == 1
    assert env.sandbox.files.data['/workspace/solution.py'] == 'new content'
    with pytest.raises(recipe._TaskInfrastructureError, match='tool:write_file'):
        env.ensure_idle()


def test_read_failure_and_local_model_error_have_distinct_health_semantics(recipe):
    env = _make_env(recipe)
    env.sandbox.files.read_error = ConnectionError('read lost')
    result = env.step('read_file', {'path': 'solution.py'})
    assert result.info['error']
    with pytest.raises(recipe._TaskInfrastructureError, match='tool:read_file'):
        env.ensure_idle()

    clean = _make_env(recipe)
    refusal = clean.step('write_file', {'path': '../outside', 'content': 'x'})
    assert refusal.observation.startswith('Error:')
    assert not refusal.info
    clean.ensure_idle()
    assert clean.sandbox.files.write_calls == 0


def test_verification_upload_and_command_failures_are_not_low_rewards(recipe):
    env = _make_env(recipe)
    env.sandbox.files.write_error = ConnectionError('upload failed')
    with pytest.raises(recipe._TaskInfrastructureError, match='verify_upload'):
        env.verify()

    command_env = _make_env(recipe)
    command_env._execute = lambda command, cwd, timeout: (1, 'verifier did not start')
    with pytest.raises(recipe._TaskInfrastructureError, match='verify_command'):
        command_env.verify()


def test_verification_result_must_match_call_and_statistics(recipe):
    env = _make_env(recipe)
    payload = {'call_id': 'wrong-call', 'completed': True, 'passed': 5, 'total': 5}
    env._execute = lambda command, cwd, timeout: (0, 'TWINKLE_VERIFY=' + json.dumps(payload))
    with pytest.raises(recipe._TaskInfrastructureError, match='verify_result'):
        env.verify()


def test_candidate_exceptions_receive_low_reward_only_from_valid_verifier(recipe, tmp_path):
    env = _make_env(recipe)
    (tmp_path / 'solution.py').write_text(
        'def smallest_positive(values):\n    raise ValueError("candidate failed")\n', encoding='utf-8')

    def execute(command, cwd, timeout):
        uploaded = next(path for path in env.sandbox.files.data if path.startswith('/tmp/twinkle_scripts/verify-'))
        source = env.sandbox.files.data[uploaded]
        script = tmp_path / 'verify.py'
        script.write_text(source, encoding='utf-8')
        result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    env._execute = execute
    assert env.verify() == 0.0


def test_verification_timeout_is_infrastructure_unknown(recipe):
    env = _make_env(recipe)

    class Commands:
        @staticmethod
        def run(command, cwd, timeout):
            raise TimeoutError('provider wait expired')

    env.sandbox.commands = Commands()
    with pytest.raises(recipe._TaskInfrastructureError, match='verify_command'):
        env.verify()
    assert env._uncertain_execution is not None


def test_training_row_validation_rejects_missing_or_misaligned_probabilities(recipe):
    with pytest.raises(RuntimeError, match='logprobs'):
        recipe._validate_training_rows([{
            'response_token_ids': [10, 11],
            'rollout_logprobs': [-0.1],
            'stop_reason': 'stop',
            'truncated': False,
        }])
