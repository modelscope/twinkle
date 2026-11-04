# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from twinkle.data_format import SamplingParams

from .validation import (
    _EVALUATION_TASKS,
    ValidationConfig,
    _CancelRecorder,
    _cancel_tool_manager,
    _code_task,
    _evaluate,
    _load_recipe,
    _probe_logps,
    _timed_tool_manager,
    _Timeline,
    _wait_for_stage,
    cancel_evidence_gate,
    learning_gate,
    parameters_changed,
    pipeline_evidence_gate,
    parse_args,
    result_payload,
    run_cancel_validation,
    run_validation,
    slow_tool_isolated,
    training_overlaps_rollout,
    write_result,
)


def _config(tmp_path):
    return ValidationConfig(
        backend='transformers',
        phase='learning',
        server_url='http://127.0.0.1:8000',
        model_id='ms://Qwen/Qwen3.5-4B',
        template_model_id='/models/Qwen3.5-4B',
        agentenv_template='python-test',
        result_dir=str(tmp_path),
    )


def test_learning_gate_requires_updates_and_independent_improvement():
    assert learning_gate(updates=2, baseline=0.25, final=0.5, minimum_improvement=0.0)
    assert not learning_gate(updates=1, baseline=0.25, final=0.5, minimum_improvement=0.0)
    assert not learning_gate(updates=2, baseline=0.25, final=0.25, minimum_improvement=0.0)
    assert not learning_gate(updates=2, baseline=0.25, final=0.3, minimum_improvement=0.1)


def test_probe_logps_uses_the_full_rollout_group_for_dp():
    recipe = _load_recipe()
    calls = {}

    def forward_only(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(result=[-0.25, -0.5])

    model = SimpleNamespace(forward_only=forward_only)
    trajectories = [{'input_ids': [1, 2]}, {'input_ids': [3, 4]}]
    result = asyncio.run(_probe_logps(recipe, model, trajectories, asyncio.Event()))
    assert calls['inputs'] is trajectories
    assert len(calls['inputs']) == 2
    assert result == [-0.25, -0.5]


def test_fake_and_real_result_modes_are_explicit(tmp_path):
    config = _config(tmp_path)
    fake = result_payload(config, mode='fake', status='completed', pipeline_gate=True)
    real = result_payload(config, status='blocked', pipeline_gate=False)
    assert fake['mode'] == 'fake'
    assert real['mode'] == 'real'
    assert fake['status'] != real['status']


def test_result_file_contains_no_server_url_or_credentials(tmp_path):
    path = write_result(_config(tmp_path), status='blocked', pipeline_gate=False, learning_gate=None)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['mode'] == 'real'
    assert 'server_url' not in payload['config']
    assert 'api_key' not in payload
    assert 'TWINKLE_SERVER_TOKEN' not in payload


def _pipeline_evidence():
    return {
        'two_updates': True,
        'parameter_change': True,
        'multi_turn_feedback': True,
        'token_alignment': True,
        'distinct_snapshots': True,
        'new_policy_requests': True,
        'sync_async_equivalent': True,
        'slow_tool_isolated': True,
        'training_overlap': True,
        'environment_isolation': True,
        'cleanup_confirmed': True,
    }


def test_pipeline_gate_fails_when_any_required_evidence_is_missing():
    evidence = _pipeline_evidence()
    assert pipeline_evidence_gate(evidence)
    for key in evidence:
        incomplete = dict(evidence)
        incomplete[key] = False
        assert not pipeline_evidence_gate(incomplete), key


def test_cancel_gate_rejects_early_env_cancel_new_work_and_cleanup_failure():
    assert not cancel_evidence_gate({
        'stage_reached': False,
        'cancelled': True,
        'post_cancel_operations': [],
        'cleanup_issues': [],
    })
    assert not cancel_evidence_gate({
        'stage_reached': True,
        'cancelled': True,
        'post_cancel_operations': ['sample'],
        'cleanup_issues': [],
    })
    assert not cancel_evidence_gate({
        'stage_reached': True,
        'cancelled': True,
        'post_cancel_operations': [],
        'cleanup_issues': ['unknown execution'],
    })


def test_learning_without_frozen_threshold_is_blocked(tmp_path):
    with pytest.raises(ValueError, match='frozen'):
        asyncio.run(run_validation(_config(tmp_path)))


def test_cli_template_is_set_before_recipe_import(tmp_path, monkeypatch):
    config = replace(_config(tmp_path), min_success_improvement=0.1)
    monkeypatch.setenv('AGENTENV_TEMPLATE', 'stale-template')
    seen = []

    def stop_at_import():
        seen.append(os.environ['AGENTENV_TEMPLATE'])
        raise RuntimeError('stop before provider setup')

    monkeypatch.setattr('tests.server.integration.agentic_rl.validation._load_recipe', stop_at_import)
    with pytest.raises(RuntimeError, match='stop before'):
        asyncio.run(run_validation(config))
    with pytest.raises(RuntimeError, match='stop before'):
        asyncio.run(run_cancel_validation(config))
    assert seen == [config.agentenv_template, config.agentenv_template]


def test_evaluation_binds_held_out_tasks_and_distinct_attempt_seeds(tmp_path):
    config = _config(tmp_path)
    created = []
    seeds = []

    @dataclass(frozen=True)
    class CodeTask:
        filename: str
        function: str
        solution: str
        smoke_test: str
        prompt: str
        hidden_cases: tuple

    class Recipe:
        NUM_GENERATIONS = 4
        ADAPTER_NAME = 'adapter'

        @staticmethod
        def _ensure_running(stop_event):
            return None

        @staticmethod
        async def _await_operation(operation, *args, stop_event, on_result=None, **kwargs):
            result = operation(*args, **kwargs)
            if on_result is not None:
                on_result(result)
            return result

        @staticmethod
        def _create_attempt(task, on_created):
            created.append(task)
            env = SimpleNamespace(verify=lambda: 1.0)
            on_created(env)
            return SimpleNamespace(env=env, manager=object(), trajectory={'messages': []})

        @staticmethod
        def _validate_training_rows(rows):
            assert len(rows) == 4

    class Rollout:
        sampling_params = SamplingParams()

        async def acall(self, trajectories, *, sampling_params, **kwargs):
            assert len(trajectories) == 1
            seeds.append(sampling_params.seed)
            return [{}]

    Recipe.CodeTask = CodeTask
    rewards, evaluations = asyncio.run(_evaluate(Recipe(), Rollout(), 'policy-0', [], asyncio.Event(), config))
    assert len(rewards) == 16
    assert len(seeds) == len(set(seeds)) == 16
    assert {task.filename for task in created} == {'count_even.py', 'last_negative.py'}
    assert all(task.function == task.filename[:-3] for task in created)
    assert {item['task_id'] for item in evaluations} == {'count-even', 'last-negative'}


_CORRECT_SOLUTIONS = {
    'count-even': 'def count_even(values):\n    return sum(value % 2 == 0 for value in values)\n',
    'last-negative': ('def last_negative(values):\n    negatives = [value for value in values if value < 0]\n'
                      '    return negatives[-1] if negatives else None\n'),
}


def _verify_locally(recipe, task, source, tmp_path):
    env = object.__new__(recipe._TaskAgentEnv)
    env.task = task
    env._workspace = '/workspace'
    uploads = {}
    env._sandbox = SimpleNamespace(
        sandbox_id='sandbox-eval', files=SimpleNamespace(write=uploads.__setitem__), set_timeout=lambda timeout: None)
    env._task_infrastructure_error = None
    env._execution_lock = threading.RLock()
    env._active_commands = 0
    env._uncertain_execution = None
    env._command_timeout = 30
    env._refresh_timeout = False
    (tmp_path / task.filename).write_text(source, encoding='utf-8')

    def execute(command, cwd, timeout):
        script = tmp_path / 'verify.py'
        script.write_text(uploads.popitem()[1], encoding='utf-8')
        result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    env._execute = execute
    return env.verify()


@pytest.mark.parametrize('item', _EVALUATION_TASKS, ids=[item['id'] for item in _EVALUATION_TASKS])
def test_held_out_verifier_scores_its_own_file_function_and_cases(item, tmp_path):
    recipe = _load_recipe()
    task = _code_task(recipe, item)
    assert task.filename != recipe.TASK.filename
    assert _verify_locally(recipe, task, _CORRECT_SOLUTIONS[item['id']], tmp_path) == 1.0
    assert _verify_locally(recipe, task, task.solution, tmp_path) < 1.0


def test_parameter_change_requires_aligned_finite_probe_shift():
    assert parameters_changed([-1.0, -2.0], [-1.0, -2.1])
    assert not parameters_changed([-1.0, -2.0], [-1.0, -2.0])
    assert not parameters_changed([], [])
    assert not parameters_changed(None, [-1.0])
    assert not parameters_changed([-1.0], [-1.0, -2.0])
    assert not parameters_changed([-1.0], [float('nan')])


def test_timeline_evidence_requires_peer_progress_and_training_overlap():
    slow = {'kind': 'tool', 'start': 1.0, 'end': 5.0, 'slow': True}
    inside = {'kind': 'sample', 'start': 2.0, 'end': 3.0}
    straddling = {'kind': 'sample', 'start': 4.0, 'end': 6.0}
    assert slow_tool_isolated([slow, inside])
    assert not slow_tool_isolated([slow, straddling])
    assert not slow_tool_isolated([dict(slow, slow=False), inside])

    train = {'kind': 'train', 'start': 10.0, 'end': 12.0}
    assert training_overlaps_rollout([train, {'kind': 'tool', 'start': 11.0, 'end': 13.0}])
    assert not training_overlaps_rollout([train, {'kind': 'sample', 'start': 12.0, 'end': 13.0}])


def test_timed_tool_manager_slows_only_the_first_batch():
    class Base:
        def call_many(self, tool_calls):
            return ['ok'] * len(tool_calls)

    timeline = _Timeline()
    manager = _timed_tool_manager(Base, timeline, 0.05)()
    assert manager.call_many([{}]) == ['ok']
    assert manager.call_many([{}, {}]) == ['ok', 'ok']
    assert [span['slow'] for span in timeline.spans] == [True, False]
    assert timeline.spans[0]['end'] - timeline.spans[0]['start'] >= 0.05


def test_cancel_tool_stage_ignores_setup_writes_and_counts_batches_after_cancel():
    class Env:
        steps = 0

        def step(self, tool_name, arguments=None):
            type(self).steps += 1

    class Base:
        def call_many(self, tool_calls):
            return ['ok'] * len(tool_calls)

    recorder = _CancelRecorder('tool')
    recorder.release.set()
    manager = _cancel_tool_manager(Base, recorder)()
    Env().step('write_file', {'path': 'solution.py'})
    assert recorder.operations == []
    manager.call_many([{}, {}])
    assert recorder.started.is_set()
    recorder.mark_cancelled()
    manager.call_many([{}])
    assert recorder.operations == ['tool', 'tool']
    assert recorder.post_cancel_operations == ['tool']


def test_wait_for_stage_returns_without_helper_threads():
    recorder = _CancelRecorder('backward')

    async def finished_early():
        return None

    async def never_reaches_stage():
        await asyncio.sleep(10)

    async def run():
        early = asyncio.create_task(finished_early())
        await asyncio.sleep(0)
        assert not await _wait_for_stage(recorder, early, timeout=5)
        slow = asyncio.create_task(never_reaches_stage())
        started = time.monotonic()
        assert not await _wait_for_stage(recorder, slow, timeout=0.2)
        assert time.monotonic() - started < 2
        slow.cancel()

    threads = threading.active_count()
    asyncio.run(run())
    assert threading.active_count() <= threads


@pytest.mark.parametrize('stage', ['sample', 'tool', 'verify', 'backward', 'step', 'save', 'unreached'])
def test_cancel_runner_drains_real_recipe_call_sites_without_a_provider(stage, tmp_path, monkeypatch):
    from . import validation

    recipe = _load_recipe()
    created = []

    class Env:
        def __init__(self, task, **kwargs):
            self.task = task
            self.sandbox_id = f'fake-{len(created)}'
            self.reward = len(created) % 2 if stage != 'unreached' else 0.0
            self.closed = False
            created.append(self)

        def reset(self):
            return None

        def step(self, *args, **kwargs):
            return SimpleNamespace(info={})

        def verify(self):
            return float(self.reward)

        def ensure_idle(self):
            return None

        def close(self):
            self.closed = True

    class Manager:
        def __init__(self, tools):
            self.env = tools[0].env

        def tool_infos(self):
            return []

        def call_many(self, calls):
            for _ in calls:
                self.env.step('write_file', {'path': 'solution.py'})
            return ['ok'] * len(calls)

    class Model:
        def forward_backward(self, **kwargs):
            return None

        def clip_grad_and_step(self, **kwargs):
            return None

        def calculate_metric(self, **kwargs):
            return {'result': {}}

        def save(self, name, **kwargs):
            return SimpleNamespace(twinkle_path=f'twinkle://{name}')

    sampler = SimpleNamespace(sample=lambda: None)
    client = SimpleNamespace(closed=False)
    client.close = lambda: setattr(client, 'closed', True)

    class Rollout:
        async def acall(self, trajectories, tool_manager, **kwargs):
            stop_event = kwargs_stop_event
            await recipe._await_operation(sampler.sample, stop_event=stop_event)
            for manager in tool_manager:
                recipe._ensure_running(stop_event)
                await recipe._await_operation(manager.call_many, [{}, {}], stop_event=stop_event)
            return [{
                'response_token_ids': [1], 'rollout_logprobs': [-0.1], 'truncated': False, 'stop_reason': 'stop'
            } for _ in trajectories]

    kwargs_stop_event = asyncio.Event()

    async def build(*args):
        return [client], Model(), sampler, Rollout(), kwargs_stop_event

    monkeypatch.setattr(recipe, '_TaskAgentEnv', Env)
    monkeypatch.setattr(recipe, 'EnvTool', SimpleNamespace(
        from_schemas=lambda env, schemas: [SimpleNamespace(env=env) for _ in schemas]))
    monkeypatch.setattr(recipe, 'ToolManager', Manager)
    monkeypatch.setattr(validation, '_load_recipe', lambda: recipe)
    monkeypatch.setattr(validation, '_build_runtime', build)
    monkeypatch.setattr(validation, '_STAGE_TIMEOUT', 2.0)
    config = replace(_config(tmp_path), phase='cancel', cancel_stage='backward' if stage == 'unreached' else stage)
    result = asyncio.run(run_cancel_validation(config))
    assert result['pipeline_gate'] is (stage != 'unreached')
    assert result['post_cancel_operations'] == []
    assert result['stage_reached'] is (stage != 'unreached')
    assert len(created) == 4 and all(env.closed for env in created)
    assert client.closed


def test_openenv_attempt_keeps_virtual_workspace_tool_schemas(tmp_path, monkeypatch):
    from . import validation
    from . import openenv_adapter

    recipe = _load_recipe()

    class Env:
        def __init__(self, task, **kwargs):
            self.task = task

        def reset(self):
            return None

        def ensure_idle(self):
            return None

        def step(self, *args, **kwargs):
            return None

        def tools(self):
            return recipe.DEFAULT_TOOLS

    monkeypatch.setattr(validation, '_load_recipe', lambda: recipe)
    monkeypatch.setattr(openenv_adapter, 'OpenEnvTaskEnv', Env)
    config = replace(_config(tmp_path), environment_backend='openenv', openenv_url='http://not-used')
    prepared = validation._prepare_recipe(config)
    attempt = prepared._create_attempt(prepared.TASK, lambda env: None)
    functions = {info['function']['name']: info['function'] for info in attempt.trajectory['tools']}
    assert functions['write_file']['parameters']['required'] == ['path', 'content']
    assert functions['run_command']['parameters']['required'] == ['command']
    assert functions['read_file']['parameters']['properties']['path']['type'] == 'string'
    prompt = attempt.trajectory['messages'][0]['content']
    assert '/workspace' in prompt and 'python smoke_test.py' in prompt
    assert 'workspace modules' in prompt and 'without repeated reruns' in prompt


@pytest.mark.parametrize('invalid', [None, {'truncated': True}, {'stop_reason': 'abort'}, {'rollout_logprobs': []}])
def test_sync_async_parity_requires_valid_complete_rollouts(tmp_path, monkeypatch, invalid):
    from . import validation
    recipe = _load_recipe()
    created = []

    def create(task, on_created):
        env = SimpleNamespace(sandbox_id=f'session-{len(created)}')
        created.append(env)
        on_created(env)
        return SimpleNamespace(env=env, manager=None, trajectory={})

    monkeypatch.setattr(recipe, '_create_attempt', create)
    row = {'response_token_ids': [1], 'rollout_logprobs': [-0.1], 'stop_reason': 'stop', 'truncated': False}
    if invalid:
        row.update(invalid)

    class Rollout:
        sampling_params = validation.SamplingParams(max_tokens=32)

        def __call__(self, *args, **kwargs):
            return [dict(row)]

        async def acall(self, *args, **kwargs):
            return [dict(row)]

    operation = validation._sync_async_evidence(recipe, Rollout(), 'policy', [], asyncio.Event(), _config(tmp_path))
    if invalid:
        with pytest.raises(RuntimeError):
            asyncio.run(operation)
    else:
        result = asyncio.run(operation)
        assert result['sync_async_equivalent'] and result['environment_isolation']


def test_openenv_cli_and_results_cannot_claim_agentenv_security(tmp_path):
    config = parse_args([
        '--backend', 'transformers', '--phase', 'pipeline', '--result-dir', str(tmp_path),
        '--environment-backend', 'openenv', '--openenv-url', 'http://127.0.0.1:18762',
    ])
    assert config.environment_backend == 'openenv'
    assert config.gradient_accumulation_steps == 2
    payload = result_payload(config, pipeline_gate=True)
    assert payload['acceptance_scope'] == 'openenv_functional'
    assert payload['agentenv_security_gate'] is None
    assert 'openenv_url' not in payload['config']


def test_cancel_result_paths_keep_every_stage(tmp_path):
    config = replace(_config(tmp_path), phase='cancel')
    sample = write_result(replace(config, cancel_stage='sample'), pipeline_gate=False)
    tool = write_result(replace(config, cancel_stage='tool'), pipeline_gate=False)
    assert sample != tool and sample.exists() and tool.exists()


@pytest.mark.parametrize('changed', [True, False])
def test_update_gate_counts_only_complete_ga_cycles_with_probe_changes(changed, tmp_path, monkeypatch):
    from . import validation

    recipe = _load_recipe()
    calls = []

    class Model:
        micro_steps = 0

        def forward_backward(self, **kwargs):
            calls.append('backward')

        def clip_grad_and_step(self, **kwargs):
            self.micro_steps += 1
            calls.append('step')

        def calculate_metric(self, **kwargs):
            return {'result': {'grad_norm': 1.0}}

        def forward_only(self, **kwargs):
            assert kwargs['compute_loss'] is False
            completed = self.micro_steps // 2 if changed else 0
            return {'result': {'logps': [[-1.0 - completed * 0.1]]}}

        def save(self, name, **kwargs):
            calls.append(name)
            return SimpleNamespace(twinkle_path='twinkle://' + name)

    model = Model()
    client = SimpleNamespace(close=lambda: None)
    sampler = SimpleNamespace(sample=lambda *args, **kwargs: [])
    rollout = SimpleNamespace(sampler=sampler, _core=SimpleNamespace(response_callback=SimpleNamespace()))

    async def build(*args):
        return [client], model, sampler, rollout, asyncio.Event()

    async def groups(*args):
        return [{'response_token_ids': [1], 'rollout_logprobs': [-0.1]}] * 4, [1.0, 0.0, 1.0, 0.0]

    async def sync_async(*args):
        return {'sync_async_equivalent': False, 'environment_isolation': False}

    monkeypatch.setattr(recipe, '_rollout_group', groups)
    monkeypatch.setattr(validation, '_load_recipe', lambda: recipe)
    monkeypatch.setattr(validation, '_build_runtime', build)
    monkeypatch.setattr(validation, '_sync_async_evidence', sync_async)
    config = replace(_config(tmp_path), phase='pipeline', max_groups=2)
    result = asyncio.run(run_validation(config))
    assert calls.count('step') == 4
    assert result['updates'] == (2 if changed else 0)
    assert result['evidence']['two_updates'] is changed
    assert [call for call in calls if call.startswith('policy')] == (
        ['policy-0', 'policy-1', 'policy-2'] if changed else ['policy-0'])


@pytest.mark.skipif(
    not os.environ.get('AGENTENV_TEMPLATE') or not os.environ.get('TWINKLE_SERVER_URL'),
    reason='real AgentENV and Twinkle Server are not configured',
)
def test_real_agentenv_configuration_is_present():
    assert os.environ['AGENTENV_TEMPLATE']
    assert os.environ['TWINKLE_SERVER_URL'].startswith(('http://', 'https://'))
