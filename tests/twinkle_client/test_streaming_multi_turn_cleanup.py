# Copyright (c) ModelScope Contributors. All rights reserved.
"""Streaming training and embedded Blackjack, without a model service."""
import contextlib
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from twinkle_agentic.rollout.base import SampleHandle
from twinkle_agentic.rollout.multi_turn import RolloutSample


@pytest.fixture
def recipe(monkeypatch):
    path = Path(__file__).parents[2] / 'cookbook/client/async_rl/streaming_multi_turn_grpo.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('streaming_blackjack_recipe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def group(key, rewards):
    return [RolloutSample(encoded={'input_ids': [1, 10], 'labels': [10, -100]},
            response_token_ids=[10], rollout_logprobs=[-0.1], response_loss_mask=[1],
            orm_reward=[reward], policy_version=0,
            extra={'trajectory_idx': index, 'adapter_uri': 'policy-0'}, group_key=key)
            for index, reward in enumerate(rewards)]


class Model:
    def __init__(self):
        self.calls = []
        self.completed_step_calls = 0

    def forward_backward(self, **kwargs):
        self.calls.append(('backward', len(kwargs['inputs'])))

    def clip_grad_and_step(self, **kwargs):
        self.calls.append(('step', 1))
        self.completed_step_calls += 1

    def calculate_metric(self, **kwargs):
        return {'loss': 0.5}

    def save(self, name, **kwargs):
        self.calls.append(('save', name))
        return SimpleNamespace(twinkle_path=f'twinkle://{name}')


class Driver:
    def __init__(self, batches):
        self.batches = batches
        self.commits = []
        self.stats = SimpleNamespace(publishes=0, optimizer_steps=0)
        self.published = []

    def __iter__(self):
        return iter(self.batches)

    def request_publish(self, push):
        self.published.append(push(None))
        self.stats.publishes += 1

    def commit(self, steps):
        self.commits.append(steps)
        self.stats.optimizer_steps += steps


def test_variable_pull_zero_advantage_budget_and_publish(recipe):
    driver = Driver([
        group('a', [1, -1, 1, -1]) + group('b', [1, 1, 1, 1]),
        group('c', [1, 0, 1, 0]) + group('d', [1, -1, 1, -1]),
    ])
    model = Model()
    recipe._train_loop(driver, model, 'test-run')
    assert driver.commits == [1, 1]
    assert driver.stats.optimizer_steps == 2
    assert driver.published == ['twinkle://test-run-policy-1']
    assert [call for call in model.calls if call[0] == 'step'] == [('step', 1), ('step', 1)]


def test_mixed_or_truncated_groups_are_not_trained(recipe):
    mixed, truncated = group('a', [1, 0, 1, 0]), group('b', [1, 0, 1, 0])
    mixed[-1].policy_version = 1
    truncated[0].truncated = True
    model = Model()
    driver = Driver([mixed + truncated])
    recipe._train_loop(driver, model, 'test-run')
    assert driver.commits == [0]
    assert model.calls == []


def test_save_failure_never_commits_a_publish(recipe):
    model = Model()
    model.save = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('save failed'))
    driver = Driver([group('a', [1, -1, 1, -1])])
    with pytest.raises(RuntimeError, match='save failed'):
        recipe._train_loop(driver, model, 'test-run')
    assert driver.stats.publishes == 0
    assert driver.commits == []


def test_later_failure_is_not_hidden_by_the_update_budget(recipe, monkeypatch):
    monkeypatch.setattr(recipe, 'MAX_STEPS', 1)
    good, bad = group('a', [1, -1, 1, -1]), group('b', [1, -1, 1, -1])
    bad[0].rollout_infos['error'] = 'provider failed'
    model = Model()
    with pytest.raises(RuntimeError, match='provider failed'):
        recipe._train_loop(Driver([good + bad]), model, 'test-run')
    assert model.calls == []


def test_blackjack_same_seed_same_initial_state_and_independent_progress(recipe):
    a, b = recipe.BlackjackEnv(42), recipe.BlackjackEnv(42)
    try:
        initial = a.reset()
        assert initial.observation == b.reset().observation
        assert 'Your cards:' in initial.observation
        assert a.system_prompt == b.system_prompt
        tools = a.tools()
        assert tools[0]['function']['name'] == 'play'
        tools[0]['function']['parameters']['properties']['action']['enum'].append('cheat')
        assert b.tools()[0]['function']['parameters']['properties']['action']['enum'] == ['hit', 'stand']
        result = a.step('play', {'action': 'stand'})
        assert result.done
        assert result.reward in (-1, 0, 1)
        # A's action must not advance B's independent chance sampler/game state.
        assert b.step('play', {'action': 'stand'}).reward == result.reward
    finally:
        a.close()
        b.close()


def test_driver_uses_environment_reward_and_rejects_unfinished_games(recipe, monkeypatch):
    seeds, calls, closed = [], [], []

    class Env(recipe.BlackjackTask):
        def __init__(self, seed):
            seeds.append(seed)

        def reset(self):
            return SimpleNamespace(observation='same hand')

        def step(self, name, arguments):
            from twinkle_agentic.envs.base import StepResult
            if arguments.get('action') == 'fail':
                return StepResult(observation='failed', info={'error': 'game process failed'})
            return StepResult(observation='game over', done=True, reward=-1)

        def ensure_idle(self):
            pass

        def close(self):
            closed.append(self)

    class Rollout:
        def admit(self, prompt, trajectory_idx, **kwargs):
            handle = SimpleNamespace(submission_id=str(trajectory_idx))
            calls.append((prompt, kwargs))
            return handle

        def poll(self, handles):
            return handles

        def collect(self, handle):
            return group('a', [0, 0, 0, 0])[int(handle.submission_id)]

    monkeypatch.setattr(recipe, 'BlackjackEnv', Env)
    resources = recipe.RolloutResources(lambda: SimpleNamespace(close=lambda: None))
    driver = recipe._build_driver(Rollout(), 'policy-0', resources)
    ref = SimpleNamespace(ordinal=0, row={'seed': 123})
    policy = SimpleNamespace(adapter_path='policy-0', version=0)
    h0, h1 = driver._submit(ref, 0, policy), driver._submit(ref, 1, policy)
    assert seeds == [123, 123]
    assert calls[0][0] == calls[1][0]
    assert calls[0][1]['tool_manager'].tool_infos() == recipe.BlackjackTask().tools()
    assert calls[0][1]['sampling_params'].seed != calls[1][1]['sampling_params'].seed
    calls[0][1]['tool_manager']({'function': {'name': 'play', 'arguments': {'action': 'stand'}}})
    assert driver._collect(h0).orm_reward == [-1]
    assert driver._collect(h1).truncated
    assert len(closed) == 2
    assert not resources._environments
    h2 = driver._submit(ref, 2, policy)
    calls[2][1]['tool_manager']({'function': {'name': 'play', 'arguments': {'action': 'fail'}}})
    failed = driver._collect(h2)
    with pytest.raises(RuntimeError, match='game process failed'):
        recipe.training_groups([failed], num_generations=recipe.NUM_GENERATIONS, drop_truncated=True)
    assert len(closed) == 3 and not resources._environments
    driver.abort()
    resources.close()


@pytest.mark.parametrize('failure', ['reset', 'empty', 'multiple', 'renamed'])
def test_failed_episode_setup_closes_environment_without_admission(recipe, failure):
    closed, admitted = [], []

    class Env(recipe.BlackjackTask):
        def __init__(self, seed):
            pass

        def reset(self):
            if failure == 'reset':
                raise RuntimeError('setup failed')
            return SimpleNamespace(observation='cards')

        def tools(self):
            if failure == 'empty':
                return []
            tools = super().tools()
            if failure == 'multiple':
                extra = super().tools()[0]
                extra['function']['name'] = 'inspect_hand'
                return tools + [extra]
            tools[0]['function']['name'] = 'other'
            return tools

        def ensure_idle(self):
            pass

        def close(self):
            closed.append(self)

    rollout = SimpleNamespace(poll=lambda handles: [], admit=lambda *a, **k: admitted.append(True))
    with recipe.RolloutResources(lambda: SimpleNamespace(close=lambda: None)) as resources:
        driver = recipe._build_driver(rollout, 'policy-0', resources, Env)
        error_type = RuntimeError if failure == 'reset' else ValueError
        message = 'setup failed' if failure == 'reset' else 'adapt submit/collect'
        try:
            with pytest.raises(error_type, match=message):
                driver._submit(SimpleNamespace(ordinal=0, row={'seed': 1}), 0, None)
        finally:
            driver.abort()
    assert admitted == [] and len(closed) == 1


@pytest.mark.parametrize('response', [(1, 'game process failed'), (0, 'invalid JSON')])
def test_e2b_process_failure_is_retained_instead_of_becoming_zero_reward(recipe, monkeypatch, response):
    path = Path(__file__).parents[2] / 'cookbook/client/async_rl/e2b_multi_turn_grpo.py'
    spec = importlib.util.spec_from_file_location('e2b_recipe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('E2B_API_KEY', 'unit-test-key')
    env = module.E2BBlackjackEnv(42)
    commands = []
    env.run_script = lambda *a, **k: (commands.append(a) or response)
    tool = recipe.EnvTool.from_schemas(env, env.tools())[0]
    from twinkle_agentic.tools.tool_manager import ToolManager

    manager = ToolManager([tool])
    assert [item['function']['name'] for item in manager.tool_infos()] == ['play']
    call = {'function': {'name': 'play', 'arguments': {'action': 'stand'}}}
    manager.call_many([call, call])
    assert tool.last_result.info['error']
    assert not tool.done
    assert len(commands) == 1  # An infrastructure failure stays failed, even if the model retries.


def test_streaming_import_does_not_load_tq(recipe):
    code = '''import importlib.abc, importlib.util, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
class BlockTQ(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'transfer_queue', 'tensordict'} or fullname.startswith('twinkle_agentic.async_rl.pipelineTQ'):
            raise ImportError(fullname)
sys.meta_path.insert(0, BlockTQ())
spec = importlib.util.spec_from_file_location('blackjack_recipe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('ok')
'''
    result = subprocess.run([sys.executable, '-c', code, recipe.__file__], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_new_uri_is_used_by_later_groups(recipe):
    admitted = []

    class Rollout:
        def submit(self, ref, index, policy):
            admitted.append((ref.ordinal, policy.adapter_path))
            return SampleHandle(str(len(admitted)), ref.ordinal, index,
                                policy_version=policy.version, adapter_path=policy.adapter_path)

        def poll(self, handles):
            return handles

        def collect(self, handle):
            sample = group('unused', [1, -1, 1, -1])[handle.trajectory_idx]
            sample.policy_version = handle.policy_version
            sample.extra['adapter_uri'] = handle.adapter_path
            return sample

        def stop(self, submission_id):
            return True

    rollout = Rollout()
    driver = recipe.StreamingDriver(prompt_stream=[1, 2], max_staleness=1, num_generations=4,
        groups_per_partition=1, buffer_depth=4, group_pull=True, need_rows=4, version_control=True,
        rollout=rollout, submit=rollout.submit, poll=rollout.poll, collect=rollout.collect,
        initial_adapter_path=lambda: 'policy-0', prune=lambda: None, consume=None, max_steps=2)
    try:
        recipe._train_loop(driver, Model(), 'test-run')
    finally:
        driver.abort()
    assert driver.stats.optimizer_steps == 2
    assert [path for ordinal, path in admitted if ordinal == 1] == ['twinkle://test-run-policy-1'] * 4


def test_complete_train_keeps_configuration_and_closes_client(recipe, monkeypatch):
    class TrainingModel(Model):
        def __getattr__(self, name):
            if name.startswith(('set_', 'add_')):
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    model = TrainingModel()
    client = SimpleNamespace(model=lambda _: model,
        sampler=lambda _: SimpleNamespace(set_template=lambda *a, **k: None))
    closed = []
    client.close = lambda: closed.append(True)
    driver = Driver([group('a', [1, -1, 1, -1])])

    class Rollout:
        is_idle = True

        def drain(self):
            pass

        @contextlib.contextmanager
        def streaming(self, value):
            assert value is driver
            yield value

    monkeypatch.setenv('TWINKLE_SERVER_URL', 'http://recipe-server:8000')
    monkeypatch.setenv('TWINKLE_SERVER_TOKEN', 'recipe-test-token')

    def connect(**kwargs):
        assert kwargs == {'base_url': 'http://recipe-server:8000', 'api_key': 'recipe-test-token'}
        return client

    monkeypatch.setattr(recipe, 'init_twinkle_client', connect)
    monkeypatch.setattr(recipe, 'Qwen3_5Template', lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(recipe, 'ClientMultiTurnRollout', lambda *a, **k: Rollout())
    monkeypatch.setattr(recipe, '_build_driver', lambda *a: driver)
    recipe.train()
    assert closed == [True]
    assert driver.commits == [1]
    names = [name for action, name in model.calls if action == 'save']
    prefix = names[0].removesuffix('-policy-0')
    assert prefix.startswith('blackjack-')
    assert names == [f'{prefix}-policy-0', f'{prefix}-policy-1']
    # Reusing the imported recipe starts a new run, not another policy-0 at the old URI.
    recipe.train()
    later = [name for action, name in model.calls if action == 'save'][2:]
    assert len(later) == 2 and set(names).isdisjoint(later)


def test_other_episode_progresses_while_training_waits_for_commit(recipe):
    training, progressed = threading.Event(), threading.Event()
    submitted, polls, ready = [], [], set()

    class Rollout:
        def submit(self, ref, index, policy):
            handle = SampleHandle(f'{ref.ordinal}-{index}', ref.ordinal, index)
            submitted.append(handle)
            if ref.ordinal == 0:
                ready.add(handle)
            else:
                def work():
                    assert training.wait(timeout=5)
                    ready.add(handle)
                    progressed.set()
                threading.Thread(target=work, daemon=True).start()
            return handle

        def poll(self, handles):
            polls.append(len(handles))
            return [h for h in handles if h in ready]

        def collect(self, handle):
            return group('unused', [1, -1, 1, -1])[handle.trajectory_idx]

        def stop(self, submission_id):
            return True

    rollout = Rollout()
    driver = recipe.StreamingDriver(prompt_stream=[1, 2], max_staleness=0, num_generations=2,
        groups_per_partition=1, buffer_depth=4, group_pull=True, need_rows=2, rollout=rollout,
        submit=rollout.submit, poll=rollout.poll, collect=rollout.collect, consume=None)
    iterator = iter(driver)
    try:
        assert len(next(iterator)) == 2
        before = (len(submitted), len(polls))
        training.set()
        assert progressed.wait(timeout=5)
        assert before == (len(submitted), len(polls))
        driver.commit(0)
        assert len(next(iterator)) == 2
        driver.commit(0)
        with pytest.raises(StopIteration):
            next(iterator)
    finally:
        driver.abort()
