# Copyright (c) ModelScope Contributors. All rights reserved.
"""Streaming training and embedded Blackjack, without a model service."""
import importlib.util
import contextlib
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from twinkle_agentic.rollout.multi_turn import RolloutSample
from twinkle_agentic.rollout.base import SampleHandle
from twinkle_client.operations import SequentialOperations


@pytest.fixture
def recipe():
    path = Path(__file__).parents[2] / 'cookbook/client/async_rl/streaming_multi_turn_grpo.py'
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

    def forward_backward(self, **kwargs):
        self.calls.append(('backward', len(kwargs['inputs'])))

    def clip_grad_and_step(self, **kwargs):
        self.calls.append(('step', 1))

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
    model, operations = Model(), SequentialOperations()
    try:
        recipe._train_loop(driver, model, operations)
    finally:
        operations.close()
    assert driver.commits == [1, 1]
    assert driver.stats.optimizer_steps == 2
    assert driver.published == [f'twinkle://{recipe.RUN_ID}-policy-1']
    assert [call for call in model.calls if call[0] == 'step'] == [('step', 1), ('step', 1)]


def test_mixed_or_truncated_groups_are_not_trained(recipe):
    mixed, truncated = group('a', [1, 0, 1, 0]), group('b', [1, 0, 1, 0])
    mixed[-1].policy_version = 1
    truncated[0].truncated = True
    model, operations = Model(), SequentialOperations()
    driver = Driver([mixed + truncated])
    try:
        recipe._train_loop(driver, model, operations)
    finally:
        operations.close()
    assert driver.commits == [0]
    assert model.calls == []


def test_hard_failure_is_not_hidden_by_incomplete_or_truncated_group(recipe):
    samples = group('bad', [1, 0, 1, 0])
    samples[0].truncated = True
    samples[1].rollout_infos['error'] = 'sampling failed'
    with pytest.raises(RuntimeError, match='sampling failed'):
        recipe._validate_group(samples[:2])


def test_save_failure_never_commits_a_publish(recipe):
    model, operations = Model(), SequentialOperations()
    model.save = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('save failed'))
    driver = Driver([group('a', [1, -1, 1, -1])])
    try:
        with pytest.raises(RuntimeError, match='save failed'):
            recipe._train_loop(driver, model, operations)
    finally:
        operations.close()
    assert driver.stats.publishes == 0
    assert driver.commits == []


def test_blackjack_same_seed_same_initial_state_and_independent_progress(recipe):
    a, b = recipe.BlackjackEnv(42), recipe.BlackjackEnv(42)
    try:
        initial = a.reset()
        assert initial.observation == b.reset().observation
        assert 'Your cards:' in initial.observation
        result = a.step('play', {'action': 'stand'})
        assert result.done
        assert result.reward in (-1, 0, 1)
        # A's action must not advance B's independent chance sampler/game state.
        assert b.step('play', {'action': 'stand'}).reward == result.reward
    finally:
        a.close()
        b.close()


def test_driver_uses_environment_reward_and_rejects_unfinished_games(recipe, monkeypatch):
    seeds, calls = [], []

    class Env:
        def __init__(self, seed):
            seeds.append(seed)

        def reset(self):
            return SimpleNamespace(observation='same hand')

        def step(self, name, arguments):
            from twinkle_agentic.envs.base import StepResult
            return StepResult(observation='game over', done=True, reward=-1)

        def ensure_idle(self):
            pass

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
    driver = recipe._build_driver(Rollout(), 'policy-0')
    ref = SimpleNamespace(ordinal=0, row={'seed': 123})
    policy = SimpleNamespace(adapter_path='policy-0', version=0)
    h0, h1 = driver._submit(ref, 0, policy), driver._submit(ref, 1, policy)
    assert seeds == [123, 123]
    assert calls[0][0] == calls[1][0]
    assert calls[0][1]['sampling_params'].seed != calls[1][1]['sampling_params'].seed
    calls[0][1]['tool_manager']({'function': {'name': 'play', 'arguments': {'action': 'stand'}}})
    assert driver._collect(h0).orm_reward == [-1]
    assert driver._collect(h1).truncated
    assert driver._version_control is True
    assert driver._initial_adapter_path() == 'policy-0'
    assert driver._consume is None
    driver.abort()


def test_streaming_import_does_not_load_tq(recipe):
    code = '''import importlib.abc, importlib.util, sys
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
    operations = SequentialOperations()
    try:
        recipe._train_loop(driver, Model(), operations)
    finally:
        driver.abort()
        operations.close()
    assert driver.stats.optimizer_steps == 2
    assert [path for ordinal, path in admitted if ordinal == 1] == [f'twinkle://{recipe.RUN_ID}-policy-1'] * 4


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

        @contextlib.contextmanager
        def streaming(self, value):
            assert value is driver
            yield value

    monkeypatch.setattr(recipe, 'init_twinkle_client', lambda: client)
    monkeypatch.setattr(recipe, 'Qwen3_5Template', lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(recipe, 'ClientMultiTurnRollout', lambda *a, **k: Rollout())
    monkeypatch.setattr(recipe, '_build_driver', lambda *a: driver)
    recipe.train()
    assert closed == [True]
    assert driver.commits == [1]
    assert [name for action, name in model.calls if action == 'save'] == [
        f'{recipe.RUN_ID}-policy-0', f'{recipe.RUN_ID}-policy-1']


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
