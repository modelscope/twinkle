# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU checks of the cookbook using real Driver/Rollout/Env/Advantage components.

Only the model, tokenizer and game outcomes are deterministic test doubles. No
server, model download or paid sandbox is needed.
"""
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import json
import pytest

from twinkle.data_format import pack_user_data, user_data_get
from twinkle.data_format.sampling import SampledSequence, SampleResponse
from twinkle.template.contract import shift_values
from twinkle_agentic.utils.token_utils import append_ids
from twinkle_client.rollout import ClientCheckpointManager, HTTPSampler


@pytest.fixture
def recipe(monkeypatch):
    path = Path(__file__).parents[2] / 'cookbook/client/async_rl/streaming_multi_turn_grpo.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('streaming_blackjack_recipe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyTemplate:
    max_length = 4096
    truncation_strategy = 'delete'

    def encode(self, trajectory, **kwargs):
        return dict(trajectory, input_ids=[1, 2], labels=[-100, -100], completion_mask=[0, 0])

    def render_delta(self, *args, **kwargs):
        return [3]

    def finalize_appended_features(self, features):
        return [dict(row, labels=shift_values(row['labels'], -100),
                     completion_mask=shift_values(row['completion_mask'], 0)) for row in features]

    def parse_tool_call(self, text):
        return [json.loads(text)] if text.startswith('{') else []

    def tool_call_errors(self, text):
        return []


def game(seed, actions):
    return {'observation': json.dumps({'seed': seed, 'actions': actions}),
            'reward': (1.0 if actions[-1] == 'stand' else -1.0) if actions else 0.0,
            'done': bool(actions)}


class ServedSampler:
    def __init__(self, template):
        self.template = template
        self.starts = {}
        self.uris = []
        self.lock = threading.Lock()

    def set_template(self, *args, **kwargs):
        pass

    def sample(self, inputs, params, *, adapter_name, adapter_uri, token_only):
        with self.lock:
            self.uris.append(adapter_uri)
            feature = inputs[0]
            if feature['messages'][-1]['role'] == 'user':
                seed = json.loads(feature['messages'][-1]['content'])['seed']
                member = self.starts.get(seed, 0)
                self.starts[seed] = member + 1
                action = 'hit' if member % 2 else 'stand'
                text = json.dumps({'name': 'play', 'arguments': {'action': action}})
            else:
                text = 'Finished.'
            extended = append_ids(feature, [4], self.template, trainable=True)
            return [SampleResponse(sequences=[SampledSequence(
                tokens=[4], logprobs=[-0.25], decoded=text, stop_reason='stop', new_input_feature=extended)])]


@pytest.mark.parametrize('all_zero,backward_fails', [(False, False), (True, False), (False, True)])
def test_streaming_recipe_skips_publish_and_update_budget(recipe, monkeypatch, all_zero, backward_fails):
    # Three skipped groups exceed the initial one-group admission budget. A missing
    # publication after a skipped group would hang before the informative fourth group.
    monkeypatch.setattr(recipe, 'MAX_STALENESS', 0)
    monkeypatch.setattr(recipe, 'MAX_GROUPS', 4 if all_zero else 6)
    monkeypatch.setattr(recipe, 'MAX_STEPS', 1)
    monkeypatch.setattr(recipe, 'NUM_GENERATIONS', 2)

    def outcome(seed, actions):
        result = game(seed, actions)
        if all_zero or seed < recipe.BASE_SEED + 3:
            result['reward'] = 0.0
        return result

    monkeypatch.setattr(recipe, 'play', outcome)
    template = TinyTemplate()
    sampler = ServedSampler(template)
    model = Mock(completed_step_calls=0)
    saved = []

    def save(name, save_optimizer=False):
        saved.append((name, model.completed_step_calls))
        return SimpleNamespace(twinkle_path=name)

    def step(**kwargs):
        model.completed_step_calls += 1

    model.save.side_effect = save
    model.clip_grad_and_step.side_effect = step
    if backward_fails:
        model.forward_backward.side_effect = RuntimeError('backward failed')
    client = Mock()
    client.model.return_value = model
    client.sampler.return_value = sampler
    monkeypatch.setattr(recipe, 'init_twinkle_client', lambda **kwargs: client)
    monkeypatch.setattr(recipe, 'Qwen3_5Template', lambda **kwargs: template)
    drivers, envs = [], []
    original_build = recipe._build_driver

    def build(*args):
        driver = original_build(*args)
        drivers.append(driver)
        return driver

    def env_factory(**kwargs):
        env = recipe.BlackjackEnv(**kwargs)
        envs.append(env)
        return env

    monkeypatch.setattr(recipe, '_build_driver', build)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(recipe.train, env_factory=env_factory)
        try:
            if backward_fails:
                with pytest.raises(RuntimeError, match='backward failed'):
                    task.result(timeout=15)
            else:
                task.result(timeout=15)
        finally:
            # Also lets a version-budget regression fail by timeout instead of hanging pytest.
            for driver in drivers:
                driver.max_steps = 0
                driver.close()

    assert drivers[0].stats()['handed'] == 4
    assert not drivers[0]._thread.is_alive()
    assert not envs[0]._sessions
    client.close.assert_called_once()
    assert len(saved) == (4 if backward_fails else 5)
    assert len({name for name, _ in saved}) == len(saved)
    assert saved[:4] == [(saved[i][0], 0) for i in range(4)]
    assert len(set(sampler.uris)) >= 4
    expected_steps = int(not all_zero and not backward_fails)
    assert model.completed_step_calls == expected_steps
    assert model.clip_grad_and_step.call_count == expected_steps
    assert saved[-1][1] == expected_steps
    assert model.forward_backward.call_count == int(not all_zero)
    for call in model.forward_backward.call_args_list:
        args = call.kwargs
        assert len(args['inputs']) == 2  # Full group, never a single survivor.
        for feature, advantages, logps in zip(args['inputs'], args['advantages'], args['old_logps'], strict=True):
            assert len(feature['input_ids']) == len(feature['labels']) == len(feature['completion_mask'])
            assert sum(feature['completion_mask']) == len(advantages) == len(logps) == 2
            assert logps == [-0.25, -0.25]


def test_environment_sessions_are_isolated_and_terminal_reward_is_paid_once(recipe, monkeypatch):
    monkeypatch.setattr(recipe, 'play', game)
    env = recipe.BlackjackEnv(max_sessions=2)
    prompt = {'user_data': pack_user_data({'seed': 42})}
    try:
        assert env.reset('a', prompt).observation == env.reset('b', prompt).observation
        assert env.step('a', 'play', {'action': 'stand'}).reward == 1.0
        assert not env.episode_outcome('b')['game_done']
        assert env.step('a', 'play', {'action': 'hit'}).reward == 0.0
        assert env.step('b', 'play', {'action': 'hit'}).reward == -1.0
        assert env.close('a') == 1.0
        assert env.close('b') == -1.0
    finally:
        env.shutdown()
    assert not env._sessions


def test_unfinished_member_rejects_whole_group_before_training(recipe, monkeypatch):
    monkeypatch.setattr(recipe, 'play', game)
    monkeypatch.setattr(recipe, 'MAX_STALENESS', 0)
    monkeypatch.setattr(recipe, 'NUM_GENERATIONS', 2)
    monkeypatch.setattr(recipe, 'MAX_GROUPS', 2)
    template = TinyTemplate()
    sampler = ServedSampler(template)
    original_sample = sampler.sample

    def sample(inputs, *args, **kwargs):
        result = original_sample(inputs, *args, **kwargs)
        if user_data_get(inputs[0]['user_data'], 'seed') == recipe.BASE_SEED and sampler.starts[recipe.BASE_SEED] == 1:
            result[0].sequences[0].decoded = 'I am done.'  # No play call: game is still open.
        return result

    sampler.sample = sample
    http_sampler = HTTPSampler(sampler, template)
    manager = ClientCheckpointManager(Mock(save=Mock(return_value=SimpleNamespace(twinkle_path='p0'))), http_sampler)
    manager.sync_weights()
    env = recipe.BlackjackEnv(max_sessions=2)
    rollout = recipe.MultiTurnRollout(sampler=http_sampler, template=template, env=env,
                                     on_episode_end=recipe._episode_end)
    driver = recipe._build_driver(rollout, manager)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(lambda: next(iter(driver)))
            try:
                rows, _ = task.result(timeout=10)
            finally:
                driver.max_steps = 0
                driver.close()
        assert len(rows) == 2
        assert {user_data_get(row['user_data'], 'seed') for row in rows} == {recipe.BASE_SEED + 1}
        assert driver.stats()['dropped_group'] == 1
    finally:
        driver.close()
        rollout.close()
        env.shutdown()


def test_weight_publication_does_not_change_an_inflight_request():
    entered, release = threading.Event(), threading.Event()
    uris = []

    def sample(inputs, params, **kwargs):
        uris.append(kwargs['adapter_uri'])
        if len(uris) == 1:
            entered.set()
            assert release.wait(5)
        return [SampleResponse(sequences=[SampledSequence(
            tokens=[4], logprobs=[-0.1], decoded='ok', stop_reason='stop', new_input_feature={'input_ids': [1, 4]})])]

    http_sampler = HTTPSampler(SimpleNamespace(sample=sample), template=None)
    model = Mock()
    model.save.side_effect = [SimpleNamespace(twinkle_path=uri) for uri in ['p0', 'p1']]
    manager = ClientCheckpointManager(model, http_sampler)
    manager.sync_weights()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(http_sampler.sample, [{'input_ids': [1]}])
        try:
            assert entered.wait(5)
            manager.sync_weights()
            http_sampler.sample([{'input_ids': [1]}])
        finally:
            release.set()
        pending.result(timeout=5)
    assert uris == ['p0', 'p1']
    assert manager.version() == 2


@pytest.mark.parametrize('fail_upload', [False, True])
def test_e2b_sessions_reuse_and_cleanup_without_cloud(recipe, monkeypatch, fail_upload):
    import shlex

    import twinkle_agentic.envs.agentenv as agentenv

    path = Path(recipe.__file__).with_name('e2b_multi_turn_grpo.py')
    spec = importlib.util.spec_from_file_location('e2b_blackjack_recipe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('E2B_API_KEY', 'test-only')
    created = []

    class Sandbox:
        def __init__(self):
            self.sandbox_id = str(len(created))
            self.files = SimpleNamespace(write=self.write)
            self.commands = SimpleNamespace(run=self.run)
            self.killed = False
            self.actions = []
            self.seed = None
            created.append(self)

        @classmethod
        def create(cls, *args, **kwargs):
            return cls()

        def write(self, path, source):
            if fail_upload:
                raise RuntimeError('upload failed')

        def run(self, command, **kwargs):
            output = ''
            if command.startswith('python3 blackjack_game.py '):
                args = shlex.split(command)
                if args[2] == 'reset':
                    self.seed, self.actions = int(args[4]), []
                else:
                    self.actions.append(args[2])
                output = json.dumps(game(self.seed, self.actions))
            return SimpleNamespace(exit_code=0, stdout=output, stderr='')

        def set_timeout(self, timeout):
            pass

        def kill(self):
            self.killed = True

    monkeypatch.setattr(agentenv, '_require_e2b', lambda: Sandbox)
    env = module.E2BBlackjackEnv(max_sessions=1)
    try:
        if fail_upload:
            with pytest.raises(RuntimeError, match='upload failed'):
                env.reset('failed', {'user_data': pack_user_data({'seed': 42})})
            assert not env._sessions and created[0].killed
            fail_upload = False  # The failed reset must return its session permit.
        env.reset('first', {'user_data': pack_user_data({'seed': 42})}, wait=0)
        assert env.step('first', 'play', {'action': 'stand'}).reward == 1.0
        assert env.step('first', 'play', {'action': 'hit'}).reward == 0.0
        assert env.close('first') == 1.0
        count = len(created)
        env.reset('second', {'user_data': pack_user_data({'seed': 99})}, wait=0)
        assert len(created) == count  # The idle sandbox is reused with a fresh game.
        assert not env.episode_outcome('second')['game_done']
        assert env.step('second', 'play', {'action': 'hit'}).reward == -1.0
        assert env.close('second') == -1.0
    finally:
        env.shutdown()
    assert not env._sessions and not env._idle
    assert all(sandbox.killed for sandbox in created)


def test_cleanup_waits_for_inflight_generation_before_closing_client(recipe, monkeypatch):
    from twinkle_client.resources import RolloutResources

    monkeypatch.setattr(recipe, 'play', game)
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    template = TinyTemplate()
    sampler = ServedSampler(template)
    original_sample = sampler.sample

    def sample(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_sample(*args, **kwargs)

    sampler.sample = sample
    client = Mock()
    resources = RolloutResources(lambda: client)
    with ThreadPoolExecutor(max_workers=1) as pool:
        resources.__enter__()
        env = resources.own_environment(recipe.BlackjackEnv(max_sessions=1))
        rollout = recipe.MultiTurnRollout(sampler=HTTPSampler(sampler, template), template=template, env=env)
        tracked = resources.own_rollout(recipe._RolloutTasks(rollout))
        original_close = tracked.close

        def close():
            closing.set()
            original_close()

        tracked.close = close
        pending = tracked.step({'messages': [], 'user_data': pack_user_data({'seed': 42})}, episode_id='a')
        try:
            assert entered.wait(5)
            cleanup = pool.submit(resources.close)
            assert closing.wait(5)
            assert not cleanup.done()
            client.close.assert_not_called()
            assert 'a' in env._sessions
        finally:
            release.set()
        pending.result(timeout=5)
        cleanup.result(timeout=5)
    client.close.assert_called_once()
    assert not env._sessions and not rollout._episodes
