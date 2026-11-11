# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from twinkle_client.resources import RolloutResources


@pytest.fixture
def recipe():
    path = Path(__file__).parents[2] / 'cookbook/client/twinkle/multi_turn_rollout.py'
    spec = importlib.util.spec_from_file_location('batch_multi_turn_recipe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('fail_setup', [False, True])
def test_batch_recipe_owns_partial_setup_and_attempts_every_close(recipe, monkeypatch, fail_setup):
    envs, closed = [], []

    class Env:
        def __init__(self, **kwargs):
            self.index = len(envs)
            envs.append(self)

        def reset(self):
            if fail_setup and self.index == 2:
                raise RuntimeError('setup failed')
            return SimpleNamespace(observation=f'hand-{self.index}')

        def ensure_idle(self):
            pass

        def close(self):
            closed.append(self.index)
            if fail_setup and self.index == 0:
                raise RuntimeError('close failed')

    monkeypatch.setattr(recipe, 'OpenEnv', Env)
    client = SimpleNamespace(close=lambda: closed.append('client'))

    def run():
        with RolloutResources(lambda: client) as resources:
            rows, managers, tools, environments = recipe.prepare_trajectories(3, resources)
            assert environments == envs
            assert [row['messages'][1]['content'] for row in rows] == ['hand-0', 'hand-1', 'hand-2']
            assert len({id(manager) for manager in managers}) == 3
            assert [manager.bound_envs()[0] for manager in managers] == envs
            assert [tool[0].tool_info()['function']['name'] for tool in tools] == ['play'] * 3

    if fail_setup:
        with pytest.raises(RuntimeError, match='setup failed') as caught:
            run()
        assert any('close failed' in note for note in caught.value.__notes__)
    else:
        run()
    assert closed == [0, 1, 2, 'client']


@pytest.mark.parametrize('backward_fails', [False, True])
def test_batch_train_keeps_policy_and_training_steps_visible(recipe, monkeypatch, backward_fails):
    client, model, sampler = Mock(), Mock(), Mock()
    client.model.return_value = model
    client.sampler.return_value = sampler
    model.save.side_effect = [SimpleNamespace(twinkle_path=f'policy-{index}') for index in range(3)]
    if backward_fails:
        model.forward_backward.side_effect = RuntimeError('backward failed')
    rolled = [{'rollout_logprobs': [-0.1], 'turns': 2} for _ in range(4)]
    rollout = Mock(return_value=rolled, is_idle=True)
    envs = []

    def prepare(count, resources):
        assert count == 4
        current = [resources.own_environment(Mock()) for _ in range(count)]
        envs.extend(current)
        tools = [[SimpleNamespace(episode_reward=reward)] for reward in (1, -1, 1, -1)]
        return [{} for _ in current], [None] * count, tools, current

    monkeypatch.setattr(recipe, 'MAX_STEPS', 2)
    monkeypatch.setattr(recipe, 'init_twinkle_client', lambda **kwargs: client)
    monkeypatch.setattr(recipe, 'Qwen3_5Template', lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(recipe, 'ClientMultiTurnRollout', lambda **kwargs: rollout)
    monkeypatch.setattr(recipe, 'prepare_trajectories', prepare)

    if backward_fails:
        with pytest.raises(RuntimeError, match='backward failed'):
            recipe.train()
    else:
        recipe.train()
    assert [call.kwargs['adapter_uri'] for call in rollout.call_args_list] == (
        ['policy-0'] if backward_fails else ['policy-0', 'policy-1'])
    assert model.clip_grad_and_step.call_count == (0 if backward_fails else 2)
    assert model.save.call_count == (1 if backward_fails else 3)
    for call in model.forward_backward.call_args_list:
        assert call.kwargs['inputs'] is rolled
        assert call.kwargs['old_logps'] == [[-0.1]] * 4
        assert len(call.kwargs['advantages']) == 4
    rollout.drain.assert_called_once()
    client.close.assert_called_once()
    for env in envs:
        env.close.assert_called_once()
