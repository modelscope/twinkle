# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest

from twinkle.advantage import StateGraphAdvantage


MODULE_PATH = Path(__file__).resolve().parents[2] / 'cookbook/rl/graphcredit/gate1.py'
SPEC = importlib.util.spec_from_file_location('graphcredit_gate1', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate1
SPEC.loader.exec_module(gate1)


def test_grid_task_and_parse_action():
    task = gate1.make_task(7, grid_size=5, distance=2)
    assert abs(task.start[0] - task.goal[0]) + abs(task.start[1] - task.goal[1]) == 2
    assert gate1.parse_action(' <action>RIGHT</action> ') == 'right'
    assert gate1.parse_action('right') == 'right'
    assert gate1.parse_action('I choose a wall') is None


def test_wrong_move_ends_episode_without_progress():
    task = gate1.GridTask(
        task_uid=1,
        grid_size=5,
        distance=2,
        start=(2, 2),
        goal=(2, 4),
    )
    episode = gate1.Episode(task=task, uid='bad', position=task.start)

    gate1.step_episode(episode, 'left')

    assert episode.done is True
    assert episode.success is False
    assert episode.reward == 0.0


def _episode(task, actions):
    episode = gate1.Episode(task=task, uid='episode', position=task.start)
    for turn_id, action in enumerate(actions):
        pre_state = task.state_key(episode.position)
        start = 4 + turn_id * 4
        gate1.step_episode(episode, action)
        post_state = (
            task.state_key(episode.position)
            if episode.success or not episode.done else task.dead_state_key
        )
        episode.turns.append(
            gate1.EpisodeTurn(
                prompt='',
                action=action,
                pre_state_key=pre_state,
                post_state_key=post_state,
                action_token_start=start,
                action_token_end=start + 2,
                terminal_success=True if episode.success else False,
            ))
        if episode.done:
            break
    episode.input_ids = list(range(4 + len(episode.turns) * 4 + 2))
    return episode


def test_graph_credit_prefers_useful_intermediate_action():
    task = gate1.GridTask(
        task_uid=2,
        grid_size=5,
        distance=2,
        start=(2, 2),
        goal=(2, 4),
    )
    success = _episode(task, ['right', 'right'])
    failure = _episode(task, ['left'])
    trajectories = [
        success.to_trajectory(policy_version=1),
        failure.to_trajectory(policy_version=1),
    ]

    advantage = StateGraphAdvantage(beta_episode=0.0, beta_graph=1.0)
    result = advantage([1.0, 0.0], trajectories, num_generations=2)

    success_start = success.turns[0].action_token_start
    failure_start = failure.turns[0].action_token_start
    assert torch.all(result[0, success_start:success_start + 2] > 0)
    assert torch.all(result[1, failure_start:failure_start + 2] < 0)


def test_train_counts_completed_updates_but_not_skipped_attempts(monkeypatch):
    task = gate1.GridTask(
        task_uid=3,
        grid_size=5,
        distance=2,
        start=(2, 2),
        goal=(2, 4),
    )
    episode = _episode(task, ['right', 'right'])
    sampled = (task, [episode])
    samples = iter([None, sampled, None, sampled])
    policy_versions = []

    monkeypatch.setattr(gate1, 'evaluate', lambda *args, **kwargs: {
        'success_rate': 0.0,
        'invalid_rate': 0.0,
    })
    monkeypatch.setattr(gate1, 'reset_peak_hbm', lambda: None)
    monkeypatch.setattr(gate1, 'synchronize', lambda device: None)
    monkeypatch.setattr(gate1, 'peak_hbm_gb', lambda: 0.0)
    monkeypatch.setattr(gate1, 'sample_training_group', lambda *args, **kwargs: next(samples))

    def compute_advantages(mode, episodes, policy_version, group_size, args):
        policy_versions.append(policy_version)
        return torch.zeros(1), {'fallback_counts': {}}

    monkeypatch.setattr(gate1, 'compute_advantages', compute_advantages)
    monkeypatch.setattr(gate1, 'policy_gradient_update', lambda *args, **kwargs: (0.0, 1))
    args = SimpleNamespace(
        learning_rate=5e-5,
        steps=2,
        max_step_attempts=3,
        eval_every=0,
        seed=20261119,
        mode='graphcredit',
        group_size=2,
        max_grad_norm=1.0,
    )
    tokenizer = SimpleNamespace(pad_token_id=0)

    result = gate1.train(torch.nn.Linear(1, 1), tokenizer, args, torch.device('cpu'))

    assert result['completed_steps'] == 2
    assert result['attempted_steps'] == 4
    assert result['skipped_steps'] == 2
    assert policy_versions == [1, 2]
    assert [record['step'] for record in result['history'] if not record.get('skipped')] == [1, 2]
