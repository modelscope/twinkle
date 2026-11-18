# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import sys
from pathlib import Path

import torch

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
