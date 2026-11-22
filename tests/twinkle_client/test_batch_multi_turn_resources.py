# Copyright (c) ModelScope Contributors. All rights reserved.
"""RolloutResources ownership: partial trajectory setup must still close every owned env.

Reanchored from cookbook/client/twinkle/multi_turn_rollout.py::prepare_trajectories: the
cookbook predates the episode-per-session EnvTool API (EnvTool now requires episode_id) and
the retired ClientMultiTurnRollout, so its module no longer loads (pending migration, tracked
in the UT audit ledger). The contract under test is the SDK's, not the cookbook's: one owned
env per trajectory, a failed setup closes every env already owned (and reports close failures
as exception notes), and the client closes last.
"""
from types import SimpleNamespace

import pytest

from twinkle_agentic.envs import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_client.resources import RolloutResources


def _prepare_trajectories(n_trajectories, resources, env_factory):
    """One independent env + play tool per trajectory, mirroring the cookbook's shape."""
    trajectories, tool_managers, env_tools_list, environments = [], [], [], []
    for index in range(n_trajectories):
        env = resources.own_environment(env_factory())
        environments.append(env)
        initial_observation = env.reset().observation
        env_tools = [EnvTool(env, f'episode-{index}', tool_name='play', description='Take an action.')]
        tool_manager = ToolManager(env_tools)
        trajectories.append({
            'messages': [
                {'role': 'system', 'content': 'You are a skilled blackjack player.'},
                {'role': 'user', 'content': initial_observation},
            ],
            'tools': tool_manager.tool_infos(),
        })
        tool_managers.append(tool_manager)
        env_tools_list.append(env_tools)
    return trajectories, tool_managers, env_tools_list, environments


@pytest.mark.parametrize('fail_setup', [False, True])
def test_batch_recipe_owns_partial_setup_and_attempts_every_close(fail_setup):
    envs, closed = [], []

    class Env:
        def __init__(self):
            self.index = len(envs)
            envs.append(self)

        def reset(self):
            if fail_setup and self.index == 2:
                raise RuntimeError('setup failed')
            return SimpleNamespace(observation=f'hand-{self.index}')

        def ensure_idle(self):
            pass

        def shutdown(self):
            # RolloutResources releases owned envs through the Env lifecycle's shutdown().
            closed.append(self.index)
            if fail_setup and self.index == 0:
                raise RuntimeError('close failed')

    client = SimpleNamespace(close=lambda: closed.append('client'))

    def run():
        with RolloutResources(lambda: client) as resources:
            rows, managers, tools, environments = _prepare_trajectories(3, resources, Env)
            assert environments == envs
            assert [row['messages'][1]['content'] for row in rows] == ['hand-0', 'hand-1', 'hand-2']
            assert len({id(manager) for manager in managers}) == 3
            assert [tool[0].session[0] for tool in tools] == envs
            assert [tool[0].tool_info()['function']['name'] for tool in tools] == ['play'] * 3

    if fail_setup:
        with pytest.raises(RuntimeError, match='setup failed') as caught:
            run()
        assert any('close failed' in note for note in caught.value.__notes__)
    else:
        run()
    assert closed == [0, 1, 2, 'client']


# Deleted: test_batch_train_keeps_policy_and_training_steps_visible drove the cookbook train()
# over the retired ClientMultiTurnRollout call shape (per-call adapter_uri + drain). That rollout
# API was folded into HTTPSampler + local MultiTurnRollout; the surviving contract (policy
# publication ordering, step/save visibility, failure stopping admission) is pinned by
# test_client_orchestrated_grpo.py and the reanchored scoped-operations recipe test.
