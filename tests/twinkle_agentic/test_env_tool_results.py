# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle_agentic.envs import Env, EnvTool, StepResult
from twinkle_agentic.tools.tool_manager import ToolManager


def test_batched_env_tools_retain_their_own_last_result():
    class Game(Env):
        def _open(self, trajectory):
            return None, StepResult()

        def _step(self, state, name, arguments):
            return StepResult(observation=name, reward=arguments['reward'], done=bool(arguments['reward']))

    env = Game()
    env.reset('game')
    hit = EnvTool(env, 'game', tool_name='hit')
    stand = EnvTool(env, 'game', tool_name='stand')
    manager = ToolManager([hit, stand])
    results = manager.call_many([
        {'function': {'name': name, 'arguments': {'reward': reward}}}
        for name, reward in [('hit', 0), ('stand', -1), ('stand', 0)]
    ])
    assert results == ['hit', 'stand', 'stand']
    assert hit.last_result.observation == 'hit'
    assert stand.last_result.observation == 'stand'
    assert not stand.last_result.done
    assert not hit.done and hit.last_result.reward == 0
    assert stand.last_result.reward == 0
    assert manager.call_many([{'function': {'name': 'stand', 'arguments': {'reward': 1}}}]) == ['stand']
    assert stand.done and stand.last_result.reward == 1
    assert env.close('game') == 0
