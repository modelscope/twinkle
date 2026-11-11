# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle_agentic.envs import Env, EnvTool, StepResult
from twinkle_agentic.tools.tool_manager import ToolManager


def test_batched_env_tools_retain_their_own_last_result():
    class Game(Env):
        def step(self, name, arguments):
            return StepResult(observation=name, reward=arguments['reward'], done=bool(arguments['reward']))

    env = Game()
    hit, stand = EnvTool(env, tool_name='hit'), EnvTool(env, tool_name='stand')
    manager = ToolManager([hit, stand])
    results = manager.call_many([
        {'function': {'name': name, 'arguments': {'reward': reward}}}
        for name, reward in [('hit', 0), ('stand', -1), ('stand', 1)]
    ])
    assert results == ['hit', 'stand', 'stand']
    assert not hit.done and hit.episode_reward == 0
    assert stand.done and stand.episode_reward == 1
