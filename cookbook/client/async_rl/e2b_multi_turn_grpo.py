# Copyright (c) ModelScope Contributors. All rights reserved.
"""Streaming Blackjack GRPO with one official E2B sandbox per active episode.

From the repo root, run pip install -e '.[client,e2b]' and set E2B_API_KEY in .env.
Leave E2B_API_URL/E2B_SANDBOX_URL/E2B_DOMAIN unset for the official service.
The base template installs open_spiel inside each sandbox; a custom template
with open_spiel==2.0.2 preinstalled avoids this startup cost (E2B_TEMPLATE).

Run against the same Twinkle Model/Sampler server as streaming_multi_turn_grpo.py:
    python cookbook/client/async_rl/e2b_multi_turn_grpo.py

MultiTurnRollout runs the model/tool loop. AgentEnv runs the game in E2B;
the SDK resource scope owns sandbox cleanup after rollout draining.
Only play(hit/stand) is exposed to the model. No code verifier is involved.
"""
import os
from dataclasses import replace
from pathlib import Path

import dotenv
import json
from blackjack_game import BlackjackTask

from twinkle.data_format import user_data_get
from twinkle_agentic.envs import AgentEnv, StepResult

dotenv.load_dotenv('.env')


class E2BBlackjackEnv(BlackjackTask, AgentEnv):
    """Keep game history and execution inside one sandbox, never on the client."""

    def __init__(self, max_sessions=8):
        if not os.environ.get('E2B_API_KEY'):
            raise ValueError('Set E2B_API_KEY in .env before creating E2B sandboxes')
        super().__init__(
            template=os.environ.get('E2B_TEMPLATE', 'base'), workspace='/home/user/blackjack',
            max_sessions=max_sessions,
            sandbox_timeout=600, command_timeout=120, include_default_tools=False,
            metadata={'task': 'twinkle-blackjack'},
            setup_commands=['python3 -m pip install --quiet open_spiel==2.0.2'])

    def _open(self, trajectory):
        state, _ = super()._open(trajectory)
        try:
            source = Path(__file__).with_name('blackjack_game.py').read_text()
            state.sandbox.files.write(f'{state.workspace}/blackjack_game.py', source)
            seed = int(user_data_get(trajectory['user_data'], 'seed'))
            result = self._run_game(state, f'reset --seed {seed}')
            if result.info.get('error'):
                raise RuntimeError(result.info['error'])
            return state, result
        except BaseException:
            self._kill(state)  # reset has not registered the session yet.
            raise

    def _step(self, state, tool_name, arguments):
        action = arguments.get('action')
        if tool_name != 'play' or action not in ('hit', 'stand'):
            return StepResult(observation='Use play(action="hit") or play(action="stand").')
        if state.last_result.done or state.last_result.info.get('error'):
            return replace(state.last_result, reward=0.0)
        return self._run_game(state, action)

    def _run_game(self, state, action):
        # This invokes the uploaded game file; no generated Python source is sent.
        code, output = self._run_script(state, f'python3 blackjack_game.py {action}', interpreter='shell')
        if code != 0:
            result = StepResult(observation=output, info={'error': f'Blackjack command failed: {output}'})
        else:
            try:
                result = StepResult(**json.loads(output))
            except (ValueError, TypeError) as error:
                result = StepResult(observation=output, info={'error': f'invalid Blackjack response: {error}'})
        result.info['sandbox_id'] = state.sandbox_id
        state.last_result = result
        self._touch(state)
        return result


if __name__ == '__main__':
    from streaming_multi_turn_grpo import train

    train(env_factory=E2BBlackjackEnv)
