# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run the streaming Blackjack tutorial with one E2B sandbox per episode.

Install the `client,e2b` extras, set E2B_API_KEY and run this script against
the same Twinkle server as streaming_multi_turn_grpo.py.
"""

import json
import os
from pathlib import Path

from twinkle.data_format import user_data_get
from twinkle_agentic.envs import AgentEnv, StepResult

if __package__:
    from .blackjack_game import BlackjackTask
    from .streaming_multi_turn_grpo import train
else:
    from blackjack_game import BlackjackTask
    from streaming_multi_turn_grpo import train


class E2BBlackjackEnv(BlackjackTask, AgentEnv):
    """Run the game and store its history inside the sandbox."""

    def __init__(self):
        if not os.environ.get('E2B_API_KEY'):
            raise ValueError('Set E2B_API_KEY before creating E2B sandboxes')
        super().__init__(
            template=os.environ.get('E2B_TEMPLATE', 'base'), workspace='/home/user/blackjack',
            sandbox_timeout=600, command_timeout=120, include_default_tools=False,
            metadata={'task': 'twinkle-blackjack'},
            setup_commands=['python3 -m pip install --quiet open_spiel==2.0.2'])

    def _open(self, trajectory):
        state, _ = super()._open(trajectory)
        try:
            source = Path(__file__).with_name('blackjack_game.py').read_text()
            state.sandbox.files.write(f'{state.workspace}/blackjack_game.py', source)
            seed = int(user_data_get(trajectory['user_data'], 'seed'))
            return state, self._run_game(state, f'reset --seed {seed}')
        except BaseException:
            self._close(state)
            raise

    def _step(self, state, tool_name, arguments):
        action = arguments.get('action')
        if tool_name != 'play' or action not in ('hit', 'stand'):
            return StepResult(observation='Use play(action="hit") or play(action="stand").')
        return self._run_game(state, action)

    def _run_game(self, state, action):
        code, output = self._run_script(state, f'python3 blackjack_game.py {action}', interpreter='shell')
        if code != 0:
            raise RuntimeError(f'Blackjack command failed: {output}')
        return StepResult(**json.loads(output), info={'sandbox_id': state.sandbox_id})

    def _close(self, state):
        self._kill(state)


if __name__ == '__main__':
    train(E2BBlackjackEnv)
