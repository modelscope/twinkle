# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in real official E2B execution; no Twinkle Model/Sampler server needed.

TWINKLE_TEST_E2B=1 pytest --noconftest -q -s tests/server/integration/agentic_rl/test_e2b_blackjack.py
"""
import importlib.util
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import json
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TWINKLE_TEST_E2B') != '1', reason='Uses real paid E2B sandboxes')


def test_e2b_blackjack_isolation_and_release(record_property, monkeypatch):
    from twinkle.data_format import pack_user_data
    path = Path(__file__).parents[4] / 'cookbook/client/async_rl/e2b_multi_turn_grpo.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('e2b_blackjack', path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    from twinkle_client.resources import RolloutResources

    sandboxes = []
    with RolloutResources(lambda: SimpleNamespace(close=lambda: None)) as resources:
        env = resources.own_environment(recipe.E2BBlackjackEnv())
        episodes = ['episode-a', 'episode-b']
        trajectory = {'user_data': pack_user_data({'seed': 42})}
        with ThreadPoolExecutor(2) as executor:
            initial = list(executor.map(lambda episode: env.reset(episode, trajectory), episodes))
        sandboxes = [env._session(episode).state.sandbox for episode in episodes]
        assert sandboxes[0].sandbox_id != sandboxes[1].sandbox_id
        assert initial[0].observation == initial[1].observation
        for index, (episode, sandbox) in enumerate(zip(episodes, sandboxes, strict=True)):
            sandbox.files.write(f'{env.workspace(episode)}/marker', str(index))
        result_a = env.step(episodes[0], 'play', {'action': 'stand'})
        assert result_a.done and result_a.reward in (-1, 0, 1)
        assert json.loads(sandboxes[1].files.read(f'{env.workspace(episodes[1])}/game.json'))['actions'] == []
        result_b = env.step(episodes[1], 'play', {'action': 'stand'})
        assert result_b.observation == result_a.observation
        assert result_b.reward == result_a.reward
        for index, (episode, sandbox) in enumerate(zip(episodes, sandboxes, strict=True)):
            assert sandbox.files.read(f'{env.workspace(episode)}/marker') == str(index)
        record_property('sandbox_ids', ','.join(sandbox.sandbox_id for sandbox in sandboxes))
        record_property('terminal_reward', result_a.reward)
    assert all(not sandbox.is_running() for sandbox in sandboxes)
    record_property('released_sandboxes', len(sandboxes))


def test_e2b_closes_sandbox_with_a_running_command():
    from dotenv import load_dotenv

    from twinkle_agentic.envs import AgentEnv

    load_dotenv('.env')
    env = AgentEnv(template='base', workspace='/home/user/blackjack', sandbox_timeout=120)
    sandbox = None
    try:
        env.reset('episode')
        sandbox = env._session('episode').state.sandbox
        workspace = env.workspace('episode')
        with ThreadPoolExecutor(1) as executor:
            work = executor.submit(env.run_script, 'episode', 'touch active; sleep 30', interpreter='shell')
            deadline = time.monotonic() + 20
            while not sandbox.files.exists(f'{workspace}/active'):
                assert time.monotonic() < deadline, 'Remote command never started'
                assert not work.done(), work.result()
                time.sleep(0.1)
            assert not work.done()
            env.close('episode')
            assert work.result(timeout=20)[0] != 0
    finally:
        env.shutdown()
    assert not sandbox.is_running()
