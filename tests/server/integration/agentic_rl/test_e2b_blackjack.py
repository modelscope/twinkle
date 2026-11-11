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
    path = Path(__file__).parents[4] / 'cookbook/client/async_rl/e2b_multi_turn_grpo.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('e2b_blackjack', path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    from twinkle_client.resources import RolloutResources

    sandboxes = []
    with RolloutResources(lambda: SimpleNamespace(close=lambda: None)) as resources:
        envs = [resources.own_environment(recipe.E2BBlackjackEnv(42)) for _ in range(2)]
        with ThreadPoolExecutor(2) as executor:
            initial = list(executor.map(lambda env: env.reset(), envs))
        sandboxes = [env.sandbox for env in envs]
        assert envs[0].sandbox_id != envs[1].sandbox_id
        assert initial[0].observation == initial[1].observation
        for index, env in enumerate(envs):
            env.sandbox.files.write(f'{env.workspace}/marker', str(index))
        result_a = envs[0].step('play', {'action': 'stand'})
        assert result_a.done and result_a.reward in (-1, 0, 1)
        assert json.loads(envs[1].sandbox.files.read(f'{envs[1].workspace}/game.json'))['actions'] == []
        result_b = envs[1].step('play', {'action': 'stand'})
        assert result_b.observation == result_a.observation
        assert result_b.reward == result_a.reward
        for index, env in enumerate(envs):
            assert env.sandbox.files.read(f'{env.workspace}/marker') == str(index)
        record_property('sandbox_ids', ','.join(env.sandbox_id for env in envs))
        record_property('terminal_reward', result_a.reward)
    assert all(not sandbox.is_running() for sandbox in sandboxes)
    record_property('released_sandboxes', len(sandboxes))


def test_e2b_refuses_close_during_real_command():
    from dotenv import load_dotenv

    from twinkle_agentic.envs import AgentEnv

    load_dotenv('.env')
    env = AgentEnv(template='base', workspace='/home/user/blackjack', sandbox_timeout=120)
    sandbox = None
    try:
        env.reset()
        sandbox = env.sandbox
        with ThreadPoolExecutor(1) as executor:
            work = executor.submit(env.run_script, 'touch active; sleep 5; touch finished', interpreter='shell')
            deadline = time.monotonic() + 20
            while not sandbox.files.exists(f'{env.workspace}/active'):
                assert time.monotonic() < deadline, 'Remote command never started'
                assert not work.done(), work.result()
                time.sleep(0.1)
            assert not work.done()
            with pytest.raises(RuntimeError, match='命令仍在执行'):
                env.ensure_idle()
            with pytest.raises(RuntimeError, match='未结束的命令'):
                env.close()
            assert work.result(timeout=20)[0] == 0
            assert sandbox.files.exists(f'{env.workspace}/finished')
        env.ensure_idle()
    finally:
        env.close()
    assert not sandbox.is_running()
