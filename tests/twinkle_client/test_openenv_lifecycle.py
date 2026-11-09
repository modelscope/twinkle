# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest

from twinkle_agentic.envs.openenv import OpenEnvClient


class Client:
    def __init__(self, **kwargs):
        self.step_error = None
        self.close_error = None
        self.close_calls = 0

    def sync(self):
        return self

    def reset(self, **kwargs):
        return SimpleNamespace(observation='ready', reward=0, done=False)

    def step(self, action):
        if self.step_error:
            raise self.step_error
        return SimpleNamespace(observation='ok', reward=0, done=False)

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise self.close_error


def make_env():
    return OpenEnvClient(env_cls=Client, base_url='http://unused', action_cls=lambda **kw: kw)


def test_transport_failure_keeps_session_unconfirmed():
    env = make_env()
    env.client.step_error = TimeoutError('remote execution unknown')
    result = env.step('run_python', {'code': 'print(1)'})
    assert result.info['error']
    with pytest.raises(RuntimeError, match='unconfirmed'):
        env.ensure_idle()
    with pytest.raises(RuntimeError, match='unconfirmed'):
        env.close()
    assert env.client.close_calls == 0


def test_close_failure_is_observable_and_not_reported_as_closed():
    env = make_env()
    env.client.close_error = RuntimeError('close not acknowledged')
    with pytest.raises(RuntimeError, match='close not acknowledged'):
        env.close()


def test_successful_session_close_is_idempotent():
    env = make_env()
    env.reset()
    assert env.step('run_python', {'code': 'print(1)'}).observation == 'ok'
    env.ensure_idle()
    env.close()
    env.close()
    assert env.client.close_calls == 1
