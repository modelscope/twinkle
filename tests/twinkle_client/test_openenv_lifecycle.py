# Copyright (c) ModelScope Contributors. All rights reserved.
"""OpenEnvClient session-lifecycle contracts (episode-per-session design).

PLAN (case | input | expected | failure it kills):
  step_transport_error_keeps_session | client.step raises TimeoutError | StepResult.info['error']
    set, session stays usable, client not closed | a transient transport error killing the episode
    or silently closing its WebSocket
  close_failure_releases_permit | client.close raises, max_sessions=1 | warning logged, the permit
    is released so the next reset succeeds | a failed close stranding its permit until the pool
    shrinks to zero and every reset times out
  double_close_idempotent | reset/step/close/close | second close returns None, client closed once,
    no double permit release (max_sessions=1 still admits a new reset) | double-release inflating
    the permit pool above max_sessions
  close_returns_accumulated_reward | steps rewarded 0.5 + 0.25 | close -> 0.75; a never-rewarded
    session -> None | folding a tool-only session's constant zero into fabricated reward signal
  unknown_episode_fails_loud | step/close on an id never reset | SessionNotFound on step; close
    returns None | step silently opening a session the driver never admitted
"""
import logging
from types import SimpleNamespace

import pytest

from twinkle_agentic.envs.base import SessionNotFound
from twinkle_agentic.envs.openenv import OpenEnvClient


class Client:
    """Stub OpenEnv client; error knobs are class-level because _open builds the instance."""
    step_error = None
    close_error = None
    step_reward = 0.0
    close_calls = 0

    def __init__(self, **kwargs):
        pass

    def sync(self):
        return self

    def reset(self, **kwargs):
        return SimpleNamespace(observation='ready', reward=0.0, done=False)

    def step(self, action):
        if self.step_error:
            raise self.step_error
        return SimpleNamespace(observation='ok', reward=self.step_reward, done=False)

    def close(self):
        type(self).close_calls += 1
        if self.close_error:
            raise self.close_error


@pytest.fixture(autouse=True)
def _clean_knobs():
    Client.step_error = None
    Client.close_error = None
    Client.step_reward = 0.0
    Client.close_calls = 0
    yield


def make_env(max_sessions=64):
    return OpenEnvClient(env_cls=Client, base_url='http://unused', action_cls=lambda **kw: kw,
                         max_sessions=max_sessions)


def test_step_transport_error_keeps_session_alive():
    env = make_env()
    env.reset('ep1')
    Client.step_error = TimeoutError('remote execution unknown')
    result = env.step('ep1', 'run_python', {'code': 'print(1)'})
    assert result.info['error']
    assert Client.close_calls == 0, 'a transient step error must not close the session'
    Client.step_error = None
    assert env.step('ep1', 'run_python', {'code': 'print(1)'}).observation == 'ok'
    env.close('ep1')


def test_close_failure_releases_permit_and_is_observable(caplog):
    env = make_env(max_sessions=1)
    env.reset('ep1')
    Client.close_error = RuntimeError('close not acknowledged')
    with caplog.at_level(logging.WARNING):
        env.close('ep1')
    assert 'closing session' in caplog.text and 'ep1' in caplog.text
    # The permit must be back even though the transport close failed, or a one-slot component
    # deadlocks after its first failed close.
    env.reset('ep2')
    env.close('ep2')


def test_double_close_is_idempotent_and_does_not_double_release():
    env = make_env(max_sessions=1)
    env.reset('ep1')
    assert env.step('ep1', 'run_python', {'code': 'print(1)'}).observation == 'ok'
    env.close('ep1')
    assert env.close('ep1') is None
    assert Client.close_calls == 1
    # A double release would let TWO sessions into a one-permit pool.
    env.reset('ep2')
    with pytest.raises(TimeoutError):
        env.reset('ep3', wait=0.05)
    env.close('ep2')


def test_close_returns_accumulated_reward_or_none():
    env = make_env()
    env.reset('scored')
    Client.step_reward = 0.5
    env.step('scored', 'run_python', {'code': '1'})
    Client.step_reward = 0.25
    env.step('scored', 'run_python', {'code': '2'})
    assert env.close('scored') == pytest.approx(0.75)
    env.reset('tool_only')
    Client.step_reward = 0.0
    env.step('tool_only', 'run_python', {'code': '3'})
    assert env.close('tool_only') == 0.0, 'an explicit zero reward is a real score'
    env.reset('never_stepped')
    assert env.close('never_stepped') is None, 'no step ever carried a reward -> None, not 0.0'


def test_unknown_episode_fails_loud():
    env = make_env()
    with pytest.raises(SessionNotFound):
        env.step('ghost', 'run_python', {'code': 'print(1)'})
    assert env.close('ghost') is None