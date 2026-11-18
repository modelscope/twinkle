# Copyright (c) ModelScope Contributors. All rights reserved.
"""Initialization errors must not return an apparently usable sandbox."""
from types import SimpleNamespace

import pytest

from twinkle_agentic.envs import AgentEnv, agentenv


@pytest.mark.parametrize('failed_command', ['mkdir', 'setup', None])
def test_initialization_checks_exit_codes_and_cleans_up_on_failure(monkeypatch, failed_command):
    calls = []
    killed = []

    def run(command, **kwargs):
        calls.append(command)
        failed = failed_command is not None and command.startswith(failed_command)
        return SimpleNamespace(exit_code=1 if failed else 0, stdout='', stderr='permission denied' if failed else '')

    sandbox = SimpleNamespace(sandbox_id='sandbox', commands=SimpleNamespace(run=run),
                              files=SimpleNamespace(write=lambda path, content: None), kill=lambda: killed.append(True))
    monkeypatch.setattr(agentenv, '_require_e2b', lambda: SimpleNamespace(create=lambda *a, **k: sandbox))
    env = AgentEnv(template='base', setup_commands=['setup', 'later'])
    try:
        if failed_command is None:
            env.reset('ep1')
            assert calls[-2:] == ['setup', 'later']
            assert env._session('ep1').state.sandbox is sandbox
            assert not killed
        else:
            with pytest.raises(RuntimeError, match='permission denied'):
                env.reset('ep1')
            assert 'later' not in calls
            # The failed sandbox is cleaned up, and no usable session is left behind.
            assert killed
            assert 'ep1' not in env._sessions
    finally:
        env.close('ep1')
    if failed_command is None:
        # Close empties the workspace but keeps the sandbox for the next episode.
        assert not killed
        assert any(state.sandbox is sandbox for state in env._idle)