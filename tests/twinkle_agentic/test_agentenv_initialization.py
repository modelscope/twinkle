# Copyright (c) ModelScope Contributors. All rights reserved.
"""Initialization errors must not return an apparently usable sandbox."""
from types import SimpleNamespace

import pytest

from twinkle_agentic.envs import AgentEnv, agentenv


@pytest.mark.parametrize('failed_command', ['mkdir', 'setup', None])
def test_initialization_checks_exit_codes_and_retains_cleanup_handle(monkeypatch, failed_command):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        failed = failed_command is not None and command.startswith(failed_command)
        return SimpleNamespace(exit_code=1 if failed else 0, stdout='', stderr='permission denied' if failed else '')

    sandbox = SimpleNamespace(sandbox_id='sandbox', commands=SimpleNamespace(run=run), kill=lambda: True)
    monkeypatch.setattr(agentenv, '_require_e2b', lambda: SimpleNamespace(create=lambda *a, **k: sandbox))
    env = AgentEnv(template='base', setup_commands=['setup', 'later'])
    try:
        if failed_command is None:
            env.reset()
            assert calls[-2:] == ['setup', 'later']
        else:
            with pytest.raises(RuntimeError, match='permission denied'):
                env.reset()
            assert 'later' not in calls
        assert env.sandbox is sandbox
    finally:
        env.close()
    assert env.sandbox_id is None
