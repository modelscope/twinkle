# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-agent, invoked as ``ms-agent run``."""
import shlex
from typing import List, Optional

from .base import CliAgent


_VALIDATE_AND_RUN = '''\
import os
from ms_agent.config import Config
from ms_agent.cli.cli import run_cmd
from omegaconf import DictConfig, ListConfig

expected_url = os.environ['OPENAI_BASE_URL']
expected_key = os.environ['OPENAI_API_KEY']
load_config = Config.from_task

def checked_config(cls, *args, **kwargs):
    config = load_config(*args, **kwargs)
    found = False
    def check(node):
        nonlocal found
        if isinstance(node, DictConfig):
            llm = node.get('llm')
            if llm is not None:
                found = True
                if not isinstance(llm, DictConfig) or str(llm.get('service', 'openai')).lower() != 'openai':
                    raise ValueError('MsAgent requires an OpenAI-compatible training-policy LLM.')
                if llm.get('openai_base_url') != expected_url or llm.get('openai_api_key') != expected_key:
                    raise ValueError('MsAgent config did not bind the training-policy endpoint and episode key.')
            for value in node.values():
                if isinstance(value, (DictConfig, ListConfig)):
                    check(value)
        elif isinstance(node, ListConfig):
            for value in node:
                check(value)
    check(config)
    if not found and not Config.is_workflow(config):
        raise ValueError('MsAgent config must explicitly bind llm.openai_base_url and llm.openai_api_key.')
    return config

Config.from_task = classmethod(checked_config)
run_cmd()
'''


class MsAgent(CliAgent):
    """Run ``ms-agent run`` against the training policy.

    ms-agent strips each stored message's content at send time, so its merged view of a
    continued turn is the strip-recurrence, declared through :attr:`merge_strips`.

    Endpoint and key go in as ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``. ms-agent's
    ``Config._update_config`` replaces config LEAVES from same-named env vars
    (case-insensitive), so the config must declare ``llm.openai_base_url`` /
    ``llm.openai_api_key`` leaves (unset is fine) for the injection to land -- they
    then win over whatever the yaml hardcodes, and the episode cannot quietly go
    to another model. A config WITHOUT those leaves falls back to the service
    default endpoint (api.openai.com): the env vars alone do not create keys.

    Args:
        config: ``--config``: a config directory or a repo id. The place to set
            ``service: openai`` and the model name.
        output_dir: parent for the per-episode ``--output_dir``. Each episode gets
            its own subdirectory named after its key, because concurrent episodes
            would otherwise write their histories over each other. Relative to the
            workspace when relative.
        install: a shell command run before the agent, joined with ``&&`` -- for
            an environment whose image does not already carry it. None to skip.
        trust_remote_code: passed through; needed by configs that load their own
            callbacks or tools.
        extra_args: appended to the command verbatim, for options this class does
            not model.
        timeout: seconds before the agent is killed.
    """

    protocol = 'chat'
    # ms-agent strips each stored message's content at send time, so its merged view of a
    # continued turn is the strip-recurrence; declaring it lets the multi-turn fold and the
    # harness-side recognition apply the SAME rule (harness/base.py default is False).
    merge_strips = True

    def __init__(
        self,
        *,
        config: Optional[str] = None,
        output_dir: str = 'ms_agent_runs',
        install: Optional[str] = None,
        trust_remote_code: bool = False,
        extra_args: str = '',
        timeout: float = CliAgent.timeout,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.install = install
        self.trust_remote_code = trust_remote_code
        self.extra_args = extra_args
        self.timeout = timeout

    def command(self, *, task: str, base_url: str, api_key: str, workspace: str) -> str:
        """Validate the config actually loaded by ms-agent inside the episode environment."""
        parts: List[str] = []
        if workspace:
            parts.append(f'cd {shlex.quote(workspace)}')
        if self.install:
            parts.append(self.install)
        run = [
            f'OPENAI_BASE_URL={shlex.quote(base_url)}',
            f'OPENAI_API_KEY={shlex.quote(api_key)}',
            f'python -c {shlex.quote(_VALIDATE_AND_RUN)} run',
            f'--query {shlex.quote(task)}',
            f'--output_dir {shlex.quote(f"{self.output_dir}/{api_key}")}',
        ]
        if self.config:
            run.append(f'--config {shlex.quote(self.config)}')
        if self.trust_remote_code:
            run.append('--trust_remote_code true')
        if self.extra_args:
            run.append(self.extra_args)
        parts.append(' '.join(run))
        return ' && '.join(parts)
