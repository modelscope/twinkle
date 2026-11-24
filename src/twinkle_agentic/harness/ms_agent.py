# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-agent, invoked as ``ms-agent run``."""
import shlex
from typing import List, Optional

from .base import CliAgent


class MsAgent(CliAgent):
    """Run ``ms-agent run`` against the training policy.

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
        parts: List[str] = []
        if workspace:
            parts.append(f'cd {shlex.quote(workspace)}')
        if self.install:
            parts.append(self.install)
        run = [
            f'OPENAI_BASE_URL={shlex.quote(base_url)}',
            f'OPENAI_API_KEY={shlex.quote(api_key)}',
            'ms-agent run',
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
