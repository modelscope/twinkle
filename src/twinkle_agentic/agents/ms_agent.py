# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-agent, invoked as ``ms-agent run``."""
import shlex
from typing import List, Optional

from .base import CliAgent


class MsAgent(CliAgent):
    """Run ``ms-agent run`` against the training policy.

    Endpoint and key go in as ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``, which
    ms-agent folds over its config as overrides -- so they win over whatever the
    yaml says, and a config that hardcodes a different endpoint cannot quietly
    send the episode to another model.

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
    """

    def __init__(
        self,
        *,
        config: Optional[str] = None,
        output_dir: str = 'ms_agent_runs',
        install: Optional[str] = None,
        trust_remote_code: bool = False,
        extra_args: str = '',
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.install = install
        self.trust_remote_code = trust_remote_code
        self.extra_args = extra_args

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
