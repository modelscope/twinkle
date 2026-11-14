# Copyright (c) ModelScope Contributors. All rights reserved.
"""Codex CLI, invoked as ``codex exec``."""
import json
import shlex
from typing import List, Optional

from .base import CliAgent

#: The variable the provider's ``env_key`` names; codex reads the API key out of it.
API_KEY_ENV = 'TWINKLE_API_KEY'


def _config(key: str, value) -> str:
    """One ``-c key=value`` override. JSON scalars are valid TOML, which is what codex parses."""
    return f'-c {shlex.quote(f"{key}={json.dumps(value)}")}'


class CodexAgent(CliAgent):
    """Run ``codex exec`` against the training policy over the Responses protocol.

    The policy is declared as a custom model provider (``model_providers.twinkle``)
    whose ``base_url`` is the endpoint and whose key is read from
    ``TWINKLE_API_KEY``, so neither a user config nor ``OPENAI_*`` in the image can
    point the episode at another model. Sandboxing and approvals are off: the env
    session is the sandbox, and nobody is there to approve.

    Args:
        model: the model name sent in each request; the endpoint ignores it.
        install: a shell command run before the agent, joined with ``&&`` (e.g.
            ``npm i -g @openai/codex``) -- for an image that does not carry it.
        extra_args: appended to ``codex exec`` verbatim, before the task.
        timeout: seconds before the agent is killed.
    """

    protocol = 'responses'

    def __init__(
        self,
        *,
        model: str = 'twinkle-policy',
        install: Optional[str] = None,
        extra_args: str = '',
        timeout: float = CliAgent.timeout,
    ) -> None:
        self.model = model
        self.install = install
        self.extra_args = extra_args
        self.timeout = timeout

    def command(self, *, task: str, base_url: str, api_key: str, workspace: str) -> str:
        parts: List[str] = []
        if self.install:
            parts.append(self.install)
        run = [
            f'{API_KEY_ENV}={shlex.quote(api_key)}',
            'codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox',
        ]
        if workspace:
            run.append(f'-C {shlex.quote(workspace)}')
        run += [
            f'-m {shlex.quote(self.model)}',
            _config('model_provider', 'twinkle'),
            _config('model_providers.twinkle.name', 'twinkle'),
            _config('model_providers.twinkle.base_url', base_url),
            _config('model_providers.twinkle.env_key', API_KEY_ENV),
            _config('model_providers.twinkle.wire_api', 'responses'),
            _config('features.multi_agent', False),
            _config('web_search', 'disabled'),
        ]
        if self.extra_args:
            run.append(self.extra_args)
        run.append(shlex.quote(task))
        parts.append(' '.join(run))
        return ' && '.join(parts)
