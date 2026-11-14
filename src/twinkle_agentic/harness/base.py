# Copyright (c) ModelScope Contributors. All rights reserved.
"""How to run one agent program on one task.

An agent here is a command line, not a loop we call. What it takes to add support
for one is a small, testable object: what to install, what to invoke, which
protocol it speaks and which environment variable that particular program reads
its endpoint out of. Nothing in this package imports a sampler, or knows a token
from a tensor -- that is :class:`~twinkle_agentic.rollout.multi_turn.MultiTurnRollout`'s
job, and keeping the two apart is what lets a new agent be added without reading
any of it.
"""
from abc import ABC, abstractmethod
from typing import Literal

from twinkle.data_format import Trajectory


def default_task_text(trajectory: Trajectory) -> str:
    """The prompt, flattened into the single string a CLI agent accepts.

    Everything the caller wrote -- system framing and question both -- in order,
    because an agent program brings its own system prompt and would drop ours if
    only the last user message were passed.
    """
    parts = [
        str(message['content']) for message in (trajectory.get('messages') or [])
        if message.get('role') in ('system', 'user') and message.get('content')
    ]
    if not parts:
        raise ValueError('cannot build a task for an external agent: the trajectory has no '
                         'system or user message to send. Override CliAgent.task for a prompt '
                         'that lives somewhere else.')
    return '\n\n'.join(parts)


class CliAgent(ABC):
    """How to run one agent program on one task. Nothing about training.

    The rollout runs :meth:`command` in the episode's env session
    (``env.run_script(episode_id, command, 'shell', timeout)``) and answers the
    requests the program makes on the way, one rollout step per request.

    Attributes:
        protocol: which OpenAI protocol the program speaks -- ``'chat'``
            (``/v1/chat/completions``) or ``'responses'`` (``/v1/responses``).
        timeout: seconds before the program is killed. There is no other bound
            on an episode: the agent decides when it is done.
    """

    protocol: Literal['chat', 'responses'] = 'chat'
    timeout: float = 1800.0

    def task(self, trajectory: Trajectory) -> str:
        """The text handed to the agent; defaults to :func:`default_task_text`."""
        return default_task_text(trajectory)

    @abstractmethod
    def command(self, *, task: str, base_url: str, api_key: str, workspace: str) -> str:
        """The shell command that runs the agent to completion on ``task``.

        Args:
            task: the task, as text. The agent's only input.
            base_url: an OpenAI-compatible endpoint serving the training policy.
                Must be what the agent actually calls -- anything else trains on
                a model that is not the one being updated.
            api_key: passed through as the API key, and doubling as the identity
                of this episode. Whatever the agent does with the conversation,
                it must keep sending this key, or its requests land in no episode.
            workspace: the directory to work in, or ``''`` when the env keeps none.

        Returns:
            One shell command, run with the shell. It must exit when the agent is
            done: this is what tells the rollout the episode is over.
        """
