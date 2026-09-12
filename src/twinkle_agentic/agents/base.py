# Copyright (c) ModelScope Contributors. All rights reserved.
"""How to run one agent program on one task.

An agent here is a command line, not a loop we call. What it takes to add support
for one is a small, testable object: what to install, what to invoke, and which
environment variable that particular program reads its endpoint out of. Nothing
in this package imports a sampler, sees a trajectory, or knows a token from a
tensor -- that is :mod:`twinkle_agentic.rollout.external`'s job, and keeping the
two apart is what lets a new agent be added without reading any of it.
"""
from abc import ABC, abstractmethod


class CliAgent(ABC):
    """How to run one agent program on one task. Nothing about training.

    Subclassing is a convenience, not a requirement:
    :class:`~twinkle_agentic.rollout.external.ExternalRollout` asks for
    ``command``, so a plain function of the same signature is an agent too, and
    one written outside this package is no worse off than one written here.

    An agent that does not speak the OpenAI protocol needs one more thing -- a
    translation in front of the endpoint -- but not a different rollout.
    """

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
                it must keep sending this key, or its rounds land in no account.
            workspace: the directory to work in, or ``''`` when the caller did not
                provide one.

        Returns:
            One shell command, run with the shell. It must exit when the agent is
            done: this is what tells the rollout the episode is over.
        """
