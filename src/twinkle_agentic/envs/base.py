# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Tool as ToolInfo

# What :meth:`Env.run_script` returns when it had to kill the script. 124 is
# what GNU ``timeout`` uses, so a caller that logs the number is logging
# something a reader already knows how to interpret.
TIMEOUT_EXIT_CODE = 124

# Truncation guard for anything that becomes an observation: a command that
# dumps a whole file would otherwise spend the episode's context on one turn.
MAX_OBSERVATION_CHARS = 32 * 1024

# The tools every general-purpose environment advertises, sandboxed or local.
# Shared rather than restated per implementation: a trajectory built against one
# env has to replay on another, and it only does if the names and the argument
# spellings are the same object.
DEFAULT_TOOLS: List[ToolInfo] = [
    {
        'type': 'function',
        'function': {
            'name': 'run_command',
            'description': 'Run a shell command inside the sandbox and return its output.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'The shell command to execute.'
                    },
                    'cwd': {
                        'type': 'string',
                        'description': 'Working directory (optional).'
                    },
                },
                'required': ['command'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'write_file',
            'description': 'Write text content to a file inside the sandbox.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {
                        'type': 'string',
                        'description': 'Absolute file path in the sandbox.'
                    },
                    'content': {
                        'type': 'string',
                        'description': 'Text content to write.'
                    },
                },
                'required': ['path', 'content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_file',
            'description': 'Read a text file from the sandbox.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {
                        'type': 'string',
                        'description': 'Absolute file path in the sandbox.'
                    },
                },
                'required': ['path'],
            },
        },
    },
]


def truncate_observation(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f'\n... [truncated, {len(text) - limit} chars omitted]'


def format_command_output(stdout: str, stderr: str, exit_code: int) -> str:
    """One command's result as the model sees it."""
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f'[stderr]\n{stderr}')
    if exit_code != 0:
        parts.append(f'[exit code: {exit_code}]')
    return truncate_observation('\n'.join(parts)) if parts else '(no output)'


@dataclass
class StepResult:
    """Result returned by :meth:`Env.step`."""
    observation: str = ''
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class Env(ABC):
    """Base class for RL execution environments.

    All environments implement this interface. Usage::

        env = SomeEnv(...)
        result = env.reset()
        result = env.step(tool_name, arguments)

    Tool-call markup is parsed upstream by
    :meth:`twinkle.template.base.Template.parse_tool_call`. This class only
    executes already-split ``(tool_name, arguments)`` pairs.
    """

    #: How many times this environment had to be rebuilt under a caller that was
    #: holding it. Reported rather than dropped: a run whose environments were
    #: rebuilt twenty times produced its numbers under different conditions than
    #: one that was rebuilt never, and that is invisible from the outputs alone.
    #: Stays at zero for an environment that cannot be lost.
    n_recoveries = 0

    def reset(self, trajectory: Optional[Trajectory] = None) -> StepResult:
        return StepResult()

    @abstractmethod
    def step(self, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        raise NotImplementedError

    def step_batch(
        self,
        calls: Sequence[Tuple[str, Dict[str, Any]]],
    ) -> List[StepResult]:
        """Execute a batch of already-parsed ``(tool_name, arguments)`` pairs.

        Default is a serial loop over :meth:`step`. Subclasses that talk to a
        remote sandbox should override this so MultiTurn can keep tools off
        the generate critical path.
        """
        return [self.step(name, args or {}) for name, args in calls]

    def run_script(self, source: str, interpreter: str = 'python',
                   timeout: Optional[int] = None) -> Tuple[int, str]:
        """Run a whole script here; returns ``(exit_code, output)``.

        The execution path a *verifier* takes, as opposed to :meth:`step`, which
        is the one the model takes. Both land in the same place, and that is the
        point: a check has to observe the filesystem the episode actually wrote
        to, so it runs in the environment rather than beside it.

        Args:
            source: the script, not a path.
            interpreter: ``'python'`` or ``'shell'``.
            timeout: seconds; ``None`` means the environment's own default.

        Returns:
            ``(exit_code, output)``. ``output`` is stdout followed by stderr, so
            a traceback lands at the end rather than interleaved. A non-zero exit
            code is the only failure signal callers should read -- the specific
            value is the script's, except for a timeout, which is
            :data:`TIMEOUT_EXIT_CODE`.
        """
        raise NotImplementedError(f'{type(self).__name__} cannot run scripts')

    def ensure_ready(self) -> bool:
        """Re-establish this environment if it has gone away. True if it did.

        For a caller that holds one environment across many jobs, losing it --
        evicted, timed out, runtime crashed -- otherwise ends the whole run. Safe
        to call only where the workspace is about to be discarded anyway: a
        mid-episode rebuild silently swaps the state a job is being judged on for
        an empty directory, which is why recovery is an explicit call rather than
        a retry hidden inside every dispatch.

        The default is ``False``: an environment that is a local process has
        nothing to lose between calls and so is never not ready.
        """
        return False

    def rebuild(self) -> None:
        """Throw this environment away and stand a fresh one up in its place.

        For the caller that has a *working* environment it no longer trusts --
        one that keeps failing an operation it should not fail -- as opposed to
        :meth:`ensure_ready`, which is about one that stopped answering. Counted
        in :attr:`n_recoveries`.

        The default is a no-op, which is the truth for an environment holding
        nothing worth rebuilding.
        """

    def clear(self) -> None:
        """Return to a clean state, ready for the next episode.

        Called by whoever owns the environment, before handing it to a job that
        must not see the previous one's files. Raising is the right answer for an
        environment that could not clean itself: a silent no-op there means the
        next job inherits a workspace, which lets a solver pass without doing
        anything and makes a difficulty measurement meaningless.

        The default is a no-op because it is the truth for an environment holding
        no state between calls -- the shape one-shot verification uses. That is
        also what lets both halves run the same sequence: the code half clears
        before every judgement too, and clearing nothing costs nothing.
        """

    def snapshot(self) -> Tuple[str, str]:
        """The end state as ``(listing, error)``; both empty when there is none.

        What an episode left behind, for a caller that has to describe it to a
        model -- writing a check against a workspace means knowing what is in it.
        The two strings are kept apart because a snapshot that returns "empty"
        when it means "I could not look" produces tasks whose only true assertion
        is that nothing happened.

        The default is the honest answer for an environment that keeps nothing:
        there is no end state to read back, which is why an env used only to run
        one-shot checks does not have to implement this.
        """
        return '', ''

    def tools(self) -> List[ToolInfo]:
        return []

    def tool_manager(self, schemas: Optional[Sequence[ToolInfo]] = None) -> Any:
        """A ``ToolManager`` that dispatches tool calls into this environment.

        What a rollout needs to let a model act here, so it is built once on the
        environment rather than restated by every caller that owns one -- and a
        caller holding N environments gets N managers that cannot be crossed,
        which is the failure this prevents: an episode acting in one workspace
        and being checked in another produces a task nobody can pass.

        Args:
            schemas: the tool contract to advertise; defaults to :meth:`tools`.
                Passed explicitly when an agent framework owns the names that go
                into the prompt and this environment only supplies the
                implementation.
        """
        # Local import: ToolManager is a consumer of this package, and the tools
        # package is not needed by an env that is only ever asked to run scripts.
        from ..tools.tool_manager import ToolManager
        from .env_tool import EnvTool
        declared = list(schemas) if schemas is not None else self.tools()
        return ToolManager(EnvTool.from_schemas(self, declared))

    def evaluate(self, trajectories: List[Trajectory], **kwargs) -> List[float]:
        return [0.0] * len(trajectories)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
