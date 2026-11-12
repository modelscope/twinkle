# Copyright (c) ModelScope Contributors. All rights reserved.
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle import remote_function
from twinkle.data_format import Trajectory
from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from .snapshot import list_workspace

logger = get_logger()

# What :meth:`Env.run_script` returns when it had to kill the script. 124 is
# what GNU ``timeout`` uses, so a caller that logs the number is logging
# something a reader already knows how to interpret.
TIMEOUT_EXIT_CODE = 124

# Truncation guard for anything that becomes an observation: a command that
# dumps a whole file would otherwise spend the episode's context on one turn.
MAX_OBSERVATION_CHARS = 32 * 1024

#: The default ceiling of concurrent method calls on one env replica: its sessions (64 by default)
#: plus room for the calls that wait for one to free up, and for ``close`` to get through meanwhile.
ENV_MAX_CONCURRENCY = 96

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


class SessionNotFound(KeyError):
    """The episode has no open session here: never reset, already closed, or its replica is gone."""


class _Session:
    __slots__ = ('state', 'reward', 'touched')

    def __init__(self, state: Any):
        self.state = state
        self.reward = 0.0
        self.touched = time.monotonic()


class Env(ABC):
    """Base class for RL execution environments: one component serving many episodes.

    Every episode is a session keyed by its ``episode_id``::

        env = SomeEnv(..., max_sessions=64, remote_group='env')
        result = env.reset(episode_id)                       # open; waits while all sessions are taken
        result = env.step(episode_id, tool_name, arguments)
        code, out = env.run_script(episode_id, source)       # a verifier's script, in the same place
        reward = env.close(episode_id)                       # the episode's reward; frees the session

    The session methods route by ``episode_id`` (calls of one episode reach the replica holding its
    session), so a deployment with several replicas looks like one env to its callers. A call for
    an episode that has no session here raises :class:`SessionNotFound`.

    Tool-call markup is parsed upstream by
    :meth:`twinkle.template.base.Template.parse_tool_call`. This class only
    executes already-split ``(tool_name, arguments)`` pairs.

    A subclass implements the session hooks: ``_open(traj) -> (state, StepResult)``,
    ``_step(state, tool_name, arguments)``, ``_run_script(state, source, interpreter, timeout)``,
    ``_close(state)`` and ``_workspace(state)``; plus :meth:`tools`.
    """

    #: What :meth:`snapshot` shows of the workspace: how many files to name, how
    #: many bytes of each body, how many bytes of bodies in total, and which
    #: directories not to walk into. Class attributes rather than constructor
    #: arguments because they are a truncation budget, not a description of the
    #: environment -- a caller that needs different numbers says so once, in a
    #: subclass, rather than at every construction site.
    snapshot_max_files = 40
    snapshot_per_file = 2000
    snapshot_budget = 20000
    snapshot_skip = ('__pycache__', '.git', '.ipynb_checkpoints')

    #: Seconds a session may go untouched before the next ``reset`` closes it: a caller that died
    #: or never called ``close`` would otherwise hold a sandbox forever.
    session_idle_timeout = 1800.0
    #: Seconds ``reset`` waits for a free session before it gives up with ``TimeoutError``.
    session_wait_timeout = 600.0

    def __init__(self, max_sessions: int = 64):
        self._max_sessions = max(1, int(max_sessions))
        self._session_slots = threading.Semaphore(self._max_sessions)
        self._sessions: Dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _open(self, trajectory: Optional[Trajectory]) -> Tuple[Any, StepResult]:
        """Start one episode; returns its state and the initial observation."""

    @abstractmethod
    def _step(self, state: Any, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        """Run one tool call in the episode."""

    def _run_script(self, state: Any, source: str, interpreter: str, timeout: Optional[float]) -> Tuple[int, str]:
        raise NotImplementedError(f'{type(self).__name__} cannot run scripts; act through its tools instead.')

    def _close(self, state: Any) -> None:
        pass

    def _workspace(self, state: Any) -> Optional[str]:
        return None

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def _session(self, episode_id: str) -> _Session:
        with self._sessions_lock:
            session = self._sessions.get(episode_id)
        if session is None:
            raise SessionNotFound(f'{type(self).__name__}: no open session for episode {episode_id!r}.')
        session.touched = time.monotonic()
        return session

    def _close_idle_sessions(self) -> None:
        now = time.monotonic()
        with self._sessions_lock:
            idle = [eid for eid, session in self._sessions.items() if now - session.touched > self.session_idle_timeout]
        for eid in idle:
            logger.warning(f'{type(self).__name__}: closing session {eid!r}, idle for over '
                           f'{self.session_idle_timeout:.0f}s.')
            self.close(eid)

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def reset(self, episode_id: str, traj: Optional[Trajectory] = None) -> StepResult:
        """Open the episode's session (closing an earlier one under the same id); returns the first observation.

        Waits while every session is taken, up to :attr:`session_wait_timeout`.
        """
        self._close_idle_sessions()
        if episode_id in self._sessions:
            self.close(episode_id)
        if not self._session_slots.acquire(timeout=self.session_wait_timeout):
            raise TimeoutError(f'{type(self).__name__}: no free session within {self.session_wait_timeout:.0f}s '
                               f'({self._max_sessions} in use).')
        try:
            state, result = self._open(traj)
        except BaseException:
            self._session_slots.release()
            raise
        with self._sessions_lock:
            self._sessions[episode_id] = _Session(state)
        return result

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def step(self, episode_id: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> StepResult:
        session = self._session(episode_id)
        result = self._step(session.state, tool_name, arguments or {})
        session.reward += float(result.reward or 0.0)
        return result

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def step_batch(self, episode_id: str, calls: Sequence[Tuple[str, Dict[str, Any]]]) -> List[StepResult]:
        """Execute already-parsed ``(tool_name, arguments)`` pairs in order, in one round trip."""
        session = self._session(episode_id)
        results = []
        for name, args in calls:
            result = self._step(session.state, name, args or {})
            session.reward += float(result.reward or 0.0)
            results.append(result)
        return results

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def run_script(self,
                   episode_id: str,
                   source: str,
                   interpreter: str = 'python',
                   timeout: Optional[float] = None) -> Tuple[int, str]:
        """Run a whole script in the episode's session; returns ``(exit_code, output)``.

        The execution path a *verifier* (or an agent program) takes, as opposed to :meth:`step`,
        which is the one the model takes. Both land in the same place, and that is the point: a
        check has to observe the filesystem the episode actually wrote to.

        Args:
            episode_id: the session.
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
        return self._run_script(self._session(episode_id).state, source, interpreter, timeout)

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def workspace(self, episode_id: str) -> Optional[str]:
        """The directory the episode works in, or None for an environment that keeps none."""
        return self._workspace(self._session(episode_id).state)

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def snapshot(self, episode_id: str) -> Tuple[str, str]:
        """The episode's end state as ``(listing, error)``; both empty when there is none.

        What an episode left behind, for a caller that has to describe it to a
        model -- writing a check against a workspace means knowing what is in it.
        The two strings are kept apart because a snapshot that returns "empty"
        when it means "I could not look" produces tasks whose only true assertion
        is that nothing happened. Truncation is set by the ``snapshot_*`` class attributes.
        """
        state = self._session(episode_id).state
        return list_workspace(
            self._workspace(state),
            lambda source: self._run_script(state, source, 'python', None),
            max_files=self.snapshot_max_files,
            per_file=self.snapshot_per_file,
            budget=self.snapshot_budget,
            skip=self.snapshot_skip)

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def close(self, episode_id: str) -> float:
        """End the episode's session and return its reward (the sum of its step rewards).

        Closing an episode with no session returns 0.0.
        """
        with self._sessions_lock:
            session = self._sessions.pop(episode_id, None)
        if session is None:
            return 0.0
        try:
            self._close(session.state)
        except Exception:  # noqa
            logger.warning(f'{type(self).__name__}: closing session {episode_id!r} failed.', exc_info=True)
        finally:
            self._session_slots.release()
        return session.reward

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def shutdown(self) -> None:
        """Close every open session."""
        with self._sessions_lock:
            episode_ids = list(self._sessions)
        for episode_id in episode_ids:
            self.close(episode_id)

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def tools(self) -> List[ToolInfo]:
        """What the model is told it can call in an episode."""
        return []

    def _pressure(self) -> float:
        with self._sessions_lock:
            return len(self._sessions) / self._max_sessions
