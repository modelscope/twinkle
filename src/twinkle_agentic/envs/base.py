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


def _record_tool_exec(episode_id: str, n: int) -> None:
    """Count model tool calls actually executed in an episode when ``TWINKLE_ROLLOUT_TRAJ_DUMP`` is set.

    The independent side of the G28 D4 depth oracle: the trajectory records the tool_calls the policy
    EMITTED (assistant messages); this records what the environment EXECUTED (step / step_batch land
    here, one count per call). An offline judge joins the two by episode_id and flags a divergence --
    a redo or a dropped tool call. run_script is the agent/verifier whole-script path, not a model
    tool_call, so it is deliberately not counted. Write-only and env-gated: a no-op when unset.
    """
    import os
    root = os.environ.get('TWINKLE_ROLLOUT_TRAJ_DUMP')
    if not root:
        return
    import json
    os.makedirs(root, exist_ok=True)
    rank = int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0)
    with open(os.path.join(root, f'tool_counts.rank{rank}.jsonl'), 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'episode_id': episode_id, 'executed': int(n)}) + '\n')


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


@dataclass(frozen=True)
class NoSlot:
    """``reset`` found every session permit taken and was asked not to wait (``wait=0``).

    A value, not an exception: the driver's admission loop treats a full pool as a scheduling
    delay (retry the member later), not a failure, so it must travel back along the normal
    ``(rows, RolloutStatus)`` return channel rather than raise. Frozen and field-less so it
    pickles by value across the Ray boundary and ``isinstance`` identifies it on either side --
    a module-level singleton compared with ``is`` would not survive the pickle.
    """


class _Session:
    __slots__ = ('state', 'reward', 'rewarded', 'touched', 'script_inflight')

    def __init__(self, state: Any):
        self.state = state
        self.reward = 0.0
        #: Whether any step ever carried a reward. A sandbox used only for its tools never
        #: scores, and its constant 0.0 sum is "no reward", not a genuine zero -- ``close`` tells
        #: the two apart so a caller does not fold an absent signal as a real score.
        self.rewarded = False
        self.touched = time.monotonic()
        #: A whole script (agent program / verifier) is executing in this session right now. Such a
        #: session is alive even when it makes no env calls for a long time, so idle reaping must
        #: skip it -- ``touched`` only refreshes on env method calls and would otherwise age out a
        #: running program and reclaim its permit mid-run.
        self.script_inflight = False


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
    ``_step(state, tool_name, arguments)``, ``_run_script(state, source, interpreter, timeout, cap_memory)``,
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

    def __init__(self, max_sessions: int = 64, session_wait_timeout: Optional[float] = 600.0):
        self._max_sessions = max(1, int(max_sessions))
        self._session_slots = threading.Semaphore(self._max_sessions)
        self._sessions: Dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        #: Instance default for how long ``reset`` waits for a free permit; a per-call ``wait``
        #: override (see :meth:`reset`) takes precedence. Three states: positive = wait up to this
        #: many seconds then raise; None = wait forever; 0 = return :class:`NoSlot` immediately.
        self.session_wait_timeout = session_wait_timeout

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _open(self, trajectory: Optional[Trajectory]) -> Tuple[Any, StepResult]:
        """Start one episode; returns its state and the initial observation."""

    @abstractmethod
    def _step(self, state: Any, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        """Run one tool call in the episode."""

    def _run_script(self, state: Any, source: str, interpreter: str, timeout: Optional[float],
                    cap_memory: bool = True) -> Tuple[int, str]:
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
            idle = [eid for eid, session in self._sessions.items()
                    if not session.script_inflight and now - session.touched > self.session_idle_timeout]
        for eid in idle:
            logger.warning(f'{type(self).__name__}: closing session {eid!r}, idle for over '
                           f'{self.session_idle_timeout:.0f}s.')
            self.close(eid)

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def reset(self, episode_id: str, traj: Optional[Trajectory] = None, wait: Optional[float] = None) -> Any:
        """Open the episode's session (closing an earlier one under the same id); returns the first observation.

        ``wait`` overrides the instance :attr:`session_wait_timeout` for this call only: a positive
        number waits up to that many seconds then raises ``TimeoutError``; ``None`` waits forever;
        ``0`` returns :class:`NoSlot` immediately when every permit is taken (the driver's admission
        loop uses this to retry a member later instead of blocking). ``wait=None`` inherits the
        instance default, so the batch ``__call__`` and ``stream`` paths keep their current behavior.
        """
        self._close_idle_sessions()
        if episode_id in self._sessions:
            self.close(episode_id)
        effective = self.session_wait_timeout if wait is None else wait
        if effective is None:
            acquired = self._session_slots.acquire()
        else:
            acquired = self._session_slots.acquire(timeout=effective)
        if not acquired:
            if effective == 0:
                return NoSlot()
            with self._sessions_lock:
                now = time.monotonic()
                holders = sorted((eid, now - session.touched) for eid, session in self._sessions.items())
            detail = ', '.join(f'{eid} idle {age:.0f}s' for eid, age in holders[:8])
            raise TimeoutError(
                f'{type(self).__name__}: no free session within {effective:.0f}s '
                f'({self._max_sessions} permits, {len(holders)} sessions registered). '
                f'Holders: {detail or "none"}. '
                'Fewer registered sessions than permits means the missing permits are held by resets '
                'still inside _open, or were stranded by an earlier overwrite.')
        try:
            state, result = self._open(traj)
        except BaseException:
            self._session_slots.release()
            raise
        with self._sessions_lock:
            displaced = self._sessions.get(episode_id)
            self._sessions[episode_id] = _Session(state)
        if displaced is not None:
            # A second reset for this id registered its session while this one was still opening. The
            # entry it displaced holds a permit of its own, and close() releases one permit per call and
            # can only ever see the entry that survived -- so overwriting without releasing here strands
            # one permit per occurrence and the pool shrinks until every reset times out.
            try:
                self._close(displaced.state)
            except Exception:  # noqa
                logger.warning(f'{type(self).__name__}: closing the session displaced by a concurrent '
                               f'reset of episode {episode_id!r} failed.', exc_info=True)
            finally:
                self._session_slots.release()
        return result

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def step(self, episode_id: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> StepResult:
        session = self._session(episode_id)
        result = self._step(session.state, tool_name, arguments or {})
        if result.reward is not None:
            session.rewarded = True
            session.reward += float(result.reward)
        _record_tool_exec(episode_id, 1)
        return result

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def step_batch(self, episode_id: str, calls: Sequence[Tuple[str, Dict[str, Any]]]) -> List[StepResult]:
        """Execute already-parsed ``(tool_name, arguments)`` pairs in order, in one round trip."""
        session = self._session(episode_id)
        results = []
        for name, args in calls:
            result = self._step(session.state, name, args or {})
            if result.reward is not None:
                session.rewarded = True
                session.reward += float(result.reward)
            results.append(result)
        _record_tool_exec(episode_id, len(results))
        return results

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def run_script(self,
                   episode_id: str,
                   source: str,
                   interpreter: str = 'python',
                   timeout: Optional[float] = None,
                   cap_memory: bool = True) -> Tuple[int, str]:
        """Run a whole script in the episode's session; returns ``(exit_code, output)``.

        The execution path a *verifier* (or an agent program) takes, as opposed to :meth:`step`,
        which is the one the model takes. Both land in the same place, and that is the point: a
        check has to observe the filesystem the episode actually wrote to.

        Args:
            episode_id: the session.
            source: the script, not a path.
            interpreter: ``'python'`` or ``'shell'``.
            timeout: seconds; ``None`` means the environment's own default.
            cap_memory: whether the environment's per-call TOOL-SCRIPT address-space cap applies.
                The harness's agent program passes ``False``: a real third-party runtime (codex,
                ms_agent) needs far more address space than a runaway-check budget, and its
                containment is the harness timeout, not this cap. Under the cap such a runtime
                stalls silently before its first request. Environments without a host-level cap
                (sandboxed) contain the program at their sandbox boundary either way.

        Returns:
            ``(exit_code, output)``. ``output`` is stdout followed by stderr, so
            a traceback lands at the end rather than interleaved. A non-zero exit
            code is the only failure signal callers should read -- the specific
            value is the script's, except for a timeout, which is
            :data:`TIMEOUT_EXIT_CODE`.
        """
        session = self._session(episode_id)
        session.script_inflight = True
        try:
            return self._run_script(session.state, source, interpreter, timeout, cap_memory)
        finally:
            session.script_inflight = False

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
    def close(self, episode_id: str) -> Optional[float]:
        """End the episode's session and return its reward (the sum of its step rewards).

        ``None`` when the session never scored (no step ever carried a reward) or there was no
        session at all: the caller keeps "no reward" apart from a genuine 0.0 score -- a sandbox
        used only for its tools has no reward of its own, and folding its constant zero as one
        would fabricate signal.
        """
        with self._sessions_lock:
            session = self._sessions.pop(episode_id, None)
        if session is None:
            return None
        try:
            self._close(session.state)
        except Exception:  # noqa
            logger.warning(f'{type(self).__name__}: closing session {episode_id!r} failed.', exc_info=True)
        finally:
            self._session_slots.release()
        return session.reward if session.rewarded else None

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
