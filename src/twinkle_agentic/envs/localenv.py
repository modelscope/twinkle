# Copyright (c) ModelScope Contributors. All rights reserved.
"""LocalEnv: the training host as an environment.

Same interface as the remote sandboxes -- :meth:`step` for the model's tool
calls, :meth:`run_script` for a verifier's script -- so a task does not have to
know which kind of environment it is being graded in. What differs is the
isolation: a subprocess in a new session with a capped address space, not a
microVM.

That makes it the right environment for a check that is a few asserts over pure
computation. A microVM round trip costs hundreds of milliseconds and a
difficulty pass makes one call per candidate per rollout, so the same
verification that takes minutes here takes hours there, for a script that cannot
tell the difference.

It makes it the wrong environment for running code against anything you would
mind that code reading or reaching: there is no filesystem or network isolation,
and the path checks below stop a mistake, not an attempt. Untrusted code belongs
in :class:`~twinkle_agentic.envs.agentenv.AgentEnv` or another sandbox.

Every episode gets its own directory under ``root`` as the working directory of all its calls;
closing the episode removes it, and kills whatever it still had running.
"""
import hashlib
import json
import os
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from twinkle import remote_class, remote_function
from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from .base import (DEFAULT_TOOLS, ENV_MAX_CONCURRENCY, TIMEOUT_EXIT_CODE, Env, StepResult, format_command_output,
                   truncate_observation)
from .resume import EnvResumeCapability, UnsafeCheckpoint

logger = get_logger()

# Kept deterministic and single-threaded: a check that changes its answer with
# the machine's core count is not a check. Matches what the sandboxes set.
_SCRIPT_ENVS = {
    'MPLBACKEND': 'Agg',
    'PYTHONHASHSEED': '0',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'TOKENIZERS_PARALLELISM': 'false',
}


class _LocalSession:
    """One episode: its directory and the processes running in it."""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.procs = set()
        self.lock = threading.Lock()
        self.closed = False


@remote_class(max_concurrency=ENV_MAX_CONCURRENCY)
class LocalEnv(Env):
    """Run scripts and tool calls on this machine. See the module docstring."""

    resume_capability = EnvResumeCapability.DURABLE_CHECKPOINT

    def __init__(self,
                 root: Optional[str] = None,
                 max_sessions: int = 64,
                 command_timeout: int = 60,
                 memory_limit_gb: Optional[float] = 2.0,
                 envs: Optional[Dict[str, str]] = None,
                 session_wait_timeout: Optional[float] = 600.0):
        """
        Args:
            root: directory holding the episodes' working directories, created if absent; a fresh
                temporary directory when omitted.
            max_sessions: how many episodes this replica runs at once.
            command_timeout: default seconds per call, when the caller does not
                pass one.
            memory_limit_gb: address-space cap per call, so one runaway script
                cannot take the trainer down with it. ``None`` to not cap.
            envs: extra environment variables for the child process.
            session_wait_timeout: default seconds ``reset`` waits for a free session (per-call
                ``wait`` overrides it); ``None`` waits forever, ``0`` returns NoSlot immediately.
        """
        super().__init__(max_sessions, session_wait_timeout=session_wait_timeout)
        self._root = os.path.abspath(root) if root else tempfile.mkdtemp(prefix='twinkle_localenv_')
        os.makedirs(self._root, exist_ok=True)
        self._command_timeout = command_timeout
        self._memory_limit_gb = memory_limit_gb
        self._envs = dict(envs or {})
        self._resume_root = Path(self._root) / '.twinkle-resume'

    # ------------------------------------------------------------------
    # Session hooks
    # ------------------------------------------------------------------

    def _open(self, trajectory) -> Tuple[_LocalSession, StepResult]:
        workspace = tempfile.mkdtemp(prefix='ep_', dir=self._root)
        return _LocalSession(workspace), StepResult()

    def _close(self, state: _LocalSession) -> None:
        with state.lock:
            state.closed = True
            procs = list(state.procs)
        for proc in procs:
            _kill_group(proc)
        shutil.rmtree(state.workspace, ignore_errors=True)

    def _workspace(self, state: _LocalSession) -> str:
        return state.workspace

    # ------------------------------------------------------------------
    # Durable session state
    # ------------------------------------------------------------------

    @staticmethod
    def _state_dir_name(state_key: str) -> str:
        # Keep the durable path short enough for Windows' classic MAX_PATH.
        return hashlib.sha256(state_key.encode('utf-8')).hexdigest()[:24]

    @staticmethod
    def _tree_digest(root: Path) -> Dict[str, Any]:
        digest = hashlib.sha256()
        files = []
        total = 0
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise UnsafeCheckpoint(f'workspace snapshot refuses symbolic link {path}')
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            file_digest = hashlib.sha256(data).hexdigest()
            digest.update(relative.encode('utf-8'))
            digest.update(b'\0')
            digest.update(file_digest.encode('ascii'))
            files.append({'path': relative, 'size': len(data), 'sha256': file_digest})
            total += len(data)
        return {'digest': digest.hexdigest(), 'files': files, 'bytes': total}

    def checkpoint_episode(self, episode_id: str, state_key: str) -> Dict[str, Any]:
        """Copy one quiescent workspace behind a new atomic generation pointer."""
        session = self._session(episode_id)
        state = session.state
        if session.script_inflight:
            raise UnsafeCheckpoint(f'episode {episode_id!r} has a script in flight')
        with state.lock:
            if state.closed:
                raise UnsafeCheckpoint(f'episode {episode_id!r} is already closed')
            if state.procs:
                raise UnsafeCheckpoint(f'episode {episode_id!r} has a running process')

        generation = f'{time.time_ns():x}-{uuid.uuid4().hex[:8]}'
        key_root = self._resume_root / self._state_dir_name(state_key)
        generation_root = key_root / generation
        workspace_copy = generation_root / 'workspace'
        generation_root.mkdir(parents=True, exist_ok=False)
        try:
            shutil.copytree(state.workspace, workspace_copy)
            tree = self._tree_digest(workspace_copy)
            manifest = {
                'format': 'twinkle.localenv.v1',
                'state_key': state_key,
                'generation': generation,
                'workspace': tree,
            }
            manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                        separators=(',', ':')).encode('utf-8')
            manifest_path = generation_root / 'manifest.json'
            manifest_path.write_bytes(manifest_bytes)
            with open(manifest_path, 'r+b') as stream:
                os.fsync(stream.fileno())
            head = {
                'generation': generation,
                'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            }
            head_bytes = json.dumps(head, sort_keys=True).encode('utf-8')
            pending = key_root / f'.head.{generation}.tmp'
            key_root.mkdir(parents=True, exist_ok=True)
            with open(pending, 'wb') as stream:
                stream.write(head_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, key_root / 'head.json')
        except BaseException:
            shutil.rmtree(generation_root, ignore_errors=True)
            raise
        for old in key_root.iterdir():
            if old.is_dir() and old.name != generation:
                shutil.rmtree(old, ignore_errors=True)
        return {
            "format": "twinkle.localenv.v1",
            "state_key": state_key,
            "generation": generation,
            "manifest_sha256": head["manifest_sha256"],
            "files": len(tree["files"]),
            "bytes": tree["bytes"],
            "reward": session.reward,
            "rewarded": session.rewarded,
        }

    def restore_episode(self, episode_id: str, state_key: str, state: dict[str, Any], trajectory=None) -> None:
        """Allocate a fresh session and rehydrate the snapshot named by ``state``."""
        if state.get('format') != 'twinkle.localenv.v1' or state.get('state_key') != state_key:
            raise UnsafeCheckpoint('LocalEnv restore metadata does not match the episode state key')
        generation = str(state.get('generation') or '')
        if not generation:
            raise UnsafeCheckpoint('LocalEnv restore metadata has no generation')
        generation_root = self._resume_root / self._state_dir_name(state_key) / generation
        manifest_path = generation_root / 'manifest.json'
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != state.get('manifest_sha256'):
            raise UnsafeCheckpoint('LocalEnv snapshot manifest checksum mismatch')
        manifest = json.loads(manifest_bytes)
        source = generation_root / 'workspace'
        if self._tree_digest(source)['digest'] != manifest['workspace']['digest']:
            raise UnsafeCheckpoint('LocalEnv snapshot workspace checksum mismatch')

        self.reset(episode_id, trajectory)
        try:
            session = self._session(episode_id)
            session.reward = float(state.get("reward") or 0.0)
            session.rewarded = bool(state.get("rewarded", session.reward != 0.0))
            target = Path(self.workspace(episode_id))
            shutil.copytree(source, target, dirs_exist_ok=True)
        except BaseException:
            self.close(episode_id)
            raise

    @remote_function(dispatch="all", collect="none", lazy_collect=False)
    def discard_episode_state(self, state_key: str) -> None:
        shutil.rmtree(self._resume_root / self._state_dir_name(state_key), ignore_errors=True)

    def _run_script(
        self,
        state: _LocalSession,
        source: str,
        interpreter: str = "python",
        timeout: float | None = None,
        cap_memory: bool = True,
    ) -> tuple[int, str]:
        timeout = self._command_timeout if timeout is None else timeout
        # The script file is never written into the workspace. A workspace gets read back -- by a
        # snapshot, or by a check that lists the directory -- and a stray _script.py in there reads
        # as something the episode created.
        holder = tempfile.mkdtemp(prefix='twinkle_local_')
        try:
            if interpreter == 'python':
                path = os.path.join(holder, '_script.py')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(source + '\n')
                argv = [sys.executable, path]
            elif interpreter in ('shell', 'bash'):
                # Not a login shell: sourcing the host's profile prepends whatever
                # banner it prints to the output of every command, and a check
                # comparing that output against an expected string then fails on
                # the banner. PATH and the rest are inherited from the trainer,
                # which is already in the right environment.
                argv = ['/bin/bash', '-c', source]
            else:
                return 1, f"unsupported interpreter {interpreter!r}; use python or shell"
            return self._spawn(state, argv, timeout, cap_memory)
        finally:
            shutil.rmtree(holder, ignore_errors=True)

    def _step(self, state: _LocalSession, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        try:
            if tool_name == 'run_command':
                observation = self.run_command(state, arguments)
            elif tool_name == 'write_file':
                observation = self._write_file(state, arguments)
            elif tool_name == 'read_file':
                observation = self._read_file(state, arguments)
            else:
                available = [t['function']['name'] for t in DEFAULT_TOOLS]
                observation = f'Error: unknown tool {tool_name!r}. Available tools: {available}.'
            return StepResult(observation=observation)
        except Exception as e:  # noqa
            # Same contract as the sandboxed envs: a tool error is an
            # observation, so the rollout loop can let the model recover.
            logger.warning(f'LocalEnv step error (tool={tool_name}): {e}')
            return StepResult(observation=f'Error: {e}', info={'error': str(e)})

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def tools(self) -> List[ToolInfo]:
        return list(DEFAULT_TOOLS)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def run_command(self, state: _LocalSession, arguments: Dict[str, Any]) -> str:
        """Run a shell command in the episode; public so custom tool handlers can reuse it."""
        command = arguments.get('command')
        if not command:
            return "Error: 'command' argument is required."
        cwd = arguments.get('cwd')
        if cwd:
            command = f'cd {shlex.quote(str(cwd))} && {command}'
        exit_code, output = self._run_script(state, command, 'shell', timeout=arguments.get('timeout'))
        # stderr is already folded into output by run_script, hence the empty
        # stream here: what this call adds is the exit-code line.
        return format_command_output(output, '', exit_code)

    def _write_file(self, state: _LocalSession, arguments: Dict[str, Any]) -> str:
        path = self._resolve(state, arguments['path'])
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(arguments.get('content', ''))
        return f"File written: {arguments['path']}"

    def _read_file(self, state: _LocalSession, arguments: Dict[str, Any]) -> str:
        with open(self._resolve(state, arguments['path']), encoding='utf-8', errors='replace') as f:
            return truncate_observation(f.read())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(state: _LocalSession, path: str) -> str:
        """Resolve a tool-supplied path inside the episode's workspace.

        The workspace is the root, so an absolute path means absolute *in here*
        -- the tool schema is shared with the sandboxed envs, where it genuinely
        is the filesystem root. An escape raises: with no isolation underneath, a
        relative path with enough ``..`` in it would otherwise be writing to the
        training host. This bounds a mistake; it is not a security boundary,
        since ``run_command`` reaches the same filesystem directly.
        """
        root = os.path.realpath(state.workspace)
        target = os.path.realpath(os.path.join(root, str(path).lstrip('/')))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f'path {path!r} escapes the workspace')
        return target

    def _spawn(self, state: _LocalSession, argv: list[str], timeout: float, cap_memory: bool = True) -> tuple[int, str]:
        env = dict(os.environ, **_SCRIPT_ENVS, **self._envs)
        # Inherited from the trainer, and a check that imports torch would
        # otherwise take a share of a GPU that is mid-generation.
        env.pop('CUDA_VISIBLE_DEVICES', None)

        def _limit():
            if cap_memory and self._memory_limit_gb:
                cap = int(self._memory_limit_gb * 1024**3)
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

        with state.lock:
            if state.closed:
                return 1, 'the episode was closed'
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=state.workspace,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors='replace',
                    start_new_session=True,
                    preexec_fn=_limit)
            except Exception as e:  # noqa
                # A spawn failure is the host's problem, not the script's, and it
                # comes back as a failed run so one bad call cannot end a whole pass.
                return 1, f'{type(e).__name__}: {e}'
            state.procs.add(proc)
        try:
            out, err = proc.communicate(timeout=timeout)
            out, err = out or '', err or ''
            # A newline between the streams: a stdout line left unterminated
            # swallows the first line of the traceback that follows it.
            if out and err and not out.endswith('\n'):
                out += '\n'
            return proc.returncode, out + err
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                proc.communicate(timeout=5)
            except Exception:  # noqa # already killed; the output is forfeit
                pass
            # A killed script has no traceback to explain itself with, so the
            # output has to say why it produced nothing.
            return TIMEOUT_EXIT_CODE, f'execution timed out after {timeout}s (possible infinite loop)'
        finally:
            with state.lock:
                state.procs.discard(proc)


def _kill_group(proc: subprocess.Popen) -> None:
    # killpg, not kill: start_new_session gave the script its own process
    # group, so a script that forked cannot leave grandchildren running.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
