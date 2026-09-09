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

Two shapes, chosen by ``workspace``:

* ``workspace=<dir>``: that directory is the working directory for every call and
  outlives them all. What a multi-turn episode needs -- the model writes a file
  with one tool call, and the check script reads it back after the episode ends.
* ``workspace=None``: every call runs in a fresh temporary directory that is
  removed afterwards. What one-shot verification needs, and the reason the code
  half has no workspace to reset: nothing survives a call to leak into the next
  one. The file tools are withdrawn in this shape, because a file written by one
  call would not be there for the next.
"""
import os
import resource
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from .base import (DEFAULT_TOOLS, TIMEOUT_EXIT_CODE, Env, StepResult, format_command_output,
                   truncate_observation)

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


class LocalEnv(Env):
    """Run scripts and tool calls on this machine. See the module docstring."""

    def __init__(self,
                 workspace: Optional[str] = None,
                 command_timeout: int = 60,
                 memory_limit_gb: Optional[float] = 2.0,
                 envs: Optional[Dict[str, str]] = None):
        """
        Args:
            workspace: persistent working directory, created if absent. ``None``
                gives every call its own temporary directory and withdraws the
                file tools.
            command_timeout: default seconds per call, when the caller does not
                pass one.
            memory_limit_gb: address-space cap per call, so one runaway script
                cannot take the trainer down with it. ``None`` to not cap.
            envs: extra environment variables for the child process.
        """
        self._workspace = os.path.abspath(workspace) if workspace else None
        if self._workspace:
            os.makedirs(self._workspace, exist_ok=True)
        self._command_timeout = command_timeout
        self._memory_limit_gb = memory_limit_gb
        self._envs = dict(envs or {})

    @property
    def workspace(self) -> Optional[str]:
        """The persistent working directory, or None in the throwaway shape."""
        return self._workspace

    # ------------------------------------------------------------------
    # Env interface
    # ------------------------------------------------------------------

    def run_script(self, source: str, interpreter: str = 'python',
                   timeout: Optional[int] = None) -> Tuple[int, str]:
        timeout = self._command_timeout if timeout is None else timeout
        # The script file is never written into the workspace. A persistent
        # workspace gets read back -- by a snapshot, or by a check that lists the
        # directory -- and a stray _script.py in there reads as something the
        # episode created. With no workspace this same directory is the working
        # directory, which is what makes that shape leave nothing behind.
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
                return 1, f'unsupported interpreter {interpreter!r}; use python or shell'
            return self._spawn(argv, self._workspace or holder, timeout)
        finally:
            shutil.rmtree(holder, ignore_errors=True)

    def step(self, tool_name: str, arguments: Dict[str, Any] = None) -> StepResult:
        arguments = arguments or {}
        try:
            if tool_name == 'run_command':
                observation = self.run_command(arguments)
            elif tool_name in ('write_file', 'read_file'):
                if self._workspace is None:
                    # Not an error the model can recover from by rephrasing, so
                    # it says what is missing rather than what went wrong.
                    observation = (f'Error: {tool_name} needs a persistent workspace; '
                                   'this environment runs every call in a fresh directory.')
                elif tool_name == 'write_file':
                    observation = self._write_file(arguments)
                else:
                    observation = self._read_file(arguments)
            else:
                available = [t['function']['name'] for t in self.tools()]
                observation = f'Error: unknown tool {tool_name!r}. Available tools: {available}.'
            return StepResult(observation=observation)
        except Exception as e:  # noqa
            # Same contract as the sandboxed envs: a tool error is an
            # observation, so the rollout loop can let the model recover.
            logger.warning(f'LocalEnv step error (tool={tool_name}): {e}')
            return StepResult(observation=f'Error: {e}', info={'error': str(e)})

    def tools(self) -> List[ToolInfo]:
        if self._workspace is None:
            # Nothing an episode could build on: every call would start from an
            # empty directory, so this shape is a verifier, not an environment.
            return []
        return list(DEFAULT_TOOLS)

    def clear(self) -> None:
        """Empty the workspace. A no-op in the throwaway shape, which has none.

        Raises rather than reporting, per :meth:`Env.clear`: a caller that clears
        before every job is depending on this, and the failure it guards against
        -- a job inheriting the previous one's files -- is invisible downstream.
        """
        if self._workspace is None:
            return
        for name in os.listdir(self._workspace):
            path = os.path.join(self._workspace, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def run_command(self, arguments: Dict[str, Any]) -> str:
        """Run a shell command; public so custom tool handlers can reuse it."""
        command = arguments.get('command')
        if not command:
            return "Error: 'command' argument is required."
        cwd = arguments.get('cwd')
        if cwd:
            command = f'cd {shlex.quote(str(cwd))} && {command}'
        exit_code, output = self.run_script(command, 'shell', timeout=arguments.get('timeout'))
        # stderr is already folded into output by run_script, hence the empty
        # stream here: what this call adds is the exit-code line.
        return format_command_output(output, '', exit_code)

    def _write_file(self, arguments: Dict[str, Any]) -> str:
        path = self._resolve(arguments['path'])
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(arguments.get('content', ''))
        return f"File written: {arguments['path']}"

    def _read_file(self, arguments: Dict[str, Any]) -> str:
        with open(self._resolve(arguments['path']), encoding='utf-8', errors='replace') as f:
            return truncate_observation(f.read())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> str:
        """Resolve a tool-supplied path inside the workspace.

        The workspace is the root, so an absolute path means absolute *in here*
        -- the tool schema is shared with the sandboxed envs, where it genuinely
        is the filesystem root. An escape raises: with no isolation underneath, a
        relative path with enough ``..`` in it would otherwise be writing to the
        training host. This bounds a mistake; it is not a security boundary,
        since ``run_command`` reaches the same filesystem directly.
        """
        root = os.path.realpath(self._workspace)
        target = os.path.realpath(os.path.join(root, str(path).lstrip('/')))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f'path {path!r} escapes the workspace')
        return target

    def _spawn(self, argv: List[str], cwd: str, timeout: int) -> Tuple[int, str]:
        env = dict(os.environ, **_SCRIPT_ENVS, **self._envs)
        # Inherited from the trainer, and a check that imports torch would
        # otherwise take a share of a GPU that is mid-generation.
        env.pop('CUDA_VISIBLE_DEVICES', None)

        def _limit():
            if self._memory_limit_gb:
                cap = int(self._memory_limit_gb * 1024**3)
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, errors='replace',
                                    start_new_session=True, preexec_fn=_limit)
        except Exception as e:  # noqa
            # A spawn failure is the host's problem, not the script's, and it
            # comes back as a failed run so one bad call cannot end a whole pass.
            return 1, f'{type(e).__name__}: {e}'
        try:
            out, err = proc.communicate(timeout=timeout)
            out, err = out or '', err or ''
            # A newline between the streams: a stdout line left unterminated
            # swallows the first line of the traceback that follows it.
            if out and err and not out.endswith('\n'):
                out += '\n'
            return proc.returncode, out + err
        except subprocess.TimeoutExpired:
            # killpg, not kill: start_new_session gave the script its own process
            # group, so a script that forked cannot leave grandchildren running.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:  # noqa # already killed; the output is forfeit
                pass
            # A killed script has no traceback to explain itself with, so the
            # output has to say why it produced nothing.
            return TIMEOUT_EXIT_CODE, f'execution timed out after {timeout}s (possible infinite loop)'
