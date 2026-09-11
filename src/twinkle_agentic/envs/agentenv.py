# Copyright (c) ModelScope Contributors. All rights reserved.
"""AgentENV adapter: thin client-side Env over an AgentENV (AENV) deployment.

AgentENV (https://github.com/kvcache-ai/AgentENV) runs Firecracker microVM
sandboxes behind an E2B-compatible HTTP API. Unlike ``OpenEnv``/``EnvPool``,
this adapter deliberately does NOT use ``@remote_class``: sandbox placement,
load balancing, pause/resume and node failover are all handled server-side by
AgentENV's gateway/scheduler/orchestrator. The adapter is a stateless HTTP
client and can be instantiated directly inside rollout workers.

Prerequisites (done once, outside training):
    1. Deploy the AgentENV server (single node) or gateway+scheduler cluster.
    2. Build a template, e.g. ``aenv pull ubuntu:22.04 --name my-env``.
    3. ``pip install 'e2b>=2.7'`` on the training side (the version that takes
       the endpoint as an argument rather than only from the environment).

Usage::

    env = AgentEnv(template='my-env', api_url='http://gateway:8080')
    result = env.reset()
    result = env.step('run_command', {'command': 'echo hello'})
    env.close()
"""
import os
import posixpath
import shlex
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from .base import (DEFAULT_TOOLS, TIMEOUT_EXIT_CODE, Env, StepResult,
                   format_command_output, truncate_observation)

logger = get_logger()

# Uploaded scripts land here, never in the workspace. The workspace is read back
# -- by a snapshot, by a check that lists it -- and a stray _script.py in there
# reads as something the episode created.
_SCRIPT_DIR = '/tmp/twinkle_scripts'

# Emptied entry by entry and then asserted empty, rather than `rm -rf`: a clear
# that silently did nothing hands the next episode the previous one's files,
# which lets a solver pass without having done anything.
_CLEAR_WORKSPACE = '''
import os, shutil
root = {root!r}
os.makedirs(root, exist_ok=True)
for name in os.listdir(root):
    path = os.path.join(root, name)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
rest = os.listdir(root)
assert not rest, 'workspace not empty after clear: %r' % rest[:10]
'''


def _require_e2b():
    """Import the e2b SDK lazily with an actionable error message."""
    try:
        from e2b import Sandbox
    except ImportError as e:
        raise ImportError('AgentEnv requires the E2B SDK to talk to an AgentENV server:\n'
                          "  pip install 'e2b>=2.7'\n"
                          'Then point it at your deployment via the api_url/api_key/sandbox_url '
                          'arguments, or the E2B_API_URL / E2B_SANDBOX_URL / E2B_API_KEY '
                          'environment variables.') from e
    return Sandbox


class AgentEnv(Env):
    """Env backed by one AgentENV sandbox per episode.

    Lifecycle mapping:
        * ``reset``  -> kill the previous sandbox (if any) and create a fresh
          one from ``template``; AgentENV's scheduler picks the node.
        * ``step``   -> execute a tool inside the sandbox (sticky-routed to
          the owning node via the sandbox id header, handled by the SDK).
        * ``clear``  -> empty the workspace, keeping the sandbox; boot one if
          there is none yet, so a slot comes up on first use.
        * ``close``  -> kill the sandbox.

    Everything an episode does happens under ``workspace``: tool calls run there
    and :meth:`run_script` runs there, which is what lets a verifier observe the
    filesystem the episode actually wrote to. A caller that holds one sandbox
    across many episodes clears between them and calls :meth:`ensure_ready` to
    survive an eviction; both are only safe where the workspace is about to be
    discarded anyway.

    Built-in tools (can be disabled via ``include_default_tools=False``):
    ``run_command``, ``write_file``, ``read_file``. Task-specific tools can
    be added with :meth:`register_tool` (arbitrary python handler) or
    :meth:`register_command_tool` (shell command template), or by
    subclassing. Tool errors never raise; they come back as observations so
    the rollout loop can continue or let the model recover.

    Note: rewards are not produced by the sandbox. Keep the default
    ``evaluate`` (zeros) and score trajectories with a separate reward
    function, or subclass and override ``step``/``evaluate``.
    """

    def __init__(self,
                 template: str,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 sandbox_url: Optional[str] = None,
                 workspace: str = '/workspace',
                 sandbox_timeout: int = 300,
                 command_timeout: int = 120,
                 setup_commands: Optional[List[str]] = None,
                 sandbox_envs: Optional[Dict[str, str]] = None,
                 metadata: Optional[Dict[str, str]] = None,
                 refresh_timeout: bool = True,
                 include_default_tools: bool = True,
                 **kwargs):
        """
        Args:
            template: AgentENV template name/ID (``aenv pull ... --name <template>``).
            api_url: AgentENV server or gateway base URL, the control plane.
                Omit to leave it to the SDK, which reads ``E2B_API_URL``.
            api_key: API key; AgentENV accepts any non-empty string on a
                trusted network. Omit to read ``E2B_API_KEY``, falling back to
                a placeholder, since the SDK requires a key to be present.
            sandbox_url: the data plane, for a deployment whose sandbox gateway
                answers on a different host than the API. Defaults to
                ``api_url``: one host serves both unless told otherwise.
            workspace: absolute path inside the sandbox that every tool call and
                script runs in, created on reset. One directory, so that what an
                episode writes is what a check reads back.
            sandbox_timeout: Sandbox idle timeout in seconds. AgentENV pauses
                (not kills) idle sandboxes and auto-resumes them on access.
            command_timeout: Per-command execution timeout in seconds.
            setup_commands: Optional commands run once after each reset.
            sandbox_envs: Environment variables injected into the sandbox.
            metadata: Sandbox metadata (visible in list APIs, useful for
                tagging the run name / trajectory id).
            refresh_timeout: Extend the sandbox timeout after every step so
                long multi-turn episodes are not paused mid-flight.
            include_default_tools: Expose the built-in run_command /
                write_file / read_file tools. Set False to expose only
                tools registered via ``register_tool``/``register_command_tool``.
        """
        if not template:
            raise ValueError("AgentEnv requires 'template'. Build one first, e.g. "
                             '`aenv pull ubuntu:22.04 --name my-env`.')
        # Where this deployment lives travels with the instance, as arguments to
        # the SDK, rather than through the E2B_* environment variables the SDK
        # would otherwise read: those are process-global, so two AgentEnvs
        # pointing at different deployments would overwrite each other and the
        # last one constructed would decide for all of them. A key left out is
        # left for the SDK to resolve, so a deployment configured entirely
        # through the environment keeps working. Passed once, at create: the
        # sandbox keeps this configuration for every later call on it.
        self._api_params: Dict[str, Any] = {
            'api_key': api_key or os.environ.get('E2B_API_KEY') or 'dummy',
            # AgentENV issues no e2b-format keys, and the SDK used to assert a
            # key matched ``e2b_[0-9a-f]+`` before it ever sent a request, which
            # rejects placeholders like 'dummy'. Newer SDKs dropped the check
            # and ignore this.
            'validate_api_key': False,
        }
        if api_url:
            self._api_params['api_url'] = api_url
        if sandbox_url or api_url:
            self._api_params['sandbox_url'] = sandbox_url or api_url

        self._template = template
        self._workspace = workspace
        self._sandbox_timeout = sandbox_timeout
        self._command_timeout = command_timeout
        self._setup_commands = setup_commands or []
        self._sandbox_envs = sandbox_envs
        self._metadata = metadata
        self._refresh_timeout = refresh_timeout
        self._include_default_tools = include_default_tools
        self._custom_tools: List[ToolInfo] = []
        self._custom_handlers: Dict[str, Callable[['AgentEnv', Dict[str, Any]], str]] = {}
        self._sandbox = None

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tool(self, tool_info: ToolInfo, handler: Callable[['AgentEnv', Dict[str, Any]], str]) -> 'AgentEnv':
        """Register a custom tool.

        Args:
            tool_info: OpenAI-format tool schema exposed to the model.
            handler: ``handler(env, arguments) -> str`` returning the
                observation; use ``env.run_command(...)`` / ``env.sandbox``
                to interact with the sandbox. Overrides a built-in tool if
                the name collides.

        Returns:
            self, to allow chained registration.
        """
        name = tool_info.get('function', {}).get('name')
        if not name:
            raise ValueError(f'tool_info must contain function.name, got: {tool_info!r}')
        self._custom_tools = [t for t in self._custom_tools if t['function']['name'] != name]
        self._custom_tools.append(tool_info)
        self._custom_handlers[name] = handler
        return self

    def register_command_tool(self, tool_info: ToolInfo, command_template: str) -> 'AgentEnv':
        """Register a tool whose handler is a shell command template.

        The template is formatted with the tool arguments, e.g.::

            env.register_command_tool(
                {'type': 'function', 'function': {
                    'name': 'run_tests',
                    'description': 'Run the task test suite.',
                    'parameters': {'type': 'object', 'properties': {
                        'test_file': {'type': 'string'}}, 'required': ['test_file']},
                }},
                'cd /workspace && pytest {test_file} -x -q')
        """

        def handler(env: 'AgentEnv', arguments: Dict[str, Any]) -> str:
            try:
                command = command_template.format(**arguments)
            except KeyError as e:
                return f'Error: missing required argument {e} for this tool.'
            return env.run_command({'command': command})

        return self.register_tool(tool_info, handler)

    # ------------------------------------------------------------------
    # Env interface
    # ------------------------------------------------------------------

    def reset(self, trajectory: Optional[Trajectory] = None) -> StepResult:
        sandbox_cls = _require_e2b()
        self._kill_sandbox()
        self._sandbox = sandbox_cls.create(
            self._template,
            timeout=self._sandbox_timeout,
            envs=self._sandbox_envs,
            metadata=self._metadata,
            **self._api_params,
        )
        setup_output = []
        # Before the setup commands, because they are written against it, and
        # because commands.run(cwd=...) fails outright on a missing directory --
        # a template without this path would otherwise break every call.
        self.run_command({'command': f'mkdir -p {shlex.quote(self._workspace)} '
                                     f'{shlex.quote(_SCRIPT_DIR)}', 'cwd': '/'})
        for cmd in self._setup_commands:
            result = self.run_command({'command': cmd})
            setup_output.append(result)
        logger.info(f'AgentEnv sandbox created: {self.sandbox_id} (template={self._template})')
        return StepResult(
            observation='\n'.join(setup_output) if setup_output else '',
            reward=0.0,
            done=False,
            info={'sandbox_id': self.sandbox_id},
        )

    def step(self, tool_name: str, arguments: Dict[str, Any] = None) -> StepResult:
        if self._sandbox is None:
            return StepResult(observation='Error: sandbox not initialized, call reset() first.', done=True)
        arguments = arguments or {}
        try:
            if tool_name in self._custom_handlers:
                observation = self._custom_handlers[tool_name](self, arguments)
            elif self._include_default_tools and tool_name == 'run_command':
                observation = self.run_command(arguments)
            elif self._include_default_tools and tool_name == 'write_file':
                self._sandbox.files.write(self._resolve(arguments['path']), arguments.get('content', ''))
                observation = f"File written: {arguments['path']}"
            elif self._include_default_tools and tool_name == 'read_file':
                observation = truncate_observation(str(self._sandbox.files.read(self._resolve(arguments['path']))))
            else:
                available = [t['function']['name'] for t in self.tools()]
                observation = f'Error: unknown tool {tool_name!r}. Available tools: {available}.'
            self._touch()
            return StepResult(observation=observation, reward=0.0, done=False, info={'sandbox_id': self.sandbox_id})
        except Exception as e:  # noqa
            # Keep the episode alive on transient tool errors; the rollout
            # loop (max_turns) bounds retries.
            logger.warning(f'AgentEnv step error (sandbox={self.sandbox_id}): {e}')
            return StepResult(observation=f'Error: {e}', reward=0.0, done=False, info={'error': str(e)})

    def run_script(self, source: str, interpreter: str = 'python',
                   timeout: Optional[int] = None) -> Tuple[int, str]:
        """Run a whole script in the workspace; returns ``(exit_code, output)``.

        The verifier's path, as opposed to :meth:`step`. Both land in the same
        directory inside the same microVM, which is the point: a check has to
        observe the filesystem the episode wrote to.

        A python script is uploaded and run by path rather than passed to
        ``python -c``: a traceback then carries real line numbers, and nothing
        has to survive shell quoting. A sandbox that has gone away comes back as
        a non-zero exit like any other failure -- recovery is
        :meth:`ensure_ready`, called where losing the workspace is acceptable.
        """
        if self._sandbox is None:
            return 1, 'sandbox not initialized, call reset() first'
        timeout = self._command_timeout if timeout is None else timeout
        if interpreter == 'python':
            path = f'{_SCRIPT_DIR}/{uuid.uuid4().hex}.py'
            try:
                self._sandbox.files.write(path, source + '\n')
            except Exception as e:  # noqa
                return 1, f'could not upload the script: {type(e).__name__}: {e}'
            command = f'python3 {shlex.quote(path)}'
        elif interpreter in ('shell', 'bash'):
            command = source
        else:
            return 1, f'unsupported interpreter {interpreter!r}; use python or shell'
        return self._execute(command, self._workspace, timeout)

    def clear(self) -> None:
        """Empty the workspace, keeping the sandbox. Raises if it could not.

        Cheaper than a fresh sandbox by a boot, which is what makes it worth
        having: a run doing this between every episode pays the difference every
        time. Raising is per :meth:`Env.clear` -- a caller that clears before
        every job depends on this, and the failure it guards against is a job
        inheriting the previous one's files, invisible downstream.

        With no sandbox yet, this boots one: :meth:`Env.clear` promises an
        environment ready for the next episode, and returning without a sandbox
        would hand over one whose every call answers ``call reset() first``. So
        this is also where a slot first comes up, from the clear its owner does
        before handing it out -- nobody has to know to call :meth:`reset`.
        """
        if self._sandbox is None:
            # A fresh microVM is already empty, and reset() runs the setup
            # commands the workspace is supposed to start with.
            self.reset()
            return
        exit_code, output = self.run_script(_CLEAR_WORKSPACE.format(root=self._workspace))
        if exit_code != 0:
            raise RuntimeError(f'could not clear {self._workspace} in sandbox '
                               f'{self.sandbox_id}: {output}')

    def healthy(self) -> bool:
        """Does the sandbox answer right now?

        A command rather than a status field: AgentENV pauses an idle sandbox and
        resumes it on access, so what matters is whether it can be reached and
        made to run something, not what a list API last recorded about it.
        """
        if self._sandbox is None:
            return False
        exit_code, _ = self._execute('true', None, timeout=10)
        return exit_code == 0

    def ensure_ready(self) -> bool:
        """Re-establish the sandbox if it has gone away. True if it did."""
        if self.healthy():
            return False
        logger.warning(f'AgentEnv sandbox {self.sandbox_id} unreachable; rebuilding')
        self.rebuild()
        return True

    def rebuild(self) -> None:
        """Throw the sandbox away and boot a replacement, counting the recovery.

        A microVM is disposable, so there is nothing to repair: :meth:`reset`
        already kills the old one. All this adds is the count, which is the part
        a run reports at the end.
        """
        self.n_recoveries += 1
        logger.warning(f'AgentEnv rebuilding the sandbox (recovery #{self.n_recoveries})')
        self.reset()

    def tools(self) -> List[ToolInfo]:
        """What the model is told it can call here.

        Registered handlers stand alongside the defaults and take precedence over
        one of the same name, since a caller adding one is naming a tool this env
        is to execute itself.
        """
        tools: List[ToolInfo] = []
        if self._include_default_tools:
            custom_names = set(self._custom_handlers)
            tools.extend(t for t in DEFAULT_TOOLS if t['function']['name'] not in custom_names)
        tools.extend(self._custom_tools)
        return tools

    def close(self) -> None:
        self._kill_sandbox()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def workspace(self) -> str:
        """The directory inside the sandbox that every call runs in."""
        return self._workspace

    @property
    def sandbox_id(self) -> Optional[str]:
        return getattr(self._sandbox, 'sandbox_id', None)

    @property
    def sandbox(self):
        """The underlying E2B sandbox handle, for custom tool handlers."""
        return self._sandbox

    def run_command(self, arguments: Dict[str, Any]) -> str:
        """Run a shell command in the sandbox; public so custom handlers can reuse it.

        Defaults to the workspace rather than the sandbox's login directory, so a
        command the model writes without a path acts on the same files the check
        will read.
        """
        command = arguments.get('command')
        if not command:
            return "Error: 'command' argument is required."
        timeout = int(arguments.get('timeout') or self._command_timeout)
        exit_code, output = self._execute(command, arguments.get('cwd') or self._workspace, timeout)
        # stderr is already folded into output by _execute, hence the empty
        # stream here: what this call adds is the exit-code line.
        return format_command_output(output, '', exit_code)

    def _touch(self) -> None:
        """Push the sandbox's expiry back, best effort.

        Called after tool work rather than on a timer: the idle clock is what
        reclaims a slot, and a slot busy running an episode's tools is exactly
        the one that must not be reclaimed. Failing here is not worth an
        episode -- the next call reports the loss with something to say about it.
        """
        if not self._refresh_timeout or self._sandbox is None:
            return
        try:
            self._sandbox.set_timeout(self._sandbox_timeout)
        except Exception:  # noqa # best-effort keepalive
            pass

    def _resolve(self, path: str) -> str:
        """A tool-supplied path, relative ones taken from the workspace.

        An absolute path is left alone: inside a microVM it means what it says,
        and a template's own directories are fair game. Relative is where the
        episode and its check have to agree, so it is anchored rather than left
        to whatever directory the SDK defaults to.
        """
        path = str(path)
        return path if path.startswith('/') else posixpath.join(self._workspace, path)

    def _execute(self, command: str, cwd: Optional[str], timeout: int) -> Tuple[int, str]:
        """One command in the sandbox as ``(exit_code, stdout + stderr)``.

        Never raises. Everything that can go wrong here -- the script failed, the
        command hung, the sandbox is gone -- is reported as a non-zero exit with
        the output that explains it, because the two callers both need that: a
        check reads the status, and :meth:`healthy` reads it to decide whether
        this sandbox still exists.
        """
        try:
            result = self._sandbox.commands.run(command, cwd=cwd, timeout=timeout)
            return int(result.exit_code or 0), self._merge(result.stdout, result.stderr)
        except Exception as e:  # noqa
            # The SDK raises on a non-zero exit, carrying the streams on the
            # exception; that is the script's own failure and belongs to the
            # caller, not to error handling.
            exit_code = getattr(e, 'exit_code', None)
            if exit_code is not None:
                # The streams are the whole truth here: a script that failed
                # silently reported the exception's own repr as its output, which
                # reads like something the script printed.
                return int(exit_code), self._merge(getattr(e, 'stdout', ''), getattr(e, 'stderr', ''))
            if 'timeout' in type(e).__name__.lower():
                # No traceback to explain itself with, so the output has to.
                return TIMEOUT_EXIT_CODE, f'execution timed out after {timeout}s (possible infinite loop)'
            return 1, f'{type(e).__name__}: {e}'

    @staticmethod
    def _merge(stdout: Optional[str], stderr: Optional[str]) -> str:
        """stdout then stderr, so a traceback lands at the end rather than inline."""
        out, err = stdout or '', stderr or ''
        # A stdout line left unterminated would otherwise swallow the first line
        # of the traceback that follows it.
        if out and err and not out.endswith('\n'):
            out += '\n'
        return out + err

    def _kill_sandbox(self) -> None:
        if self._sandbox is None:
            return
        sandbox_id = self.sandbox_id
        try:
            self._sandbox.kill()
        except Exception as e:  # noqa # best-effort: AgentENV auto-evicts on timeout anyway
            logger.warning(f'AgentEnv failed to kill sandbox {sandbox_id}: {e}')
        finally:
            self._sandbox = None
