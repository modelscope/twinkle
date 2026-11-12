# Copyright (c) ModelScope Contributors. All rights reserved.
"""AgentENV adapter: Env over an AgentENV (AENV) deployment, one microVM sandbox per episode.

AgentENV (https://github.com/kvcache-ai/AgentENV) runs Firecracker microVM
sandboxes behind an E2B-compatible HTTP API. Sandbox placement, load balancing,
pause/resume and node failover are all handled server-side by AgentENV's
gateway/scheduler/orchestrator; this component only holds the sandbox handles of
the episodes it serves, and keeps the sandboxes of finished episodes (emptied) to
hand to the next ones instead of booting a fresh microVM every time.

Prerequisites (done once, outside training):
    1. Deploy the AgentENV server (single node) or gateway+scheduler cluster.
    2. Build a template, e.g. ``aenv pull ubuntu:22.04 --name my-env``.
    3. ``pip install 'e2b>=2.7'`` on the training side (the version that takes
       the endpoint as an argument rather than only from the environment).

Usage::

    env = AgentEnv(template='my-env', api_url='http://gateway:8080')
    env.reset(episode_id)
    result = env.step(episode_id, 'run_command', {'command': 'echo hello'})
    env.close(episode_id)
"""
import json
import os
import posixpath
import shlex
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from twinkle import remote_class, remote_function
from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from .base import (DEFAULT_TOOLS, ENV_MAX_CONCURRENCY, TIMEOUT_EXIT_CODE, Env, StepResult, format_command_output,
                   truncate_observation)

logger = get_logger()

# Uploaded scripts land here, never in the workspace. The workspace is read back
# -- by a snapshot, by a check that lists it -- and a stray _script.py in there
# reads as something the episode created.
_SCRIPT_DIR = '/tmp/twinkle_scripts'


def _as_file_content(content: Any) -> Union[str, bytes]:
    """Coerce a ``write_file`` payload to something the sandbox FS accepts.

    A model calling ``write_file`` routinely hands back structured content -- a
    dict or list, not a string -- while e2b's ``files.write`` takes only
    str/bytes/IO. That is a data shape, not a failure: JSON-encode containers
    and stringify everything else so the write lands rather than sinking the
    whole step (and, downstream, the episode as ``empty_workspace``).
    """
    if isinstance(content, (str, bytes)):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


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


def _merge(stdout: Optional[str], stderr: Optional[str]) -> str:
    """stdout then stderr, so a traceback lands at the end rather than inline."""
    out, err = stdout or '', stderr or ''
    # A stdout line left unterminated would otherwise swallow the first line
    # of the traceback that follows it.
    if out and err and not out.endswith('\n'):
        out += '\n'
    return out + err


class AgentSandbox:
    """One episode's sandbox; what a custom tool handler receives.

    ``sandbox`` is the E2B handle; :meth:`run_command` runs a shell command in the workspace. A
    handler may keep its own per-episode values as attributes here.
    """

    def __init__(self, sandbox: Any, workspace: str, command_timeout: int):
        self.sandbox = sandbox
        self.workspace = workspace
        self.command_timeout = command_timeout
        #: Commands running right now; a sandbox closed with one still running is killed, not reused.
        self.running = 0
        self._running_lock = threading.Lock()

    @property
    def sandbox_id(self) -> Optional[str]:
        return getattr(self.sandbox, 'sandbox_id', None)

    def execute(self, command: str, cwd: Optional[str], timeout: float) -> Tuple[int, str]:
        """One command in the sandbox as ``(exit_code, stdout + stderr)``.

        Never raises. Everything that can go wrong here -- the script failed, the
        command hung, the sandbox is gone -- is reported as a non-zero exit with
        the output that explains it: a check reads the status, and the health probe
        reads it to decide whether this sandbox still exists.
        """
        with self._running_lock:
            self.running += 1
        try:
            result = self.sandbox.commands.run(command, cwd=cwd, timeout=timeout)
            if result.exit_code is None:
                return 1, 'AgentEnv returned no exit code for the command.'
            return int(result.exit_code), _merge(result.stdout, result.stderr)
        except Exception as e:  # noqa
            # The SDK raises on a non-zero exit, carrying the streams on the
            # exception; that is the script's own failure and belongs to the
            # caller, not to error handling.
            exit_code = getattr(e, 'exit_code', None)
            if exit_code is not None:
                return int(exit_code), _merge(getattr(e, 'stdout', ''), getattr(e, 'stderr', ''))
            if 'timeout' in type(e).__name__.lower():
                return TIMEOUT_EXIT_CODE, f'execution timed out after {timeout}s'
            return 1, f'{type(e).__name__}: {e}'
        finally:
            with self._running_lock:
                self.running -= 1

    def run_command(self, arguments: Dict[str, Any]) -> str:
        """Run a shell command in the sandbox; public so custom handlers can reuse it.

        Defaults to the workspace rather than the sandbox's login directory, so a
        command the model writes without a path acts on the same files the check
        will read.
        """
        command = arguments.get('command')
        if not command:
            return "Error: 'command' argument is required."
        timeout = int(arguments.get('timeout') or self.command_timeout)
        exit_code, output = self.execute(command, arguments.get('cwd') or self.workspace, timeout)
        # stderr is already folded into output by execute, hence the empty
        # stream here: what this call adds is the exit-code line.
        return format_command_output(output, '', exit_code)

    def resolve(self, path: str) -> str:
        """A tool-supplied path, relative ones taken from the workspace.

        An absolute path is left alone: inside a microVM it means what it says,
        and a template's own directories are fair game. Relative is where the
        episode and its check have to agree, so it is anchored rather than left
        to whatever directory the SDK defaults to.
        """
        path = str(path)
        return path if path.startswith('/') else posixpath.join(self.workspace, path)


@remote_class(max_concurrency=ENV_MAX_CONCURRENCY)
class AgentEnv(Env):
    """Env backed by one AgentENV sandbox per episode.

    Lifecycle mapping:
        * ``reset``  -> take an emptied sandbox from the finished episodes, or create a fresh one
          from ``template`` (AgentENV's scheduler picks the node), and run ``setup_commands``.
        * ``step``   -> execute a tool inside the sandbox (sticky-routed to
          the owning node via the sandbox id header, handled by the SDK).
        * ``close``  -> empty the workspace and keep the sandbox for the next episode; one that
          cannot be emptied, or more than ``max_sessions`` kept, is killed.

    Everything an episode does happens under ``workspace``: tool calls run there
    and :meth:`run_script` runs there, which is what lets a verifier observe the
    filesystem the episode actually wrote to.

    Built-in tools (can be disabled via ``include_default_tools=False``):
    ``run_command``, ``write_file``, ``read_file``. Task-specific tools can
    be added with :meth:`register_tool` (arbitrary python handler) or
    :meth:`register_command_tool` (shell command template), or by
    subclassing. Tool errors never raise; they come back as observations so
    the rollout loop can continue or let the model recover.

    Note: rewards are not produced by the sandbox; score trajectories with a separate reward
    function, or subclass and override ``_step``.
    """

    def __init__(self,
                 template: str,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 sandbox_url: Optional[str] = None,
                 workspace: str = '/workspace',
                 max_sessions: int = 64,
                 sandbox_timeout: int = 300,
                 command_timeout: int = 120,
                 setup_commands: Optional[List[str]] = None,
                 sandbox_envs: Optional[Dict[str, str]] = None,
                 metadata: Optional[Dict[str, str]] = None,
                 refresh_timeout: bool = True,
                 include_default_tools: bool = True,
                 pre_tool_call: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None,
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
            max_sessions: how many episodes (sandboxes) this replica runs at once.
            sandbox_timeout: Sandbox idle timeout in seconds. AgentENV pauses
                (not kills) idle sandboxes and auto-resumes them on access.
            command_timeout: Per-command execution timeout in seconds.
            setup_commands: Optional commands run at the start of every episode.
            sandbox_envs: Environment variables injected into the sandbox.
            metadata: Sandbox metadata (visible in list APIs, useful for
                tagging the run name).
            refresh_timeout: Extend the sandbox timeout after every step so
                long multi-turn episodes are not paused mid-flight.
            include_default_tools: Expose the built-in run_command /
                write_file / read_file tools. Set False to expose only
                tools registered via ``register_tool``/``register_command_tool``.
            pre_tool_call: Optional ``(tool_name, arguments) -> Optional[str]``
                gate run before every :meth:`step` dispatch. Return a string to
                refuse the call -- it becomes the tool observation and the tool
                does not run -- or None to let it through. The single choke
                point for a per-run policy (e.g. confining writes to the
                workspace); internal calls (setup, run_script) bypass
                step and are never gated. Off by default.
        """
        if not template:
            raise ValueError("AgentEnv requires 'template'. Build one first, e.g. "
                             '`aenv pull ubuntu:22.04 --name my-env`.')
        super().__init__(max_sessions)
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
        self._workspace_dir = workspace
        self._sandbox_timeout = sandbox_timeout
        self._command_timeout = command_timeout
        self._setup_commands = setup_commands or []
        self._sandbox_envs = sandbox_envs
        self._metadata = metadata
        self._refresh_timeout = refresh_timeout
        self._include_default_tools = include_default_tools
        self._pre_tool_call = pre_tool_call
        self._custom_tools: List[ToolInfo] = []
        self._custom_handlers: Dict[str, Callable[[AgentSandbox, Dict[str, Any]], str]] = {}
        # Emptied sandboxes of finished episodes, handed to the next ones.
        self._idle: List[AgentSandbox] = []
        self._idle_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def register_tool(self, tool_info: ToolInfo, handler: Callable[[AgentSandbox, Dict[str, Any]], str]) -> None:
        """Register a custom tool on every replica.

        Args:
            tool_info: OpenAI-format tool schema exposed to the model.
            handler: ``handler(sandbox, arguments) -> str`` returning the observation, where
                ``sandbox`` is the episode's :class:`AgentSandbox` (``sandbox.run_command(...)``,
                ``sandbox.sandbox``). Overrides a built-in tool if the name collides.

        Replicas added later are built from the constructor arguments only, so a deployment that
        grows its env replicas declares its tools in a subclass instead.
        """
        name = tool_info.get('function', {}).get('name')
        if not name:
            raise ValueError(f'tool_info must contain function.name, got: {tool_info!r}')
        self._custom_tools = [t for t in self._custom_tools if t['function']['name'] != name]
        self._custom_tools.append(tool_info)
        self._custom_handlers[name] = handler

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def register_command_tool(self, tool_info: ToolInfo, command_template: str) -> None:
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

        def handler(sandbox: AgentSandbox, arguments: Dict[str, Any]) -> str:
            try:
                command = command_template.format(**arguments)
            except KeyError as e:
                return f'Error: missing required argument {e} for this tool.'
            return sandbox.run_command({'command': command})

        name = tool_info.get('function', {}).get('name')
        if not name:
            raise ValueError(f'tool_info must contain function.name, got: {tool_info!r}')
        self._custom_tools = [t for t in self._custom_tools if t['function']['name'] != name]
        self._custom_tools.append(tool_info)
        self._custom_handlers[name] = handler

    # ------------------------------------------------------------------
    # Session hooks
    # ------------------------------------------------------------------

    def _open(self, trajectory) -> Tuple[AgentSandbox, StepResult]:
        state = self._take_idle() or self._create()
        setup_output = []
        try:
            for cmd in self._setup_commands:
                exit_code, output = state.execute(cmd, self._workspace_dir, self._command_timeout)
                if exit_code != 0:
                    raise RuntimeError(f'sandbox {state.sandbox_id} setup failed: {output}')
                setup_output.append(format_command_output(output, '', exit_code))
        except BaseException:
            self._kill(state)
            raise
        # The setup output is the operator's, not something the model is shown at the start of the episode.
        return state, StepResult(info={'sandbox_id': state.sandbox_id, 'setup_output': '\n'.join(setup_output)})

    def _close(self, state: AgentSandbox) -> None:
        """Empty the workspace and keep the sandbox for the next episode; kill it if that fails.

        A sandbox with a command still running (an agent program the episode was stopped under) is
        killed: emptying the workspace would not stop the program, which would go on in the next
        episode's sandbox.
        """
        with self._idle_lock:
            keep = len(self._idle) < self._max_sessions and not state.running
        if keep:
            exit_code, _ = self._run_script(state, _CLEAR_WORKSPACE.format(root=self._workspace_dir), 'python',
                                            self._command_timeout)
            if exit_code == 0:
                with self._idle_lock:
                    self._idle.append(state)
                return
        self._kill(state)

    def _workspace(self, state: AgentSandbox) -> str:
        return self._workspace_dir

    def _step(self, state: AgentSandbox, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        try:
            if self._pre_tool_call is not None:
                refusal = self._pre_tool_call(tool_name, arguments)
                if refusal is not None:
                    return StepResult(observation=refusal, info={'sandbox_id': state.sandbox_id})
            if tool_name in self._custom_handlers:
                observation = self._custom_handlers[tool_name](state, arguments)
            elif self._include_default_tools and tool_name == 'run_command':
                observation = state.run_command(arguments)
            elif self._include_default_tools and tool_name == 'write_file':
                state.sandbox.files.write(state.resolve(arguments['path']), _as_file_content(arguments.get('content', '')))
                observation = f"File written: {arguments['path']}"
            elif self._include_default_tools and tool_name == 'read_file':
                observation = truncate_observation(str(state.sandbox.files.read(state.resolve(arguments['path']))))
            else:
                available = [t['function']['name'] for t in self._tool_infos()]
                observation = f'Error: unknown tool {tool_name!r}. Available tools: {available}.'
            self._touch(state)
            return StepResult(observation=observation, info={'sandbox_id': state.sandbox_id})
        except Exception as e:  # noqa
            # Keep the episode alive on transient tool errors; the rollout
            # loop (max_turns) bounds retries.
            logger.warning(f'AgentEnv step error (sandbox={state.sandbox_id}): {e}')
            return StepResult(observation=f'Error: {e}', info={'error': str(e)})

    def _run_script(self, state: AgentSandbox, source: str, interpreter: str = 'python',
                    timeout: Optional[float] = None) -> Tuple[int, str]:
        """Run a whole script in the workspace; returns ``(exit_code, output)``.

        A python script is uploaded and run by path rather than passed to
        ``python -c``: a traceback then carries real line numbers, and nothing
        has to survive shell quoting. A sandbox that has gone away comes back as
        a non-zero exit like any other failure.
        """
        timeout = self._command_timeout if timeout is None else timeout
        if interpreter == 'python':
            path = f'{_SCRIPT_DIR}/{uuid.uuid4().hex}.py'
            try:
                state.sandbox.files.write(path, source + '\n')
            except Exception as e:  # noqa
                return 1, f'could not upload the script: {type(e).__name__}: {e}'
            command = f'python3 {shlex.quote(path)}'
        elif interpreter in ('shell', 'bash'):
            command = source
        else:
            return 1, f'unsupported interpreter {interpreter!r}; use python or shell'
        return state.execute(command, self._workspace_dir, timeout)

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def tools(self) -> List[ToolInfo]:
        return self._tool_infos()

    def _tool_infos(self) -> List[ToolInfo]:
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

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def shutdown(self) -> None:
        """Close every open session and kill the kept sandboxes."""
        super().shutdown()
        with self._idle_lock:
            idle, self._idle = self._idle, []
        for state in idle:
            self._kill(state)

    # ------------------------------------------------------------------
    # Sandboxes
    # ------------------------------------------------------------------

    def _create(self) -> AgentSandbox:
        sandbox_cls = _require_e2b()
        sandbox = sandbox_cls.create(
            self._template,
            timeout=self._sandbox_timeout,
            envs=self._sandbox_envs,
            metadata=self._metadata,
            **self._api_params,
        )
        state = AgentSandbox(sandbox, self._workspace_dir, self._command_timeout)
        # Before the setup commands, because they are written against it, and
        # because commands.run(cwd=...) fails outright on a missing directory --
        # a template without this path would otherwise break every call.
        exit_code, output = state.execute(
            f'mkdir -p {shlex.quote(self._workspace_dir)} {shlex.quote(_SCRIPT_DIR)}', '/', self._command_timeout)
        if exit_code != 0:
            self._kill(state)
            raise RuntimeError(f'could not initialize sandbox {state.sandbox_id}: {output}')
        logger.info(f'AgentEnv sandbox created: {state.sandbox_id} (template={self._template})')
        return state

    def _take_idle(self) -> Optional[AgentSandbox]:
        """A kept sandbox that still answers, or None.

        A command rather than a status field: AgentENV pauses an idle sandbox and
        resumes it on access, so what matters is whether it can be reached and
        made to run something, not what a list API last recorded about it.
        """
        while True:
            with self._idle_lock:
                if not self._idle:
                    return None
                state = self._idle.pop()
            exit_code, _ = state.execute('true', None, timeout=10)
            if exit_code == 0:
                self._touch(state)
                return state
            logger.warning(f'AgentEnv kept sandbox {state.sandbox_id} no longer answers; dropping it')
            self._kill(state)

    def _touch(self, state: AgentSandbox) -> None:
        """Push the sandbox's expiry back, best effort.

        Called after tool work rather than on a timer: the idle clock is what
        reclaims a slot, and a slot busy running an episode's tools is exactly
        the one that must not be reclaimed. Failing here is not worth an
        episode -- the next call reports the loss with something to say about it.
        """
        if not self._refresh_timeout:
            return
        try:
            state.sandbox.set_timeout(self._sandbox_timeout)
        except Exception:  # noqa # best-effort keepalive
            pass

    @staticmethod
    def _kill(state: AgentSandbox) -> None:
        try:
            state.sandbox.kill()
        except Exception:  # noqa
            logger.warning(f'AgentEnv could not kill sandbox {state.sandbox_id}.', exc_info=True)
