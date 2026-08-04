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
    3. ``pip install e2b`` on the training side.

Usage::

    env = AgentEnv(template='my-env', api_url='http://gateway:8080')
    result = env.reset()
    result = env.step('run_command', {'command': 'echo hello'})
    env.close()
"""
import os
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from .base import Env, StepResult

logger = get_logger()

_MAX_OBSERVATION_CHARS = 32 * 1024

_DEFAULT_TOOLS: List[ToolInfo] = [
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


def _require_e2b():
    """Import the e2b SDK lazily with an actionable error message."""
    try:
        from e2b import Sandbox
    except ImportError as e:
        raise ImportError('AgentEnv requires the E2B SDK to talk to an AgentENV server:\n'
                          '  pip install e2b\n'
                          'Then point it at your deployment via api_url/api_key or the '
                          'E2B_API_URL / E2B_SANDBOX_URL / E2B_API_KEY environment variables.') from e
    return Sandbox


def _truncate(text: str, limit: int = _MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f'\n... [truncated, {len(text) - limit} chars omitted]'


def _format_command_output(stdout: str, stderr: str, exit_code: int) -> str:
    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f'[stderr]\n{stderr}')
    if exit_code != 0:
        parts.append(f'[exit code: {exit_code}]')
    return _truncate('\n'.join(parts)) if parts else '(no output)'


class AgentEnv(Env):
    """Env backed by one AgentENV sandbox per episode.

    Lifecycle mapping:
        * ``reset``  -> kill the previous sandbox (if any) and create a fresh
          one from ``template``; AgentENV's scheduler picks the node.
        * ``step``   -> execute a tool inside the sandbox (sticky-routed to
          the owning node via the sandbox id header, handled by the SDK).
        * ``close``  -> kill the sandbox.

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
            api_url: AgentENV server or gateway base URL. Falls back to the
                ``E2B_API_URL`` environment variable.
            api_key: API key; AgentENV accepts any non-empty string on a
                trusted network. Falls back to ``E2B_API_KEY``. Client-side
                format validation is disabled by default because AgentENV does
                not issue ``e2b_``-prefixed keys; set
                ``E2B_VALIDATE_API_KEY=true`` to re-enable it.
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
        # The E2B SDK reads its endpoint config from env vars; explicit args win.
        if api_url:
            os.environ['E2B_API_URL'] = api_url
            os.environ.setdefault('E2B_SANDBOX_URL', api_url)
        if api_key:
            os.environ['E2B_API_KEY'] = api_key
        os.environ.setdefault('E2B_API_KEY', 'dummy')
        os.environ.setdefault('E2B_ACCESS_TOKEN', 'dummy')
        # AgentENV has no authorization, so any non-empty key works — but the
        # SDK client-side asserts the key matches ``e2b_[0-9a-f]+`` before it
        # ever sends a request, which rejects placeholders like 'dummy'. The
        # SDK exposes this opt-out for exactly this case (deployments that do
        # not issue e2b-format keys); set E2B_VALIDATE_API_KEY=true to restore
        # validation when pointing at e2b.dev itself.
        os.environ.setdefault('E2B_VALIDATE_API_KEY', 'false')

        self._template = template
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
            raise ValueError("tool_info must contain function.name, got: {!r}".format(tool_info))
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
        )
        setup_output = []
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
                self._sandbox.files.write(arguments['path'], arguments.get('content', ''))
                observation = f"File written: {arguments['path']}"
            elif self._include_default_tools and tool_name == 'read_file':
                observation = _truncate(str(self._sandbox.files.read(arguments['path'])))
            else:
                available = [t['function']['name'] for t in self.tools()]
                observation = f'Error: unknown tool {tool_name!r}. Available tools: {available}.'
            if self._refresh_timeout:
                try:
                    self._sandbox.set_timeout(self._sandbox_timeout)
                except Exception:  # noqa # best-effort keepalive
                    pass
            return StepResult(observation=observation, reward=0.0, done=False, info={'sandbox_id': self.sandbox_id})
        except Exception as e:  # noqa
            # Keep the episode alive on transient tool errors; the rollout
            # loop (max_turns) bounds retries.
            logger.warning(f'AgentEnv step error (sandbox={self.sandbox_id}): {e}')
            return StepResult(observation=f'Error: {e}', reward=0.0, done=False, info={'error': str(e)})

    def tools(self) -> List[ToolInfo]:
        tools: List[ToolInfo] = []
        if self._include_default_tools:
            custom_names = set(self._custom_handlers)
            tools.extend(t for t in _DEFAULT_TOOLS if t['function']['name'] not in custom_names)
        tools.extend(self._custom_tools)
        return tools

    def close(self) -> None:
        self._kill_sandbox()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def sandbox_id(self) -> Optional[str]:
        return getattr(self._sandbox, 'sandbox_id', None)

    @property
    def sandbox(self):
        """The underlying E2B sandbox handle, for custom tool handlers."""
        return self._sandbox

    def run_command(self, arguments: Dict[str, Any]) -> str:
        """Run a shell command in the sandbox; public so custom handlers can reuse it."""
        command = arguments.get('command')
        if not command:
            return "Error: 'command' argument is required."
        try:
            result = self._sandbox.commands.run(
                command,
                cwd=arguments.get('cwd'),
                timeout=int(arguments.get('timeout', self._command_timeout)),
            )
            return _format_command_output(result.stdout or '', result.stderr or '', result.exit_code or 0)
        except Exception as e:  # noqa
            # The SDK raises on non-zero exit codes; surface the output
            # instead of failing the step so the model can react to it.
            stdout = getattr(e, 'stdout', '') or ''
            stderr = getattr(e, 'stderr', '') or str(e)
            exit_code = getattr(e, 'exit_code', None)
            if exit_code is None:
                raise
            return _format_command_output(stdout, stderr, exit_code)

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
