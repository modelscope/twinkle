# Copyright (c) ModelScope Contributors. All rights reserved.
"""OpenEnv adapters: embedded and server mode.

OpenEnv environments can be consumed in two ways:

- **Embedded** (:class:`OpenEnv`): every session is an environment object
  instantiated in-process, bypassing OpenEnv's FastAPI server. Zero network
  overhead, no isolation. Give it a ``remote_group`` to spread sessions over
  several replicas.
- **Server** (:class:`OpenEnvClient`): every session is one WebSocket session on
  a running OpenEnv server. The environment runs in a separate process or
  container, one server hosts many concurrent sessions.

Neither runs scripts (:meth:`Env.run_script` raises): act through the tools.
"""
import importlib
import importlib.util
import inspect
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from twinkle import remote_class, remote_function
from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from .base import ENV_MAX_CONCURRENCY, Env, StepResult

logger = get_logger()

# ==========================================================================
# Utilities
# ==========================================================================


def _import_env_class(path: str):
    """Import a class from 'module:ClassName' or 'module.ClassName'."""
    if ':' in path:
        module_path, class_name = path.rsplit(':', 1)
    elif '.' in path:
        module_path, class_name = path.rsplit('.', 1)
    else:
        raise ValueError(f"env_cls must be 'module.ClassName' or 'module:ClassName', got {path!r}")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        missing = getattr(e, 'name', None) or ''
        if missing == module_path or missing == module_path.split('.')[0]:
            raise ModuleNotFoundError(f'Cannot import module {module_path!r}. '
                                      f'Make sure it is installed or on PYTHONPATH.') from e
        raise
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f'Cannot find class {class_name!r} in module {module_path!r}')
    return cls


def _format_observation(obs) -> str:
    """Normalize observation to string."""
    if obs is None:
        return ''
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        for key in ('result', 'output', 'content', 'text', 'message'):
            if key in obs:
                return str(obs[key])
        try:
            return json.dumps(obs, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(obs)
    if hasattr(obs, 'model_dump'):
        try:
            return json.dumps(obs.model_dump(), ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            pass
    return str(obs)


def _normalize_result(result) -> Dict[str, Any]:
    """Normalize an environment result to a standard dict."""
    if isinstance(result, dict):
        return {
            'observation': _format_observation(result.get('observation', '')),
            'reward': float(result.get('reward', 0.0)),
            'done': bool(result.get('done', False)),
        }
    if hasattr(result, 'observation'):
        return {
            'observation': _format_observation(result.observation),
            'reward': float(getattr(result, 'reward', 0.0) or 0.0),
            'done': bool(getattr(result, 'done', False)),
        }
    return {
        'observation': _format_observation(result),
        'reward': float(getattr(result, 'reward', 0.0) or 0.0),
        'done': bool(getattr(result, 'done', False)),
    }


# ==========================================================================
# OpenEnv: embedded adapter (replaces HTTP server)
# ==========================================================================


def _is_env_class_name(name: str) -> bool:
    """Whether a class name names a CONCRETE environment to instantiate.

    The ecosystem mostly uses the full 'Environment' suffix, but the code-action family
    stops at 'Env' (``PythonCodeActEnv``, ``JuliaCodeActEnv``). The bare suffixes name
    the base classes themselves, which must never be picked.
    """
    return name not in ('Environment', 'Env', 'MCPEnvironment') and name.endswith(('Environment', 'Env'))


def _tool_from_action(action_cls: type) -> ToolInfo:
    """Derive the OpenAI tool schema for an OpenEnv Action class.

    OpenEnv's ``Environment`` interface has no ``tools()`` contract and no official
    environment implements one, so the embedded adapter falls back to the Action class:
    every OpenEnv action is a pydantic model whose fields ARE the tool arguments
    (``step`` builds the action straight from ``action_cls(**arguments)``), so the schema
    derived here is exactly the arguments the environment will accept. Fields declared on
    a base class (OpenEnv's ``Action`` carries a framework-side ``metadata``) are dropped:
    only what the environment's own class declares is a model-facing parameter. The name
    comes from the class (``CodeAction`` -> ``code_action``) and the description from its
    docstring, which OpenEnv authors use to document the fields.
    """
    inherited = set()
    for base in action_cls.mro()[1:]:
        # ``model_fields`` (not ``model_json_schema``): it reads fine on BaseModel itself,
        # which only carries an empty mapping.
        fields = getattr(base, 'model_fields', None)
        if fields:
            inherited.update(fields)
    schema = action_cls.model_json_schema()
    properties = {key: value for key, value in schema.get('properties', {}).items()
                  if key not in inherited}
    # pydantic echoes the field names as 'title'; the parameter key already says it.
    for prop in properties.values():
        prop.pop('title', None)
    required = [key for key in schema.get('required', []) if key in properties]
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2',
                  re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', action_cls.__name__)).lower()
    description = ' '.join((inspect.getdoc(action_cls) or '').split()) \
        or f'Invoke a {name} action on the environment.'
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required,
            },
        },
    }


def _discover_openenv_classes(env_name: str):
    """Auto-discover Environment and Action classes from an OpenEnv package."""
    import pkgutil
    try:
        pkg = importlib.import_module(env_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Cannot import '{env_name}'. Install the environment package:\n"
                                  f"  pip install openenv-{env_name.replace('_', '-')}\n"
                                  f'Or from source:\n'
                                  f'  pip install -e /path/to/OpenEnv/envs/{env_name}')

    action_cls = None
    for name in getattr(pkg, '__all__', dir(pkg)):
        obj = getattr(pkg, name, None)
        if isinstance(obj, type) and name.endswith('Action') and name != 'Action':
            action_cls = obj
            break

    env_cls = None
    try:
        server_pkg = importlib.import_module(f'{env_name}.server')
    except ImportError:
        raise ImportError(f"Cannot import '{env_name}.server'. "
                          f'Make sure the package is correctly installed.')
    for _importer, modname, _ispkg in pkgutil.iter_modules(server_pkg.__path__):
        if modname.endswith('_environment') or modname.endswith('_env'):
            mod = importlib.import_module(f'{env_name}.server.{modname}')
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if isinstance(obj, type) and _is_env_class_name(attr_name):
                    env_cls = obj
                    break
            if env_cls:
                break

    if env_cls is None:
        raise ImportError(f"Cannot discover Environment class in '{env_name}.server'. "
                          f"Expected a '*Environment' or '*Env' class in a '*_environment.py' "
                          f"or '*_env.py' module.")
    return env_cls, action_cls


@remote_class(max_concurrency=ENV_MAX_CONCURRENCY)
class OpenEnv(Env):
    """Embedded OpenEnv adapter: every session is one environment object, built in this process.

    Usage::

        env = OpenEnv(env_name='openspiel_env', env_kwargs={'game_name': 'blackjack'}, remote_group='env')
        result = env.reset(episode_id)
        result = env.step(episode_id, 'play', {'action': 'hit'})
        reward = env.close(episode_id)
    """

    def __init__(self,
                 env_name=None,
                 env_cls=None,
                 env_kwargs=None,
                 action_cls=None,
                 action_mapper=None,
                 max_sessions: int = 64,
                 **kwargs):
        super().__init__(max_sessions=max_sessions)
        if env_cls is None and env_name is None:
            raise ValueError("Either 'env_name' or 'env_cls' is required. "
                             "Example: OpenEnv(env_name='openspiel_env', env_kwargs={'game_name': 'blackjack'})")
        if env_name and env_cls is None:
            resolved_cls, discovered_action_cls = _discover_openenv_classes(env_name)
            if action_cls is None:
                action_cls = discovered_action_cls
        else:
            resolved_cls = _import_env_class(env_cls) if isinstance(env_cls, str) else env_cls

        self._env_cls = resolved_cls
        self._env_kwargs = dict(env_kwargs or {})
        self._action_cls = _import_env_class(action_cls) if isinstance(action_cls, str) else action_cls
        self._action_mapper = action_mapper
        self._advertised: Optional[List[ToolInfo]] = None

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def tools(self) -> List[ToolInfo]:
        # Prefer the environment's own advertisement when it has one. OpenEnv's Environment
        # interface defines no tools() and no official environment implements it, so fall
        # back to deriving the schema from the Action class (_tool_from_action): its fields
        # are exactly the arguments step() accepts, which keeps the advertised schema and
        # the invocation contract from ever drifting apart.
        if self._advertised is None:
            advertised = []
            if callable(getattr(self._env_cls, 'tools', None)):
                probe = self._env_cls(**self._env_kwargs)
                try:
                    advertised = list(probe.tools())
                finally:
                    if hasattr(probe, 'close'):
                        probe.close()
            if not advertised and self._action_cls is not None:
                advertised = [_tool_from_action(self._action_cls)]
            self._advertised = advertised
        return list(self._advertised)

    def _open(self, trajectory) -> Tuple[Any, StepResult]:
        env = self._env_cls(**self._env_kwargs)
        try:
            normalized = _normalize_result(env.reset())
        except BaseException:
            if hasattr(env, 'close'):
                env.close()
            raise
        return env, StepResult(**normalized)

    def _step(self, state, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        if self._action_mapper is not None:
            action = self._action_mapper(tool_name, arguments)
        else:
            action = arguments
        if self._action_cls is not None and isinstance(action, dict):
            action = self._action_cls(**action)
        return StepResult(**_normalize_result(state.step(action)))

    def _close(self, state) -> None:
        if hasattr(state, 'close'):
            state.close()


# ==========================================================================
# OpenEnvClient: server mode (WebSocket sessions against a running server)
# ==========================================================================

_DEFAULT_CODE_TOOL: ToolInfo = {
    'type': 'function',
    'function': {
        'name': 'run_python',
        'description': 'Execute Python code in the environment and return its output.',
        'parameters': {
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': 'Python source to execute. Must print the result.'
                },
            },
            'required': ['code'],
        },
    },
}


def _require_env_client():
    """Import OpenEnv's EnvClient lazily with an actionable error message."""
    try:
        from openenv.core.env_client import EnvClient
    except ImportError as e:
        raise ImportError('OpenEnvClient requires the openenv package:\n'
                          '  pip install openenv\n'
                          'Plus the environment client package, e.g.\n'
                          '  pip install openenv-coding_env') from e
    return EnvClient


def _discover_openenv_client_classes(env_name: str):
    """Auto-discover the ``EnvClient`` subclass and Action class of a package."""
    env_client_cls = _require_env_client()
    try:
        pkg = importlib.import_module(env_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Cannot import '{env_name}'. Install the environment package:\n"
                                  f"  pip install openenv-{env_name.replace('_env', '').replace('_', '-')}\n"
                                  f'Or from source:\n'
                                  f'  pip install -e /path/to/OpenEnv/envs/{env_name}')

    client_cls = None
    action_cls = None
    for name in getattr(pkg, '__all__', dir(pkg)):
        obj = getattr(pkg, name, None)
        if not isinstance(obj, type):
            continue
        if client_cls is None and issubclass(obj, env_client_cls) and obj is not env_client_cls:
            client_cls = obj
        elif action_cls is None and name.endswith('Action') and name != 'Action':
            action_cls = obj

    if client_cls is None:
        raise ImportError(f"Cannot discover an EnvClient subclass in '{env_name}'. "
                          f"Pass env_cls explicitly, e.g. env_cls='{env_name}:MyEnv'.")
    return client_cls, action_cls


def _format_client_observation(obs) -> str:
    """Render a server observation as text for the model.

    Understands the two shapes OpenEnv code environments use — flat
    ``stdout``/``stderr``/``exit_code`` (``coding_env``) and a nested
    ``result`` block (``repl_env``) — and falls back to the generic
    formatter for everything else.
    """
    if obs is None:
        return ''
    inner = getattr(obs, 'result', None)
    if inner is not None and hasattr(inner, 'stdout'):
        obs = inner
    if not hasattr(obs, 'stdout'):
        return _format_observation(obs)

    parts = []
    stdout = getattr(obs, 'stdout', '') or ''
    stderr = getattr(obs, 'stderr', '') or ''
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f'[stderr]\n{stderr}')
    exit_code = getattr(obs, 'exit_code', 0) or 0
    if exit_code:
        parts.append(f'[exit code: {exit_code}]')
    return '\n'.join(parts) if parts else '(no output)'


class ClientSession:
    """One episode's session on the OpenEnv server: a synchronous client with its own WebSocket.

    What a registered tool handler receives (``handler(session, arguments)``).
    """

    def __init__(self, client, action_builder: Callable[[Dict[str, Any]], Any]):
        self.client = client
        self.last_result = None
        self._action_builder = action_builder

    def execute(self, action) -> Any:
        """Send an action (an Action instance or a dict of its fields); returns the server's raw ``StepResult``."""
        if isinstance(action, dict):
            action = self._action_builder(action)
        self.last_result = self.client.step(action)
        return self.last_result


@remote_class(max_concurrency=ENV_MAX_CONCURRENCY)
class OpenEnvClient(Env):
    """Env backed by a running OpenEnv server: every session is one WebSocket session on it.

    Usage::

        env = OpenEnvClient(env_name='coding_env', base_url='http://host:8000')
        env.reset(episode_id)
        result = env.step(episode_id, 'run_python', {'code': 'print(1 + 1)'})
        reward = env.close(episode_id)

    The server must have capacity for every concurrent session
    (``MAX_CONCURRENT_ENVS``), and its environment class must declare
    ``SUPPORTS_CONCURRENT_SESSIONS = True``; otherwise extra connections are
    rejected.
    """

    def __init__(self,
                 env_name: Optional[str] = None,
                 env_cls: Optional[Union[str, Type]] = None,
                 base_url: Optional[str] = None,
                 action_cls: Optional[Union[str, Type]] = None,
                 action_mapper: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
                 tools: Optional[List[ToolInfo]] = None,
                 reset_kwargs: Optional[Dict[str, Any]] = None,
                 connect_timeout_s: float = 10.0,
                 message_timeout_s: float = 120.0,
                 client_kwargs: Optional[Dict[str, Any]] = None,
                 max_sessions: int = 64,
                 **kwargs):
        """
        Args:
            env_name: OpenEnv environment package, e.g. ``'coding_env'``. The
                client and Action classes are auto-discovered from it.
            env_cls: Explicit client class or ``'module:ClassName'``, used
                instead of ``env_name``.
            base_url: Server address, ``http(s)://`` or ``ws(s)://`` (converted
                automatically). Falls back to ``OPENENV_BASE_URL``.
            action_cls: Action class or ``'module:ClassName'``; auto-discovered
                from ``env_name`` when omitted.
            action_mapper: ``(tool_name, arguments) -> action`` (an Action
                instance or a dict of Action fields). Defaults to passing the
                tool arguments straight to ``action_cls``.
            tools: Tool schemas exposed to the model. Defaults to a single
                ``run_python(code)`` tool, matching OpenEnv's code environments.
            reset_kwargs: Extra kwargs forwarded to the server's ``reset()``
                (e.g. ``task_prompt`` / ``expected_answer`` for ``repl_env``).
            connect_timeout_s: WebSocket connect timeout.
            message_timeout_s: Per-message timeout; raise it when the
                environment runs long-executing code.
            client_kwargs: Extra kwargs for the OpenEnv client constructor.
            max_sessions: Sessions this component holds open at once.
        """
        super().__init__(max_sessions=max_sessions)
        if env_cls is None and env_name is None:
            raise ValueError("Either 'env_name' or 'env_cls' is required. "
                             "Example: OpenEnvClient(env_name='coding_env', base_url='http://host:8000')")
        base_url = base_url or os.environ.get('OPENENV_BASE_URL')
        if not base_url:
            raise ValueError("OpenEnvClient requires 'base_url' (or the OPENENV_BASE_URL env var), "
                             'pointing at a running OpenEnv server.')

        if env_cls is not None:
            resolved_cls = _import_env_class(env_cls) if isinstance(env_cls, str) else env_cls
        else:
            resolved_cls, discovered_action_cls = _discover_openenv_client_classes(env_name)
            if action_cls is None:
                action_cls = discovered_action_cls

        self._client_cls = resolved_cls
        self._base_url = base_url
        self._client_kwargs = dict(
            connect_timeout_s=connect_timeout_s, message_timeout_s=message_timeout_s, **(client_kwargs or {}))
        self._action_cls = _import_env_class(action_cls) if isinstance(action_cls, str) else action_cls
        self._action_mapper = action_mapper
        self._tools = list(tools) if tools is not None else [_DEFAULT_CODE_TOOL]
        self._custom_tools: List[ToolInfo] = []
        self._custom_handlers: Dict[str, Callable[[ClientSession, Dict[str, Any]], str]] = {}
        self._reset_kwargs: Dict[str, Any] = dict(reset_kwargs or {})

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def register_tool(self, tool_info: ToolInfo, handler: Callable[[ClientSession, Dict[str, Any]], str]) -> None:
        """Register a tool handled locally instead of being sent to the server, on every replica.

        Useful for bookkeeping tools such as ``submit_solution``, which record
        the model's answer on the session for a check to read later.

        Args:
            tool_info: OpenAI-format tool schema exposed to the model.
            handler: ``handler(session, arguments) -> str`` returning the
                observation, where ``session`` is the episode's :class:`ClientSession`.
                Shadows a server-backed tool of the same name.

        Replicas added later are built from the constructor arguments only, so a deployment that
        grows its env replicas declares its tools in a subclass instead.
        """
        name = tool_info.get('function', {}).get('name')
        if not name:
            raise ValueError(f'tool_info must contain function.name, got: {tool_info!r}')
        self._custom_tools = [t for t in self._custom_tools if t['function']['name'] != name]
        self._custom_tools.append(tool_info)
        self._custom_handlers[name] = handler

    # ------------------------------------------------------------------
    # Env interface
    # ------------------------------------------------------------------

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def tools(self) -> List[ToolInfo]:
        custom_names = set(self._custom_handlers)
        tools = [t for t in self._tools if t['function']['name'] not in custom_names]
        tools.extend(self._custom_tools)
        return tools

    @remote_function(enable_continous_work=True, sticky_key='episode_id')
    def execute(self, episode_id: str, action) -> Any:
        """Send an action in the episode's session and return the server's raw ``StepResult``.

        Public so a check can run scoring code in the same session (e.g. append
        unit tests to a submitted solution) and inspect typed fields such as
        ``exit_code`` instead of the rendered text.
        """
        return self._session(episode_id).state.execute(action)

    def _open(self, trajectory) -> Tuple[ClientSession, StepResult]:
        # .sync() gives each session a dedicated background event loop, which
        # keeps concurrent episodes safe.
        client = self._client_cls(base_url=self._base_url, **self._client_kwargs).sync()
        try:
            result = client.reset(**self._reset_kwargs)
        except BaseException:
            client.close()
            raise
        session = ClientSession(client, self._build_action_from_fields)
        session.last_result = result
        return session, StepResult(
            observation=_format_client_observation(getattr(result, 'observation', None)),
            reward=float(getattr(result, 'reward', 0.0) or 0.0),
            done=bool(getattr(result, 'done', False)),
        )

    def _step(self, state: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        if tool_name in self._custom_handlers:
            try:
                observation = self._custom_handlers[tool_name](state, arguments)
            except Exception as e:  # noqa
                logger.warning(f'OpenEnvClient handler error (tool={tool_name}): {e}')
                observation = f'Error: {e}'
            return StepResult(observation=observation)
        try:
            result = state.execute(self._build_action(tool_name, arguments))
        except Exception as e:  # noqa
            # Keep the episode alive on transient errors; max_turns bounds it.
            logger.warning(f'OpenEnvClient step error (tool={tool_name}): {e}')
            return StepResult(observation=f'Error: {e}', info={'error': str(e)})
        return StepResult(
            observation=_format_client_observation(getattr(result, 'observation', None)),
            reward=float(getattr(result, 'reward', 0.0) or 0.0),
            done=bool(getattr(result, 'done', False)),
        )

    def _close(self, state: ClientSession) -> None:
        state.client.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_action(self, tool_name: str, arguments: Dict[str, Any]):
        if self._action_mapper is not None:
            action = self._action_mapper(tool_name, arguments)
        else:
            action = arguments
        return self._build_action_from_fields(action) if isinstance(action, dict) else action

    def _build_action_from_fields(self, fields: Dict[str, Any]):
        if self._action_cls is None:
            raise ValueError('No action_cls available: pass action_cls, or an action_mapper '
                             'that returns a ready-made Action instance.')
        return self._action_cls(**fields)
