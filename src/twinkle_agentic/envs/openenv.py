# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import json
import importlib.util
import os
from typing import Any, Dict, List, Optional, Type
from twinkle.utils import get_logger
from twinkle.data_format import Trajectory
from twinkle.data_format.message import Tool as ToolInfo
from .base import Env, StepResult

logger = get_logger()


def _import_class(dotted_path: str):
    """Dynamically import a class from a dotted path.

    Example: ``'coding_env.CodingEnv'`` → imports ``CodingEnv`` from the
    ``coding_env`` package.
    """
    parts = dotted_path.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(
            f'env_cls must be "module.ClassName", got {dotted_path!r}'
        )
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(
            f'Cannot find class {class_name!r} in module {module_path!r}'
        )
    return cls


def _get_generic_env_client():
    """Import GenericEnvClient from openenv, handling broken sub-imports."""
    try:
        from openenv.core.generic_client import GenericEnvClient
        return GenericEnvClient
    except ImportError:
        pass
    # Fallback: try direct submodule import bypassing core __init__
    try:
        import openenv
        pkg_dir = os.path.dirname(openenv.__file__)
        spec = importlib.util.spec_from_file_location(
            'openenv.core.generic_client',
            os.path.join(pkg_dir, 'core', 'generic_client.py'),
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            # We need client_types to be importable
            ct_spec = importlib.util.spec_from_file_location(
                'openenv.core.client_types',
                os.path.join(pkg_dir, 'core', 'client_types.py'),
            )
            if ct_spec and ct_spec.loader:
                import sys
                ct_mod = importlib.util.module_from_spec(ct_spec)
                sys.modules['openenv.core.client_types'] = ct_mod
                ct_spec.loader.exec_module(ct_mod)
            spec.loader.exec_module(mod)
            return mod.GenericEnvClient
    except Exception:
        pass
    raise ImportError(
        'Cannot import GenericEnvClient from openenv. '
        'Please install openenv: pip install openenv'
    )


class OpenEnv(Env):
    """Adapter that wraps an OpenEnv ``EnvClient`` as a Twinkle :class:`Env`.

    OpenEnv environments communicate via WebSocket (async). This adapter
    provides a synchronous interface via the ``.sync()`` wrapper, making it
    compatible with Twinkle's synchronous :class:`MultiTurnRollout`.

    Args:
        base_url: URL of the running OpenEnv environment server
            (e.g. ``'http://localhost:8000'``).
        env_cls: Optional dotted import path for a typed OpenEnv client class,
            e.g. ``'coding_env.CodingEnv'`` or ``'echo_env.EchoEnv'``.
            Alternatively pass the class object directly.
            If *None*, uses ``GenericEnvClient`` (works with any environment
            using plain dict actions/observations).
        env_kwargs: Extra keyword arguments forwarded to the EnvClient
            constructor (e.g. ``connect_timeout_s``, ``message_timeout_s``).
        tool_schema: Optional list of tool definitions (OpenAI function-call
            schema). If provided, :meth:`tools` will return them.
    """

    def __init__(
        self,
        base_url: str,
        env_cls: Any = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        tool_schema: Optional[List[ToolInfo]] = None,
    ):
        # Resolve env_cls
        if env_cls is None:
            self._env_cls: Type = _get_generic_env_client()
        elif isinstance(env_cls, str):
            self._env_cls = _import_class(env_cls)
        else:
            self._env_cls = env_cls

        self._base_url = base_url
        self._env_kwargs = env_kwargs or {}
        self._tool_schema = tool_schema
        self._sync_client = None
        self._episode_reward: float = 0.0

    def _ensure_client(self):
        """Lazily create and connect the synchronous client."""
        if self._sync_client is not None:
            return
        client = self._env_cls(
            base_url=self._base_url,
            **self._env_kwargs,
        )
        # .sync() returns a SyncEnvClient with __enter__/__exit__
        self._sync_client = client.sync()
        self._sync_client.__enter__()

    def reset(self, trajectory: Optional[Trajectory] = None) -> StepResult:
        """Reset the OpenEnv environment for a new episode.

        Args:
            trajectory: Ignored for OpenEnv (state is server-managed).

        Returns:
            StepResult with the initial observation.
        """
        self._ensure_client()
        self._episode_reward = 0.0
        result = self._sync_client.reset()
        obs = self._format_observation(result)
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={'raw_result': result},
        )

    def step(self, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        """Execute a tool call in the OpenEnv environment.

        The action is sent as a dict ``{tool_name: ..., arguments: ...}``
        which is the standard format accepted by ``GenericEnvClient`` and
        typed clients via ``_step_payload``.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Tool arguments dict.

        Returns:
            StepResult with observation, step reward, and done flag.
        """
        self._ensure_client()
        try:
            action = {'tool_name': tool_name, 'arguments': arguments}
            result = self._sync_client.step(action)

            obs = self._format_observation(result)
            reward = getattr(result, 'reward', 0.0) or 0.0
            done = getattr(result, 'done', False) or False
            self._episode_reward += reward

            return StepResult(
                observation=obs,
                reward=reward,
                done=done,
                info={'raw_result': result, 'episode_reward': self._episode_reward},
            )
        except Exception as e:
            logger.warning(f'OpenEnv step error: {e}')
            return StepResult(
                observation=f'Error: {e}',
                reward=0.0,
                done=True,
                info={'error': str(e)},
            )

    def tools(self) -> List[ToolInfo]:
        """Return tool definitions from the OpenEnv environment."""
        if self._tool_schema is not None:
            return self._tool_schema
        return []

    def evaluate(self, trajectories: List[Trajectory], **kwargs) -> List[float]:
        """OpenEnv environments provide per-step rewards; episode reward is
        accumulated in ``info['episode_reward']``. This method is a no-op."""
        return [0.0] * len(trajectories)

    def close(self) -> None:
        """Disconnect from the OpenEnv server."""
        if self._sync_client is not None:
            try:
                self._sync_client.__exit__(None, None, None)
            except Exception:
                pass
            self._sync_client = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_observation(result) -> str:
        """Extract a string observation from an OpenEnv StepResult.

        OpenEnv's ``StepResult.observation`` may be a dict (GenericEnvClient),
        a typed object, or a string depending on the client class.
        """
        obs = getattr(result, 'observation', None)
        if obs is None:
            return ''
        if isinstance(obs, str):
            return obs
        # Dict observations (GenericEnvClient)
        if isinstance(obs, dict):
            # Common patterns in tool-based envs
            for key in ('result', 'output', 'content', 'text', 'message'):
                if key in obs:
                    return str(obs[key])
            # Return full dict as JSON
            try:
                return json.dumps(obs, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(obs)
        # Typed observation objects
        for attr in ('result', 'content', 'output', 'text'):
            if hasattr(obs, attr):
                return str(getattr(obs, attr))
        try:
            return json.dumps(obs, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(obs)
