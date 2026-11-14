# Copyright (c) ModelScope Contributors. All rights reserved.
"""EnvTool: bridges one episode's Env session to ToolManager."""
from typing import Any, Dict, List, Optional, Tuple

from twinkle.data_format.message import Tool as ToolInfo
from .base import Env, StepResult


class EnvTool:
    """One tool of an Env, bound to the session of ``episode_id``."""

    def __init__(self,
                 env: Env,
                 episode_id: str,
                 tool_name: str = 'env_action',
                 description: str = 'Execute an action in the environment.',
                 parameters: Optional[Dict[str, Any]] = None):
        self._env = env
        self._episode_id = episode_id
        self._tool_name = tool_name
        self._description = description
        self._parameters = parameters or {'type': 'object', 'properties': {}}
        self.last_result: Optional[StepResult] = None

    @property
    def session(self) -> Tuple[Env, str]:
        """The ``(env, episode_id)`` this tool acts in; tools sharing it can be batched."""
        return self._env, self._episode_id

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        result = self._env.step(self._episode_id, tool_name, arguments)
        self.last_result = result
        return result.observation

    def call_many(self, calls: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """Batch through ``Env.step_batch``."""
        results = self._env.step_batch(self._episode_id, list(calls))
        if results:
            self.last_result = results[-1]
        return [r.observation for r in results]

    def tool_info(self) -> ToolInfo:
        return {
            'type': 'function',
            'function': {
                'name': self._tool_name,
                'description': self._description,
                'parameters': self._parameters,
            },
        }

    @property
    def done(self) -> bool:
        return self.last_result.done if self.last_result else False

    @classmethod
    def from_schemas(cls, env: Env, episode_id: str, schemas: List[ToolInfo]) -> List['EnvTool']:
        """Bind an externally-declared tool list to the episode's session on ``env``.

        Used when an agent framework owns the tool names/schemas that go into
        the prompt and the Env only supplies the implementation, so training
        and serving advertise the same tools. Every returned tool shares the
        session, which lets :meth:`ToolManager.call_many` collapse a whole turn
        into one :meth:`Env.step_batch`.

        Each name is forwarded to ``env.step`` verbatim, so the Env must accept
        exactly these names.
        """
        tools = []
        for info in schemas or []:
            fn = info.get('function', {}) if isinstance(info, dict) else {}
            name = fn.get('name')
            if not name:
                raise ValueError(f'tool schema without function.name cannot be bound to an '
                                 f'Env; the prompt would advertise an uncallable tool: {info!r}')
            tools.append(
                cls(
                    env=env,
                    episode_id=episode_id,
                    tool_name=name,
                    description=fn.get('description', ''),
                    parameters=fn.get('parameters') or {
                        'type': 'object',
                        'properties': {}
                    },
                ))
        return tools

    @classmethod
    def from_env(cls, env: Env, episode_id: str, tool_infos: Optional[List[ToolInfo]] = None) -> List['EnvTool']:
        """The env's tools bound to the episode's session; ``tool_infos`` saves a round trip to ``env.tools()``."""
        tool_infos = env.tools() if tool_infos is None else tool_infos
        if not tool_infos:
            return [cls(env, episode_id)]
        tools = []
        for info in tool_infos:
            fn = info.get('function', {}) if isinstance(info, dict) else {}
            tools.append(
                cls(
                    env=env,
                    episode_id=episode_id,
                    tool_name=fn.get('name', 'env_action'),
                    description=fn.get('description', ''),
                    parameters=fn.get('parameters', {
                        'type': 'object',
                        'properties': {}
                    }),
                ))
        return tools
