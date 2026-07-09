# Copyright (c) ModelScope Contributors. All rights reserved.
"""EnvTool: bridges any Env to ToolManager."""
from typing import Any, Dict, List, Optional

from twinkle.data_format.message import Tool as ToolInfo
from .base import Env, StepResult


class EnvTool:
    """Wraps an Env as a Tool for ToolManager."""

    def __init__(self,
                 env: Env,
                 tool_name: str = 'env_action',
                 description: str = 'Execute an action in the environment.',
                 parameters: Optional[Dict[str, Any]] = None):
        self._env = env
        self._tool_name = tool_name
        self._description = description
        self._parameters = parameters or {'type': 'object', 'properties': {}}
        self.last_result: Optional[StepResult] = None

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        result = self._env.step(tool_name, arguments)
        self.last_result = result
        return result.observation

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

    @property
    def episode_reward(self) -> float:
        if self.last_result and 'episode_reward' in self.last_result.info:
            return self.last_result.info['episode_reward']
        return self.last_result.reward if self.last_result else 0.0

    @classmethod
    def from_env(cls, env: Env) -> List['EnvTool']:
        tool_infos = env.tools()
        if not tool_infos:
            return [cls(env)]
        tools = []
        for info in tool_infos:
            fn = info.get('function', {}) if isinstance(info, dict) else {}
            tools.append(
                cls(
                    env=env,
                    tool_name=fn.get('name', 'env_action'),
                    description=fn.get('description', ''),
                    parameters=fn.get('parameters', {
                        'type': 'object',
                        'properties': {}
                    }),
                ))
        return tools
