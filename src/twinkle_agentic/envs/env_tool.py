# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, List, Optional

from twinkle.data_format.message import Tool as ToolInfo
from twinkle_agentic.tools.base import Tool
from .base import Env, StepResult


class EnvTool(Tool):
    """Wraps a :class:`Env` environment as a :class:`Tool` for ToolManager.

    Each ``EnvTool`` instance maps to one tool name. When the LLM generates a
    tool call with that name, it is dispatched to ``env.step(tool_name, args)``.

    The observation string is returned as the tool response content. The reward
    and done flag are stored on the instance for the caller to inspect.

    Args:
        env: The Env environment instance.
        tool_name: Name of the tool this adapter represents.
        description: Human-readable description for the LLM.
        parameters: JSON Schema dict describing the tool's parameters.
            Default to an empty object schema (accepts any arguments).
    """

    def __init__(
        self,
        env: Env,
        tool_name: str = 'env_action',
        description: str = 'Execute an action in the environment.',
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self._env = env
        self._tool_name = tool_name
        self._description = description
        self._parameters = parameters or {'type': 'object', 'properties': {}}
        # Last step result for inspection by callers
        self.last_result: Optional[StepResult] = None

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch tool call to the underlying Env environment.

        Args:
            tool_name: Tool name (should match ``self._tool_name``).
            arguments: Tool call arguments from the LLM.

        Returns:
            Observation string from the environment.
        """
        result = self._env.step(tool_name, arguments)
        self.last_result = result
        return result.observation

    def tool_info(self) -> ToolInfo:
        """Return OpenAI-compatible tool schema."""
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
        """Whether the last step terminated the episode."""
        return self.last_result.done if self.last_result else False

    @property
    def episode_reward(self) -> float:
        """Cumulative reward from the last result info."""
        if self.last_result and 'episode_reward' in self.last_result.info:
            return self.last_result.info['episode_reward']
        return self.last_result.reward if self.last_result else 0.0

    @classmethod
    def from_env(cls, env: Env) -> List['EnvTool']:
        """Create one EnvTool per tool defined in the Env.

        If the env exposes multiple tools via :meth:`Env.tools`, this creates
        a list of ``EnvTool`` instances (one per tool) that can all be
        registered into the same :class:`ToolManager`.

        If the env has no tool definitions, returns a single generic adapter.
        """
        tool_infos = env.tools()
        if not tool_infos:
            return [cls(env)]

        tools = []
        for info in tool_infos:
            fn = info.get('function', {}) if isinstance(info, dict) else {}
            name = fn.get('name', 'env_action')
            desc = fn.get('description', '')
            params = fn.get('parameters', {'type': 'object', 'properties': {}})
            tools.append(cls(
                env=env,
                tool_name=name,
                description=desc,
                parameters=params,
            ))
        return tools
