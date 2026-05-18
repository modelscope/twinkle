# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from typing import Any, Dict, Iterable, List, Optional, Union

from twinkle.data_format import ToolCall
from twinkle.data_format.message import Tool as ToolInfo
from twinkle_agentic.tools.base import Tool


def _extract_name(info: Any) -> Optional[str]:
    """Read ``function.name`` from an OpenAI-shaped tool / tool-call dict."""
    if not isinstance(info, dict):
        return None
    fn = info.get('function')
    if isinstance(fn, dict):
        name = fn.get('name')
        if isinstance(name, str) and name:
            return name
    return None


class ToolManager:

    def __init__(
        self,
        tools: Optional[Union[Dict[str, Tool], Iterable[Tool]]] = None,
    ):
        if tools is None:
            self._tools: Dict[str, Tool] = {}
            return
        if isinstance(tools, dict):
            self._tools = dict(tools)
            return
        if isinstance(tools, (list, tuple)):
            self._tools = {}
            for t in tools:
                info = t.tool_info() if hasattr(t, 'tool_info') else None
                name = _extract_name(info)
                if not name:
                    raise ValueError(f'tool {type(t).__name__} must expose a non-empty '
                                     f'tool_info()["function"]["name"]')
                self._tools[name] = t
            return
        raise TypeError(f'ToolManager expects dict | Iterable[Tool] | None; '
                        f'got {type(tools).__name__}')

    def register(self, tool: Tool):
        info = tool.tool_info()
        name = _extract_name(info)
        if not name:
            raise ValueError(f'tool {type(tool).__name__} must expose a non-empty '
                             f'tool_info()["function"]["name"]')
        self._tools[name] = tool

    def unregister(self, name: str) -> Optional[Tool]:
        return self._tools.pop(name, None)

    def names(self) -> List[str]:
        return list(self._tools)

    def copy(self) -> 'ToolManager':
        return ToolManager(dict(self._tools))

    def tool_infos(self) -> List[ToolInfo]:
        return [t.tool_info() for t in self._tools.values()]

    def __call__(self, tool_call: Union[ToolCall, Dict[str, Any]]) -> str:
        if not isinstance(tool_call, dict):
            return f'Error: tool_call must be an object, got {type(tool_call).__name__}.'
        fn = tool_call.get('function')
        if not isinstance(fn, dict):
            return 'Error: tool_call missing "function" object.'
        name = fn.get('name')
        if not name:
            return 'Error: tool_call missing "function.name".'
        if (tool := self._tools.get(name)) is None:
            available = ', '.join(sorted(self._tools)) or '(none)'
            return f'Error: unknown tool {name!r}. Available: {available}.'

        raw_args = fn.get('arguments')
        if raw_args is None:
            args: Dict[str, Any] = {}
        elif isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as e:
                return f'Error: invalid JSON in arguments: {e}'
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return (f'Error: "arguments" must be a JSON string or object, '
                    f'got {type(raw_args).__name__}.')

        try:
            return str(tool(name, args))
        except Exception as e:  # noqa
            return f'Error: tool {name!r} raised {type(e).__name__}: {e}'
