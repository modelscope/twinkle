# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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


def _unpack_tool_call(tool_call: Any) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Split an OpenAI-shaped tool_call into ``(name, args, error)``.

    These dicts come from :meth:`twinkle.template.base.Template.parse_tool_call`.
    ``error`` is set when the payload cannot be executed.
    """
    if not isinstance(tool_call, dict):
        return None, {}, f'Error: tool_call must be an object, got {type(tool_call).__name__}.'
    fn = tool_call.get('function')
    if not isinstance(fn, dict):
        return None, {}, 'Error: tool_call missing "function" object.'
    name = fn.get('name')
    if not name:
        return None, {}, 'Error: tool_call missing "function.name".'
    raw_args = fn.get('arguments')
    if raw_args is None:
        return str(name), {}, None
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as e:
            return str(name), {}, f'Error: invalid JSON in arguments: {e}'
        if not isinstance(args, dict):
            return str(name), {}, 'Error: "arguments" JSON must be an object.'
        return str(name), args, None
    if isinstance(raw_args, dict):
        return str(name), raw_args, None
    return None, {}, (f'Error: "arguments" must be a JSON string or object, '
                      f'got {type(raw_args).__name__}.')


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
        info = tool.tool_info() if hasattr(tool, 'tool_info') else None
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
        name, args, err = _unpack_tool_call(tool_call)
        if err:
            return err
        if (tool := self._tools.get(name)) is None:
            available = ', '.join(sorted(self._tools)) or '(none)'
            return f'Error: unknown tool {name!r}. Available: {available}.'
        try:
            return str(tool(name, args))
        except Exception as e:  # noqa
            return f'Error: tool {name!r} raised {type(e).__name__}: {e}'

    def call_many(
        self,
        tool_calls: Iterable[Union[ToolCall, Dict[str, Any]]],
        max_workers: Optional[int] = None,
    ) -> List[str]:
        """Execute many tool calls, preserving input order.

        ``tool_calls`` are the OpenAI-shaped dicts produced by
        :meth:`~twinkle.template.base.Template.parse_tool_call`. This method
        unpacks them to ``(name, arguments)`` and, when every tool wraps the
        same :class:`~twinkle_agentic.envs.base.Env`, dispatches through
        ``Env.step_batch``. Otherwise a thread pool of :meth:`__call__`.
        """
        calls = list(tool_calls)
        if not calls:
            return []
        if len(calls) == 1:
            return [self(calls[0])]

        unpacked = [_unpack_tool_call(tc) for tc in calls]
        env = self._shared_env()
        can_batch = env is not None and all(
            err is None and name in self._tools for name, _args, err in unpacked)
        if can_batch:
            try:
                results = env.step_batch([(name, args) for name, args, _err in unpacked])
                return [
                    r.observation if hasattr(r, 'observation') else str(r) for r in results
                ]
            except Exception:
                pass

        workers = max_workers or min(32, len(calls))
        out: List[Optional[str]] = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self, tc): i for i, tc in enumerate(calls)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return ['' if x is None else x for x in out]

    def _shared_env(self):
        """Return the Env shared by every registered EnvTool, else None."""
        env = None
        for tool in self._tools.values():
            wrapped = getattr(tool, '_env', None)
            if wrapped is None:
                return None
            if env is None:
                env = wrapped
            elif wrapped is not env:
                return None
        return env
