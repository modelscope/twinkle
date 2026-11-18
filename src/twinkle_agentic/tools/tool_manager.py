# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from twinkle.data_format import ToolCall
from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from twinkle_agentic.tools.base import Tool, ToolExecutionResult

logger = get_logger()


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


def _suggest(name: str, available: Iterable[str]) -> Optional[str]:
    """The registered tool ``name`` was probably meant to be, if there is one.

    Only one mistake is guessed at: a name given without its namespace, or under
    the wrong one. Agent frameworks hand out qualified names -- ms-agent's are
    ``{server}---{tool}`` -- and a model that has seen the bare verb in a
    docstring writes ``shell_executor``, or files it under the server it was last
    using. Measured over 5793 calls: 201 bare ``shell_executor`` and 30
    ``file_system---shell_executor``, all for one tool that does exist.

    Deliberately only a suggestion: the call is still refused. Resolving it
    silently would train the policy to emit a name that no serving deployment
    accepts, and the unqualified form is ambiguous the moment two servers export
    the same verb -- which is why a suffix shared by several tools yields nothing.
    """
    wanted = name.rsplit('---', 1)[-1]
    if not wanted:
        return None
    matches = [n for n in available if n != name and n.rsplit('---', 1)[-1] == wanted]
    return matches[0] if len(matches) == 1 else None


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
        return self.call_result(tool_call).observation

    def call_result(self, tool_call: Union[ToolCall, Dict[str, Any]]) -> ToolExecutionResult:
        name, args, err = _unpack_tool_call(tool_call)
        if err:
            return ToolExecutionResult(observation=err, info={'error': err})
        if (tool := self._tools.get(name)) is None:
            available = ', '.join(sorted(self._tools)) or '(none)'
            hint = ''
            if (suggestion := _suggest(name, self._tools)) is not None:
                hint = (f' Did you mean {suggestion!r}? Tool names must be given in '
                        f'full, including the part before "---".')
            error = f'Error: unknown tool {name!r}.{hint} Available: {available}.'
            return ToolExecutionResult(observation=error, info={'error': error})
        try:
            if hasattr(tool, 'call_result'):
                result = tool.call_result(name, args)
                return ToolExecutionResult(observation=str(result.observation), info=dict(result.info or {}))
            return ToolExecutionResult(observation=str(tool(name, args)))
        except Exception as e:  # noqa
            error = f'Error: tool {name!r} raised {type(e).__name__}: {e}'
            return ToolExecutionResult(observation=error, info={'error': error})

    def call_many(
        self,
        tool_calls: Iterable[Union[ToolCall, Dict[str, Any]]],
        max_workers: Optional[int] = None,
    ) -> List[str]:
        return [result.observation for result in self.call_many_results(tool_calls, max_workers=max_workers)]

    def call_many_results(
        self,
        tool_calls: Iterable[Union[ToolCall, Dict[str, Any]]],
        max_workers: Optional[int] = None,
    ) -> List[ToolExecutionResult]:
        """Execute many tool calls, preserving input order.

        ``tool_calls`` are the OpenAI-shaped dicts produced by
        :meth:`~twinkle.template.base.Template.parse_tool_call`. This method
        unpacks them to ``(name, arguments)`` and, when the tools wrap the same
        :class:`~twinkle_agentic.envs.base.Env`, dispatches through
        ``Env.step_batch``. Otherwise a thread pool of :meth:`__call__`.

        A call this manager can answer by itself -- an unknown name, a malformed
        payload -- is answered here and *excluded* from the batch rather than
        disqualifying it. It used to disqualify it: one bare ``shell_executor``
        in a turn of five sent the whole turn down the thread pool, and
        concurrent dispatch is where the environment is least likely to be safe.
        It was not: in ex4's episode 8 four calls fired at once and all four came
        back with the same glob listing, so the model was told its python had run
        when it never did. Nothing in that turn needed concurrency -- the reason
        it was used was a tool name the host could have refused on the spot.

        Once the tools share an Env, ``step_batch`` is the only way the batch
        runs: a raise or a short result list is reported as the result of those
        calls, not retried down the thread pool. The thread pool is for tools
        that have no Env in common.
        """
        calls = list(tool_calls)
        if not calls:
            return []
        if len(calls) == 1:
            return [self.call_result(calls[0])]

        unpacked = [_unpack_tool_call(tc) for tc in calls]
        env = self._shared_env()
        if env is not None:
            out: List[Optional[ToolExecutionResult]] = [None] * len(calls)
            batched: List[Tuple[int, str, Dict[str, Any]]] = []
            for i, (name, args, err) in enumerate(unpacked):
                if err is None and name in self._tools:
                    batched.append((i, name, args))
                else:
                    out[i] = self.call_result(calls[i])
            try:
                results = env.step_batch([(name, args) for _i, name, args in batched])
            except Exception as e:  # noqa
                # The exception text is the only account of why the batch did not
                # run, and the model is what has to react to it, so it goes back
                # as the result of every call in the batch. Retrying down the
                # thread pool instead -- which is what this used to do, silently
                # and without even a log line -- sends the turn along the path the
                # docstring above exists to keep it off.
                logger.warning(f'{type(env).__name__}.step_batch raised '
                               f'{type(e).__name__}: {e}')
                failure = f'Error: tool batch did not run: {type(e).__name__}: {e}'
                for i, _name, _args in batched:
                    out[i] = ToolExecutionResult(observation=failure, info={'error': failure})
                return [ToolExecutionResult() if x is None else x for x in out]
            if len(results) != len(batched):
                # Same reasoning: a short result list means the calls did not all
                # run, and pairing them up by position would report one call's
                # result under another's name.
                logger.warning(f'{type(env).__name__}.step_batch returned '
                               f'{len(results)} results for {len(batched)} calls')
                failure = (f'Error: tool batch did not run: the environment returned '
                           f'{len(results)} results for {len(batched)} calls.')
                for i, _name, _args in batched:
                    out[i] = ToolExecutionResult(observation=failure, info={'error': failure})
                return [ToolExecutionResult() if x is None else x for x in out]
            for (i, _name, _args), r in zip(batched, results):
                observation = r.observation if hasattr(r, 'observation') else str(r)
                info = dict(getattr(r, 'info', {}) or {})
                out[i] = ToolExecutionResult(observation=observation, info=info)
            return [ToolExecutionResult() if x is None else x for x in out]

        workers = max_workers or min(32, len(calls))
        out: List[Optional[ToolExecutionResult]] = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.call_result, tc): i for i, tc in enumerate(calls)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return [ToolExecutionResult() if x is None else x for x in out]

    def current_state_key(self) -> Optional[str]:
        env = self._shared_env()
        if env is None:
            return None
        return env.state_key()

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
