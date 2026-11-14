# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from twinkle.data_format import ToolCall
from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from twinkle_agentic.tools.base import Tool

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
        name, args, err = _unpack_tool_call(tool_call)
        if err:
            return err
        if (tool := self._tools.get(name)) is None:
            available = ', '.join(sorted(self._tools)) or '(none)'
            hint = ''
            if (suggestion := _suggest(name, self._tools)) is not None:
                hint = (f' Did you mean {suggestion!r}? Tool names must be given in '
                        f'full, including the part before "---".')
            return f'Error: unknown tool {name!r}.{hint} Available: {available}.'
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
        unpacks them to ``(name, arguments)``; the calls whose tools are bound to
        the same :class:`~twinkle_agentic.envs.base.Env` session (``tool.session``)
        run as one ``Env.step_batch``, in order. The others go through a thread
        pool of :meth:`__call__`.

        A call this manager can answer by itself -- an unknown name, a malformed
        payload -- is answered here and *excluded* from the batch rather than
        disqualifying it. It used to disqualify it: one bare ``shell_executor``
        in a turn of five sent the whole turn down the thread pool, and
        concurrent dispatch is where the environment is least likely to be safe.
        It was not: in ex4's episode 8 four calls fired at once and all four came
        back with the same glob listing, so the model was told its python had run
        when it never did. Nothing in that turn needed concurrency -- the reason
        it was used was a tool name the host could have refused on the spot.

        For calls bound to a session, ``step_batch`` is the only way they
        run: a raise or a short result list is reported as the result of those
        calls, not retried down the thread pool. The thread pool is for tools
        that have no session.
        """
        calls = list(tool_calls)
        if not calls:
            return []
        if len(calls) == 1:
            return [self(calls[0])]

        out: List[Optional[str]] = [None] * len(calls)
        batches: Dict[Tuple[int, str], List[Tuple[int, str, Dict[str, Any]]]] = {}
        rest: List[int] = []
        for i, (name, args, err) in enumerate(_unpack_tool_call(tc) for tc in calls):
            session = getattr(self._tools.get(name), 'session', None) if err is None else None
            if session is None:
                rest.append(i)
            else:
                batches.setdefault((id(session[0]), session[1]), []).append((i, name, args))
        for batched in batches.values():
            self._run_batch(batched, out)
        if len(rest) == 1:
            out[rest[0]] = self(calls[rest[0]])
        elif rest:
            with ThreadPoolExecutor(max_workers=max_workers or min(32, len(rest))) as pool:
                futs = {pool.submit(self, calls[i]): i for i in rest}
                for fut in as_completed(futs):
                    out[futs[fut]] = fut.result()
        return ['' if x is None else x for x in out]

    def _run_batch(self, batched: List[Tuple[int, str, Dict[str, Any]]], out: List[Optional[str]]) -> None:
        """Run the calls bound to one env session as one ``step_batch``, writing observations into ``out``."""
        tool = self._tools[batched[0][1]]
        env, episode_id = tool.session
        try:
            results = env.step_batch(episode_id, [(name, args) for _i, name, args in batched])
        except Exception as e:  # noqa
            # The exception text is the only account of why the batch did not
            # run, and the model is what has to react to it, so it goes back
            # as the result of every call in the batch. Retrying down the
            # thread pool instead sends the turn along the concurrent path the
            # docstring above exists to keep it off.
            logger.warning(f'{type(env).__name__}.step_batch raised '
                           f'{type(e).__name__}: {e}')
            failure = f'Error: tool batch did not run: {type(e).__name__}: {e}'
            for i, _name, _args in batched:
                out[i] = failure
            return
        if len(results) != len(batched):
            # Same reasoning: a short result list means the calls did not all
            # run, and pairing them up by position would report one call's
            # result under another's name.
            logger.warning(f'{type(env).__name__}.step_batch returned '
                           f'{len(results)} results for {len(batched)} calls')
            failure = (f'Error: tool batch did not run: the environment returned '
                       f'{len(results)} results for {len(batched)} calls.')
            for i, _name, _args in batched:
                out[i] = failure
            return
        for (i, name, _args), r in zip(batched, results):
            self._tools[name].last_result = r
            out[i] = r.observation if hasattr(r, 'observation') else str(r)
