# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-agent harness: LLMAgent owns prompt/message evolution, not generate/execute.

Training path:

    harness.start(query)            # create_messages + tool schema
    MultiTurnRollout                # sampler.sample + Env.step_batch
        harness.before_generate     # memory / hooks / (optional) skill refresh
        harness.after_generate      # handle_new_response
        harness.after_tools         # tool-message shape (not tool execution)

ms-agent owns the tool names and schemas so the prompt is identical in
training and serving; the Env owns the implementation. Wire the executing
side from the same list, or the prompt advertises tools the Env cannot run::

    harness = MsAgentHarness(config)
    harness.prepare()
    tool_manager = ToolManager(EnvTool.from_schemas(env, harness.tool_schemas()))
    rollout = MultiTurnRollout(sampler, template,
                               tool_manager=tool_manager, harness=harness)
    outs = rollout([harness.start(q) for q in queries])

Serving path keeps using ``LLMAgent.run()`` with the same ``agent.yaml`` and
the same :class:`~twinkle_agentic.envs.base.Env` backend. This class must
**not** call ``llm.generate`` or ``parallel_tool_call`` (those execute tools).
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Union

from twinkle import requires
from twinkle.data_format import Trajectory
from twinkle.utils import run_sync

from .base import AgentHarness


class MsAgentHarness(AgentHarness):
    """Harness that *calls* LLMAgent methods instead of copying their prompts.

    Args:
        config: ms-agent ``DictConfig`` / dict / yaml path. Ignored when
            ``agent`` is passed.
        agent: an existing :class:`ms_agent.agent.llm_agent.LLMAgent`.
        auto_prepare: run ``prepare_runtime`` / ``prepare_tools`` / skills /
            memory on first :meth:`start`. Skip LLM init (training generate
            is vLLM). Set ``False`` in unit tests that only need
            ``create_messages``.
        freeze_system: if True (default, RL-safe), do not rewrite
            ``messages[0]`` after the episode starts. Skill/memory *append*
            paths still run.
        permission_mode: forced onto the agent so training never blocks on
            a TUI/CLI confirm. ``auto`` matches non-interactive LLMAgent.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        agent: Any = None,
        auto_prepare: bool = True,
        freeze_system: bool = True,
        permission_mode: str = 'auto',
        trust_remote_code: bool = False,
        **agent_kwargs,
    ):
        requires('ms-agent')
        from omegaconf import DictConfig, OmegaConf

        from ms_agent.agent.llm_agent import LLMAgent

        if agent is not None:
            self.agent = agent
        else:
            if config is None:
                cfg: Any = DictConfig({})
            elif isinstance(config, str):
                cfg = OmegaConf.load(config)
            elif isinstance(config, dict):
                cfg = OmegaConf.create(config)
            else:
                cfg = config
            self.agent = LLMAgent(
                cfg,
                trust_remote_code=trust_remote_code,
                **agent_kwargs,
            )
        self.auto_prepare = auto_prepare
        self.freeze_system = freeze_system
        self.permission_mode = permission_mode
        self._prepared = False
        self._apply_rl_stubs()

    # ------------------------------------------------------------------ public

    def prepare(self) -> None:
        """Initialize tools / skills / memory (sync wrapper). Idempotent."""
        if self._prepared:
            return
        run_sync(self._prepare_async)
        self._prepared = True

    def start(self, query: str, **kwargs) -> Trajectory:
        if self.auto_prepare:
            self.prepare()
        messages = run_sync(self.agent.create_messages, query)
        tools = self.tool_schemas()
        traj: Trajectory = {
            'messages': self._messages_to_dicts(messages),
            'tools': tools,
        }
        traj.update(kwargs)
        return traj

    def before_generate(self, trajectory: Trajectory) -> Trajectory:
        from ms_agent.hooks.context import condense_hook_attachments_for_llm

        if self.auto_prepare:
            self.prepare()
        messages = self._dicts_to_messages(trajectory.get('messages') or [])
        frozen_system = messages[0].content if (self.freeze_system and messages
                                                and messages[0].role == 'system') else None

        messages = self.agent._append_task_notifications(messages)
        messages = condense_hook_attachments_for_llm(messages)

        if getattr(self.agent, 'runtime', None) is not None:
            run_sync(self.agent.on_generate_response, messages)

        if getattr(self.agent, 'context_assembler', None) is not None and not self.freeze_system:
            # Compaction rewrites earlier turns — incompatible with
            # new_input_feature extension. Only run when the caller opts in.
            assembled = self.agent.context_assembler.assemble()
            if assembled:
                messages = self._dicts_to_messages(assembled)

        messages = run_sync(self.agent.condense_memory, messages)

        skill_runtime = getattr(self.agent, '_skill_runtime', None)
        if skill_runtime is not None and not self.freeze_system:
            skill_runtime.maybe_refresh_system_prompt(messages)

        if frozen_system is not None and messages and messages[0].role == 'system':
            messages[0].content = frozen_system

        trajectory['messages'] = self._messages_to_dicts(messages)
        return trajectory

    def after_generate(
        self,
        trajectory: Trajectory,
        decoded: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        messages = self._dicts_to_messages(trajectory.get('messages') or [])
        response = self._assistant_message(decoded, tool_calls, messages)
        self.agent.handle_new_response(messages, response)
        if getattr(self.agent, 'runtime', None) is not None and response.tool_calls:
            run_sync(self.agent.on_tool_call, messages)
        trajectory['messages'] = self._messages_to_dicts(messages)
        return trajectory

    def after_tools(
        self,
        trajectory: Trajectory,
        observations: List[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        """Format Env observations as ms-agent ``role=tool`` messages.

        Mirrors the *message construction* half of ``parallel_tool_call``;
        does not execute tools.
        """
        from ms_agent.llm.utils import Message, ToolResult

        messages = self._dicts_to_messages(trajectory.get('messages') or [])
        calls = self._ms_tool_calls(tool_calls or self._last_assistant_calls(messages))
        for i, raw in enumerate(observations):
            formatted = ToolResult.from_raw(raw)
            tc = calls[i] if i < len(calls) else {}
            tid = tc.get('id') or str(uuid.uuid4())[:8]
            name = tc.get('tool_name') or ''
            messages.append(
                Message(
                    role='tool',
                    content=formatted.text,
                    tool_call_id=tid,
                    name=name,
                    resources=formatted.resources,
                    tool_detail=formatted.tool_detail,
                    hook_attachments=formatted.hook_attachments,
                    is_error=formatted.is_error,
                ))
            if i < len(calls) and not tc.get('id'):
                calls[i]['id'] = tid

        skill_runtime = getattr(self.agent, '_skill_runtime', None)
        if skill_runtime is not None and not self.freeze_system:
            skill_runtime.maybe_refresh_system_prompt(messages)

        messages = run_sync(self.agent.condense_memory, messages)
        if getattr(self.agent, 'runtime', None) is not None:
            run_sync(self.agent.after_tool_call, messages)

        trajectory['messages'] = self._messages_to_dicts(messages)
        return trajectory

    # ------------------------------------------------------------------ prepare

    def _apply_rl_stubs(self) -> None:
        """Non-interactive: never block on TUI / permission prompts / stdin."""
        patch_ms_agent_python_executor()
        try:
            from omegaconf import open_dict
            with open_dict(self.agent.config):
                self.agent.config.interactive = False
                if self.permission_mode:
                    self.agent.config.permission_mode = self.permission_mode
        except Exception:
            pass
        self.agent._interactive = False
        self.agent._event_sink = None
        self.agent._input_source = None

    async def _prepare_async(self) -> None:
        agent = self.agent
        if getattr(agent, 'runtime', None) is None:
            agent.prepare_runtime()
        if getattr(agent, 'tool_manager', None) is None:
            await agent.prepare_tools()
        await agent.prepare_skills()
        await agent.load_memory()
        if hasattr(agent, 'prepare_rag'):
            await agent.prepare_rag()
        if hasattr(agent, 'prepare_knowledge_search'):
            await agent.prepare_knowledge_search()

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """ms-agent's own tool list, OpenAI-shaped.

        This is the list that reaches the prompt. Feed the same list to
        ``EnvTool.from_schemas`` so the Env executes exactly what was
        advertised.
        """
        if self.auto_prepare:
            self.prepare()
        tm = getattr(self.agent, 'tool_manager', None)
        if tm is None:
            return []
        raw = run_sync(tm.get_tools)
        return _ms_tools_to_openai(raw)

    # ------------------------------------------------------------------ convert

    def _assistant_message(self, decoded: str, tool_calls, messages):
        from ms_agent.llm.utils import Message

        ms_calls = self._ms_tool_calls(tool_calls)
        if messages and messages[-1].role == 'assistant':
            response = messages[-1]
            if ms_calls and not response.tool_calls:
                response.tool_calls = ms_calls
            if decoded and not response.content:
                response.content = decoded
            return response
        return Message(role='assistant', content=decoded or '', tool_calls=ms_calls)

    @staticmethod
    def _last_assistant_calls(messages) -> List[Dict[str, Any]]:
        for msg in reversed(messages):
            if getattr(msg, 'role', None) == 'assistant':
                return list(getattr(msg, 'tool_calls', None) or [])
        return []

    @staticmethod
    def _ms_tool_calls(tool_calls: Optional[List[Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get('function') if isinstance(tc.get('function'), dict) else None
            if fn is not None:
                args = fn.get('arguments', '{}')
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                out.append({
                    'id': tc.get('id') or '',
                    'type': tc.get('type', 'function'),
                    'tool_name': fn.get('name') or '',
                    'arguments': args if isinstance(args, str) else '{}',
                })
                continue
            args = tc.get('arguments', '{}')
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            out.append({
                'id': tc.get('id') or '',
                'type': tc.get('type', 'function'),
                'tool_name': tc.get('tool_name') or tc.get('name') or '',
                'arguments': args if isinstance(args, str) else '{}',
            })
        return out

    @staticmethod
    def _messages_to_dicts(messages) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                out.append(dict(msg))
                continue
            d: Dict[str, Any] = {
                'role': msg.role,
                'content': msg.content if msg.content is not None else '',
            }
            if getattr(msg, 'tool_calls', None):
                d['tool_calls'] = _ms_calls_to_openai(msg.tool_calls)
            if getattr(msg, 'tool_call_id', None):
                d['tool_call_id'] = msg.tool_call_id
            if getattr(msg, 'name', None):
                d['name'] = msg.name
            if getattr(msg, 'reasoning_content', ''):
                d['reasoning_content'] = msg.reasoning_content
            out.append(d)
        return out

    @staticmethod
    def _dicts_to_messages(messages: List[Dict[str, Any]]):
        from ms_agent.llm.utils import Message

        out = []
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            kwargs: Dict[str, Any] = {
                'role': m.get('role') or 'user',
                'content': m.get('content') if m.get('content') is not None else '',
            }
            tcs = m.get('tool_calls')
            if tcs:
                kwargs['tool_calls'] = MsAgentHarness._ms_tool_calls(tcs)
            if m.get('tool_call_id'):
                kwargs['tool_call_id'] = m['tool_call_id']
            if m.get('name'):
                kwargs['name'] = m['name']
            if m.get('reasoning_content'):
                kwargs['reasoning_content'] = m['reasoning_content']
            out.append(Message(**kwargs))
        return out


def _ms_calls_to_openai(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get('function') if isinstance(tc.get('function'), dict) else None
        if fn is not None:
            args = fn.get('arguments', '{}')
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            item = {
                'id': tc.get('id') or '',
                'type': tc.get('type', 'function'),
                'function': {
                    'name': fn.get('name') or '',
                    'arguments': args if isinstance(args, str) else '{}',
                },
            }
            out.append(item)
            continue
        args = tc.get('arguments', '{}')
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        out.append({
            'id': tc.get('id') or '',
            'type': tc.get('type', 'function'),
            'function': {
                'name': tc.get('tool_name') or tc.get('name') or '',
                'arguments': args if isinstance(args, str) else '{}',
            },
        })
    return out


_SINGLE_NS_FLAG = '_twinkle_single_namespace'


def single_namespace_source(code: str) -> str:
    """Wrap ``code`` so it runs in one namespace and cannot exit the process.

    Two things happen here, both of them repairs (see
    :func:`patch_ms_agent_python_executor`):

    * the inner ``exec`` passes one dict twice, which is what ordinary module
      execution does, so nested scopes see top-level names;
    * ``SystemExit`` / ``KeyboardInterrupt`` are caught and turned into stderr
      output, so a ``sys.exit(3)`` in a script fails that one call instead of
      escaping into the caller's event loop.

    A non-zero status is reported the way any other failure is -- text on
    stderr, which is what ms-agent turns into ``success: false`` -- so stdout
    written before the exit survives. ``sys.exit()`` and ``sys.exit(0)`` stay
    successes: that is a script saying it is done.

    The wrapper only assigns and reads at top level, which works under split
    globals/locals. ``repr`` handles all quoting, so the original source
    survives byte for byte.
    """
    return ('import builtins as _tw_builtins\n'
            'import sys as _tw_sys\n'
            '_tw_src = ' + repr(code) + '\n'
            "_tw_ns = {'__name__': '__main__', '__builtins__': _tw_builtins}\n"
            'try:\n'
            "    exec(compile(_tw_src, '<tool>', 'exec'), _tw_ns, _tw_ns)\n"
            'except (SystemExit, KeyboardInterrupt) as _tw_exit:\n'
            "    _tw_status = getattr(_tw_exit, 'code', 1)\n"
            '    if _tw_status not in (0, None):\n'
            "        _tw_sys.stderr.write('%s: %s\\n' % (type(_tw_exit).__name__, _tw_status))\n")


def patch_ms_agent_python_executor() -> bool:
    """Give ms-agent's local ``python_executor`` ordinary module semantics.

    ``LocalCodeExecutionTool.python_executor`` calls
    ``exec(code, globals_dict, locals_dict)`` with two *different* dicts
    (ms_agent/tools/code/local_code_executor.py:670). Python then runs the
    submitted code the way it runs a class body: top-level assignments land in
    ``locals_dict``, but every nested scope -- a function body, a generator
    expression -- resolves free names against ``globals_dict`` alone. So::

        import os
        paths = ['a.txt']
        assert all(os.path.exists(p) for p in paths)

    raises ``NameError: name 'os' is not defined``, which reads as if the model
    wrote broken code. For RSI that is worse than noise: the check script *is*
    the reward's ground truth, so this scores a correct check as a failure.

    The same method catches only ``Exception`` around the ``exec``, so a script
    calling ``sys.exit(3)`` raises ``SystemExit`` out of the ``asyncio.to_thread``
    call. ``asyncio.Task`` re-raises that one after storing it, which unwinds
    whatever loop is driving the tool: with a long-lived loop (the RSI sandbox
    server keeps one, so notebook and MCP state survive across turns) the loop
    thread dies and every later tool call in the run hangs. One model-written
    ``sys.exit`` would take out the rest of the episode.

    Temporary local fix pending an upstream PR. It wraps the source instead of
    reimplementing the method, so ms-agent keeps owning timeouts, output capture
    and the JSON result shape.

    Idempotent. Returns True when it patched, False when ms-agent is missing or
    the patch is already in place.
    """
    try:
        from ms_agent.tools.code.local_code_executor import LocalCodeExecutionTool
    except Exception:  # noqa -- ms-agent is optional for most of twinkle
        return False

    original = LocalCodeExecutionTool.python_executor
    if getattr(original, _SINGLE_NS_FLAG, False):
        return False

    async def python_executor(self, code: str, description: str = '', timeout=None):
        return await original(self, single_namespace_source(code),
                              description=description, timeout=timeout)

    setattr(python_executor, _SINGLE_NS_FLAG, True)
    LocalCodeExecutionTool.python_executor = python_executor
    return True


def _ms_tools_to_openai(raw: Union[Dict[str, Any], List[Any], None]) -> List[Dict[str, Any]]:
    if not raw:
        return []
    items: List[Any] = []
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                items.extend(v)
            else:
                items.append(v)
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    out: List[Dict[str, Any]] = []
    for t in items:
        if not isinstance(t, dict):
            continue
        if t.get('type') == 'function' and isinstance(t.get('function'), dict):
            out.append(t)
            continue
        name = t.get('tool_name') or t.get('name')
        if not name:
            continue
        out.append({
            'type': 'function',
            'function': {
                'name': name,
                'description': t.get('description', ''),
                'parameters': t.get('parameters') or {
                    'type': 'object',
                    'properties': {},
                },
            },
        })
    return out
