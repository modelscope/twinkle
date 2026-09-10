# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-agent harness: LLMAgent owns prompt/message evolution, not generate/execute.

Training path::

    harness.start(query)            # create_messages + tool schema
    MultiTurnRollout                # sampler.sample + Env.step_batch
        harness.before_generate     # memory / hooks
        harness.after_generate      # handle_new_response
        harness.after_tools         # tool-message shape (not tool execution)

ms-agent owns the tool names and schemas so the prompt is identical in training
and serving; the Env owns the implementation. Wire the executing side from the
same list, or the prompt advertises tools the Env cannot run.

One harness per trajectory: each holds an ``LLMAgent`` with memory and context
of its own, and episodes run in parallel threads::

    harnesses = [MsAgentHarness(config) for _ in queries]
    tool_managers = [env.tool_manager(h.tool_schemas()) for h, env in zip(harnesses, envs)]
    rollout = MultiTurnRollout(sampler, template)
    outs = rollout([h.start(q) for h, q in zip(harnesses, queries)],
                   tool_manager=tool_managers, harness=harnesses)

Serving keeps using ``LLMAgent.run()`` with the same ``agent.yaml`` and the same
:class:`~twinkle_agentic.envs.base.Env` backend. This class must **not** call
``llm.generate`` or ``parallel_tool_call`` -- those execute tools.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

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
        auto_prepare: run runtime / tools / skills / memory on first
            :meth:`start`. Skips LLM init -- training generation is vLLM. Set
            ``False`` in tests that only need ``create_messages``.
        freeze_system: if True (default, RL-safe), do not rewrite
            ``messages[0]`` after the episode starts. Skill/memory *append*
            paths still run.
        permission_mode: forced onto the agent so training never blocks on a
            TUI/CLI confirm. ``auto`` matches non-interactive LLMAgent.
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
            # A partial config makes LLMAgent load ms-agent's own agent.yaml
            # through Config.from_task, which folds every ``--key value`` pair in
            # sys.argv into the config -- matching leaf names case-insensitively
            # (config.py:_update_config). Built inside a Ray worker, sys.argv
            # carries the training driver's flags, so a ``--model`` or
            # ``--output_dir`` there would silently retarget the agent. Hide them.
            saved_argv = sys.argv
            sys.argv = [saved_argv[0]]
            try:
                self.agent = LLMAgent(cfg, trust_remote_code=trust_remote_code, **agent_kwargs)
            finally:
                sys.argv = saved_argv
        self.auto_prepare = auto_prepare
        self.freeze_system = freeze_system
        self._prepared = False
        self._go_non_interactive(permission_mode)

    # ------------------------------------------------------------------ public

    def prepare(self) -> None:
        """Initialize tools / skills / memory (sync wrapper). Idempotent."""
        if self._prepared:
            return
        run_sync(self._prepare_async)
        self._prepared = True

    def close(self) -> None:
        """Let go of the tool runtimes, which own subprocesses and MCP sessions.

        ``reusable`` stays False: ``LLMAgent`` holds memory, an assembled context
        and skill state, and offers no way to empty them, so
        :class:`~twinkle_agentic.harness.base.HarnessLeases` builds a fresh one
        per episode and this is where the old one is released.
        """
        if self._prepared and self.agent.tool_manager is not None:
            run_sync(self.agent.cleanup_tools)
        self._prepared = False

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
        return [_tool_to_openai(t) for t in run_sync(tm.get_tools) if isinstance(t, dict)]

    def start(self, query: str, **kwargs) -> Trajectory:
        if self.auto_prepare:
            self.prepare()
        messages = run_sync(self.agent.create_messages, query)
        traj: Trajectory = {
            'messages': self._to_dicts(messages),
            'tools': self.tool_schemas(),
        }
        traj.update(kwargs)
        return traj

    def before_generate(self, trajectory: Trajectory) -> Trajectory:
        from ms_agent.hooks.context import condense_hook_attachments_for_llm

        if self.auto_prepare:
            self.prepare()
        messages = self._to_messages(trajectory.get('messages') or [])
        frozen_system = messages[0].content if (self.freeze_system and messages
                                                and messages[0].role == 'system') else None

        messages = self.agent._append_task_notifications(messages)
        messages = condense_hook_attachments_for_llm(messages)

        if self.agent.runtime is not None:
            run_sync(self.agent.on_generate_response, messages)

        if self.agent.context_assembler is not None and not self.freeze_system:
            # Compaction rewrites earlier turns -- incompatible with
            # new_input_feature extension. Only when the caller opts in.
            assembled = self.agent.context_assembler.assemble()
            if assembled:
                messages = self._to_messages(assembled)

        messages = run_sync(self.agent.condense_memory, messages)
        self._maybe_refresh_skills(messages)

        if frozen_system is not None and messages and messages[0].role == 'system':
            messages[0].content = frozen_system

        trajectory['messages'] = self._to_dicts(messages)
        return trajectory

    def after_generate(
        self,
        trajectory: Trajectory,
        decoded: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        messages = self._to_messages(trajectory.get('messages') or [])
        response = self._assistant_message(decoded, tool_calls, messages)
        self.agent.handle_new_response(messages, response)
        if self.agent.runtime is not None and response.tool_calls:
            run_sync(self.agent.on_tool_call, messages)
        trajectory['messages'] = self._to_dicts(messages)
        return trajectory

    def after_tools(
        self,
        trajectory: Trajectory,
        observations: List[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        """Format Env observations as ms-agent ``role=tool`` messages.

        Mirrors the *message construction* half of ``parallel_tool_call``; does
        not execute tools.
        """
        from ms_agent.llm.utils import Message, ToolResult

        messages = self._to_messages(trajectory.get('messages') or [])
        calls = _flat_calls(tool_calls) or _last_assistant_calls(messages)
        for i, raw in enumerate(observations):
            result = ToolResult.from_raw(raw)
            call = calls[i] if i < len(calls) else {}
            call_id = call.get('id') or uuid.uuid4().hex[:8]
            messages.append(
                Message(role='tool',
                        content=result.text,
                        tool_call_id=call_id,
                        name=call.get('tool_name') or '',
                        resources=result.resources,
                        tool_detail=result.tool_detail,
                        hook_attachments=result.hook_attachments,
                        is_error=result.is_error,
                        attachments=result.attachments))
            if i < len(calls):
                calls[i]['id'] = call_id

        self._maybe_refresh_skills(messages)
        messages = run_sync(self.agent.condense_memory, messages)
        if self.agent.runtime is not None:
            run_sync(self.agent.after_tool_call, messages)

        trajectory['messages'] = self._to_dicts(messages)
        return trajectory

    # ------------------------------------------------------------------ prepare

    def _go_non_interactive(self, permission_mode: str) -> None:
        """Never block on a TUI prompt, a permission confirm or stdin.

        ``permission.mode`` is the nested key ``PermissionConfig.from_dict``
        reads; a flat ``permission_mode`` at the top level is not read by
        anything. Written before :meth:`prepare`, which is what builds the
        SafetyGuard and the enforcer from it.
        """
        from omegaconf import OmegaConf, open_dict

        patch_ms_agent_python_executor()
        with open_dict(self.agent.config):
            self.agent.config.interactive = False
            if permission_mode:
                OmegaConf.update(self.agent.config, 'permission.mode', permission_mode, merge=True)
        self.agent._interactive = False
        self.agent._event_sink = None
        self.agent._input_source = None

    async def _prepare_async(self) -> None:
        agent = self.agent
        if agent.runtime is None:
            agent.prepare_runtime()
        if agent.tool_manager is None:
            await agent.prepare_tools()
        # After prepare_tools: skills register a toolset into the tool manager.
        await agent.prepare_skills()
        await agent.load_memory()
        await agent.prepare_rag()
        await agent.prepare_knowledge_search()

    def _maybe_refresh_skills(self, messages) -> None:
        """Let the skill runtime rewrite the system prompt, when allowed to."""
        runtime = getattr(self.agent, '_skill_runtime', None)
        if runtime is not None and not self.freeze_system:
            runtime.maybe_refresh_system_prompt(messages)

    # ------------------------------------------------------------------ convert

    def _assistant_message(self, decoded: str, tool_calls, messages):
        from ms_agent.llm.utils import Message

        calls = _flat_calls(tool_calls)
        if messages and messages[-1].role == 'assistant':
            response = messages[-1]
            if calls and not response.tool_calls:
                response.tool_calls = calls
            if decoded and not response.content:
                response.content = decoded
            return response
        return Message(role='assistant', content=decoded or '', tool_calls=calls)

    @staticmethod
    def _to_dicts(messages) -> List[Dict[str, Any]]:
        """ms-agent ``Message`` objects to the plain dicts a Trajectory holds."""
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                out.append(dict(msg))
                continue
            d: Dict[str, Any] = {
                'role': msg.role,
                'content': msg.content if msg.content is not None else '',
            }
            if msg.tool_calls:
                d['tool_calls'] = [{
                    'id': c.get('id') or '',
                    'type': c.get('type', 'function'),
                    'function': {
                        'name': c.get('tool_name') or '',
                        'arguments': c.get('arguments') or '{}',
                    },
                } for c in _flat_calls(msg.tool_calls)]
            for field in ('tool_call_id', 'name', 'reasoning_content'):
                value = getattr(msg, field, None)
                if value:
                    d[field] = value
            out.append(d)
        return out

    @staticmethod
    def _to_messages(messages: List[Dict[str, Any]]):
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
            if m.get('tool_calls'):
                kwargs['tool_calls'] = _flat_calls(m['tool_calls'])
            for field in ('tool_call_id', 'name', 'reasoning_content'):
                if m.get(field):
                    kwargs[field] = m[field]
            out.append(Message(**kwargs))
        return out


def _flat_calls(tool_calls: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Tool calls in ms-agent's flat ``ToolCall`` shape, whichever came in.

    The sampler hands over OpenAI's nested ``{function: {name, arguments}}``,
    ms-agent stores ``{tool_name, arguments}`` (llm/utils.py:8), and both shapes
    reach every conversion site here. Arguments are always a JSON string, which
    is what ``ToolCall`` declares and what ``single_call_tool`` re-parses.
    """
    out: List[Dict[str, Any]] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get('function') if isinstance(call.get('function'), dict) else call
        args = fn.get('arguments', '{}')
        out.append({
            'id': call.get('id') or '',
            'type': call.get('type', 'function'),
            'tool_name': fn.get('name') or fn.get('tool_name') or '',
            'arguments': args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
        })
    return out


def _last_assistant_calls(messages) -> List[Dict[str, Any]]:
    for msg in reversed(messages):
        if getattr(msg, 'role', None) == 'assistant':
            return _flat_calls(getattr(msg, 'tool_calls', None))
    return []


def _tool_to_openai(tool: Dict[str, Any]) -> Dict[str, Any]:
    """One ms-agent ``Tool`` to the OpenAI shape the prompt and Env speak."""
    if tool.get('type') == 'function' and isinstance(tool.get('function'), dict):
        return tool
    return {
        'type': 'function',
        'function': {
            'name': tool.get('tool_name') or tool.get('name') or '',
            'description': tool.get('description', ''),
            'parameters': tool.get('parameters') or {
                'type': 'object',
                'properties': {},
            },
        },
    }


_SINGLE_NS_FLAG = '_twinkle_single_namespace'


def single_namespace_source(code: str) -> str:
    """Wrap ``code`` so it runs in one namespace and cannot exit the process.

    The inner ``exec`` passes one dict twice, which is what ordinary module
    execution does, so nested scopes see top-level names; and ``SystemExit`` /
    ``KeyboardInterrupt`` become stderr text -- which ms-agent reads as
    ``success: false`` -- instead of escaping into the caller's event loop.
    Stdout written before the exit survives, and ``sys.exit(0)`` stays a
    success. ``repr`` handles the quoting, so the source survives byte for byte.
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

    Three repairs, all of them for bugs in one method
    (``ms_agent/tools/code/local_code_executor.py:773``):

    * it ``exec``s with two *different* dicts (line 786), so the code runs the
      way a class body does: top-level assignments land in the locals, but every
      nested scope resolves free names against the globals alone. ``import os``
      followed by ``all(os.path.exists(p) for p in paths)`` raises ``NameError``,
      which reads as if the model wrote broken code. For RSI that is worse than
      noise -- the check script is the reward's ground truth, so a correct check
      scores as a failure.
    * it catches only ``Exception``, so a script calling ``sys.exit(3)`` raises
      ``SystemExit`` out of its ``asyncio.to_thread`` call and unwinds whatever
      loop drives the tool. With a long-lived loop every later tool call in the
      run then hangs.
    * that ``exec`` runs in the host process, so a relative path resolves
      against the process's cwd, while ``shell_executor`` and every
      ``file_system`` tool pass ``cwd=self._ws.root``. Measured before the fix:
      ``write_file 'a.txt'`` answered "Save file successfully" and the next
      python call got ``No such file or directory: 'a.txt'`` -- 41 of one run's
      58 such failures, and files python wrote landed outside the directory an
      episode's end state is read from.

    Temporary local fix pending an upstream PR. It wraps the source rather than
    reimplementing the method, so ms-agent keeps owning timeouts, output capture
    and the JSON result shape. Idempotent; False when ms-agent is missing or the
    patch is already in place.
    """
    try:
        from ms_agent.tools.code.local_code_executor import LocalCodeExecutionTool
    except Exception:  # noqa -- ms-agent is optional for most of twinkle
        return False

    original = LocalCodeExecutionTool.python_executor
    if getattr(original, _SINGLE_NS_FLAG, False):
        return False

    async def python_executor(self, code: str, description: str = '', timeout=None):
        root = getattr(self, 'output_dir', None) or getattr(getattr(self, '_ws', None), 'root', None)
        if root:
            os.makedirs(root, exist_ok=True)
            os.chdir(root)
        return await original(self, single_namespace_source(code), description=description, timeout=timeout)

    setattr(python_executor, _SINGLE_NS_FLAG, True)
    LocalCodeExecutionTool.python_executor = python_executor
    return True
