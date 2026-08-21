# Copyright (c) ModelScope Contributors. All rights reserved.
"""Env backed by an ms-agent tool runtime.

The harness declares which tools exist (``AgentHarness.tool_schemas``); this
Env is where those calls actually run. It owns no tool logic of its own -- it
forwards to the ``ToolManager`` that ms-agent already built (filesystem, shell,
python, notebook sandbox, web search, todo list) and adds the two things RL
needs on top: one workspace per episode, and batched dispatch so a turn's tool
calls do not run one at a time while the GPUs idle.

Isolation comes from ``config.output_dir``: both ``FileSystemTool`` and
``CodeExecutionTool`` root themselves there, so giving every trajectory its own
directory keeps concurrent episodes from reading each other's files.

Reward is deliberately absent. ``step`` always reports ``reward=0.0`` and
``done=False``; an agentic episode is scored after the fact from the state it
left behind (:mod:`twinkle_agentic.verifier.result_check`), and the rollout
ends when the model stops emitting tool calls.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle.utils import run_sync

from .base import Env, StepResult

__all__ = ['MsAgentToolEnv']

# Marker used to recover an exit status from a tool that only returns text.
_RC_MARK = '__TWINKLE_RC__'
_RC_RE = re.compile(rf'{_RC_MARK}:(-?\d+)')

_PY_WRAPPER = """\
import sys, traceback
try:
{body}
except SystemExit as _e:
    print('{mark}:%d' % (_e.code or 0))
    sys.exit(0)
except BaseException:
    traceback.print_exc()
    print('{mark}:1')
else:
    print('{mark}:0')
"""


class MsAgentToolEnv(Env):
    """Execute ms-agent tool calls for one episode.

    Args:
        agent: an ``LLMAgent`` whose tools are already prepared -- normally
            ``MsAgentHarness.agent`` after ``harness.prepare()``. Sharing the
            harness's agent is what guarantees the executing tool set is the
            one the prompt advertised.
        tool_manager: an ms-agent ``ToolManager`` to use instead of the
            agent's. Only one of ``agent`` / ``tool_manager`` is needed.
        workspace: directory this episode reads and writes. Defaults to the
            agent's ``config.output_dir``.
        max_observation_chars: truncate a tool result before it becomes a
            message. A single ``grep`` can otherwise blow the context window
            and truncate the trajectory mid-episode.
    """

    def __init__(
        self,
        agent: Any = None,
        *,
        tool_manager: Any = None,
        workspace: str = '',
        max_observation_chars: int = 8000,
    ):
        if agent is None and tool_manager is None:
            raise ValueError('MsAgentToolEnv needs either agent= or tool_manager=')
        self._agent = agent
        self._tm = tool_manager if tool_manager is not None else getattr(agent, 'tool_manager', None)
        if self._tm is None:
            raise ValueError('no ms-agent ToolManager available; call harness.prepare() '
                             'before constructing the Env so tools are initialised')
        self.workspace = workspace or self._workspace_from_agent(agent)
        if self.workspace:
            os.makedirs(self.workspace, exist_ok=True)
        self.max_observation_chars = max_observation_chars
        self._names: Optional[List[str]] = None

    # ------------------------------------------------------------------ Env

    def tool_names(self) -> List[str]:
        """Names of the tools actually registered, as the runtime spells them."""
        if self._names is None:
            raw = run_sync(self._tm.get_tools)
            items: List[Any] = []
            if isinstance(raw, dict):
                for value in raw.values():
                    items.extend(value if isinstance(value, list) else [value])
            elif isinstance(raw, list):
                items = raw
            names = []
            for item in items:
                if isinstance(item, dict):
                    fn = item.get('function')
                    name = (fn or {}).get('name') if isinstance(fn, dict) else None
                    name = name or item.get('tool_name') or item.get('name')
                    if name:
                        names.append(str(name))
            self._names = names
        return list(self._names)

    def resolve_tool(self, name: str) -> str:
        """Map a plain tool name onto the runtime's own spelling.

        ms-agent namespaces its tools as ``{server}---{tool}``, so a caller that
        asks for ``shell_executor`` means ``code_executor---shell_executor``.
        An unknown name raises instead of being passed through: a mistyped tool
        comes back as a failed call, which for a checker is indistinguishable
        from a failed check, and a whole GRPO group would silently score zero.
        """
        names = self.tool_names()
        if name in names:
            return name
        matches = [n for n in names if n.rsplit('---', 1)[-1] == name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f'no registered tool named {name!r}; available: {names}')
        raise ValueError(f'{name!r} is ambiguous across servers: {matches}')

    def step(self, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        result = run_sync(self._tm.single_call_tool, self._call(tool_name, arguments))
        return StepResult(observation=self._observation(result))

    def step_batch(self, calls: Sequence[Tuple[str, Dict[str, Any]]]) -> List[StepResult]:
        """Run a turn's calls concurrently through ms-agent's own gather."""
        calls = list(calls)
        if not calls:
            return []
        if len(calls) == 1:
            return [self.step(calls[0][0], calls[0][1] or {})]
        payload = [self._call(name, args or {}) for name, args in calls]
        results = run_sync(self._tm.parallel_call_tool, payload)
        return [StepResult(observation=self._observation(r)) for r in results]

    def close(self) -> None:
        cleanup = getattr(self._tm, 'cleanup', None)
        if cleanup is not None:
            try:
                run_sync(cleanup)
            except Exception:  # noqa
                # Teardown must not take down a training step; a leaked sandbox
                # is recoverable, a crashed trainer loses the whole batch.
                pass

    # ------------------------------------------------------- for the checker

    def runner(self, shell_tool: str = 'shell_executor', python_tool: str = 'python_executor'):
        """A ``result_check`` runner that executes inside this episode's sandbox.

        Verification has to see the same filesystem the agent wrote to, so the
        check goes back through the same tools rather than a local subprocess.
        Those tools return prose, not an exit status, so the command is made to
        print a marker and the status is read back out of the output.

        The tool names are resolved against what is actually registered, so
        plain names work regardless of how the runtime namespaces them.
        """
        shell_name = self.resolve_tool(shell_tool)
        python_name = self.resolve_tool(python_tool)

        def _run(source: str, interpreter: str) -> Tuple[int, str]:
            if interpreter == 'python':
                body = '\n'.join('    ' + line for line in source.splitlines()) or '    pass'
                code = _PY_WRAPPER.format(body=body, mark=_RC_MARK)
                out = self.step(python_name, {'code': code}).observation
            else:
                out = self.step(shell_name, {'command': f'{source}\necho "{_RC_MARK}:$?"'}).observation
            match = _RC_RE.search(out or '')
            if match is None:
                # No marker means the tool itself failed (timeout, sandbox down)
                # rather than the check failing; report non-zero and keep output.
                return 1, out or 'check produced no output and no exit marker'
            return int(match.group(1)), _RC_RE.sub('', out or '').strip()

        return _run

    # -------------------------------------------------------------- private

    @staticmethod
    def _call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {'tool_name': tool_name, 'arguments': arguments or {}}

    def _observation(self, result: Any) -> str:
        if result is None:
            text = ''
        elif isinstance(result, str):
            text = result
        else:
            try:
                text = json.dumps(result, ensure_ascii=False)
            except (TypeError, ValueError):
                text = str(result)
        limit = self.max_observation_chars
        if limit and len(text) > limit:
            head = text[:limit]
            text = f'{head}\n...[truncated {len(text) - limit} chars]'
        return text

    @staticmethod
    def _workspace_from_agent(agent: Any) -> str:
        config = getattr(agent, 'config', None)
        return str(getattr(config, 'output_dir', '') or '') if config is not None else ''
