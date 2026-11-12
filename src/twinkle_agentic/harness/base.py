# Copyright (c) ModelScope Contributors. All rights reserved.
"""Agent harness: framework-owned message/tool lifecycle, minus generate/execute.

``MultiTurnRollout`` owns batched sampling and ``new_input_feature`` extension.
``Env`` owns tool execution. A harness mutates the same :class:`Trajectory`
the rest of the stack already uses (``messages`` / ``tools`` / ``user_data``).

Only *append-only* mutations of ``messages`` are safe after the first encode:
rewriting earlier turns would break the token-id chain MultiTurn keeps in
``new_input_feature``. Implementations that compact/rewrite history must do
it in :meth:`start` / the first :meth:`before_generate` (before encode), or
opt in explicitly.
"""
from abc import ABC
from typing import Any, Dict, List, Optional

from twinkle.data_format import Trajectory


class AgentHarness(ABC):
    """Per-episode agent-framework hooks.

    Default implementations are no-ops so MultiTurn can take ``harness=None``
    or a subclass that only overrides some phases. Subclasses that wrap a
    specific framework (ms-agent, …) live next to this file, not in
    ``rollout/`` or ``rsi/``. Harness-private runtime (LLMAgent, session)
    lives on the harness instance, not on the trajectory.
    """

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI-shaped tool list this harness puts in the prompt.

        The harness owns the tool *names and schemas* so training and serving
        advertise the identical set; the Env owns the *implementation*. Build
        the executing side from the same list::

            tm = ToolManager(EnvTool.from_schemas(env, harness.tool_schemas()))

        Skipping that step lets the prompt advertise tools the Env cannot run,
        and every call comes back as an unknown-tool error.
        """
        return []

    def start(self, query: str, **kwargs) -> Trajectory:
        """Open an episode: system + user (+ tool schema).

        Called by the training driver *before* MultiTurn encodes. Not invoked
        by MultiTurn itself. Extra kwargs are merged onto the trajectory
        (``user_data``, ``tools``, …).
        """
        traj: Trajectory = {'messages': [{'role': 'user', 'content': query}]}
        traj.update(kwargs)
        return traj

    def before_generate(self, trajectory: Trajectory) -> Trajectory:
        """Mutate ``trajectory`` immediately before a generate turn.

        First call happens before the initial ``template.encode``. Later calls
        must be append-only relative to ``messages`` already in the pif,
        or MultiTurn will ignore the rewrite to protect token alignment.
        """
        return trajectory

    def after_generate(
        self,
        trajectory: Trajectory,
        decoded: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        """Normalize the assistant turn (content / tool_calls / reasoning).

        ``decoded`` and ``tool_calls`` come from the sampler; the pif already
        contains the generated tokens. This hook only updates message metadata
        so the next encode-bridge and the serving agent see the same shape.
        """
        return trajectory

    def after_tools(
        self,
        trajectory: Trajectory,
        observations: List[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Trajectory:
        """Turn raw Env observations into ``role=tool`` messages (append).

        Default: one tool message per observation, copying ``id`` / ``name``
        from the corresponding tool call when present.
        """
        msgs = trajectory.setdefault('messages', [])
        calls = list(tool_calls or [])
        for i, obs in enumerate(observations):
            msg: Dict[str, Any] = {'role': 'tool', 'content': obs if obs is not None else ''}
            if i < len(calls):
                tc = calls[i] if isinstance(calls[i], dict) else {}
                fn = tc.get('function') if isinstance(tc.get('function'), dict) else {}
                tid = tc.get('id') or tc.get('tool_call_id')
                name = fn.get('name') or tc.get('name') or tc.get('tool_name')
                if tid:
                    msg['tool_call_id'] = tid
                if name:
                    msg['name'] = name
            msgs.append(msg)
        return trajectory
