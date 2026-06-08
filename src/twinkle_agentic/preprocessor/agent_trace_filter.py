# Copyright (c) ModelScope Contributors. All rights reserved.
"""Detect agent-rollout data so downstream filters can adapt their rules.

Agent SFT datasets (Cline / OpenClaw / Claude Code) carry trajectories whose
tool calls are encoded as text inside assistant content (e.g.
``<read_file><path>foo</path></read_file>``) rather than as the OpenAI
``tool_calls`` field, and whose tool execution results are ``role='tool'``.

Two consequences this preprocessor exists to handle:

1. ``MessageSanityFilter`` strict role-order rules reject these traces.
2. ``DeadLoopFilter`` over-fires on long agent trajectories whose phrasing
   ("Let me read the file...") matches hesitation regexes designed for
   short reasoning traces.

Detection-only: rows are tagged ``is_agent=True`` and never dropped.
Downstream filters read the flag and adapt.
"""
from typing import Any, Dict, List

from twinkle.preprocessor import Preprocessor
from twinkle.template.tools import ToolCallRegistry

from .message_utils import normalize_tool_calls as _normalize_tool_calls


def _msg_text(m: Dict[str, Any]) -> str:
    c = m.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(p.get('text', '') for p in c
                        if isinstance(p, dict) and p.get('type') == 'text')
    return ''


def _is_agent_row(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role')
        if role == 'tool':
            return True
        tcs = _normalize_tool_calls(m)
        if tcs:
            return True
        # Text-embedded tool calls (Cline / OpenClaw / Claude-Code style):
        # delegate detection to the parser registry — no hardcoded tag list.
        if role == 'assistant' and ToolCallRegistry.detect_first(_msg_text(m)) is not None:
            return True
    return False


class AgentTraceFilter(Preprocessor):
    """Tag rows that look like agent rollouts; never drops rows."""

    def __call__(self, rows) -> List[Dict[str, Any]]:
        # Set is_agent on every row (not just matches) so map_row_to_col sees a
        # uniform schema; otherwise rows[0].keys() may miss 'is_agent' and KeyError later.
        return [
            dict(row, is_agent=_is_agent_row(row.get('messages')))
            for row in rows
        ]
