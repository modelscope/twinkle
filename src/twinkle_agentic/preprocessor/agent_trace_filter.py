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
import re
from typing import Any, Dict, List

from twinkle.preprocessor import Preprocessor

# Conservative whitelist of well-known agent tool tag names. Generic names like
# 'bash' / 'shell' / 'python_exec' are deliberately excluded — they appear in
# regular code blocks (``<bash>echo hi</bash>``) and would falsely suppress
# DeadLoopFilter on plain technical content.
_AGENT_TAG_RE = re.compile(
    r'<(?:read_file|write_to_file|replace_in_file|execute_command|list_files|'
    r'search_files|browser_action|use_mcp_tool|access_mcp_resource|'
    r'attempt_completion|new_task|plan_mode_respond|ask_followup_question|'
    r'list_code_definition_names|feishu_doc|feishu_message|bark_\w+)\b',
    re.IGNORECASE,
)


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
        tcs = m.get('tool_calls')
        if isinstance(tcs, list) and tcs:
            return True
        if role == 'assistant' and _AGENT_TAG_RE.search(_msg_text(m)):
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
