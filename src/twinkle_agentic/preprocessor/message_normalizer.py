# Copyright (c) ModelScope Contributors. All rights reserved.
"""Normalize message sequences to standard OpenAI multi-turn schema.

Three passes (each idempotent):
    0. **Heartbeat strip** — drop heartbeat polling rounds (user + assistant).
    1. **Tool-call normalization** — rewrite embedded tool calls (Cline XML,
       ReAct, VCP, Hermes) to ``tool_calls`` + ``role=tool``.
    2. **Consecutive-role merge** — merge adjacent same-role messages into one
       (content joined by newline). Empty messages inside a run are dropped.
       Single-element runs are preserved verbatim (keeps multimodal list
       content intact).

All passes use ``msg_content_text`` to project content (str | list-of-parts)
to plain text for inspection. List content with only text parts is treated
identically to plain strings; truly multimodal single-element runs are
preserved verbatim by the merge pass.
"""
import json
import re
from typing import Any, Dict, List, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.template.tools import ToolCallRegistry
from .utils import msg_content_text, msg_has_media

# IGNORECASE absorbs every variant ("Read HEARTBEAT.md", "HEARTBEAT_OK",
# "duplicate heartbeat", etc.) under the single token "heartbeat".
_HEARTBEAT_USER_RE = re.compile(r'heartbeat|keep.?alive', re.IGNORECASE)
_HEARTBEAT_ASST_RE = re.compile(r'heartbeat', re.IGNORECASE)

# ── Pass 0: heartbeat strip ─────────────────────────────────────────────────


def _strip_heartbeat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    skip_next_assistant = False
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role', '')
        if role == 'developer':
            m = dict(m)
            m['role'] = 'system'
            role = 'system'
        text = msg_content_text(m)[:300]
        if role == 'user' and _HEARTBEAT_USER_RE.search(text):
            skip_next_assistant = True
            continue
        if role == 'assistant' and not m.get('tool_calls'):
            if skip_next_assistant or _HEARTBEAT_ASST_RE.search(text):
                skip_next_assistant = False
                continue
        skip_next_assistant = False
        out.append(m)
    while out and out[0].get('role') not in ('user', 'system'):
        out.pop(0)
    return out


# ── Pass 1: tool-call normalization ─────────────────────────────────────────


def _normalize_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite embedded tool calls in assistant messages to OpenAI schema."""
    out: List[Dict[str, Any]] = []
    call_counter = 0
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        text = msg_content_text(msg)
        parser = (
            ToolCallRegistry.detect_first(text)
            if msg.get('role') == 'assistant' and not msg.get('tool_calls') and text else None)
        parsed = parser.parse(text) if parser else None
        if not parsed:
            out.append(msg)
            i += 1
            continue

        tc_list = []
        for tc in parsed:
            call_counter += 1
            args = tc['function']['arguments']
            tc_list.append({
                'id': f'call_norm_{call_counter:04d}',
                'type': 'function',
                'function': {
                    'name': tc['function']['name'],
                    'arguments': json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                },
            })
        out.append({
            'role': 'assistant',
            'content': parser.clean(text),
            'tool_calls': json.dumps(tc_list, ensure_ascii=False),
            'tool_call_id': '',
        })

        # Consume following user messages as tool results — one per tool call.
        j = i + 1
        for tc_idx, tc in enumerate(tc_list):
            if j >= n or messages[j].get('role') != 'user':
                break
            nxt_text = msg_content_text(messages[j])
            if not nxt_text:
                break
            if parser.detect_result(nxt_text):
                body = parser.parse_result(nxt_text)
            elif tc_idx == 0 and len(tc_list) == 1:
                body = nxt_text
            else:
                break
            out.append({
                'role': 'tool',
                'content': body,
                'tool_calls': '',
                'tool_call_id': tc['id'],
            })
            j += 1
        i = j
    return out


# ── Pass 2: consecutive-role merge ──────────────────────────────────────────


def _is_atomic(msg: Dict[str, Any]) -> bool:
    """Atomic = never merge: tool results + assistant turns carrying tool_calls."""
    role = msg.get('role', '')
    return role == 'tool' or (role == 'assistant' and msg.get('tool_calls'))


def _is_blank_content(msg: Dict[str, Any]) -> bool:
    if msg_has_media(msg):
        return False
    return not msg_content_text(msg).strip()


def _merge_consecutive(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]

        if _is_atomic(msg):
            # Drop only blank-string tool messages; preserve everything else verbatim.
            if msg.get('role') == 'tool' and _is_blank_content(msg):
                i += 1
                continue
            out.append(msg)
            i += 1
            continue

        role = msg.get('role', '')
        j = i + 1
        run = [msg]
        while j < n and messages[j].get('role') == role and not _is_atomic(messages[j]):
            run.append(messages[j])
            j += 1

        if len(run) == 1:
            # Preserve original shape (incl. multimodal list content); drop only blank strings.
            if not _is_blank_content(msg):
                out.append(msg)
        else:
            non_blank = [m for m in run if not _is_blank_content(m)]
            if not non_blank:
                i = j
                continue
            has_str = any(isinstance(m.get('content'), str) for m in non_blank)
            has_list = any(isinstance(m.get('content'), list) for m in non_blank)
            if has_str and has_list:
                # Mixed types — keep each individually, don't merge.
                out.extend(non_blank)
            elif has_list:
                parts: list = []
                for m in non_blank:
                    parts.extend(m.get('content'))
                merged = dict(non_blank[0])
                merged['content'] = parts
                out.append(merged)
            else:
                merged = dict(non_blank[0])
                merged['content'] = '\n'.join(msg_content_text(m).strip() for m in non_blank)
                out.append(merged)
        i = j
    return out


# ── Combined normalizer ─────────────────────────────────────────────────────


class MessageNormalizer(Preprocessor):
    """Three-pass message normalizer (heartbeat strip + tool-call rewrite + role merge).

    Multimodal list-shaped content passes through every stage untouched.
    This is a mapper — it never drops rows.
    """

    def __call__(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        for row in rows:
            msgs = row.get('messages')
            if not isinstance(msgs, list) or not msgs:
                continue
            msgs = _strip_heartbeat(msgs)
            msgs = _normalize_tool_calls(msgs)
            msgs = _merge_consecutive(msgs)
            row['messages'] = msgs
        return rows, []
