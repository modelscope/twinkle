# Copyright (c) ModelScope Contributors. All rights reserved.
"""MessageSanityFilter — structural and content sanity for messages-format datasets.

Architecture: check-pipeline pattern. Each check is a standalone function with
signature ``(messages, is_agent, cfg) -> bool`` (True = pass). The filter class
iterates enabled checks in order.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from .utils import (build_sensitive_regex, cjk_ratio, is_agent_row, load_sensitive_words, msg_content_text,
                    msg_has_media, msg_has_payload, normalize_tool_calls)

# Backward-compat re-exports.
_msg_content_text = msg_content_text
_normalize_tool_calls = normalize_tool_calls

_VALID_ROLES = {'system', 'user', 'assistant', 'tool'}
_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_.\-]*$')

# ══════════════════════════════════════════════════════════════════════════════
# Transforms (applied before checks, may modify messages)
# ══════════════════════════════════════════════════════════════════════════════


def consolidate_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold multiple system messages into one at index 0."""
    sys_count = sum(1 for m in messages if isinstance(m, dict) and m.get('role') == 'system')
    misplaced = any(isinstance(m, dict) and m.get('role') == 'system' and i != 0 for i, m in enumerate(messages))
    if sys_count <= 1 and not misplaced:
        return messages
    sys_chunks: List[str] = []
    rest: List[Dict[str, Any]] = []
    template: Optional[Dict[str, Any]] = None
    for m in messages:
        if isinstance(m, dict) and m.get('role') == 'system':
            if template is None:
                template = m
            text = msg_content_text(m).strip()
            if text:
                sys_chunks.append(text)
        else:
            rest.append(m)
    return [dict(template, content='\n\n'.join(sys_chunks))] + rest


def trim_to_last_assistant(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim trailing messages so the conversation ends with an assistant that has visible content."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get('role') == 'assistant' and msg_has_payload(m):
            return messages[:i + 1]
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Check functions: (messages, is_agent, cfg) -> bool   (True = pass)
# ══════════════════════════════════════════════════════════════════════════════


def check_role_order(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """Validate conversational role ordering."""
    if not messages:
        return False
    seen_user = False
    seen_assistant = False
    saw_first_non_system = False
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return False
        role = m.get('role')
        if role not in _VALID_ROLES:
            return False
        if role == 'system':
            if i != 0:
                return False
            continue
        if not saw_first_non_system:
            if role != 'user':
                return False
            saw_first_non_system = True
        if role == 'user':
            seen_user = True
        elif role == 'assistant':
            if not seen_user:
                return False
            seen_assistant = True
        elif role == 'tool':
            if is_agent:
                if not seen_assistant:
                    return False
            else:
                prev = messages[i - 1] if i > 0 else None
                if not isinstance(prev, dict):
                    return False
                prev_role = prev.get('role')
                if prev_role not in ('assistant', 'tool'):
                    return False
                if prev_role == 'assistant' and not normalize_tool_calls(prev):
                    return False
    return True


def check_tool_matching(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """Tool_call_id matching.

    Non-agent: bidirectional strict equality (every call has response AND vice versa).
    Agent: forward-only (every tool message must reference an existing call).
    """
    all_call_ids: set = set()
    i = 0
    while i < len(messages):
        m = messages[i]
        if not isinstance(m, dict) or m.get('role') != 'assistant':
            i += 1
            continue
        norm_tcs = normalize_tool_calls(m)
        if not norm_tcs:
            i += 1
            continue
        expected_ids = {tc['id'] for tc in norm_tcs if isinstance(tc, dict) and tc.get('id')}
        if not expected_ids:
            i += 1
            continue
        all_call_ids.update(expected_ids)
        actual_ids: set = set()
        j = i + 1
        while j < len(messages):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get('role') != 'tool':
                break
            tid = nxt.get('tool_call_id')
            if tid:
                actual_ids.add(tid)
            j += 1
        if not is_agent:
            if actual_ids != expected_ids:
                return False
        i = j
    # Agent forward check: every tool message's tool_call_id must exist in some assistant's tool_calls.
    if is_agent and all_call_ids:
        for m in messages:
            if isinstance(m, dict) and m.get('role') == 'tool':
                tid = m.get('tool_call_id')
                if tid and tid not in all_call_ids:
                    return False
    return True


def check_content_integrity(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """Min turns, max length, duplicate detection, tool_calls structural validity."""
    min_turns = cfg.get('min_turns', 2)
    max_msg_chars = cfg.get('max_msg_chars', 50000)
    user_count = 0
    assistant_count = 0
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return False
        role = m.get('role')
        content = msg_content_text(m)
        norm_tcs = normalize_tool_calls(m)
        if role == 'user':
            user_count += 1
        elif role == 'assistant':
            assistant_count += 1
            if not content.strip() and not norm_tcs:
                return False
        elif role == 'system' and not content.strip():
            return False
        if content and len(content) > max_msg_chars:
            return False
        if norm_tcs is not None:
            for tc in norm_tcs:
                func = tc.get('function')
                name = func.get('name', '') if isinstance(func, dict) else ''
                if not name or not _IDENTIFIER_RE.match(name):
                    return False
                args = func.get('arguments') if isinstance(func, dict) else None
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except (ValueError, json.JSONDecodeError):
                        return False
        # Consecutive-duplicate detection — skip tool messages and messages carrying REAL tool_calls.
        if i > 0 and role != 'tool' and norm_tcs is None and content:
            prev = messages[i - 1]
            if (isinstance(prev, dict) and prev.get('role') == role and normalize_tool_calls(prev) is None
                    and msg_content_text(prev) == content):
                return False
    if user_count < 1 or assistant_count < 1:
        return False
    if (user_count + assistant_count) < min_turns:
        return False
    return True


def check_lang_match(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """False if user is CJK-dominant but assistant is pure Latin (or vice versa)."""
    cjk_threshold = 0.3
    mismatch_threshold = 0.02
    user_text = ''
    asst_text = ''
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get('role') == 'user':
            user_text += msg_content_text(m)
        elif m.get('role') == 'assistant':
            asst_text += msg_content_text(m)
    if len(asst_text) < 50:
        return True
    user_cjk = cjk_ratio(user_text)
    asst_cjk = cjk_ratio(asst_text)
    if user_cjk >= cjk_threshold and asst_cjk < mismatch_threshold:
        return False
    if user_cjk < mismatch_threshold and asst_cjk >= cjk_threshold:
        return False
    return True


def check_agent_min_visible(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """Agent rows must have minimum visible text across all assistant turns."""
    if not is_agent:
        return True
    min_chars = cfg.get('min_agent_visible_chars', 200)
    if min_chars <= 0:
        return True
    total = 0
    for m in messages:
        if isinstance(m, dict) and m.get('role') == 'assistant':
            total += len(msg_content_text(m).strip())
            rc = m.get('reasoning_content')
            if isinstance(rc, str):
                total += len(rc.strip())
    return total >= min_chars


def check_sensitive_words(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """False if any message content matches the sensitive-word regex."""
    regex = cfg.get('_sensitive_re')
    if not regex:
        return True
    return not any(regex.search(msg_content_text(m)) for m in messages if isinstance(m, dict))


# ══════════════════════════════════════════════════════════════════════════════
# Filter class
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_CHECKS = [
    ('role_order', check_role_order),
    ('tool_matching', check_tool_matching),
    ('content_integrity', check_content_integrity),
    ('lang_match', check_lang_match),
    ('agent_min_visible', check_agent_min_visible),
    ('sensitive_words', check_sensitive_words),
]


class MessageSanityFilter(Preprocessor):
    """Structural and content sanity filter for messages-format datasets.

    Each check is a named function returning True (pass) or False (drop), and
    is individually enable-able via the constructor flags.
    """

    def __init__(
        self,
        check_role_order: bool = True,
        check_tool_matching: bool = True,
        check_content_integrity: bool = True,
        check_lang_match: bool = True,
        check_agent_min_visible: bool = True,
        trim_to_assistant: bool = True,
        filter_sensitive: bool = True,
        sensitive_words_file: Optional[str] = None,
        extra_sensitive_words: Optional[List[str]] = None,
        min_turns: int = 2,
        max_msg_chars: int = 80000,
        min_agent_visible_chars: int = 50,
    ) -> None:
        super().__init__()
        self._trim = trim_to_assistant

        words = load_sensitive_words(sensitive_words_file) if sensitive_words_file else set()
        if extra_sensitive_words:
            words.update(w.strip() for w in extra_sensitive_words if w and w.strip())

        self._cfg: Dict[str, Any] = {
            'min_turns': min_turns,
            'max_msg_chars': max_msg_chars,
            'min_agent_visible_chars': min_agent_visible_chars,
            '_sensitive_re': build_sensitive_regex(words),
        }

        enabled = {
            'role_order': check_role_order,
            'tool_matching': check_tool_matching,
            'content_integrity': check_content_integrity,
            'lang_match': check_lang_match,
            'agent_min_visible': check_agent_min_visible,
            'sensitive_words': filter_sensitive,
        }
        self._checks = [(name, fn) for name, fn in _DEFAULT_CHECKS if enabled.get(name, True)]

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for row in rows:
            messages = row.get('messages')
            if not isinstance(messages, list) or not messages:
                dropped.append(dict(row, drop_reason='invalid_messages'))
                continue
            is_agent = is_agent_row(messages)

            normalized = consolidate_system_messages(messages)
            if normalized is not messages:
                messages = normalized
                row = dict(row, messages=messages)

            if self._trim:
                messages = trim_to_last_assistant(messages)
                if not messages:
                    dropped.append(dict(row, drop_reason='no_assistant'))
                    continue
                row = dict(row, messages=messages)

            reason = self._run_checks(messages, is_agent)
            if reason is None:
                out.append(row)
            else:
                dropped.append(dict(row, drop_reason=reason))
        return out, dropped

    def _run_checks(self, messages: List[Dict[str, Any]], is_agent: bool) -> Optional[str]:
        for name, fn in self._checks:
            if not fn(messages, is_agent, self._cfg):
                return name
        return None
