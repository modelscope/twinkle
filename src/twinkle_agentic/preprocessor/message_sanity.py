# Copyright (c) ModelScope Contributors. All rights reserved.
"""MessageSanityFilter — structural and content sanity for messages-format datasets.

Architecture: check-pipeline pattern. Each check is a standalone function with
signature ``(messages, is_agent, cfg) -> bool`` (True = pass). The filter class
simply iterates enabled checks in order.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor

from .utils import (
    build_sensitive_regex,
    cjk_ratio,
    is_agent_row,
    load_sensitive_words,
    msg_content_text,
    normalize_tool_calls,
)

# backward compat: other modules import these from here
_msg_content_text = msg_content_text
_normalize_tool_calls = normalize_tool_calls

_VALID_ROLES = {'system', 'user', 'assistant', 'tool'}
_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_.\-]*$')


# ══════════════════════════════════════════════════════════════════════════════
# Transforms (applied before checks, may modify messages)
# ══════════════════════════════════════════════════════════════════════════════

def consolidate_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold multiple system messages into one at index 0."""
    sys_count = 0
    misplaced = False
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get('role') == 'system':
            sys_count += 1
            if i != 0:
                misplaced = True
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
        if isinstance(m, dict) and m.get('role') == 'assistant':
            if (msg_content_text(m) or '').strip():
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
                prev = messages[i - 1]
                if not isinstance(prev, dict):
                    return False
                prev_role = prev.get('role')
                if prev_role not in ('assistant', 'tool'):
                    return False
                if prev_role == 'assistant' and not normalize_tool_calls(prev):
                    return False
    return True


def check_tool_matching(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """Verify tool_call_id bidirectional matching (skipped for agent rows)."""
    if is_agent:
        return True
    i = 0
    while i < len(messages):
        m = messages[i]
        if not isinstance(m, dict):
            i += 1
            continue
        if m.get('role') == 'assistant':
            norm_tcs = normalize_tool_calls(m)
            if norm_tcs:
                expected_ids = set()
                for tc in norm_tcs:
                    if isinstance(tc, dict) and tc.get('id'):
                        expected_ids.add(tc['id'])
                if not expected_ids:
                    i += 1
                    continue
                actual_ids = set()
                j = i + 1
                while j < len(messages):
                    nxt = messages[j]
                    if not isinstance(nxt, dict) or nxt.get('role') != 'tool':
                        break
                    tid = nxt.get('tool_call_id')
                    if tid:
                        actual_ids.add(tid)
                    j += 1
                if not actual_ids or not actual_ids.issubset(expected_ids):
                    return False
                i = j
            else:
                i += 1
        else:
            i += 1
    return True


def check_content_integrity(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """Content-level integrity: min turns, max length, duplicate detection, tool_calls validity."""
    import json as _json
    min_turns = cfg.get('min_turns', 2)
    max_msg_chars = cfg.get('max_msg_chars', 50000)
    user_count = 0
    assistant_count = 0
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return False
        role = m.get('role')
        content = msg_content_text(m)
        if role == 'user':
            user_count += 1
        elif role == 'assistant':
            assistant_count += 1
            if not content.strip() and not normalize_tool_calls(m):
                return False
        elif role == 'system':
            if not content.strip():
                return False
        if content and len(content) > max_msg_chars:
            return False
        # tool_calls structural validity
        norm_tcs = normalize_tool_calls(m)
        if norm_tcs is not None:
            for tc in norm_tcs:
                if not isinstance(tc, dict):
                    return False
                func = tc.get('function')
                if not isinstance(func, dict):
                    return False
                name = func.get('name', '')
                if not name or not _IDENTIFIER_RE.match(name):
                    return False
                args = func.get('arguments')
                if isinstance(args, str):
                    try:
                        _json.loads(args)
                    except (ValueError, _json.JSONDecodeError):
                        return False
        # consecutive duplicate detection (skip tool — parallel calls may repeat)
        if i > 0 and role != 'tool' and isinstance(messages[i - 1], dict):
            prev = messages[i - 1]
            if prev.get('role') == role and msg_content_text(prev) == content and content:
                return False
    if user_count < 1 or assistant_count < 1:
        return False
    if (user_count + assistant_count) < min_turns:
        return False
    return True


def check_lang_match(messages: List[Dict[str, Any]], is_agent: bool, cfg: dict) -> bool:
    """Return False if user is CJK-dominant but assistant is pure Latin (or vice versa)."""
    cjk_threshold = 0.3
    mismatch_threshold = 0.02
    user_text = ''
    asst_text = ''
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role')
        if role == 'user':
            user_text += msg_content_text(m)
        elif role == 'assistant':
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
    """Return False if any message content matches sensitive word regex."""
    regex = cfg.get('_sensitive_re')
    if not regex:
        return True
    for m in messages:
        if regex.search(msg_content_text(m)):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Filter class
# ══════════════════════════════════════════════════════════════════════════════

# Default check pipeline (order matters)
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

    Uses a check-pipeline pattern: each check is a named function that returns
    True (pass) or False (drop). Checks can be individually enabled/disabled.
    """

    def __init__(
        self,
        check_role_order: bool = True,
        check_tool_matching: bool = True,
        check_content_integrity: bool = True,
        check_lang_match: bool = True,
        trim_to_assistant: bool = True,
        filter_sensitive: bool = True,
        sensitive_words_file: Optional[str] = None,
        extra_sensitive_words: Optional[List[str]] = None,
        min_turns: int = 2,
        max_msg_chars: int = 50000,
        min_agent_visible_chars: int = 200,
    ) -> None:
        super().__init__()
        self._trim = trim_to_assistant

        # build config dict shared across checks
        self._cfg: Dict[str, Any] = {
            'min_turns': min_turns,
            'max_msg_chars': max_msg_chars,
            'min_agent_visible_chars': min_agent_visible_chars,
        }

        # sensitive regex
        all_words = load_sensitive_words(sensitive_words_file) if sensitive_words_file else set()
        if extra_sensitive_words:
            all_words.update(w.strip() for w in extra_sensitive_words if w and w.strip())
        self._cfg['_sensitive_re'] = build_sensitive_regex(all_words)

        # build enabled check list
        enabled = {
            'role_order': check_role_order,
            'tool_matching': check_tool_matching,
            'content_integrity': check_content_integrity,
            'lang_match': check_lang_match,
            'agent_min_visible': True,
            'sensitive_words': filter_sensitive,
        }
        self._checks = [(name, fn) for name, fn in _DEFAULT_CHECKS if enabled.get(name, True)]

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        out = []
        dropped = []
        for row in rows:
            messages = row.get('messages')
            if not isinstance(messages, list) or not messages:
                dropped.append(dict(row, drop_reason='invalid_messages'))
                continue
            is_agent = is_agent_row(messages)

            # pre-transform: consolidate system messages
            normalized = consolidate_system_messages(messages)
            if normalized is not messages:
                messages = normalized
                row = dict(row, messages=messages)

            # pre-transform: trim to last assistant
            if self._trim:
                messages = trim_to_last_assistant(messages)
                if not messages:
                    dropped.append(dict(row, drop_reason='no_assistant'))
                    continue
                row = dict(row, messages=messages)

            # run check pipeline
            reason = self._run_checks(messages, is_agent)
            if reason is None:
                out.append(row)
            else:
                dropped.append(dict(row, drop_reason=reason))
        return out, dropped

    def _run_checks(self, messages: List[Dict[str, Any]], is_agent: bool) -> Optional[str]:
        """Return None if all checks pass, else the name of the first failing check."""
        for name, fn in self._checks:
            if not fn(messages, is_agent, self._cfg):
                return name
        return None
