# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from twinkle.preprocessor import Preprocessor

# ── Valid role set ────────────────────────────────────────────────────────────
_VALID_ROLES = {'system', 'user', 'assistant', 'tool'}

_DEFAULT_SENSITIVE: Set[str] = set()


def _load_sensitive_words(path: Optional[str]) -> Set[str]:
    """Load sensitive words from an external file (one word per line).

    Blank lines and #-comments are ignored.
    """
    if not path or not os.path.isfile(path):
        return set()
    words: Set[str] = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                words.add(line)
    return words


def _build_sensitive_regex(words: Set[str]) -> Optional['re.Pattern']:
    """Build a compiled regex from a set of words. Returns None if empty."""
    if not words:
        return None
    cjk_words = []
    latin_words = []
    cjk_re = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]')
    for w in words:
        if cjk_re.search(w):
            cjk_words.append(re.escape(w))
        else:
            latin_words.append(re.escape(w))
    parts = []
    if latin_words:
        parts.append(r'\b(' + '|'.join(latin_words) + r')\b')
    if cjk_words:
        parts.append('(' + '|'.join(cjk_words) + ')')
    return re.compile('|'.join(parts), re.IGNORECASE)


def _msg_content_text(msg: Dict[str, Any]) -> str:
    """Extract plain text from a message's content (str | list | dict)."""
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(
            p.get('text', '') for p in c
            if isinstance(p, dict) and p.get('type') == 'text'
        )
    if isinstance(c, dict) and c.get('type') == 'text':
        return c.get('text', '')
    return ''


# ── Role order validation ────────────────────────────────────────────────────

def _validate_role_order(messages: List[Dict[str, Any]]) -> bool:
    """Check that message roles follow a sane conversational order.

    Rules:
    - Every message must have a valid role.
    - system (if present) must be at index 0.
    - tool messages must immediately follow an assistant message (that has tool_calls).
    - user/assistant should roughly alternate (we allow tool in between).
    """
    if not messages:
        return False

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return False
        role = m.get('role')
        if role not in _VALID_ROLES:
            return False
        if role == 'system' and i != 0:
            return False
        if role == 'tool':
            if i == 0:
                return False
            prev = messages[i - 1]
            prev_role = prev.get('role')
            if prev_role == 'assistant' and not prev.get('tool_calls'):
                return False
            if prev_role not in ('assistant', 'tool'):
                return False
    return True


_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_.\-]*$')


def _validate_content_integrity(
    messages: List[Dict[str, Any]],
    min_turns: int = 2,
    max_msg_chars: int = 50000,
) -> bool:
    """Check content-level integrity of a conversation."""
    user_count = 0
    assistant_count = 0

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return False
        role = m.get('role')
        content = _msg_content_text(m)

        if role == 'user':
            user_count += 1
        elif role == 'assistant':
            assistant_count += 1
            # Assistant must have content or tool_calls
            if not content.strip() and not m.get('tool_calls'):
                return False
        elif role == 'system':
            if not content.strip():
                return False

        # Single message length bounds
        if content and len(content) > max_msg_chars:
            return False

        # tool_calls structural validity
        if m.get('tool_calls'):
            for tc in m['tool_calls']:
                if not isinstance(tc, dict):
                    return False
                func = tc.get('function')
                if not isinstance(func, dict):
                    return False
                name = func.get('name', '')
                if not name or not _IDENTIFIER_RE.match(name):
                    return False
                # arguments must be valid JSON string (or dict)
                args = func.get('arguments')
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        return False

        # Duplicate consecutive detection (skip tool — parallel calls may return same result)
        if i > 0 and role != 'tool' and isinstance(messages[i - 1], dict):
            prev = messages[i - 1]
            if prev.get('role') == role and _msg_content_text(prev) == content and content:
                return False

    # Minimum conversation depth
    if user_count < 1 or assistant_count < 1:
        return False
    if (user_count + assistant_count) < min_turns:
        return False

    return True


def _validate_tool_call_matching(messages: List[Dict[str, Any]]) -> bool:
    """Verify tool_call_id bidirectional matching between assistant and tool messages."""
    i = 0
    while i < len(messages):
        m = messages[i]
        if not isinstance(m, dict):
            i += 1
            continue
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            # Collect expected IDs from this assistant's tool_calls
            expected_ids = set()
            for tc in m['tool_calls']:
                if isinstance(tc, dict) and tc.get('id'):
                    expected_ids.add(tc['id'])
            if not expected_ids:
                i += 1
                continue
            # Collect actual tool response IDs that follow
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
            # Must have at least one matching response; all responses must reference valid calls
            if not actual_ids or not actual_ids.issubset(expected_ids):
                return False
            i = j
        else:
            i += 1
    return True


def _trim_to_last_assistant(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim trailing messages so the conversation ends with an assistant message.

    Returns the trimmed list, or empty list if no assistant message exists.
    """
    last_asst = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get('role') == 'assistant':
            last_asst = i
            break
    if last_asst < 0:
        return []
    return messages[:last_asst + 1]


# ── Preprocessor ─────────────────────────────────────────────────────────────

class MessageSanityFilter(Preprocessor):
    """Structural and content sanity filter for messages-format datasets.

    1. Role order validation (system at 0, tool after assistant, valid roles).
    2. Trim to last assistant (discard if no assistant remains).
    3. Sensitive word filtering (discard row if any message contains bad words).

    Sensitive words source:
    - ``sensitive_words_file``: external text file (one word per line, # for comments)
    - ``extra_sensitive_words``: additional words merged programmatically
    """

    def __init__(
        self,
        check_role_order: bool = True,
        check_tool_matching: bool = True,
        check_content_integrity: bool = True,
        trim_to_assistant: bool = True,
        filter_sensitive: bool = True,
        sensitive_words_file: Optional[str] = None,
        extra_sensitive_words: Optional[List[str]] = None,
        min_turns: int = 2,
        max_msg_chars: int = 50000,
    ) -> None:
        super().__init__()
        self.check_role_order = check_role_order
        self.check_tool_matching = check_tool_matching
        self.check_content_integrity = check_content_integrity
        self.trim_to_assistant = trim_to_assistant
        self.filter_sensitive = filter_sensitive
        self._min_turns = min_turns
        self._max_msg_chars = max_msg_chars

        # Build unified sensitive word set
        if sensitive_words_file:
            all_words = _load_sensitive_words(sensitive_words_file)
        else:
            all_words = set(_DEFAULT_SENSITIVE)
        if extra_sensitive_words:
            all_words.update(w.strip() for w in extra_sensitive_words if w and w.strip())
        self._sensitive_re = _build_sensitive_regex(all_words)

    def __call__(self, rows) -> List[Dict[str, Any]]:
        out = []
        for row in rows:
            messages = row.get('messages')
            if not isinstance(messages, list) or not messages:
                continue

            # Step 1: role order check
            if self.check_role_order and not _validate_role_order(messages):
                continue

            # Step 1.5: tool_call_id matching
            if self.check_tool_matching and not _validate_tool_call_matching(messages):
                continue

            # Step 2: trim to last assistant
            if self.trim_to_assistant:
                messages = _trim_to_last_assistant(messages)
                if not messages:
                    continue
                row = dict(row, messages=messages)

            # Step 2.5: content integrity (after trim so we validate the final sample)
            if self.check_content_integrity and not _validate_content_integrity(
                messages,
                min_turns=self._min_turns,
                max_msg_chars=self._max_msg_chars,
            ):
                continue

            # Step 3: sensitive word check
            if self.filter_sensitive and self._sensitive_re:
                has_bad = False
                for m in messages:
                    text = _msg_content_text(m)
                    if self._sensitive_re.search(text):
                        has_bad = True
                        break
                if has_bad:
                    continue

            out.append(row)
        return out
