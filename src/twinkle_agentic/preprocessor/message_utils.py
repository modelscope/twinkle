"""Message-format utilities shared across active preprocessor steps.

Split out of ``utils.py`` (AUDIT A2): content projection, tool-call
normalization, CJK ratio, sensitive-word regex, and agent-row detection. These
are the helpers every cleaning step depends on, independent of the log-prob
scoring math (see :mod:`logprob_utils`).
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Set


def msg_content_text(msg: Dict[str, Any]) -> str:
    """Extract plain text from a message's content (str | list | dict)."""
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(p.get('text', '') for p in c if isinstance(p, dict) and p.get('type') == 'text')
    if isinstance(c, dict) and c.get('type') == 'text':
        return c.get('text', '')
    return ''


def msg_has_media(msg: Dict[str, Any]) -> bool:
    """True if message content contains non-text parts (image/audio/video)."""
    c = msg.get('content')
    return isinstance(c, list) and any(isinstance(p, dict) and p.get('type') not in ('text', None) for p in c)


def msg_has_payload(msg: Dict[str, Any]) -> bool:
    """True if a message carries any substantive payload (text, tool_calls, reasoning, or media)."""
    return bool(
        msg_content_text(msg).strip() or msg.get('tool_calls') or msg.get('reasoning_content') or msg.get('thinking')
        or msg_has_media(msg))


_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]')


def normalize_tool_calls(msg: Dict[str, Any]) -> Optional[List[Any]]:
    """Return ``tool_calls`` as a list of dicts, handling PyArrow/HF serialization artifacts."""
    tcs = msg.get('tool_calls')
    if isinstance(tcs, str):
        s = tcs.strip()
        if not s:
            return None
        try:
            decoded = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(decoded, list) or not decoded:
            return None
        tcs = decoded
    if not isinstance(tcs, list) or not tcs:
        return None
    result = []
    for tc in tcs:
        if isinstance(tc, str):
            try:
                tc = json.loads(tc)
            except (json.JSONDecodeError, ValueError):
                return None
        if not isinstance(tc, dict):
            return None
        func = tc.get('function')
        if isinstance(func, str):
            try:
                func = json.loads(func)
            except (json.JSONDecodeError, ValueError):
                return None
            tc = dict(tc, function=func)
        result.append(tc)
    return result


CJK_CHARS_RE = _CJK_RE


def cjk_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are CJK."""
    chars = text.replace(' ', '').replace('\n', '').replace('\t', '')
    if not chars:
        return 0.0
    return len(CJK_CHARS_RE.findall(chars)) / len(chars)


def load_sensitive_words(path: Optional[str]) -> Set[str]:
    """Load from external file (one word per line). Blank lines and #-comments ignored."""
    if not path or not os.path.isfile(path):
        return set()
    words: Set[str] = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                words.add(line)
    return words


def build_sensitive_regex(words: Set[str]) -> Optional['re.Pattern']:
    """Build a compiled regex from a set of words. Returns None if empty."""
    if not words:
        return None
    cjk_words = []
    latin_words = []
    cjk_re = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]')
    for w in sorted(words):
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


def is_agent_row(messages) -> bool:
    """Return True if the conversation contains tool interactions (agent trace).

    After MessageNormalizer runs, all non-standard formats are already converted
    to standard tool_calls / role=tool — so checking those two signals suffices.
    """
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get('role') == 'tool':
            return True
        if normalize_tool_calls(m):
            return True
    return False
