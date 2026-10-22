# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reading what messages carry.

A message's ``content`` is a plain string in the simple case and a list of typed
parts when it is multimodal, so every caller that wants the text has to handle
both shapes. ``tool_calls`` has the same problem one level up: a round trip
through PyArrow or a JSONL dataset can leave it as a string holding JSON, or a
list of such strings, so asking "did the model call a tool" means decoding
before looking. A whole conversation raises the same kind of question -- which
turn is the model's answer, did it use tools at all -- answered the same way,
by looking rather than trusting the shape.

These live here rather than under any one consumer because none of the questions
is a preprocessing one: a challenger reading a model's reply, a reward scoring
one, and a cleaning step filtering one all ask them. Each place that answered on
its own answered differently -- handing back the raw list, or raising on it.

Kept to a plain ``Dict`` rather than :class:`~twinkle.data_format.Message` on
purpose: rows read straight off disk go through these too, before anything has
promised they match the type.
"""
import json
from typing import Any, Dict, List, Optional

from twinkle.data_format import last_assistant_text, msg_content_text

__all__ = [
    'assistant_text',
    'is_agent_row',
    'msg_content_text',
    'msg_has_media',
    'msg_has_payload',
    'normalize_tool_calls',
]


def msg_has_media(msg: Dict[str, Any]) -> bool:
    """True if message content contains non-text parts (image/audio/video)."""
    c = msg.get('content')
    return isinstance(c, list) and any(isinstance(p, dict) and p.get('type') not in ('text', None) for p in c)


def msg_has_payload(msg: Dict[str, Any]) -> bool:
    """True if a message carries any substantive payload (text, tool_calls, reasoning, or media)."""
    return bool(
        msg_content_text(msg).strip() or msg.get('tool_calls') or msg.get('reasoning_content') or msg.get('thinking')
        or msg_has_media(msg))


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


def assistant_text(trajectory: Dict[str, Any]) -> str:
    """The last assistant message's text, or '' if the model produced none.

    Thin alias of :func:`twinkle.data_format.last_assistant_text` -- the one
    implementation both twinkle rewards and agentic consumers share.
    """
    return last_assistant_text(trajectory)
