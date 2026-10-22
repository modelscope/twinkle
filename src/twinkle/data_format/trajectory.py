# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import sys
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

from .message import Message, Tool

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class Trajectory(TypedDict, total=False):
    messages: List[Message]
    tools: List[Tool]
    # PyArrow-stable encoding: each entry is (key, json.dumps(value)). Use the helpers below.
    user_data: List[Tuple[str, str]]
    images: Optional[List[Union[str, Any]]]
    videos: Optional[List[Union[str, Any]]]
    audios: Optional[List[Union[str, Any]]]
    prompt: Optional[str]


def pack_value(value: Any) -> str:
    """Encode a single user_data value to a JSON string."""
    return json.dumps(value, ensure_ascii=False, default=str)


def pack_user_data(values: Any) -> List[Tuple[str, str]]:
    """Build a canonical ``user_data`` payload from a plain mapping.

    ``user_data`` must be a list of ``(key, json_string)`` pairs: a dict cannot be written to
    PyArrow (its struct schema would differ from shard to shard), and readers go through
    :func:`user_data_get`, which only understands the packed form — a dict is silently ignored
    there, so e.g. ``key_rounds`` would be dropped without any error. Always pack before
    attaching to a trajectory.
    """
    if values is None:
        return []
    if isinstance(values, Mapping):
        return [(k, v if isinstance(v, str) else pack_value(v)) for k, v in values.items()]
    return [(k, v if isinstance(v, str) else pack_value(v)) for k, v in values]


def attach_user_data(trajectory: Trajectory, **values: Any) -> Trajectory:
    """Return ``trajectory`` with ``values`` merged into its packed ``user_data``.

    ``user_data`` is a list of ``(key, json_string)`` pairs rather than a dict,
    so it cannot be updated in place with ``update()``; going through
    :func:`pack_user_data` keeps it in the one shape readers understand.
    """
    merged: Dict[str, Any] = {}
    for entry in trajectory.get('user_data') or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            merged[entry[0]] = entry[1]
    merged.update(values)
    out = dict(trajectory)
    out['user_data'] = pack_user_data(merged)
    return out


def ground_truth(trajectory: Trajectory, key: str = 'ground_truth') -> Any:
    """A trajectory's gold label: the packed ``user_data`` entry first, the flat column second.

    Both spellings carry the same label -- a twinkle pipeline packs ``user_data``, a swift scoring
    row keeps dataset columns flat -- so a Reward reads one helper and works in either world.
    """
    value = user_data_get(trajectory.get('user_data'), key)
    if value is None:
        value = trajectory.get(key)
    return value


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


def last_assistant_text(trajectory: Trajectory) -> str:
    """The newest assistant turn's text -- the completion a reward scores.

    Multimodal messages carry content as a list of parts, so text goes through
    :func:`msg_content_text` instead of being read as a plain string. The *last* assistant
    turn: a conversation that went through tools has several, and the model's answer is
    the turn it finished on.
    """
    for message in reversed(trajectory.get('messages') or []):
        if isinstance(message, dict) and message.get('role') == 'assistant':
            return msg_content_text(message)
    return ''


def user_data_get(items: Any, key: str, default: Any = None) -> Any:
    """Look up the first value matching ``key`` in packed user_data, decoded."""
    if isinstance(items, Mapping):
        raise TypeError('user_data must be a list of (key, json_string) pairs, got a mapping. '
                        'Wrap it with twinkle.data_format.pack_user_data().')
    if not isinstance(items, list):
        return default
    for entry in items:
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] == key:
            v = entry[1]
            if not isinstance(v, str):
                return v
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return v
    return default
