# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import sys
from collections.abc import Mapping
from typing import Any, List, Optional, Tuple, Union

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
