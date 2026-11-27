# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lightweight JSON conversion helpers shared by component clients and servers."""
from __future__ import annotations

import base64
from io import BytesIO
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from numbers import Number
from pydantic import BaseModel
from typing import Any

_PRIMITIVE_TYPES = (str, Number, bool, bytes, type(None))


def json_safe(obj: Any) -> Any:
    """Recursively convert models, tensors, and arrays into JSON-compatible values.

    This module intentionally does not import :mod:`twinkle`: component protocol
    types are imported while the top-level package is still being initialized.
    """
    if isinstance(obj, BaseModel):
        return json_safe(obj.model_dump())
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_safe({field.name: getattr(obj, field.name) for field in fields(obj)})
    if isinstance(obj, Mapping):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_safe(value) for value in obj]
    tolist = getattr(obj, 'tolist', None)
    if callable(tolist) and not isinstance(obj, _PRIMITIVE_TYPES):
        try:
            return json_safe(tolist())
        except Exception:
            pass
    return obj


def media_ref(value: Any) -> str:
    """A URL passes as is; a local file, raw bytes or a PIL image is uploaded as a ``data:`` URI."""
    if isinstance(value, dict) and ('bytes' in value or 'path' in value):
        value = value.get('bytes') or value.get('path')
    if isinstance(value, str):
        if value.startswith(('http://', 'https://', 'data:')):
            return value
        with open(value, 'rb') as file:
            value = file.read()
    if isinstance(value, (bytes, bytearray)):
        return 'data:application/octet-stream;base64,' + base64.b64encode(bytes(value)).decode()
    if callable(getattr(value, 'save', None)):  # a PIL image: PNG keeps the pixels the template encoded
        buffer = BytesIO()
        value.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    raise TypeError(f'cannot upload a {type(value).__name__} media entry; pass a path, URL, bytes or PIL image.')


def wire_trajectory(value: Any) -> Any:
    """Serialize trajectory media without interpreting arbitrary tool arguments as media fields."""
    media_keys = ('images', 'videos', 'audios')

    def wire_message(message):
        result = json_safe({key: item for key, item in message.items() if key not in media_keys and key != 'content'})
        for key in media_keys:
            if message.get(key) is not None:
                entries = message[key] if isinstance(message[key], (list, tuple)) else [message[key]]
                result[key] = [media_ref(entry) for entry in entries]
        content = message.get('content')
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, Mapping):
                    block = dict(block)
                    kind = block.get('type')
                    if kind in ('image', 'video', 'audio', 'image_url', 'video_url', 'audio_url'):
                        for key in (kind, 'url', 'path'):
                            item = block.get(key)
                            if item is not None:
                                block[key] = ({**item, 'url': media_ref(item['url'])}
                                              if isinstance(item, Mapping) and 'url' in item else media_ref(item))
                                break
                blocks.append(json_safe(block))
            result['content'] = blocks
        elif 'content' in message:
            result['content'] = json_safe(content)
        return result

    if isinstance(value, Mapping):
        if 'role' in value and 'content' in value and 'messages' not in value:
            return wire_message(value)
        conversation = any(key in value for key in ('messages', 'input_ids', 'input_embedding'))
        result = {}
        for key, item in value.items():
            if conversation and key in media_keys and item is not None:
                entries = item if isinstance(item, (list, tuple)) else [item]
                result[key] = [media_ref(entry) for entry in entries]
            elif key == 'messages' and item is not None:
                result[key] = [wire_message(message) for message in item]
            elif key == 'train_input' and isinstance(item, Mapping):
                result[key] = wire_trajectory(item)
            else:
                result[key] = json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [wire_trajectory(item) for item in value]
    return json_safe(value)
