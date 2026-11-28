# Copyright (c) ModelScope Contributors. All rights reserved.
"""JSON-safe encoding for rollout component state.

Model checkpoints persist ``component_states`` in ``trainer_state.json``. Agentic RL rows can
contain tensors (``pixel_values`` / media grids), numpy arrays, PIL images and raw image bytes, all
of which the standard JSON encoder rejects. This codec keeps those values self-contained without
falling back to pickle:

* tensor / ndarray: dtype, shape and C-order raw bytes;
* PIL image: PNG bytes;
* bytes-like value: base64, with zlib compression when it reduces the payload.

Plain JSON values and containers are returned structurally unchanged, so existing text-only
component states remain backwards compatible.
"""
from __future__ import annotations
import base64
import os
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MARKER = '__twinkle_component_state_v1__'
_VERSION = 1
_SCALARS = (str, int, float, bool)


def encode_component_state(value: Any) -> Any:
    """Return a JSON-serializable copy of ``value`` for ``trainer_state.json``."""
    return _encode(value, '$')


def decode_component_state(value: Any) -> Any:
    """Restore values encoded by :func:`encode_component_state`."""
    return _decode(value, '$')


def _encode(value: Any, path: str) -> Any:
    if value is None or isinstance(value, _SCALARS):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _marker('bytes', **_blob(bytes(value)))
    if isinstance(value, os.PathLike):
        return _marker('path', value=os.fspath(value))
    if isinstance(value, tuple):
        return _marker('tuple', items=[_encode(item, f'{path}[{index}]')
                                       for index, item in enumerate(value)])
    if isinstance(value, list):
        return [_encode(item, f'{path}[{index}]') for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if _MARKER not in value and all(isinstance(key, str) for key in value):
            return {key: _encode(item, f'{path}.{key}') for key, item in value.items()}
        return _marker(
            'mapping',
            items=[[_encode(key, f'{path}.<key>'), _encode(item, f'{path}[{key!r}]')]
                   for key, item in value.items()],
        )

    value_type = type(value)
    module = value_type.__module__
    if module.startswith('torch'):
        import torch

        if isinstance(value, torch.Tensor):
            return _encode_tensor(value, path)
    if module.startswith('numpy'):
        import numpy as np

        if isinstance(value, np.ndarray):
            return _encode_ndarray(value, path)
        if isinstance(value, np.generic):
            return value.item()
    if module.startswith('PIL.'):
        from PIL import Image

        if isinstance(value, Image.Image):
            return _encode_image(value, path)

    raise TypeError(
        f'component state at {path} contains unsupported {value_type.__module__}.'
        f'{value_type.__qualname__}; use JSON values, tensor, ndarray, PIL.Image or bytes'
    )


def _decode(value: Any, path: str) -> Any:
    if isinstance(value, list):
        return [_decode(item, f'{path}[{index}]') for index, item in enumerate(value)]
    if not isinstance(value, Mapping):
        return value

    kind = value.get(_MARKER)
    if kind is None or value.get('version') != _VERSION:
        return {key: _decode(item, f'{path}.{key}') for key, item in value.items()}

    if kind == 'bytes':
        return _decode_blob(value, path)
    if kind == 'path':
        return Path(_required(value, 'value', path, str))
    if kind == 'tuple':
        items = _required(value, 'items', path, list)
        return tuple(_decode(item, f'{path}[{index}]') for index, item in enumerate(items))
    if kind == 'mapping':
        items = _required(value, 'items', path, list)
        return {
            _decode(key, f'{path}.<key>'):
            _decode(item, f'{path}[{key!r}]')
            for key, item in items
        }
    if kind == 'tensor':
        return _decode_tensor(value, path)
    if kind == 'ndarray':
        return _decode_ndarray(value, path)
    if kind == 'image':
        return _decode_image(value, path)

    # Unknown future markers remain ordinary dictionaries rather than making old readers fail.
    return {key: _decode(item, f'{path}.{key}') for key, item in value.items()}


def _encode_tensor(value: Any, path: str) -> dict[str, Any]:
    import torch

    if value.layout != torch.strided:
        raise TypeError(f'component state at {path} has unsupported tensor layout {value.layout}')
    if value.is_quantized:
        raise TypeError(f'component state at {path} cannot persist a quantized tensor')
    if value.device.type == 'meta':
        raise TypeError(f'component state at {path} cannot persist a meta tensor')
    cpu = value.detach().to('cpu').contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return _marker(
        'tensor',
        dtype=str(value.dtype).removeprefix('torch.'),
        shape=list(value.shape),
        **_blob(raw),
    )


def _decode_tensor(value: Mapping[str, Any], path: str) -> Any:
    import torch

    dtype_name = _required(value, 'dtype', path, str)
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f'component state at {path} has unknown tensor dtype {dtype_name!r}')
    shape = tuple(_required(value, 'shape', path, list))
    raw = _decode_blob(value, path)
    if not raw:
        return torch.empty(shape, dtype=dtype)
    try:
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f'component state at {path} cannot restore tensor {dtype_name}{shape}') from exc


def _encode_ndarray(value: Any, path: str) -> dict[str, Any]:
    if value.dtype.hasobject:
        raise TypeError(f'component state at {path} cannot persist an object-dtype ndarray')
    return _marker(
        'ndarray',
        dtype=value.dtype.str,
        shape=list(value.shape),
        **_blob(value.tobytes(order='C')),
    )


def _decode_ndarray(value: Mapping[str, Any], path: str) -> Any:
    import numpy as np

    dtype = _required(value, 'dtype', path, str)
    shape = tuple(_required(value, 'shape', path, list))
    raw = _decode_blob(value, path)
    if not raw:
        return np.empty(shape, dtype=dtype)
    try:
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f'component state at {path} cannot restore ndarray {dtype}{shape}') from exc


def _encode_image(value: Any, path: str) -> dict[str, Any]:
    from io import BytesIO

    buffer = BytesIO()
    try:
        value.save(buffer, format='PNG')
    except Exception as exc:
        raise TypeError(f'component state at {path} cannot encode PIL image as PNG') from exc
    return _marker('image', format='PNG', **_blob(buffer.getvalue()))


def _decode_image(value: Mapping[str, Any], path: str) -> Any:
    from io import BytesIO

    from PIL import Image

    image_format = _required(value, 'format', path, str)
    try:
        with Image.open(BytesIO(_decode_blob(value, path)), formats=(image_format,)) as image:
            return image.copy()
    except Exception as exc:
        raise ValueError(f'component state at {path} cannot restore {image_format} image') from exc


def _marker(kind: str, **fields: Any) -> dict[str, Any]:
    return {_MARKER: kind, 'version': _VERSION, **fields}


def _blob(raw: bytes) -> dict[str, Any]:
    compression = 'none'
    payload = raw
    if len(raw) >= 256:
        compressed = zlib.compress(raw, level=6)
        if len(compressed) < len(raw):
            compression = 'zlib'
            payload = compressed
    return {
        'compression': compression,
        'encoding': 'base64',
        'data': base64.b64encode(payload).decode('ascii'),
    }


def _decode_blob(value: Mapping[str, Any], path: str) -> bytes:
    encoding = _required(value, 'encoding', path, str)
    if encoding != 'base64':
        raise ValueError(f'component state at {path} has unsupported byte encoding {encoding!r}')
    try:
        raw = base64.b64decode(_required(value, 'data', path, str), validate=True)
    except ValueError as exc:
        raise ValueError(f'component state at {path} contains invalid base64 data') from exc
    compression = _required(value, 'compression', path, str)
    if compression == 'none':
        return raw
    if compression == 'zlib':
        try:
            return zlib.decompress(raw)
        except zlib.error as exc:
            raise ValueError(f'component state at {path} contains invalid zlib data') from exc
    raise ValueError(f'component state at {path} has unsupported compression {compression!r}')


def _required(value: Mapping[str, Any], key: str, path: str, expected_type: type) -> Any:
    item = value.get(key)
    if not isinstance(item, expected_type):
        raise ValueError(f'component state at {path} requires {expected_type.__name__} field {key!r}')
    return item


__all__ = ['decode_component_state', 'encode_component_state']
