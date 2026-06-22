# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

MODEL_ID_ALIASES_ENV = 'TWINKLE_MODEL_ID_ALIASES'


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_model_alias(route_prefix: str, service_type: str = 'model') -> str | None:
    marker = f'/{service_type}/'
    if not route_prefix or marker not in route_prefix:
        return None
    alias = route_prefix.split(marker, 1)[1].strip('/')
    return alias or None


def build_model_alias_map(applications: Iterable[Any]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for application in applications or []:
        if _get_value(application, 'import_path') != 'model':
            continue
        alias = extract_model_alias(_get_value(application, 'route_prefix', ''), service_type='model')
        args = _get_value(application, 'args')
        model_id = _get_value(args, 'model_id') if args is not None else None
        if alias and model_id and alias != model_id:
            alias_map[alias] = model_id
    return alias_map


def load_model_alias_map(raw: str | Mapping[str, str] | None = None) -> dict[str, str]:
    if raw is None:
        raw = os.environ.get(MODEL_ID_ALIASES_ENV)
    if not raw:
        return {}
    if isinstance(raw, Mapping):
        return {str(key): str(value) for key, value in raw.items() if key and value}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if key and value}


def resolve_model_id_alias(model_id_or_path: str | None, alias_map: Mapping[str, str] | None = None) -> str | None:
    if not model_id_or_path:
        return model_id_or_path
    aliases = alias_map if alias_map is not None else load_model_alias_map()
    return aliases.get(model_id_or_path, model_id_or_path)
