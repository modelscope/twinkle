"""Reward manager registry."""
from typing import Dict, Type

_REGISTRY: Dict[str, Type] = {}


def _normalize(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("reward manager name must be a non-empty string")
    return name.strip().lower().replace("-", "_")


def register(name: str):
    key = _normalize(name)

    def decorator(cls):
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"reward manager already registered: {key}")
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_reward_manager_cls(name: str):
    key = _normalize(name)
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown reward manager {name!r}; available: {available}") from exc


def registered_managers():
    return dict(_REGISTRY)
