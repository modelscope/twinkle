# Copyright (c) ModelScope Contributors. All rights reserved.
import functools
from typing import Any, Tuple, Callable

from fastapi import Request
from fastapi.responses import JSONResponse


async def verify_request_token(request: Request, call_next):
    authorization = request.headers.get("Authorization")
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else authorization
    if not is_token_valid(token):
        return JSONResponse(status_code=403, content={"detail": "Invalid token"})

    request_id = request.headers.get("X-Ray-Serve-Request-Id")
    if not request_id:
        return JSONResponse(
            status_code=400,
            content={"detail": "Missing X-Ray-Serve-Request-Id header, required for sticky session"}
        )
    request.state.request_id = request_id
    request.state.token = token
    response = await call_next(request)
    return response


def is_token_valid(token: str) -> bool:
    return True


class ConfigRegistryProxy:

    def __init__(self, actor_handle):
        self._actor = actor_handle

    def add_config(self, key: str, value: Any):
        import ray
        return ray.get(self._actor.add_config.remote(key, value))

    def add_or_get(self, key: str, value: Any) -> Any:
        import ray
        return ray.get(self._actor.add_or_get.remote(key, value))

    def get_config(self, key: str):
        import ray
        return ray.get(self._actor.get_config.remote(key))

    def pop(self, key: str):
        import ray
        return ray.get(self._actor.pop.remote(key))

    def clear(self):
        import ray
        return ray.get(self._actor.clear.remote())


class ConfigRegistry:

    def __init__(self):
        self.config = {}

    def add_config(self, key: str, value: Any):
        self.config[key] = value

    def add_or_get(self, key: str, value: Any) -> Tuple[bool, Any]:
        if key in self.config:
            return self.config[key]
        self.config[key] = value
        return value

    def get_config(self, key: str):
        return self.config.get(key)

    def pop(self, key: str):
        self.config.pop(key, None)

    def clear(self):
        self.config.clear()


def init_config_registry() -> ConfigRegistryProxy:
    import ray

    _registry = None
    _ConfigRegistry = ray.remote(ConfigRegistry)

    try:
        _registry = ray.get_actor('adapter_registry')
    except ValueError:
        try:
            _registry = _ConfigRegistry.options(
                name='adapter_registry',
                lifetime='detached',
            ).remote()
        except ValueError:
            _registry = ray.get_actor('adapter_registry')
    assert _registry is not None
    return ConfigRegistryProxy(_registry)
