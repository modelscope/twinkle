# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Client-API contract harness.

Builds the five FastAPI apps used by the Ray Serve deployments (Data Plane,
Gateway, Model, Sampler, Processor) by registering their route-registration helpers against a
fresh FastAPI instance, then extracts route paths, methods, parameters, and
recursive request/response type shapes as a stable JSON dict.

Used to:
- snapshot the current surface into ``client_api_baseline.json`` before the
  refactor begins, and
- assert post-refactor equality (cross-cutting freeze guard).

Notes:
- The handler factories accept ``(app, self_fn)``; we pass a no-op ``self_fn``
  because route registration only inspects the app object — the closures are
  never invoked here.
- We restrict the surface to the client-facing endpoints. The Tinker-public
  surface on the Gateway is at ``/*`` (flat, by design), and the Twinkle
  surface is at ``/twinkle/*`` everywhere. Internal ``/tinker/*`` routes
  registered on Model and Sampler are also captured because the Gateway proxy
  forwards Tinker compute requests to them — their request/response schemas
  are part of the externally observed Tinker contract.
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
import types as pytypes
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from fastapi import FastAPI
from fastapi.routing import APIRoute
from pathlib import Path
from pydantic import BaseModel
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

# ----- App build helpers --------------------------------------------------- #


def _noop_self() -> None:
    return None


def build_data_plane_app() -> FastAPI:
    from twinkle.server.data_plane.handlers import register_data_plane_routes

    app = FastAPI()
    register_data_plane_routes(app, _noop_self)
    return app


def build_gateway_app() -> FastAPI:
    from twinkle.server.gateway.openai_handlers import _register_openai_routes
    from twinkle.server.gateway.tinker_handlers import _register_gateway_tinker_routes
    from twinkle.server.gateway.twinkle_handlers import _register_gateway_twinkle_routes

    app = FastAPI()
    _register_gateway_tinker_routes(app, _noop_self)
    _register_gateway_twinkle_routes(app, _noop_self)
    _register_openai_routes(app, _noop_self)
    return app


def build_model_app() -> FastAPI:
    from twinkle.server.model.tinker_handlers import _register_model_tinker_routes
    from twinkle.server.model.twinkle_handlers import _register_model_twinkle_routes

    app = FastAPI()
    _register_model_tinker_routes(app, _noop_self)
    _register_model_twinkle_routes(app, _noop_self)
    return app


def build_sampler_app() -> FastAPI:
    from twinkle.server.sampler.tinker_handlers import _register_tinker_sampler_routes
    from twinkle.server.sampler.twinkle_handlers import _register_twinkle_sampler_routes

    app = FastAPI()
    _register_tinker_sampler_routes(app, _noop_self)
    _register_twinkle_sampler_routes(app, _noop_self)
    return app


def build_processor_app() -> FastAPI:
    from twinkle.server.processor.twinkle_handlers import _register_processor_routes

    app = FastAPI()
    _register_processor_routes(app, _noop_self)
    return app


APP_BUILDERS: dict[str, Callable[[], FastAPI]] = {
    'data_plane': build_data_plane_app,
    'gateway': build_gateway_app,
    'model': build_model_app,
    'sampler': build_sampler_app,
    'processor': build_processor_app,
}

# ----- Surface extraction -------------------------------------------------- #

_HTTP_METHODS = {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}


def _type_contract(annotation: Any, seen: frozenset[str] = frozenset()) -> Any:
    """Build a stable field-level schema for Pydantic models and SDK dataclasses."""
    if annotation is None or annotation is type(None):
        return {'type': 'null'}
    if annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _type_contract(args[0], seen)
    if origin in (Union, pytypes.UnionType):
        return {'anyOf': [_type_contract(arg, seen) for arg in args]}
    if origin in (list, set, tuple, Sequence):
        return {'type': 'array', 'items': _type_contract(args[0], seen) if args else {}}
    if origin in (dict, Mapping):
        return {'type': 'object', 'additionalProperties': _type_contract(args[1], seen) if len(args) > 1 else {}}
    if origin is Literal:
        return {'enum': list(args)}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {'enum': [item.value for item in annotation]}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_json_schema()
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        name = f'{annotation.__module__}.{annotation.__qualname__}'
        if name in seen:
            return {'$ref': name}
        module = sys.modules.get(annotation.__module__)
        try:
            hints = get_type_hints(annotation, globalns=vars(module) if module else None)
        except (NameError, TypeError):
            hints = annotation.__annotations__
        properties = {}
        required = []
        for field in dataclasses.fields(annotation):
            if not field.init or field.name.startswith('_'):
                continue
            properties[field.name] = _type_contract(hints.get(field.name, Any), seen | {name})
            if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
                required.append(field.name)
        result = {'type': 'object', 'properties': properties}
        if required:
            result['required'] = required
        return result
    primitive = {str: 'string', int: 'integer', float: 'number', bool: 'boolean'}
    if annotation in primitive:
        return {'type': primitive[annotation]}
    return {'pythonType': getattr(annotation, '__qualname__', repr(annotation))}


def _parameter_contract(field: Any) -> dict[str, Any]:
    field_info = field.field_info
    return {
        'name': field.alias,
        'required': bool(field_info.is_required()),
        'schema': _type_contract(field_info.annotation),
    }


def _extract_app_surface(app: FastAPI) -> dict[str, Any]:
    """Return every route's complete request and response type shape."""
    paths: dict[str, dict[str, Any]] = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        extra_responses = {}
        for status, response in route.responses.items():
            extra_responses[str(status)] = {
                'description': response.get('description'),
                'content': response.get('content'),
                'model': _type_contract(response.get('model')) if response.get('model') else None,
            }
        operation = {
            'operationId': route.operation_id or route.name,
            'body': [_parameter_contract(field) for field in route.dependant.body_params],
            'path': [_parameter_contract(field) for field in route.dependant.path_params],
            'query': [_parameter_contract(field) for field in route.dependant.query_params],
            'headers': [_parameter_contract(field) for field in route.dependant.header_params],
            'cookies': [_parameter_contract(field) for field in route.dependant.cookie_params],
            'response': _type_contract(route.response_model),
            'responses': extra_responses,
            'statusCode': route.status_code or 200,
        }
        for method in sorted(route.methods & _HTTP_METHODS):
            client_path = _client_path(route.path)
            paths.setdefault(client_path, {})[method] = operation
    return {'paths': paths}


def extract_full_surface() -> dict[str, Any]:
    """Build all five apps and return a per-app contract surface dict."""
    surface: dict[str, Any] = {}
    for name, builder in APP_BUILDERS.items():
        app = builder()
        surface[name] = _extract_app_surface(app)
    return surface


def _client_path(route_path: str) -> str:
    """Strip FastAPI path-converter suffixes so ``{id:path}`` reads as ``{id}``."""
    return re.sub(r'{([^}:]+):[^}]+}', r'{\1}', route_path)


def _model_name(annotation: Any) -> str | None:
    """A stable, human-readable name for a request/response model annotation."""
    if annotation is None:
        return None
    return getattr(annotation, '__qualname__', None) or repr(annotation)


def extract_route_inventory() -> dict[str, dict[str, Any]]:
    """A compact, reviewable projection of the wire surface.

    One entry per route -- ``"<METHOD> <path>" -> {response, body, statusCode}`` --
    naming the model classes instead of inlining their field schemas.

    This is the projection that gets committed. A route appearing, disappearing, or
    changing its response/body model shows up as a few readable lines in a PR diff,
    whereas the full field-level surface is ~8k lines and nobody reads that diff.
    The trade-off is explicit: this catches route-level and model-level changes, not
    field-level drift inside a model.
    """
    inventory: dict[str, dict[str, Any]] = {}
    for name, builder in APP_BUILDERS.items():
        app = builder()
        routes: dict[str, Any] = {}
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue
            client_path = _client_path(route.path)
            body = [_model_name(field.field_info.annotation) for field in route.dependant.body_params]
            for method in sorted(route.methods & _HTTP_METHODS):
                routes[f'{method} {client_path}'] = {
                    'response': _model_name(route.response_model),
                    'body': body,
                    'statusCode': route.status_code or 200,
                }
        inventory[name] = routes
    return inventory


# ----- Snapshot I/O -------------------------------------------------------- #

# The full field-level surface (:func:`extract_full_surface`). A GENERATED artifact,
# deliberately NOT committed: an 8k-line diff on every intentional wire change is noise
# nobody reads. Being regenerated from the code under test, it cannot by itself detect an
# unintended change -- ``ROUTES_PATH`` is the guard that can. Keep that asymmetry in mind
# before treating a green baseline test as evidence of anything.
BASELINE_PATH = Path(__file__).parent / 'client_api_baseline.json'

# The compact route inventory (:func:`extract_route_inventory`). COMMITTED to git: this
# is the actual regression guard, so it has to stay tracked for the guard to mean
# anything.
ROUTES_PATH = Path(__file__).parent / 'client_api_routes.json'

_REGEN_HINT = 'Regenerate with: python -m tests.server.contract.update_baseline'


def write_baseline(path: Path | None = None) -> Path:
    """Snapshot the current client-API surface to ``client_api_baseline.json``."""
    p = Path(path) if path is not None else BASELINE_PATH
    surface = extract_full_surface()
    p.write_text(json.dumps(surface, indent=2, sort_keys=True) + '\n')
    return p


def write_route_inventory(path: Path | None = None) -> Path:
    """Snapshot the compact route inventory to ``client_api_routes.json``."""
    p = Path(path) if path is not None else ROUTES_PATH
    p.write_text(json.dumps(extract_route_inventory(), indent=2, sort_keys=True) + '\n')
    return p


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    """Load the generated full surface, failing with a fix hint rather than a bare OSError."""
    p = Path(path) if path is not None else BASELINE_PATH
    if not p.is_file():
        raise FileNotFoundError(f'Contract baseline missing: {p}\n'
                                f'It is a generated artifact and is deliberately not committed. '
                                f'{_REGEN_HINT}')
    return json.loads(p.read_text())


def load_route_inventory(path: Path | None = None) -> dict[str, Any]:
    """Load the committed route inventory, failing loudly if it went missing."""
    p = Path(path) if path is not None else ROUTES_PATH
    if not p.is_file():
        raise FileNotFoundError(f'Committed route inventory missing: {p}\n'
                                f'This file IS tracked by git -- restore it instead of regenerating '
                                f'blindly, or the guard silently becomes a tautology. {_REGEN_HINT}')
    return json.loads(p.read_text())
