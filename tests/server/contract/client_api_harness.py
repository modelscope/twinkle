# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Client-API contract harness.

Builds the four FastAPI apps used by the Ray Serve deployments (Gateway, Model,
Sampler, Processor) by registering their route-registration helpers against a
fresh FastAPI instance, then extracts the client-facing surface (route paths,
HTTP methods, and request/response schemas) as a stable JSON dict.

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

import json
from collections.abc import Callable
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pathlib import Path
from typing import Any

# ----- App build helpers --------------------------------------------------- #


def _noop_self() -> None:
    return None


def build_gateway_app() -> FastAPI:
    from twinkle.server.gateway.openai_handlers import _register_openai_routes
    from twinkle.server.gateway.tinker_handlers import _register_tinker_routes
    from twinkle.server.gateway.twinkle_handlers import _register_twinkle_routes

    app = FastAPI()
    _register_tinker_routes(app, _noop_self)
    _register_twinkle_routes(app, _noop_self)
    _register_openai_routes(app, _noop_self)
    return app


def build_model_app() -> FastAPI:
    from twinkle.server.model.tinker_handlers import _register_tinker_routes
    from twinkle.server.model.twinkle_handlers import _register_twinkle_routes

    app = FastAPI()
    _register_tinker_routes(app, _noop_self)
    _register_twinkle_routes(app, _noop_self)
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
    'gateway': build_gateway_app,
    'model': build_model_app,
    'sampler': build_sampler_app,
    'processor': build_processor_app,
}

# ----- Surface extraction -------------------------------------------------- #

_HTTP_METHODS = {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}


def _extract_app_surface(app: FastAPI) -> dict[str, Any]:
    """Return a SLIM client-contract view of ``app``'s OpenAPI surface.

    Snapshots, per path and HTTP method, only the stable client-facing contract:
    the ``operationId``, the ``parameters``, and the set of response status
    codes. The full ``components.schemas`` body and per-operation ``requestBody``
    schema are intentionally NOT snapshotted — they churn on Pydantic / FastAPI
    version bumps without representing a real client-contract change. Route
    paths, HTTP methods, and response status codes remain frozen.
    """
    spec = get_openapi(
        title='contract',
        version='0.0.0',
        routes=app.routes,
    )

    paths: dict[str, dict[str, Any]] = {}
    for path, ops in (spec.get('paths') or {}).items():
        clean_ops: dict[str, Any] = {}
        for method, op in ops.items():
            if method.upper() not in _HTTP_METHODS:
                continue
            clean_ops[method.upper()] = {
                'operationId': op.get('operationId'),
                'parameters': op.get('parameters', []),
                'responses': sorted((op.get('responses') or {}).keys()),
            }
        if clean_ops:
            paths[path] = clean_ops

    return {'paths': paths}


def extract_full_surface() -> dict[str, Any]:
    """Build all four apps and return a per-app contract surface dict."""
    surface: dict[str, Any] = {}
    for name, builder in APP_BUILDERS.items():
        app = builder()
        surface[name] = _extract_app_surface(app)
    return surface


# ----- Baseline I/O -------------------------------------------------------- #

BASELINE_PATH = Path(__file__).parent / 'client_api_baseline.json'


def write_baseline(path: Path | None = None) -> Path:
    """Snapshot the current client-API surface to ``client_api_baseline.json``."""
    p = Path(path) if path is not None else BASELINE_PATH
    surface = extract_full_surface()
    p.write_text(json.dumps(surface, indent=2, sort_keys=True) + '\n')
    return p


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else BASELINE_PATH
    return json.loads(p.read_text())
