# Copyright (c) ModelScope Contributors. All rights reserved.
"""Characterization tests for the four App_Builders.

These freeze the externally observable behavior of ``build_gateway_app``,
``build_model_app``, ``build_sampler_app`` and ``build_processor_app`` BEFORE
the Shared_App_Scaffold is extracted, so the extraction can be shown to be
behavior-preserving. For each builder they assert, as fixed expectations:

1. the **registered route set** — the complete set of route paths together with
   their HTTP methods (compared against the committed contract baseline);
2. the **externally observable middleware ordering/effect**, to the extent it is
   black-box observable — the ``verify_token`` auth middleware rejects a
   non-``/healthz`` request that is missing the sticky-session request-id header
   while letting ``/healthz`` through, which only holds when the middleware
   stack is wired in front of the routes;
3. the **bound deployment** identified by the name passed to ``serve.deployment``
   (``GatewayServer``, ``ModelManagement``, ``SamplerManagement``,
   ``ProcessorManagement``).

They MUST NOT assert internal object identity or internal middleware-stack
structure. They are built by capturing the FastAPI app the builder
constructs (via monkeypatched ``serve.ingress`` / ``serve.deployment``) so no
live Ray Serve cluster is needed.
"""
from __future__ import annotations

import json
import pytest
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.testclient import TestClient
from pathlib import Path

# ----- contract baseline (route-set oracle) -------------------------------- #

_BASELINE = json.loads((Path(__file__).parent / 'contract' / 'client_api_baseline.json').read_text())

_HTTP_METHODS = {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}


def _route_set(app: FastAPI) -> set[tuple[str, str]]:
    """Return the (path, method) pairs of ``app``'s OpenAPI surface.

    Uses the same ``get_openapi`` extraction the contract harness uses, so the
    FastAPI built-in doc routes (``/docs``, ``/openapi.json``, ``/redoc``) and
    auto-added ``HEAD`` / ``OPTIONS`` methods are excluded — leaving exactly the
    documented client-facing operations.
    """
    spec = get_openapi(title='c', version='0', routes=app.routes)
    pairs: set[tuple[str, str]] = set()
    for path, ops in (spec.get('paths') or {}).items():
        for method in ops:
            if method.upper() in _HTTP_METHODS:
                pairs.add((path, method.upper()))
    return pairs


def _baseline_route_set(app_name: str) -> set[tuple[str, str]]:
    paths = _BASELINE[app_name]['paths']
    return {(path, method) for path, ops in paths.items() for method in ops}


# ----- builder capture ----------------------------------------------------- #


class _CaptureResult:

    def __init__(self) -> None:
        self.app: FastAPI | None = None
        self.deployment_name: str | None = None


def _capture_builder(monkeypatch, build_callable, *args, **kwargs) -> _CaptureResult:
    """Invoke a real builder, capturing the FastAPI app + deployment name.

    ``serve.ingress(app)(cls)`` and ``serve.deployment(name=...)(...)`` are
    replaced with capturing stubs in the shared ``app_scaffold`` (where every
    builder now performs binding), and the returned ``DeploymentClass`` exposes
    a no-op ``.options(...).bind(...)`` so the builder never touches a live
    cluster.
    """
    result = _CaptureResult()

    def fake_ingress(app):
        result.app = app

        def _decorate(cls):
            return cls

        return _decorate

    def fake_deployment(*d_args, **d_kwargs):
        result.deployment_name = d_kwargs.get('name') or (d_args[0] if d_args else None)

        def _decorate(cls):

            class _Bound:

                def options(self, **_opts):
                    return self

                def bind(self, *_a, **_k):
                    return self

            return _Bound()

        return _decorate

    from twinkle.server import deployment as app_scaffold

    monkeypatch.setattr(app_scaffold.serve, 'ingress', fake_ingress)
    monkeypatch.setattr(app_scaffold.serve, 'deployment', fake_deployment)
    build_callable(*args, **kwargs)
    assert result.app is not None, 'builder did not call serve.ingress(app)'
    return result


# ----- per-builder expectations -------------------------------------------- #


def test_gateway_builder_characterization(monkeypatch) -> None:
    from twinkle.server.gateway import app as gateway_mod

    res = _capture_builder(
        monkeypatch,
        gateway_mod.build_gateway_app,
        deploy_options={},
    )
    assert res.deployment_name == 'GatewayServer'
    assert _route_set(res.app) == _baseline_route_set('gateway')
    _assert_auth_middleware_effect(res.app)
    _assert_middleware_lifo_order(res.app, expect_cleanup=True)


def test_model_builder_characterization(monkeypatch) -> None:
    from twinkle.server.model import app as model_mod

    res = _capture_builder(
        monkeypatch,
        model_mod.build_model_app,
        model_id='m',
        nproc_per_node=1,
        device_group={},
        device_mesh={},
        deploy_options={},
        backend='mock',
    )
    assert res.deployment_name == 'ModelManagement'
    assert _route_set(res.app) == _baseline_route_set('model')
    _assert_auth_middleware_effect(res.app)
    _assert_middleware_lifo_order(res.app, expect_cleanup=False, expect_replica_id=True)


def test_sampler_builder_characterization(monkeypatch) -> None:
    from twinkle.server.sampler import app as sampler_mod

    res = _capture_builder(
        monkeypatch,
        sampler_mod.build_sampler_app,
        model_id='m',
        nproc_per_node=1,
        device_group={},
        device_mesh={},
        deploy_options={},
        sampler_type='mock',
    )
    assert res.deployment_name == 'SamplerManagement'
    assert _route_set(res.app) == _baseline_route_set('sampler')
    _assert_auth_middleware_effect(res.app)
    _assert_middleware_lifo_order(res.app, expect_cleanup=False, expect_replica_id=True)


def test_processor_builder_characterization(monkeypatch) -> None:
    from twinkle.server.processor import app as processor_mod

    res = _capture_builder(
        monkeypatch,
        processor_mod.build_processor_app,
        ncpu_proc_per_node=1,
        device_group={},
        device_mesh={},
        deploy_options={},
    )
    assert res.deployment_name == 'ProcessorManagement'
    assert _route_set(res.app) == _baseline_route_set('processor')
    _assert_auth_middleware_effect(res.app)
    _assert_middleware_lifo_order(res.app, expect_cleanup=False)


# ----- black-box middleware-effect oracle ---------------------------------- #


def _assert_auth_middleware_effect(app: FastAPI) -> None:
    """The ``verify_token`` middleware is wired in front of the routes.

    Observable contract: a non-``/healthz`` POST that carries a Bearer token but
    omits the ``X-Ray-Serve-Request-Id`` header is rejected with 400 by the
    middleware before reaching any handler. This is a black-box assertion about
    the middleware's *effect* — it does not inspect the middleware stack.
    """
    client = TestClient(app)

    # Find a non-healthz POST route from the registered set.
    target = None
    for path, method in sorted(_route_set(app)):
        if method == 'POST' and not path.endswith('/healthz'):
            target = path
            break
    assert target is not None, 'no POST route to probe'

    # Skip path-parameterized routes (would 404 on a literal request before the
    # body is parsed); pick a static path.
    if '{' in target:
        static = [p for p, m in sorted(_route_set(app)) if m == 'POST' and '{' not in p and not p.endswith('/healthz')]
        if static:
            target = static[0]

    resp = client.post(target, headers={'Twinkle-Authorization': 'Bearer t'}, json={})
    # The verify_token middleware rejects the missing sticky-session header with
    # 400 before the handler runs. (Without the middleware the request would
    # reach the handler and fail differently — typically 422/500.)
    assert resp.status_code == 400, (
        f'expected 400 from verify_token for {target!r}, got {resp.status_code}: {resp.text[:200]}')
    assert 'x-request-id' in resp.text


# ----- middleware ordering oracle (metrics outermost → tracing → auth) ----- #


def _registered_http_middleware_names(app: FastAPI) -> list[str]:
    """Return the dispatch-function names of the app's HTTP middlewares.

    FastAPI's ``app.middleware('http')(func)`` decorator records the function
    on ``app.user_middleware`` as ``Middleware(BaseHTTPMiddleware,
    dispatch=func)``. Starlette prepends each new middleware to the list, so
    ``user_middleware[0]`` is the OUTERMOST wrapper at request time and
    ``user_middleware[-1]`` is the INNERMOST.

    We expose just the dispatch ``__name__`` per entry so the test asserts
    against the externally observable identity of each middleware (the names
    are stable parts of the scaffold's public surface), not the
    BaseHTTPMiddleware wrapping that Starlette imposes internally.
    """
    names: list[str] = []
    for mw in app.user_middleware:
        dispatch = mw.kwargs.get('dispatch') if hasattr(mw, 'kwargs') else None
        if dispatch is None and len(getattr(mw, 'args', ())) >= 1:
            # Starlette historically stored the dispatch positionally.
            dispatch = mw.args[0]
        if dispatch is not None:
            names.append(getattr(dispatch, '__name__', repr(dispatch)))
    return names


def _assert_middleware_lifo_order(app: FastAPI, *, expect_cleanup: bool, expect_replica_id: bool = False) -> None:
    """Assert the scaffold middleware stack preserves the intended LIFO order.

    ``user_middleware`` is ordered OUTERMOST → INNERMOST (Starlette prepends
    each new entry). Model and Sampler add ``inject_replica_id`` after the
    scaffold stack, making it the outermost middleware when enabled.
    """
    names = _registered_http_middleware_names(app)
    expected = []
    if expect_replica_id:
        expected.append('inject_replica_id')
    expected.extend(['metrics_middleware', 'tracing_middleware', 'verify_token'])
    expected.append('catch_unhandled_exceptions')
    if expect_cleanup:
        expected.append('ensure_state_cleanup_started')
    assert names == expected, (f'middleware ordering mismatch — expected (outermost→innermost) '
                               f'{expected!r}, got {names!r}')
