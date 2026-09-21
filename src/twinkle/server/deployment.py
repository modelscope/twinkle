# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared deployment-application construction.

Top-level deployment-construction infrastructure shared by Gateway, Model,
Sampler, Processor, and DataPlane. It is intentionally NOT under
``utils/`` — it is core to how every deployment is built, not a generic helper.

It consolidates, in one place, the construction logic the App_Builders
used to repeat:

- ``get_servable()`` — the single servable-object accessor;
- ``build_deployment_app(component, register_routes, ...)`` — the canonical
  FastAPI lifespan + middleware stack + route registration;
- ``bind_deployment(app, cls, deploy_options, ...)`` — the
  ``serve.ingress`` → ``serve.deployment`` → ``.options().bind()`` chain;
- ``LazyCleanupMixin`` — the single lazy first-request ``ServerState``
  cleanup-start behavior (adopted by the deployments in a later step).

The middleware ordering is the load-bearing invariant: FastAPI runs ``http``
middleware in LIFO order, so the LAST registered is the OUTERMOST. The fixed
registration sequence here — optional cleanup → exception boundary →
``verify_token`` → tracing → metrics → optional replica-id header — reproduces
the per-builder execution order exactly
(``[replica-id] → metrics → tracing → verify_token → exception boundary →
[cleanup] → handler``), with metrics wrapping tracing and the replica-id header
wrapping the full response path when enabled.
"""
from __future__ import annotations

import traceback
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ray import serve
from typing import Any

from twinkle.protocol.types.errors import ErrorCategory
from twinkle.server.exceptions import TwinkleServerError
from twinkle.server.middleware.auth import verify_request_token
from twinkle.server.task_errors import build_error_payload
from twinkle.server.telemetry.http_middleware import create_metrics_middleware
from twinkle.server.telemetry.tracing import create_tracing_middleware
from twinkle.utils.logger import get_logger

logger = get_logger()

# Type aliases for the per-builder customization points.
RegisterRoutes = Callable[[FastAPI, Callable[[], Any]], None]
OnShutdown = Callable[[Any], Awaitable[None]]


async def twinkle_server_error_handler(request: Request, exc: TwinkleServerError) -> JSONResponse:
    """Map a TwinkleServerError to a structured response, fields at the top level.

    Status code is the exception's ``error_code``; the body is an ``ErrorPayload``
    (``error`` / ``category`` / ``error_code`` / ``request_id``) placed at the top
    level rather than nested under ``detail``. A Decision_Boundary-left rejection
    (``category=user``) carries no traceback.
    """
    request_id = getattr(request.state, 'request_id', None) or ''
    payload = build_error_payload(
        str(exc) or exc.__class__.__name__,
        category=exc.category,
        error_code=exc.error_code,
        request_id=request_id,
    )
    return JSONResponse(status_code=exc.error_code, content=payload.model_dump(mode='json', exclude_none=True))


# A body can produce hundreds of errors (one per element of a mis-typed tensor), and a
# response listing all of them helps nobody while costing bandwidth on every retry.
_MAX_VALIDATION_DETAILS = 20


def _validation_detail(error: dict[str, Any]) -> dict[str, Any]:
    """One pydantic error as a JSON-safe detail entry."""
    location = [str(part) for part in error.get('loc', ())]
    return {
        'field': location[-1] if location else '',
        'path': '.'.join(location),
        'type': error.get('type', ''),
        'message': error.get('msg', ''),
    }


def _validation_summary(errors: list[dict[str, Any]]) -> str:
    fields = []
    for error in errors:
        path = '.'.join(str(part) for part in error.get('loc', ()))
        if path and path not in fields:
            fields.append(path)
    shown = ', '.join(fields[:_MAX_VALIDATION_DETAILS]) or 'request body'
    suffix = '' if len(fields) <= _MAX_VALIDATION_DETAILS else f' (+{len(fields) - _MAX_VALIDATION_DETAILS} more)'
    return f'Request body validation failed for: {shown}{suffix}'


def _validation_mentions_unknown_field(errors: list[dict[str, Any]]) -> bool:
    return any(error.get('type') == 'extra_forbidden' for error in errors)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Map a body validation failure to a 422 carrying an ``ErrorPayload``.

    Lives next to ``twinkle_server_error_handler`` because both are the same concern --
    the wire shape of a failure -- and both are registered by ``build_deployment_app``
    for all four deployments. It used to sit in ``validation/``, whose ``__init__``
    docstring admitted the fit was awkward ("another half of the story").

    FastAPI's default handler answers with ``{"detail": [...]}``, a second error shape
    on the wire; this makes the Model, Sampler and Processor deployments answer
    identically to every other twinkle failure. The per-field ``details`` name the
    offending field, its path, and why it was rejected; there is no traceback because a
    rejected body is the caller's problem, not a crash.
    """
    errors = list(exc.errors())
    message = _validation_summary(errors)
    if _validation_mentions_unknown_field(errors):
        # An unknown top-level field is what an older client looks like against a newer
        # server, so say so instead of leaving the caller to infer it from a field list.
        message += ('. Unknown fields are rejected; if this worked before, upgrade '
                    'twinkle-kit on the client to match the server version.')
    payload = build_error_payload(
        message,
        category=ErrorCategory.User,
        error_code=422,
        request_id=getattr(request.state, 'request_id', None) or '',
        details=[_validation_detail(error) for error in errors[:_MAX_VALIDATION_DETAILS]],
    )
    return JSONResponse(status_code=422, content=payload.model_dump(mode='json', exclude_none=True))


def get_servable() -> Any:
    """The single definition of the servable-object accessor used by every builder.

    Ray Serve binds ``servable_object`` AFTER FastAPI lifespan startup, so this
    returns ``None`` during lifespan startup and the live instance once a
    request is being served.
    """
    return serve.get_replica_context().servable_object


def build_deployment_app(
    component: str,
    register_routes: RegisterRoutes,
    *,
    fastapi_kwargs: dict[str, Any] | None = None,
    on_shutdown: OnShutdown | None = None,
    attach_cleanup_middleware: bool = False,
    attach_replica_id_header: bool = False,
) -> FastAPI:
    """Build the FastAPI app with the canonical lifespan + middleware stack + routes.

    Reproduces, in one place, the construction every builder repeats:

    1. lifespan startup → ``ensure_telemetry_initialized()``;
       shutdown → ``on_shutdown(get_servable())`` (best-effort) then
       ``flush_telemetry_safely()`` so buffered OTLP batches flush on graceful
       replica termination;
    2. the ``TwinkleServerError`` and ``RequestValidationError`` handlers, so a
       rejected request body carries the same ``ErrorPayload`` shape as any other
       failure;
    3. [if ``attach_cleanup_middleware``] the gateway-only lazy-cleanup
       middleware (registered first ⇒ innermost), since the Gateway has no
       per-handler hook;
    4. ``catch_unhandled_exceptions`` middleware, inside auth/tracing/metrics
       and outside cleanup/routes;
    5. ``verify_token`` middleware;
    6. ``create_tracing_middleware(component)``;
    7. ``create_metrics_middleware(component)``;
    8. [if ``attach_replica_id_header``] replica-id response header middleware
       (registered last ⇒ outermost);
    9. ``register_routes(app, get_servable)``.

    Args:
        component: ``'Gateway' | 'Model' | 'Sampler' | 'Processor'`` — used as
            the tracing/metrics component label.
        register_routes: Callback that registers the deployment's routes on the
            app, given ``(app, get_servable)``.
        fastapi_kwargs: Extra kwargs forwarded to ``FastAPI(...)`` (Sampler passes
            ``title``/``description``/``version``).
        on_shutdown: Optional async teardown given the servable instance, run on
            lifespan shutdown (Gateway: ``proxy.close``; Model: ``shutdown``).
        attach_cleanup_middleware: Gateway-only — install the lazy-cleanup
            middleware because the Gateway has no per-handler request hook.
        attach_replica_id_header: Add ``X-Twinkle-Replica-Id`` to every
            response (Model + Sampler deployments).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize telemetry in the worker process (after deserialization).
        from twinkle.server.telemetry.worker_init import ensure_telemetry_initialized
        ensure_telemetry_initialized()
        # NOTE: ``state.start_cleanup_task()`` cannot run here — Ray Serve binds
        # ``servable_object`` AFTER lifespan startup. It is lazy-started from the
        # first request (see ``LazyCleanupMixin._ensure_state_cleanup_started``).
        yield
        if on_shutdown is not None:
            try:
                await on_shutdown(get_servable())
            except Exception:
                pass
        # Flush buffered OTLP batches on graceful replica termination.
        import asyncio

        from twinkle.server.telemetry import flush_telemetry_safely
        await asyncio.to_thread(flush_telemetry_safely)

    app = FastAPI(lifespan=lifespan, **(fastapi_kwargs or {}))

    app.add_exception_handler(TwinkleServerError, twinkle_server_error_handler)
    # Request-body validation failures answer with the same ``ErrorPayload`` shape as
    # every other error, registered here so all deployments behave identically rather
    # than each app keeping (or forgetting) its own copy.
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    # Registration order matters: FastAPI runs middleware LIFO, so the LAST
    # registered wraps the outermost layer. Register cleanup (if any) first so
    # it stays innermost, then the exception boundary, auth, tracing, metrics,
    # and finally the optional replica-id header as the outermost layer.
    if attach_cleanup_middleware:

        @app.middleware('http')
        async def ensure_state_cleanup_started(request: Request, call_next):
            # ``LazyCleanupMixin._ensure_state_cleanup_started`` already
            # warn-and-swallows backend errors itself, so no outer try is
            # needed here — anything escaping it would be a programming bug
            # worth surfacing.
            await get_servable()._ensure_state_cleanup_started()
            return await call_next(request)

    @app.middleware('http')
    async def catch_unhandled_exceptions(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(tb)
            # Unify the last-resort 500 with the rest of the wire: an
            # ``ErrorPayload`` body (Server category keeps the traceback) instead
            # of the legacy ``{'detail': <traceback>}`` shape.
            request_id = getattr(request.state, 'request_id', None) or ''
            payload = build_error_payload(
                str(exc) or exc.__class__.__name__,
                category=ErrorCategory.Server,
                error_code=500,
                request_id=request_id,
                traceback_text=tb,
            )
            return JSONResponse(status_code=500, content=payload.model_dump(mode='json', exclude_none=True))

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    app.middleware('http')(create_tracing_middleware(component))
    app.middleware('http')(create_metrics_middleware(component))

    if attach_replica_id_header:

        @app.middleware('http')
        async def inject_replica_id(request: Request, call_next):
            response = await call_next(request)
            try:
                ctx = serve.get_replica_context()
                response.headers['X-Twinkle-Replica-Id'] = ctx.replica_id.unique_id
            except Exception:
                pass
            return response

    register_routes(app, get_servable)
    return app


def bind_deployment(
        app: FastAPI,
        servable_cls: type,
        deploy_options: dict[str, Any],
        *,
        deployment_name: str,
        request_router_config: Any | None = None,
        bind_args: tuple = (),
        bind_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Run ``serve.ingress(app)(cls)`` → ``serve.deployment(...)`` → ``.options().bind()``.

    Args:
        app: The FastAPI app to mount as the deployment ingress.
        servable_cls: The deployment class.
        deploy_options: Ray Serve ``.options(...)`` kwargs.
        deployment_name: Name passed to ``serve.deployment(name=...)``.
        request_router_config: Optional ``RequestRouterConfig`` (Model only).
        bind_args: Positional args forwarded to ``.bind(...)``.
        bind_kwargs: Keyword args forwarded to ``.bind(...)``.
    """
    ingress_cls = serve.ingress(app)(servable_cls)
    deployment_kwargs: dict[str, Any] = {'name': deployment_name}
    if request_router_config is not None:
        deployment_kwargs['request_router_config'] = request_router_config
    deployment_cls = serve.deployment(**deployment_kwargs)(ingress_cls)
    return deployment_cls.options(**deploy_options).bind(*bind_args, **(bind_kwargs or {}))


class LazyCleanupMixin:
    """Single source of the lazy first-request ServerState cleanup-start behavior.

    Mixed into all four deployment classes so the four copy-pasted
    ``_ensure_state_cleanup_started`` methods collapse into one. The method name
    is preserved so existing call sites (``_on_request_start``,
    ``_ensure_sticky``, the Gateway cleanup middleware) are unchanged.

    ``_state_cleanup_started`` is declared here with a class-level default so the four
    deployment classes need no ``__init__`` change and the ``getattr`` fallback can go:
    only ``GatewayServer`` used to initialise it explicitly, and Model/Sampler/Processor
    relied on ``getattr``'s default. The first write of ``True`` shadows the
    class attribute on the instance -- expected, since the class attribute is only a
    default.
    """

    _state_cleanup_started: bool = False

    async def _ensure_state_cleanup_started(self) -> None:
        if self._state_cleanup_started:
            return
        try:
            # Idempotent via the ResourceCleanupCoordinator's own start guard.
            await self.state.start_cleanup_task()
        except Exception as e:
            logger.warning(f'Failed to start ServerState cleanup task: {e}')
        self._state_cleanup_started = True
