# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared deployment-application construction.

Top-level, central deployment-construction infrastructure shared by all four
deployments (Gateway, Model, Sampler, Processor). It is intentionally NOT under
``utils/`` — it is core to how every deployment is built, not a generic helper.

It consolidates, in one place, the construction logic the four App_Builders
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
from fastapi.responses import JSONResponse
from ray import serve
from typing import Any

from twinkle.server.telemetry.middleware import create_metrics_middleware
from twinkle.server.telemetry.tracing import create_tracing_middleware
from twinkle.server.utils.validation import verify_request_token
from twinkle.utils.logger import get_logger

logger = get_logger()

# Type aliases for the per-builder customization points.
RegisterRoutes = Callable[[FastAPI, Callable[[], Any]], None]
OnShutdown = Callable[[Any], Awaitable[None]]


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
    2. [if ``attach_cleanup_middleware``] the gateway-only lazy-cleanup
       middleware (registered first ⇒ innermost), since the Gateway has no
       per-handler hook;
    3. ``catch_unhandled_exceptions`` middleware, inside auth/tracing/metrics
       and outside cleanup/routes;
    4. ``verify_token`` middleware;
    5. ``create_tracing_middleware(component)``;
    6. ``create_metrics_middleware(component)``;
    7. [if ``attach_replica_id_header``] replica-id response header middleware
       (registered last ⇒ outermost);
    8. ``register_routes(app, get_servable)``.

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
        except Exception:
            error = traceback.format_exc()
            logger.error(error)
            return JSONResponse(status_code=500, content={'detail': error})

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


def init_twinkle_runtime(
    is_mock: bool,
    nproc_per_node: int,
    device_group: Any,
    device_mesh_dict: dict[str, Any],
) -> Any | None:
    """Initialize the Twinkle distributed runtime and build a DeviceMesh.

    Shared by ModelManagement and SamplerManagement ``__init__``.
    Returns ``None`` for mock backends (CPU-only, no device mesh).
    """
    import twinkle
    from twinkle import DeviceMesh

    if is_mock:
        twinkle.initialize(
            mode='ray', nproc_per_node=nproc_per_node, ncpu_proc_per_node=1, groups=[device_group], lazy_collect=False)
        return None

    twinkle.initialize(mode='ray', nproc_per_node=nproc_per_node, groups=[device_group], lazy_collect=False)
    if 'mesh_dim_names' in device_mesh_dict:
        return DeviceMesh(**device_mesh_dict)
    return DeviceMesh.from_sizes(**device_mesh_dict)


class LazyCleanupMixin:
    """Single source of the lazy first-request ServerState cleanup-start behavior.

    Mixed into all four deployment classes so the four copy-pasted
    ``_ensure_state_cleanup_started`` methods collapse into one. The method name
    is preserved so existing call sites (``_on_request_start``,
    ``_ensure_sticky``, the Gateway cleanup middleware) are unchanged.
    """

    async def _ensure_state_cleanup_started(self) -> None:
        if getattr(self, '_state_cleanup_started', False):
            return
        try:
            # Idempotent via ServerState's own ``_cleanup_running`` guard.
            await self.state.start_cleanup_task()
        except Exception as e:
            logger.warning(f'Failed to start ServerState cleanup task: {e}')
        self._state_cleanup_started = True
