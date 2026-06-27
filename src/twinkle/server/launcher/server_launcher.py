# Copyright (c) ModelScope Contributors. All rights reserved.
"""``ServerLauncher`` and the ``launch_server`` entry point.

The real launcher module. It lives here (rather than in the package
``__init__.py``) so the package's ``__init__`` stays a thin aggregator; the
public dotted paths ``twinkle.server.launcher.ServerLauncher`` /
``launch_server`` are preserved by the ``__init__`` re-export.
"""
from __future__ import annotations

import json
import os
import signal
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from twinkle import get_logger
from twinkle.hub.model_alias import MODEL_ID_ALIASES_ENV, build_model_alias_map
from twinkle.server.config import ServerConfig
from twinkle.server.config.application_spec import ApplicationSpec
from twinkle.server.utils.ray_serve_patch import apply_ray_serve_patches, get_runtime_env_for_patches
from .builder_registry import get_builders, resolve_builder
from .env_propagation import build_propagated_env_vars

logger = get_logger()


class ServerLauncher:
    """
    Unified server launcher.

    This class handles Ray/Serve initialization and application deployment.

    Attributes:
        config: The server configuration dictionary
        ray_namespace: The Ray namespace for the cluster
    """

    def __init__(
        self,
        config: ServerConfig,
        ray_namespace: str | None = None,
    ):
        """
        Initialize the server launcher.

        Args:
            config: A validated :class:`ServerConfig` instance. Raw dicts are
                rejected — operators must build ``ServerConfig`` via
                ``ServerConfig.from_yaml`` or its constructor so cross-field
                validation runs before the launcher consumes anything.
            ray_namespace: Ray namespace (default: 'twinkle_cluster')
        """
        if not isinstance(config, ServerConfig):
            raise TypeError('ServerLauncher requires a typed ServerConfig instance; '
                            f'got {type(config).__name__}. Build one with '
                            'ServerConfig.from_yaml(path) or ServerConfig(...).')
        self.config: ServerConfig = config
        self.ray_namespace = ray_namespace
        self._builders: dict[str, Callable] = {}
        self._ray_initialized = False
        self._serve_started = False

    def _get_builders(self) -> dict[str, Callable]:
        """Get (and cache) the builder functions for all app types."""
        if not self._builders:
            self._builders = get_builders()
        return self._builders

    def _resolve_builder(self, import_path: str) -> Callable:
        """Resolve an import_path to a builder function.

        Raises:
            ValueError: If the import_path cannot be resolved.
        """
        return resolve_builder(import_path, self._get_builders())

    def _init_ray(self) -> None:
        """Initialize Ray if not already initialized."""
        if self._ray_initialized:
            return

        import ray

        namespace = self.ray_namespace or self.config.ray_namespace or 'twinkle_cluster'

        if not ray.is_initialized():
            # Use runtime_env to apply patches in worker processes
            # This is required because Ray Serve's ProxyActor runs in separate processes
            runtime_env = get_runtime_env_for_patches()
            # Propagate telemetry + persistence env vars to all Ray workers
            propagated_env_vars = build_propagated_env_vars()
            if propagated_env_vars:
                merged_env_vars = dict(runtime_env.get('env_vars') or {})
                merged_env_vars.update(propagated_env_vars)
                runtime_env['env_vars'] = merged_env_vars
            # Connect to existing cluster if available, otherwise start local instance
            ray.init(
                address='auto',
                namespace=namespace,
                runtime_env=runtime_env,
            )
            logger.info(f'Ray initialized with namespace={namespace}')

        self._ray_initialized = True

    def _start_serve(self) -> None:
        """Start Ray Serve with http_options from config."""
        if self._serve_started:
            return

        from ray import serve

        try:
            from ray.serve.context import _get_global_client
            _get_global_client()
            # Serve is running, shut it down before re-starting
            serve.shutdown()
        except Exception:
            # Serve not running — nothing to shut down
            pass

        http_options = self.config.http_options.model_dump()
        serve_kwargs: dict[str, Any] = {'http_options': http_options}
        # ``proxy_location`` controls where the Ray Serve HTTP proxy runs
        # (``EveryNode`` / ``HeadOnly`` / ``Disabled``). The example configs
        # set this field, so honour it here instead of silently ignoring.
        if self.config.proxy_location:
            serve_kwargs['proxy_location'] = self.config.proxy_location
        serve.start(**serve_kwargs)
        logger.info(f'Ray Serve started with http_options={http_options}, '
                    f'proxy_location={self.config.proxy_location!r}')

        self._serve_started = True

    def _deploy_application(self, app_spec: ApplicationSpec) -> None:
        """Deploy a single application.

        Args:
            app_spec: Validated :class:`ApplicationSpec` from the typed config.
        """
        from ray import serve

        name = app_spec.name
        route_prefix = app_spec.route_prefix
        import_path = app_spec.import_path
        # Shallow-dump the typed args to a kwargs dict WITHOUT recursing into
        # nested models: ``dict(model)`` yields top-level (field, value) pairs
        # with nested models left as instances, so ``queue_config`` stays a
        # typed ``TaskQueueConfig`` and is not re-serialized to a dict only to be
        # revived via ``from_dict`` inside the builder (the prior triple
        # validation). ``None`` values are dropped to match ``exclude_none``.
        args = {k: v for k, v in dict(app_spec.args).items() if v is not None}
        deployments = list(app_spec.deployments or [])

        logger.info(f'Starting {name} at {route_prefix}...')

        builder = self._resolve_builder(import_path)

        deploy_options = {}
        if deployments:
            if len(deployments) > 1:
                logger.warning(f'Application "{name}" has {len(deployments)} deployments configured, '
                               f'but only the first deployment will be used.')
            deploy_config = deployments[0]
            if isinstance(deploy_config, dict):
                deploy_options = {k: v for k, v in deploy_config.items() if k != 'name'}

        # Inject telemetry + persistence env vars into the deployment's
        # runtime_env so that Ray Serve replicas (worker processes) can
        # initialize telemetry and resolve the configured persistence backend
        # regardless of deployment startup order.
        # User-specified env_vars take precedence over our defaults.
        propagated_env_vars = build_propagated_env_vars()
        if propagated_env_vars:
            ray_actor_options = dict(deploy_options.get('ray_actor_options') or {})
            runtime_env = dict(ray_actor_options.get('runtime_env') or {})
            env_vars = dict(runtime_env.get('env_vars') or {})
            for k, v in propagated_env_vars.items():
                env_vars.setdefault(k, v)
            runtime_env['env_vars'] = env_vars
            ray_actor_options['runtime_env'] = runtime_env
            deploy_options['ray_actor_options'] = ray_actor_options

        # Pass http_options to server apps for internal proxy routing
        if import_path == 'server':
            args.setdefault('http_options', self.config.http_options.model_dump())

        app = builder(deploy_options=deploy_options, **args)

        serve.run(app, name=name, route_prefix=route_prefix)
        logger.info(f'Deployed {name} at {route_prefix}')

    def launch(self) -> None:
        """Launch the server with all configured applications.

        Blocks the calling thread to keep the server running. Installs signal
        handlers for SIGINT/SIGTERM so that ``serve.shutdown()`` is called on
        termination instead of leaving orphaned deployments.
        """
        # Apply Ray Serve patches before initializing Ray
        apply_ray_serve_patches()

        # Initialize telemetry if configured
        telemetry = self.config.telemetry
        if telemetry.enabled:
            from twinkle.server.telemetry import init_telemetry
            init_telemetry(telemetry)
            # Export config to env vars for Ray worker processes
            os.environ['TWINKLE_TELEMETRY_ENABLED'] = '1'
            os.environ['TWINKLE_TELEMETRY_DEBUG'] = '1' if telemetry.debug else '0'
            os.environ['TWINKLE_TELEMETRY_SERVICE'] = telemetry.service_name
            os.environ['TWINKLE_TELEMETRY_ENDPOINT'] = telemetry.otlp_endpoint
            os.environ['TWINKLE_TELEMETRY_INTERVAL'] = str(telemetry.export_interval_ms)

        # Export top-level persistence to env vars so any worker
        # (not just Gateway) can build the same backend on first call to
        # get_server_state().
        persistence = self.config.persistence
        for k, v in persistence.to_env_vars().items():
            os.environ[k] = v
        logger.info(f'Persistence backend configured: mode={persistence.mode}')

        model_alias_map = build_model_alias_map(self.config.applications)
        if model_alias_map:
            os.environ[MODEL_ID_ALIASES_ENV] = json.dumps(model_alias_map, ensure_ascii=False)
        else:
            os.environ.pop(MODEL_ID_ALIASES_ENV, None)

        self._init_ray()
        self._start_serve()

        applications = self.config.applications
        if not applications:
            logger.warning('No applications configured')
            return

        for app_spec in applications:
            self._deploy_application(app_spec)

        host = self.config.http_options.host
        port = self.config.http_options.port

        logger.info('All applications started!')
        for app_spec in applications:
            logger.info('  Endpoint: http://%s:%s%s', host, port, app_spec.route_prefix)

        # Graceful shutdown via signal handling
        shutdown_event = threading.Event()

        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f'Received {sig_name}, shutting down gracefully...')
            shutdown_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # Block until a termination signal is received
        shutdown_event.wait()

        self._shutdown()

    def _shutdown(self) -> None:
        """Tear down Serve and flush telemetry on graceful termination.

        Telemetry flush runs AFTER ``serve.shutdown()`` and is wrapped so a
        telemetry-shutdown failure cannot mask the user-facing shutdown path.
        """
        from ray import serve
        try:
            serve.shutdown()
            logger.info('Ray Serve shut down successfully')
        except Exception:
            logger.warning('Error during Ray Serve shutdown', exc_info=True)

        # Flush buffered OTLP batches (traces / metrics / logs) on the driver
        # after Serve is down. Errors are swallowed by the helper.
        from twinkle.server.telemetry import flush_telemetry_safely
        flush_telemetry_safely()


def launch_server(
    config: ServerConfig | None = None,
    config_path: str | Path | None = None,
    ray_namespace: str | None = None,
) -> None:
    """Launch a twinkle server — the single launch choke point.

    Exactly one of ``config`` (a :class:`ServerConfig` instance) or
    ``config_path`` (a YAML file) must be provided. When a path is given the
    config is loaded via :meth:`ServerConfig.from_yaml` here, so there is one
    construction path: load config → ``ServerLauncher(config=...)`` →
    ``launch()``. The call blocks until a SIGINT/SIGTERM signal is received.

    Raises:
        ValueError: neither ``config`` nor ``config_path`` was provided.
        TypeError: ``config`` is not a :class:`ServerConfig` instance — raw
            dicts are not accepted.

    Examples:
        launch_server(config_path="server_config.yaml")
        launch_server(config=ServerConfig(...))
    """
    if config is None and config_path is None:
        raise ValueError("Either 'config' or 'config_path' must be provided")

    if config is None:
        # ``FileNotFoundError`` / ``ConfigParseError`` / ``pydantic.ValidationError``
        # propagate so the caller can surface a precise message.
        config = ServerConfig.from_yaml(config_path)

    launcher = ServerLauncher(
        config=config,
        ray_namespace=ray_namespace or config.ray_namespace,
    )
    launcher.launch()
