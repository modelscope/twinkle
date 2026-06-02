# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified Server Launcher for Twinkle.

This module provides a unified way to launch the server with support for
YAML config files, Python dict config, and CLI.

Usage:
    # From YAML config
    from twinkle.server import launch_server
    launch_server(config_path="server_config.yaml")

    # From Python dict
    launch_server(config={
        "http_options": {"host": "0.0.0.0", "port": 8000},
        "applications": [...]
    })

    # CLI
    python -m twinkle.server --config server_config.yaml
"""
from __future__ import annotations

import os
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Dict, NoReturn, Optional, Union

from twinkle import get_logger
from twinkle.server.config import ServerConfig
from twinkle.server.utils.ray_serve_patch import apply_ray_serve_patches, get_runtime_env_for_patches

logger = get_logger()


class ServerLauncher:
    """
    Unified server launcher.

    This class handles Ray/Serve initialization and application deployment.

    Attributes:
        config: The server configuration dictionary
        ray_namespace: The Ray namespace for the cluster
    """

    # Mapping of simplified import_path names to builder function names
    _BUILDERS: dict[str, str] = {
        'server': 'build_server_app',
        'model': 'build_model_app',
        'sampler': 'build_sampler_app',
        'processor': 'build_processor_app',
    }

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
                validation runs before the launcher consumes anything (R6.6).
            ray_namespace: Ray namespace (default: 'twinkle_cluster')
        """
        if not isinstance(config, ServerConfig):
            raise TypeError(
                'ServerLauncher requires a typed ServerConfig instance; '
                f'got {type(config).__name__}. Build one with '
                'ServerConfig.from_yaml(path) or ServerConfig(...).'
            )
        self.config: ServerConfig = config
        self.ray_namespace = ray_namespace
        self._builders: dict[str, Callable] = {}
        self._ray_initialized = False
        self._serve_started = False

    # Telemetry env var keys that need to be propagated to Ray worker processes
    _TELEMETRY_ENV_KEYS: tuple[str, ...] = (
        'TWINKLE_TELEMETRY_ENABLED',
        'TWINKLE_TELEMETRY_DEBUG',
        'TWINKLE_TELEMETRY_SERVICE',
        'TWINKLE_TELEMETRY_ENDPOINT',
        'TWINKLE_TELEMETRY_INTERVAL',
    )

    def _build_telemetry_env_vars(self) -> dict[str, str]:
        """Collect telemetry env vars from os.environ for propagation to Ray workers.

        These vars are read by ``ensure_telemetry_initialized()`` inside the
        FastAPI startup hook running in each worker process.
        """
        return {k: os.environ[k] for k in self._TELEMETRY_ENV_KEYS if k in os.environ}

    def _build_persistence_env_vars(self) -> dict[str, str]:
        """Collect persistence env vars from os.environ for propagation to Ray workers.

        These vars are read by ``PersistenceConfig.from_env()`` inside any
        worker that calls ``get_server_state()`` without an explicit config,
        which makes the chosen backend independent of deployment startup order.
        """
        from twinkle.server.state.backend.factory import PERSISTENCE_ENV_KEYS
        return {k: os.environ[k] for k in PERSISTENCE_ENV_KEYS if k in os.environ}

    def _build_propagated_env_vars(self) -> dict[str, str]:
        """Aggregate all env vars that must reach Ray worker processes."""
        merged: dict[str, str] = {}
        merged.update(self._build_telemetry_env_vars())
        merged.update(self._build_persistence_env_vars())
        return merged

    def _get_builders(self) -> dict[str, Callable]:
        """Get the builder functions for all app types."""
        if self._builders:
            return self._builders

        from twinkle.server.gateway import build_server_app
        from twinkle.server.model import build_model_app
        from twinkle.server.processor import build_processor_app
        from twinkle.server.sampler import build_sampler_app

        self._builders = {
            'build_server_app': build_server_app,
            'build_model_app': build_model_app,
            'build_sampler_app': build_sampler_app,
            'build_processor_app': build_processor_app,
        }

        return self._builders

    def _resolve_builder(self, import_path: str) -> Callable:
        """
        Resolve an import_path to a builder function.

        Args:
            import_path: The import path from config (e.g., 'server', 'model')

        Returns:
            The builder function

        Raises:
            ValueError: If the import_path cannot be resolved
        """
        builders = self._get_builders()

        # Try to resolve through the simplified name mapping
        if import_path in self._BUILDERS:
            builder_name = self._BUILDERS[import_path]
            if builder_name in builders:
                return builders[builder_name]

        # Direct builder name
        if import_path in builders:
            return builders[import_path]

        raise ValueError(f"Unknown import_path '{import_path}'. "
                         f'Available: {list(self._BUILDERS.keys())}')

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
            propagated_env_vars = self._build_propagated_env_vars()
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
        serve.start(http_options=http_options)
        logger.info(f'Ray Serve started with http_options={http_options}')

        self._serve_started = True

    def _deploy_application(self, app_spec: 'ApplicationSpec') -> None:
        """Deploy a single application.

        Args:
            app_spec: Validated :class:`ApplicationSpec` from the typed config.
        """
        from ray import serve

        name = app_spec.name
        route_prefix = app_spec.route_prefix
        import_path = app_spec.import_path
        # Re-serialize the typed args back to a kwargs dict for the builder.
        # Using ``mode='python'`` keeps nested Pydantic models as dicts (which
        # the legacy builders expect) without losing field-level validation.
        args = app_spec.args.model_dump(mode='python', exclude_none=True)
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
        propagated_env_vars = self._build_propagated_env_vars()
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

        print('\nAll applications started!')
        print('Endpoints:')
        for app_spec in applications:
            print(f'  - http://{host}:{port}{app_spec.route_prefix}')

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

        from ray import serve
        try:
            serve.shutdown()
            logger.info('Ray Serve shut down successfully')
        except Exception:
            logger.warning('Error during Ray Serve shutdown', exc_info=True)

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        ray_namespace: str | None = None,
    ) -> ServerLauncher:
        """Build a ``ServerLauncher`` from a YAML config file.

        Thin wrapper over :meth:`ServerConfig.from_yaml`. ``FileNotFoundError``
        / ``ConfigParseError`` / ``pydantic.ValidationError`` propagate so the
        caller can surface a precise message before the launcher is constructed.
        """
        config = ServerConfig.from_yaml(config_path)
        return cls(
            config=config,
            ray_namespace=ray_namespace or config.ray_namespace,
        )


def launch_server(
    config: ServerConfig | None = None,
    config_path: str | Path | None = None,
    ray_namespace: str | None = None,
) -> None:
    """Launch a twinkle server.

    Exactly one of ``config`` (a :class:`ServerConfig` instance) or
    ``config_path`` (a YAML file) must be provided. The call blocks until a
    SIGINT/SIGTERM signal is received.

    Raises:
        ValueError: neither ``config`` nor ``config_path`` was provided.
        TypeError: ``config`` is not a :class:`ServerConfig` instance — raw
            dicts are no longer accepted (R6.6).

    Examples:
        launch_server(config_path="server_config.yaml")
        launch_server(config=ServerConfig(...))
    """
    if config is None and config_path is None:
        raise ValueError("Either 'config' or 'config_path' must be provided")

    if config is not None:
        launcher = ServerLauncher(
            config=config,
            ray_namespace=ray_namespace or config.ray_namespace,
        )
    else:
        launcher = ServerLauncher.from_yaml(
            config_path=config_path,
            ray_namespace=ray_namespace,
        )

    launcher.launch()
