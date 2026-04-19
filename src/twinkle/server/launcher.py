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

import signal
import threading
from pathlib import Path
from typing import Any, Callable, Dict, NoReturn, Optional, Union

from twinkle import get_logger
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
        config: dict[str, Any] | None = None,
        ray_namespace: str | None = None,
    ):
        """
        Initialize the server launcher.

        Args:
            config: Configuration dictionary
            ray_namespace: Ray namespace (default: 'twinkle_cluster')
        """
        self.config = config or {}
        self.ray_namespace = ray_namespace
        self._builders: dict[str, Callable] = {}
        self._ray_initialized = False
        self._serve_started = False

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

        namespace = self.ray_namespace or self.config.get('ray_namespace') or 'twinkle_cluster'

        if not ray.is_initialized():
            # Use runtime_env to apply patches in worker processes
            # This is required because Ray Serve's ProxyActor runs in separate processes
            runtime_env = get_runtime_env_for_patches()
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

        http_options = self.config.get('http_options', {})
        if isinstance(http_options, dict):
            http_options = dict(http_options)
        else:
            http_options = dict(http_options) if http_options else {}

        serve.start(http_options=http_options)
        logger.info(f'Ray Serve started with http_options={http_options}')

        self._serve_started = True

    def _deploy_application(self, app_config: dict[str, Any]) -> None:
        """Deploy a single application.

        Args:
            app_config: Application configuration dictionary
        """
        from ray import serve

        name = app_config.get('name', 'app')
        route_prefix = app_config.get('route_prefix', '/')
        import_path = app_config.get('import_path', 'server')
        args = app_config.get('args', {}) or {}
        deployments = app_config.get('deployments', [])

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

        # Pass http_options to server apps for internal proxy routing
        http_options = self.config.get('http_options', {})
        if import_path == 'server' and http_options:
            args['http_options'] = http_options

        app = builder(deploy_options=deploy_options, **{k: v for k, v in args.items()})

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

        self._init_ray()
        self._start_serve()

        applications = self.config.get('applications', [])
        if not applications:
            logger.warning('No applications configured')
            return

        for app_config in applications:
            if isinstance(app_config, dict):
                self._deploy_application(app_config)
            else:
                self._deploy_application(dict(app_config))

        http_options = self.config.get('http_options', {})
        host = http_options.get('host', 'localhost')
        port = http_options.get('port', 8000)

        print('\nAll applications started!')
        print('Endpoints:')
        for app_config in applications:
            route_prefix = app_config.get('route_prefix', '/') if isinstance(app_config,
                                                                             dict) else app_config.route_prefix
            print(f'  - http://{host}:{port}{route_prefix}')

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
        """
        Create a ServerLauncher from a YAML config file.

        Args:
            config_path: Path to the YAML config file
            ray_namespace: Override Ray namespace from config

        Returns:
            Configured ServerLauncher instance
        """
        from omegaconf import OmegaConf

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')

        config = OmegaConf.load(config_path)
        config_dict = OmegaConf.to_container(config, resolve=True)

        return cls(
            config=config_dict,
            ray_namespace=ray_namespace or config_dict.get('ray_namespace'),
        )


def launch_server(
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    ray_namespace: str | None = None,
) -> None:
    """
    Launch a twinkle server with flexible configuration options.

    This is the main entry point for launching servers programmatically.
    The call blocks until a SIGINT/SIGTERM signal is received.

    Args:
        config: Configuration dictionary (takes precedence over config_path)
        config_path: Path to YAML config file
        ray_namespace: Ray namespace

    Raises:
        ValueError: If neither config nor config_path is provided

    Examples:
        # From YAML config
        launch_server(config_path="server_config.yaml")

        # From Python dict
        launch_server(config={
            "http_options": {"host": "0.0.0.0", "port": 8000},
            "applications": [...]
        })
    """
    if config is None and config_path is None:
        raise ValueError("Either 'config' or 'config_path' must be provided")

    if config is not None:
        launcher = ServerLauncher(
            config=config,
            ray_namespace=ray_namespace or config.get('ray_namespace'),
        )
    else:
        launcher = ServerLauncher.from_yaml(
            config_path=config_path,
            ray_namespace=ray_namespace,
        )

    launcher.launch()
