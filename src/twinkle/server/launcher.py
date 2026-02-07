# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified Server Launcher for Twinkle.

This module provides a unified way to launch both tinker and twinkle servers
with support for YAML config files, Python dict config, and CLI.

Usage:
    # From YAML config
    from twinkle.server import launch_server
    launch_server(config_path="server_config.yaml")

    # From Python dict
    launch_server(config={
        "server_type": "tinker",
        "http_options": {"host": "0.0.0.0", "port": 8000},
        "applications": [...]
    })

    # CLI
    python -m twinkle.server --config server_config.yaml
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from twinkle import get_logger

logger = get_logger()


class ServerLauncher:
    """
    Unified server launcher for tinker and twinkle servers.

    This class handles Ray/Serve initialization and application deployment
    for both tinker and twinkle server types.

    Attributes:
        server_type: The type of server ('tinker' or 'twinkle')
        config: The server configuration dictionary
        ray_namespace: The Ray namespace for the cluster
    """

    # Mapping of simplified import_path names to actual builder functions
    # These will be populated lazily to avoid circular imports
    _TINKER_BUILDERS: Dict[str, str] = {
        'server': 'build_server_app',
        'model': 'build_model_app',
        'sampler': 'build_sampler_app',
    }

    _TWINKLE_BUILDERS: Dict[str, str] = {
        'server': 'build_server_app',
        'model': 'build_model_app',
        'sampler': 'build_sampler_app',
        'processor': 'build_processor_app',
    }

    def __init__(
        self,
        server_type: str = "twinkle",
        config: Optional[Dict[str, Any]] = None,
        ray_namespace: Optional[str] = None,
    ):
        """
        Initialize the server launcher.

        Args:
            server_type: Server type ('tinker' or 'twinkle')
            config: Configuration dictionary
            ray_namespace: Ray namespace (default: 'twinkle_cluster' for tinker, None for twinkle)
        """
        if server_type not in ("tinker", "twinkle"):
            raise ValueError(
                f"server_type must be 'tinker' or 'twinkle', got '{server_type}'")

        self.server_type = server_type
        self.config = config or {}
        self.ray_namespace = ray_namespace
        self._builders: Dict[str, Callable] = {}
        self._ray_initialized = False
        self._serve_started = False

    def _get_builders(self) -> Dict[str, Callable]:
        """
        Get the appropriate builder functions for the server type.

        Returns:
            Dictionary mapping import_path names to builder functions
        """
        if self._builders:
            return self._builders

        if self.server_type == "tinker":
            from twinkle.server.tinker import (
                build_model_app,
                build_sampler_app,
                build_server_app,
            )
            self._builders = {
                'build_server_app': build_server_app,
                'build_model_app': build_model_app,
                'build_sampler_app': build_sampler_app,
            }
        else:  # twinkle
            from twinkle.server import (
                build_model_app,
                build_processor_app,
                build_sampler_app,
                build_server_app,
            )
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
            import_path: The import path from config (e.g., 'server', 'main:build_server_app')

        Returns:
            The builder function

        Raises:
            ValueError: If the import_path cannot be resolved
        """
        builders = self._get_builders()
        builder_map = self._TINKER_BUILDERS if self.server_type == "tinker" else self._TWINKLE_BUILDERS

        # Try to resolve through the mapping
        if import_path in builder_map:
            builder_name = builder_map[import_path]
            if builder_name in builders:
                return builders[builder_name]

        # Direct builder name
        if import_path in builders:
            return builders[import_path]

        raise ValueError(
            f"Unknown import_path '{import_path}' for server_type '{self.server_type}'. "
            f"Available: {list(builder_map.keys())}"
        )

    def _init_ray(self) -> None:
        """Initialize Ray if not already initialized."""
        if self._ray_initialized:
            return

        import ray

        # Determine namespace
        namespace = self.ray_namespace
        if namespace is None:
            namespace = self.config.get('ray_namespace')
        if namespace is None and self.server_type == "tinker":
            namespace = "twinkle_cluster"

        init_kwargs = {}
        if namespace:
            init_kwargs['namespace'] = namespace

        if not ray.is_initialized():
            ray.init(**init_kwargs)
            logger.info(f"Ray initialized with namespace={namespace}")

        self._ray_initialized = True

    def _start_serve(self) -> None:
        """Start Ray Serve with http_options from config."""
        if self._serve_started:
            return

        from ray import serve

        # Shutdown any existing serve instance
        try:
            serve.shutdown()
            time.sleep(2)  # Wait for cleanup
        except Exception:
            pass

        # Get http_options from config
        http_options = self.config.get('http_options', {})
        if isinstance(http_options, dict):
            http_options = dict(http_options)
        else:
            # Handle OmegaConf or other config objects
            http_options = dict(http_options) if http_options else {}

        serve.start(http_options=http_options)
        logger.info(f"Ray Serve started with http_options={http_options}")

        self._serve_started = True

    def _deploy_application(self, app_config: Dict[str, Any]) -> None:
        """
        Deploy a single application.

        Args:
            app_config: Application configuration dictionary
        """
        from ray import serve

        name = app_config.get('name', 'app')
        route_prefix = app_config.get('route_prefix', '/')
        import_path = app_config.get('import_path', 'server')
        args = app_config.get('args', {}) or {}
        deployments = app_config.get('deployments', [])

        logger.info(f"Starting {name} at {route_prefix}...")

        # Resolve builder function
        builder = self._resolve_builder(import_path)

        # Build deploy_options from deployments config
        deploy_options = {}
        if deployments:
            deploy_config = deployments[0]
            if isinstance(deploy_config, dict):
                # Copy all deployment options from the config, except 'name'.
                deploy_options = {k: v for k, v in deploy_config.items() if k != 'name'}

        
        # Build and deploy the application
        app = builder(
            deploy_options=deploy_options,
            **{k: v for k, v in args.items()}
        )

        serve.run(app, name=name, route_prefix=route_prefix)
        logger.info(f"Deployed {name} at {route_prefix}")

    def launch(self, wait: bool = True) -> None:
        """
        Launch the server with all configured applications.

        Args:
            wait: If True, block and wait for Enter to stop the server
        """
        self._init_ray()
        self._start_serve()

        applications = self.config.get('applications', [])
        if not applications:
            logger.warning("No applications configured")
            return

        # Deploy each application
        for app_config in applications:
            if isinstance(app_config, dict):
                self._deploy_application(app_config)
            else:
                # Handle OmegaConf or other config objects
                self._deploy_application(dict(app_config))

        # Print endpoints
        http_options = self.config.get('http_options', {})
        host = http_options.get('host', 'localhost')
        port = http_options.get('port', 8000)

        print("\nAll applications started!")
        print("Endpoints:")
        for app_config in applications:
            route_prefix = app_config.get(
                'route_prefix', '/') if isinstance(app_config, dict) else app_config.route_prefix
            print(f"  - http://{host}:{port}{route_prefix}")

        if wait:
            input("\nPress Enter to stop the server...")

    @classmethod
    def from_yaml(
        cls,
        config_path: Union[str, Path],
        server_type: str = "twinkle",
        ray_namespace: Optional[str] = None,
    ) -> "ServerLauncher":
        """
        Create a ServerLauncher from a YAML config file.

        Args:
            config_path: Path to the YAML config file
            server_type: Server type ('tinker' or 'twinkle'), default is 'twinkle'
            ray_namespace: Override Ray namespace from config

        Returns:
            Configured ServerLauncher instance
        """
        from omegaconf import OmegaConf

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = OmegaConf.load(config_path)
        config_dict = OmegaConf.to_container(config, resolve=True)

        # Override server_type from config if specified
        if 'server_type' in config_dict:
            server_type = config_dict['server_type']

        return cls(
            server_type=server_type,
            config=config_dict,
            ray_namespace=ray_namespace or config_dict.get('ray_namespace'),
        )


def launch_server(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    server_type: str = "twinkle",
    ray_namespace: Optional[str] = None,
    wait: bool = True,
) -> ServerLauncher:
    """
    Launch a twinkle server with flexible configuration options.

    This is the main entry point for launching servers programmatically.

    Args:
        config: Configuration dictionary (takes precedence over config_path)
        config_path: Path to YAML config file
        server_type: Server type ('tinker' or 'twinkle'), default is 'twinkle'
        ray_namespace: Ray namespace
        wait: If True, block and wait for Enter to stop the server

    Returns:
        The ServerLauncher instance

    Raises:
        ValueError: If neither config nor config_path is provided

    Examples:
        # From YAML config (twinkle mode)
        launch_server(config_path="server_config.yaml")

        # From YAML config (tinker mode)
        launch_server(config_path="server_config.yaml", server_type="tinker")

        # From Python dict
        launch_server(config={
            "server_type": "tinker",
            "http_options": {"host": "0.0.0.0", "port": 8000},
            "applications": [...]
        })
    """
    if config is None and config_path is None:
        raise ValueError("Either 'config' or 'config_path' must be provided")

    launcher: ServerLauncher

    if config is not None:
        # From Python dict config - override with config's server_type if specified
        final_server_type = config.get('server_type', server_type)
        launcher = ServerLauncher(
            server_type=final_server_type,
            config=config,
            ray_namespace=ray_namespace or config.get('ray_namespace'),
        )
    else:
        # From YAML config file
        launcher = ServerLauncher.from_yaml(
            config_path=config_path,
            server_type=server_type,
            ray_namespace=ray_namespace,
        )

    launcher.launch(wait=wait)
    return launcher
