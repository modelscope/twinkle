# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified Server Launcher for Twinkle.

This package provides a unified way to launch the server from a YAML config
file or a typed ``ServerConfig``. It is split into focused submodules:

- ``server_launcher`` — the ``ServerLauncher`` class and the ``launch_server``
  entry point (the real launcher logic);
- ``builder_registry`` — ``import_path`` → deployment-builder resolution;
- ``env_propagation`` — telemetry / persistence env-var collection for workers.

This ``__init__`` is a thin aggregator: it only re-exports the public surface so
the dotted paths ``twinkle.server.launcher.ServerLauncher`` / ``launch_server``
keep working (no shim — this is allowed package-``__init__`` aggregation).

Usage:
    # From YAML config
    from twinkle.server import launch_server
    launch_server(config_path="server_config.yaml")

    # From a typed ServerConfig
    launch_server(config=ServerConfig(...))

    # CLI
    python -m twinkle.server launch --config server_config.yaml
"""
from __future__ import annotations

from .server_launcher import ServerLauncher, launch_server

__all__ = ['ServerLauncher', 'launch_server']
