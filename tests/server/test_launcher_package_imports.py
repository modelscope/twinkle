# Copyright (c) ModelScope Contributors. All rights reserved.
"""Import test for the same-named ``launcher/`` package (R13.2, R13.8).

The decomposition preserves the dotted paths ``twinkle.server.launcher.ServerLauncher``
and ``twinkle.server.ServerLauncher``/``launch_server`` (no shim), and exposes the
extracted ``env_propagation`` and ``builder_registry`` submodules.
"""
from __future__ import annotations

import importlib


def test_server_package_exports_unchanged() -> None:
    from twinkle.server import ServerLauncher, launch_server  # noqa: F401


def test_launcher_dotted_path_preserved() -> None:
    mod = importlib.import_module('twinkle.server.launcher')
    assert hasattr(mod, 'ServerLauncher')
    assert hasattr(mod, 'launch_server')


def test_launcher_submodules_resolve() -> None:
    sl = importlib.import_module('twinkle.server.launcher.server_launcher')
    env = importlib.import_module('twinkle.server.launcher.env_propagation')
    reg = importlib.import_module('twinkle.server.launcher.builder_registry')
    # The real launcher logic lives in server_launcher (not in __init__).
    assert hasattr(sl, 'ServerLauncher')
    assert hasattr(sl, 'launch_server')
    assert hasattr(env, 'build_propagated_env_vars')
    assert hasattr(env, 'TELEMETRY_ENV_KEYS')
    assert hasattr(reg, 'BUILDERS')
    assert hasattr(reg, 'resolve_builder')
    assert hasattr(reg, 'get_builders')


def test_launcher_init_is_thin_aggregator() -> None:
    # __init__ should only re-export, not define the launcher class itself.
    import twinkle.server.launcher as pkg
    assert pkg.ServerLauncher.__module__ == 'twinkle.server.launcher.server_launcher'


def test_operator_facing_server_literal_maps_to_gateway_builder() -> None:
    from twinkle.server.launcher.builder_registry import BUILDERS
    # The YAML literal 'server' is operator-facing and must keep selecting the
    # gateway builder even though the function was renamed.
    assert BUILDERS['server'] == 'build_gateway_app'
    assert set(BUILDERS) == {'server', 'model', 'sampler', 'processor'}


def test_cli_import_path_unchanged() -> None:
    # cli/app.py does `from twinkle.server.launcher import ServerLauncher`.
    from twinkle.server.launcher import ServerLauncher  # noqa: F401
