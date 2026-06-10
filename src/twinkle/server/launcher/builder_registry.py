# Copyright (c) ModelScope Contributors. All rights reserved.
"""``import_path`` → deployment-builder resolution.

Extracted from the former single-file ``launcher.py`` (TIER 3 same-named-package
decomposition). No logic change.

The operator-facing YAML ``import_path`` literals (``"server"``, ``"model"``,
``"sampler"``, ``"processor"``) are unchanged; only the internal builder
function the ``"server"`` literal resolves to was renamed
(``build_server_app`` → ``build_gateway_app``).
"""
from __future__ import annotations

from collections.abc import Callable

# Mapping of operator-facing import_path literals to builder function names.
BUILDERS: dict[str, str] = {
    'server': 'build_gateway_app',
    'model': 'build_model_app',
    'sampler': 'build_sampler_app',
    'processor': 'build_processor_app',
}


def get_builders() -> dict[str, Callable]:
    """Import and return the deployment builder functions by name.

    Imported lazily so that importing the launcher package does not eagerly
    pull in every deployment module.
    """
    from twinkle.server.gateway import build_gateway_app
    from twinkle.server.model import build_model_app
    from twinkle.server.processor import build_processor_app
    from twinkle.server.sampler import build_sampler_app

    return {
        'build_gateway_app': build_gateway_app,
        'build_model_app': build_model_app,
        'build_sampler_app': build_sampler_app,
        'build_processor_app': build_processor_app,
    }


def resolve_builder(import_path: str, builders: dict[str, Callable]) -> Callable:
    """Resolve an ``import_path`` to a builder function.

    Args:
        import_path: The import path from config (e.g., 'server', 'model').
        builders: The name → builder mapping (from :func:`get_builders`).

    Raises:
        ValueError: If the import_path cannot be resolved.
    """
    # Try to resolve through the operator-facing name mapping.
    if import_path in BUILDERS:
        builder_name = BUILDERS[import_path]
        if builder_name in builders:
            return builders[builder_name]

    # Direct builder name.
    if import_path in builders:
        return builders[import_path]

    raise ValueError(f"Unknown import_path '{import_path}'. "
                     f'Available: {list(BUILDERS.keys())}')
