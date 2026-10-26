"""Control-plane primitives for async RL.

The public names resolve LAZILY (PEP 562 ``__getattr__``): ``StreamingDriver`` and
``ClusterAdapter`` sit on heavier dependency chains than the plain-data names, and a consumer
that only needs one primitive must not be forced to import anything heavier. Each name is
imported from its defining submodule on first access instead of eagerly here.

The former TransferQueue runtime (pipeline / workers / scheduler / data_plane / native_tq / the
``*_sampler_tq`` backends / generation_submissions / context_manager) is archived under ``legacy/``
for preservation only, not for use: it modelled a server-side multi-tenant context pipeline, which
this architecture retires -- the streaming driver runs on the client, the server is a stateless
execution device, and the two never shared state. Nothing outside ``legacy/`` may import from it,
and the live modules carry no fields or parameters that exist only to serve it.
"""

from __future__ import annotations

import importlib

#: Maps each public name to the submodule that defines it, so ``__getattr__`` imports only what is
#: accessed, and only when. Kept next to ``__all__`` so the two cannot drift. Only names with real
#: consumers OUTSIDE this package live here; the rest is reachable via its defining submodule.
_EXPORTS = {
    'ClusterAdapter': 'cluster_adapter',
    'REF_ONLY_REMOTE_GROUP': 'score_worker',
    'StreamingDriver': 'streaming_driver',
    'estimate_plan': 'cluster_adapter',
    'resource_device_groups': 'cluster_adapter',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Import and cache ``name`` from its defining submodule on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(f'.{module_name}', __name__), name)
        globals()[name] = value  # cache, so __getattr__ is not hit again for that name
        return value
    # Not a re-exported name: fall back to importing a submodule of the same name, so
    # ``from twinkle_agentic.async_rl import <submodule>`` keeps working.
    try:
        module = importlib.import_module(f'.{name}', __name__)
    except ImportError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc
    globals()[name] = module
    return module


def __dir__():
    return sorted(__all__)
