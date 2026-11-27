# Copyright (c) ModelScope Contributors. All rights reserved.
"""Twinkle distributed-runtime initialisation for the deployment classes.

Moved out of ``deployment.py``: that module is deployment-*construction*
infrastructure -- the FastAPI scaffold, the middleware stack, the ``serve.ingress``
chain -- and ``twinkle.initialize`` + ``DeviceMesh`` construction is neither. It lived
there only because Model and Sampler both needed it; now Processor reuses it too, so its
formerly-inlined copy is gone.
"""
from __future__ import annotations

from typing import Any


def init_twinkle_runtime(
    is_mock: bool,
    nproc_per_node: int,
    device_group: Any,
    device_mesh_dict: dict[str, Any],
    *,
    ncpu_proc_per_node: int | None = None,
) -> Any | None:
    """Initialize the Twinkle distributed runtime and build a DeviceMesh.

    Shared by ModelManagement, SamplerManagement and ProcessorManagement ``__init__``.
    Returns ``None`` for mock backends (CPU-only, no device mesh).

    ``ncpu_proc_per_node`` is forwarded only when provided; Model/Sampler leave it unset
    (preserving their prior behaviour), while Processor passes its own value -- reusing
    this function without that parameter would have silently changed the processor's CPU
    process count.
    """
    import twinkle
    from twinkle import DeviceMesh

    if is_mock:
        twinkle.initialize(
            mode='ray', nproc_per_node=nproc_per_node, ncpu_proc_per_node=1, groups=[device_group], lazy_collect=False)
        return None

    init_kwargs: dict[str, Any] = {
        'mode': 'ray',
        'nproc_per_node': nproc_per_node,
        'groups': [device_group],
        'lazy_collect': False,
    }
    if ncpu_proc_per_node is not None:
        init_kwargs['ncpu_proc_per_node'] = ncpu_proc_per_node
    twinkle.initialize(**init_kwargs)
    if 'mesh_dim_names' in device_mesh_dict:
        return DeviceMesh(**device_mesh_dict)
    return DeviceMesh.from_sizes(**device_mesh_dict)
