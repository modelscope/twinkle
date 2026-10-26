# Copyright (c) ModelScope Contributors. All rights reserved.
"""Platform -> delta transport resolution."""
from __future__ import annotations

_HCCL_MODULE = 'twinkle.checkpoint_engine.delta.transport.hccl_engine'

# Trainer-side torch device type per supported platform.
DELTA_DEVICES = {'GPU': 'cuda', 'NPU': 'npu'}


def resolve_delta_engine_cls(platform_name: str) -> type:
    """Return the delta transport class for ``platform_name`` ('GPU' or 'NPU').

    Imports are lazy so the GPU path never touches NPU-specific modules.
    """
    if platform_name == 'GPU':
        from .transport.nccl_engine import DeltaShardedCheckpointEngine
        return DeltaShardedCheckpointEngine
    if platform_name == 'NPU':
        try:
            from .transport.hccl_engine import DeltaHCCLCheckpointEngine
        except ModuleNotFoundError as error:
            if error.name != _HCCL_MODULE:
                raise
            raise NotImplementedError(
                'delta_sharded on NPU requires the HCCL delta transport, which is not implemented yet') from error
        return DeltaHCCLCheckpointEngine
    raise ValueError(f'delta_sharded does not support platform {platform_name!r}; supported platforms: GPU, NPU')
