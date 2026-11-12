# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run checkpoint loading and generation as one owned sampling job."""
from __future__ import annotations

import glob
import os
from safetensors import SafetensorError

from twinkle.exceptions import FullWeightLoadError
from twinkle.server.exceptions import RequestRejectedError
from twinkle.server.task_queue.backend_gate import BACKEND_TIMEOUT_EXCEPTIONS


def is_full_checkpoint(resolved_uri: str | None) -> bool:
    if not resolved_uri:
        return False
    if os.path.isfile(os.path.join(resolved_uri, 'adapter_config.json')):
        return False
    if os.path.isabs(resolved_uri) and not os.path.isdir(resolved_uri):
        raise RequestRejectedError(f'Checkpoint directory does not exist: {resolved_uri}')
    if os.path.isdir(resolved_uri) and not (os.path.isfile(os.path.join(resolved_uri, 'model.safetensors.index.json'))
                                            or glob.glob(os.path.join(resolved_uri, '*.safetensors'))):
        raise RequestRejectedError(f'No .safetensors weights found under {resolved_uri}')
    return True


async def resolve_sampler_weights(service, resolved_uri: str | None, *, full_checkpoint=None) -> str | None:
    """Load base weights or return the adapter path."""
    if full_checkpoint is None:
        full_checkpoint = is_full_checkpoint(resolved_uri)
        if full_checkpoint:
            return await service.sampling_guard.run(
                lambda: resolve_sampler_weights(service, resolved_uri, full_checkpoint=True), exclusive=True)
    if not full_checkpoint:
        return resolved_uri or None
    try:
        await service.call_backend(service.sampler.load_full_weights_from_path, resolved_uri)
    except (FullWeightLoadError, *BACKEND_TIMEOUT_EXCEPTIONS) as error:
        service.sampling_guard.invalidate_weights(error)
        raise
    except (ValueError, OSError, KeyError, SafetensorError) as error:
        raise RequestRejectedError(str(error)) from error
    return None


async def sample_with_weights(service, resolved_uri: str | None, generate):
    """Keep a full checkpoint's load and generation in the same exclusive job."""
    full_checkpoint = is_full_checkpoint(resolved_uri)

    async def job():
        adapter_path = await resolve_sampler_weights(service, resolved_uri, full_checkpoint=full_checkpoint)
        return await generate(adapter_path)

    return await service.sampling_guard.run(job, exclusive=full_checkpoint)
