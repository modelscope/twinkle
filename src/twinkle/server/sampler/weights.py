# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run checkpoint loading and generation as one owned sampling job."""
from __future__ import annotations

import os


def is_full_checkpoint(resolved_uri: str | None) -> bool:
    return bool(resolved_uri and not os.path.exists(os.path.join(resolved_uri, 'adapter_config.json')))


async def resolve_sampler_weights(service, resolved_uri: str | None, *, full_checkpoint=None) -> str | None:
    """Load base weights or return the adapter path."""
    full_checkpoint = is_full_checkpoint(resolved_uri) if full_checkpoint is None else full_checkpoint
    if not full_checkpoint:
        return resolved_uri or None
    await service.call_backend(service.sampler.load_full_weights_from_path, resolved_uri, mutation=True)
    return None


async def sample_with_weights(service, resolved_uri: str | None, generate):
    """Keep a full checkpoint's load and generation in the same exclusive job."""
    full_checkpoint = is_full_checkpoint(resolved_uri)

    async def job():
        adapter_path = await resolve_sampler_weights(service, resolved_uri, full_checkpoint=full_checkpoint)
        return await generate(adapter_path)

    return await service.sampling_guard.run(job, exclusive=full_checkpoint)
