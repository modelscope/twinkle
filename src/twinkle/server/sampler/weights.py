# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared sampler weight-resolution rule (F004 / P004).

The "a resolved checkpoint is either a LoRA adapter dir or a full-parameter HF
checkpoint" rule was re-derived from a filesystem probe in three sampler handlers
(``sample`` / ``sample_stream`` on the twinkle dialect and ``asample`` on tinker),
and the copies had begun to diverge. It lives here so the storage-layout decision
has a single owner.

Prefix-cache invalidation is deliberately NOT handled here: each caller keeps its
own ``reset_prefix_cache`` policy (the tinker endpoint resets unconditionally on
every request; the twinkle endpoints reset only when an ``adapter_uri`` is
present), which is an observable behaviour difference this helper must not erase.
"""
from __future__ import annotations

import os
from typing import Any


async def resolve_sampler_weights(service: Any, resolved_uri: str | None) -> str | None:
    """Resolve a checkpoint path into a LoRA adapter path (or load full weights).

    Returns the LoRA ``adapter_path`` when ``resolved_uri`` is a directory holding
    an ``adapter_config.json``. Otherwise the path is a full-parameter checkpoint:
    it is loaded into the sampler base model via ``load_full_weights_from_path`` and
    ``None`` is returned (no LoRA adapter to pass). ``None``/empty input returns
    ``None`` unchanged.
    """
    if not resolved_uri:
        return None
    if os.path.exists(os.path.join(resolved_uri, 'adapter_config.json')):
        return resolved_uri
    await service.call_backend(service.sampler.load_full_weights_from_path, resolved_uri)
    return None
