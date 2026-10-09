# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the shared sampler weight-resolution helper (F004 / P004)."""
from __future__ import annotations

import pytest

from twinkle.server.sampler.weights import resolve_sampler_weights


class _FakeSampler:

    def load_full_weights_from_path(self, path):
        return None


class _FakeService:
    """Minimal stand-in exposing the two attributes the helper touches."""

    def __init__(self):
        self.sampler = _FakeSampler()
        self.full_weight_loads: list[str] = []

    async def call_backend(self, fn, *args, **kwargs):
        # The helper only routes load_full_weights_from_path through call_backend.
        if getattr(fn, '__name__', None) == 'load_full_weights_from_path':
            self.full_weight_loads.append(args[0])
        return fn(*args, **kwargs)


@pytest.mark.asyncio
async def test_lora_dir_returns_adapter_path_and_loads_no_full_weights(tmp_path):
    (tmp_path / 'adapter_config.json').write_text('{}')
    svc = _FakeService()
    result = await resolve_sampler_weights(svc, str(tmp_path))
    assert result == str(tmp_path)
    assert svc.full_weight_loads == []


@pytest.mark.asyncio
async def test_full_checkpoint_loads_weights_and_returns_none(tmp_path):
    # A directory without adapter_config.json is a full-parameter checkpoint.
    svc = _FakeService()
    result = await resolve_sampler_weights(svc, str(tmp_path))
    assert result is None
    assert svc.full_weight_loads == [str(tmp_path)]


@pytest.mark.asyncio
async def test_empty_uri_is_a_noop():
    svc = _FakeService()
    assert await resolve_sampler_weights(svc, None) is None
    assert await resolve_sampler_weights(svc, '') is None
    assert svc.full_weight_loads == []
