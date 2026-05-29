"""Tests for config signature validation."""
from __future__ import annotations

import pytest

from twinkle.server.state.backend.memory_backend import MemoryBackend
from twinkle.server.state.config_signature import (
    SignatureMismatchPolicy,
    compute_signature,
    validate_config_signature,
)


# ---- compute_signature ----

def test_compute_signature_deterministic():
    """Same input should produce same output."""
    config = {"model": "qwen", "batch_size": 8}
    sig1 = compute_signature(config)
    sig2 = compute_signature(config)
    assert sig1 == sig2


def test_compute_signature_different_inputs():
    """Different inputs should produce different outputs."""
    config_a = {"model": "qwen", "batch_size": 8}
    config_b = {"model": "llama", "batch_size": 8}
    assert compute_signature(config_a) != compute_signature(config_b)


def test_compute_signature_key_order_independent():
    """Key order should not affect the signature (sort_keys=True)."""
    config_a = {"b": 2, "a": 1}
    config_b = {"a": 1, "b": 2}
    assert compute_signature(config_a) == compute_signature(config_b)


def test_compute_signature_is_hex_string():
    """Signature should be a valid hex SHA256 string."""
    sig = compute_signature({"key": "value"})
    assert len(sig) == 64  # SHA256 hex = 64 chars
    assert all(c in "0123456789abcdef" for c in sig)


# ---- validate_config_signature ----

@pytest.mark.asyncio
async def test_first_run_stores_signature():
    """First run with no stored sig should store it and return True."""
    backend = MemoryBackend()
    config = {"model": "test"}
    result = await validate_config_signature(backend, config)
    assert result is True
    # Signature should be stored
    stored = await backend.get("_meta::config_signature")
    assert stored == compute_signature(config)


@pytest.mark.asyncio
async def test_same_config_passes():
    """Same config on second run should pass validation."""
    backend = MemoryBackend()
    config = {"model": "test", "lr": 0.001}
    # First run
    await validate_config_signature(backend, config)
    # Second run same config
    result = await validate_config_signature(backend, config)
    assert result is True


@pytest.mark.asyncio
async def test_different_config_warn_policy():
    """Different config with WARN policy should return False and update sig."""
    backend = MemoryBackend()
    config_v1 = {"model": "v1"}
    config_v2 = {"model": "v2"}

    await validate_config_signature(backend, config_v1)
    result = await validate_config_signature(
        backend, config_v2, policy=SignatureMismatchPolicy.WARN
    )
    assert result is False
    # Signature should be updated to v2
    stored = await backend.get("_meta::config_signature")
    assert stored == compute_signature(config_v2)


@pytest.mark.asyncio
async def test_different_config_clear_policy():
    """CLEAR policy should clear non-meta data, preserve _meta, return False."""
    backend = MemoryBackend()
    config_v1 = {"model": "v1"}
    config_v2 = {"model": "v2"}

    # Store initial config
    await validate_config_signature(backend, config_v1)
    # Add some user data
    await backend.set("session::abc", {"data": 123})
    await backend.set("model::xyz", {"data": 456})
    await backend.set("_meta::other", "keep_this")

    result = await validate_config_signature(
        backend, config_v2, policy=SignatureMismatchPolicy.CLEAR
    )
    assert result is False

    # User data should be cleared
    assert await backend.get("session::abc") is None
    assert await backend.get("model::xyz") is None

    # _meta keys should be preserved
    assert await backend.get("_meta::other") == "keep_this"
    # Signature should be updated
    stored = await backend.get("_meta::config_signature")
    assert stored == compute_signature(config_v2)


@pytest.mark.asyncio
async def test_different_config_abort_policy():
    """ABORT policy should raise ConfigMismatchError."""
    from twinkle.server.exceptions import ConfigMismatchError

    backend = MemoryBackend()
    config_v1 = {"model": "v1"}
    config_v2 = {"model": "v2"}

    await validate_config_signature(backend, config_v1)

    with pytest.raises(ConfigMismatchError):
        await validate_config_signature(
            backend, config_v2, policy=SignatureMismatchPolicy.ABORT
        )


@pytest.mark.asyncio
async def test_abort_policy_does_not_update_signature():
    """ABORT policy should NOT update the stored signature."""
    from twinkle.server.exceptions import ConfigMismatchError

    backend = MemoryBackend()
    config_v1 = {"model": "v1"}
    config_v2 = {"model": "v2"}

    await validate_config_signature(backend, config_v1)

    with pytest.raises(ConfigMismatchError):
        await validate_config_signature(
            backend, config_v2, policy=SignatureMismatchPolicy.ABORT
        )

    # Signature should still be v1
    stored = await backend.get("_meta::config_signature")
    assert stored == compute_signature(config_v1)
