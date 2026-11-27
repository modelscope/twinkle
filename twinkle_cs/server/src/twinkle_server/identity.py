# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint and telemetry fingerprint primitives and request identity."""

import hashlib
import hmac
import os
from contextvars import ContextVar

CHECKPOINT_SALT_DEFAULT = "twinkle-path-salt-v1"


def fingerprint(value: str, key: str) -> str:
    return hmac.digest(key.encode("utf-8"), value.encode("utf-8"), "sha256").hex()


def sampler_controller_name(namespace, pool_id):
    """Actor-name convention shared by the sampling pool and any router that looks it up.

    Lives in the identity layer so HTTP components (gateway) can resolve the name without
    importing the pool composition layer above them.
    """
    return f"sampler-controller-{namespace}-{pool_id}"


def sampling_pool_id(route):
    return hashlib.sha256(route.encode()).hexdigest()[:16]


USER_ID_KEY_ENV = "TWINKLE_TELEMETRY_USER_ID_KEY"
current_tenant_fingerprint: ContextVar[str | None] = ContextVar("twinkle_tenant_fingerprint", default=None)


def user_id_key() -> str | None:
    key = os.getenv(USER_ID_KEY_ENV)
    if key and key != os.getenv("TWINKLE_TOKEN_SALT", CHECKPOINT_SALT_DEFAULT):
        return key
    return None


def tenant_fingerprint(token: str | None) -> str | None:
    key = user_id_key()
    if key:
        return fingerprint(f"token:{token}", key) if token else "unassigned"
    return None
