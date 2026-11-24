# Copyright (c) ModelScope Contributors. All rights reserved.
"""User fingerprints for the current authenticated request."""

import os
from contextvars import ContextVar

from twinkle.server.identity import CHECKPOINT_SALT_DEFAULT, fingerprint

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
