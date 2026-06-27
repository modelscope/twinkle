# Copyright (c) ModelScope Contributors. All rights reserved.
"""Cross-process-stable deterministic seed utility."""
from __future__ import annotations

import hashlib
from typing import Any


def stable_seed(*parts: Any) -> int:
    """Cross-process-stable numpy seed (uint32) derived from arbitrary parts.

    Uses SHA-256 over a canonical string form rather than Python's built-in
    ``hash()``: the latter is salted per process (PYTHONHASHSEED) for tuples
    containing strings, which would make identical requests on different
    replicas / restarts produce different outputs.
    """
    canonical = '\x1f'.join(str(p) for p in parts).encode('utf-8')
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:4], 'big')
