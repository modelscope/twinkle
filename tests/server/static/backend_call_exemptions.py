# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared exemption list for the "no direct backend call" static checks.

This file is the SINGLE source of allowed Blocking_Backend_Call bypasses. It is
consumed by this spec's check (``test_no_direct_backend_call.py``) and is intended
to be consumed unchanged by the ``server-request-lifecycle`` spec's equivalent
check -- there must be exactly one physical copy, not one per spec (R2#8).

Each entry is ``(module_relpath, function_name)`` where ``module_relpath`` is
relative to ``src/twinkle/server`` and ``function_name`` is the innermost enclosing
function of the exempted call.

The allowed exemptions are:

- the ray ``Queue.get`` inside ``sample_stream``'s ``_stream_queue``: it bridges the
  sampler actor's process boundary and is bounded by the dedicated double-timeout of
  R4#10-11 (T5.5), not by ``call_backend``;
- the ``<actor>.sample_stream_to_queue.remote(...)`` call inside ``sample_stream``
  itself: streaming generation must keep producing while the HTTP response streams,
  so it cannot use ``call_backend`` as-is and carries its own double timeout. This is
  a ``remote_function`` bypass that the guard now *detects* (via backend-derived
  local tracking) and that is *explicitly* accepted here — replacing the previous
  "No remote_function call is exempt" claim, which was true only because the guard
  could not see this shape.
"""
from __future__ import annotations

# (module_relpath under src/twinkle/server, innermost enclosing function name)
BACKEND_CALL_EXEMPTIONS: frozenset[tuple[str, str]] = frozenset({
    ('sampler/twinkle_handlers.py', '_stream_queue'),
    ('sampler/twinkle_handlers.py', 'sample_stream'),
})
