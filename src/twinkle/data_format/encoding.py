# Copyright (c) ModelScope Contributors. All rights reserved.
"""The single definition of "is this entry already encoded model input?".

One predicate, one place. Before this module the same rule existed three times
(``MegatronModel._not_encoded``, ``TransformersModel._not_encoded``,
``Sampler._not_encoded``), one of them carrying a comment that it was "aligned
with" another -- an invariant only a human could maintain. The wire schema needs
the same rule to decide whether an ``inputs`` entry is an ``InputFeature`` or a
``Trajectory``, so a fourth copy would have made divergence a matter of time.

``input_embedding`` matters as much as ``input_ids``: a batch carrying only
embeddings is already encoded, and misreading it as a ``Trajectory`` sends it
through ``template.batch_encode``, which fails far away from the cause.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Presence of any of these keys means the entry carries encoded model input.
ENCODED_INPUT_KEYS: tuple[str, ...] = ('input_ids', 'input_embedding')


def is_encoded(entry: Any) -> bool:
    """True when ``entry`` is an already-encoded ``InputFeature``-shaped mapping.

    A non-mapping is not encoded -- callers that need a type error raise it
    themselves; this predicate answers only the classification question.
    """
    if not isinstance(entry, Mapping):
        return False
    return any(key in entry for key in ENCODED_INPUT_KEYS)
