# Copyright (c) ModelScope Contributors. All rights reserved.
"""Generic validate-then-dispatch helper for backend selectors (TIER 1, R12).

The Model backend selector (``mock | transformers | megatron``) and the Sampler
type selector (``mock | vllm | torch``) share one validate-then-dispatch shape.
``BackendSelector`` expresses it once, parameterized by the permitted value set
and the per-value construction callbacks. The callbacks stay defined in
``model/app.py`` and ``sampler/app.py`` so the lazy backend imports remain local
— importing the mock backend on a CPU-only host must not pull in
torch/vllm/megatron.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from twinkle.server.exceptions import ConfigError


class BackendSelector:
    """Validate a selector value against a permitted set, then dispatch construction.

    The mapping's insertion order defines the reported permitted set.
    """

    def __init__(self, field: str, builders: dict[str, Callable[[dict[str, Any]], Any]]) -> None:
        self.field = field
        self.builders = builders

    def validate(self, value: Any) -> str:
        """Pure validation; raises :class:`ConfigError` on absent/empty/non-string/unknown.

        Preserves the current failure behavior: the same exception class
        (``ConfigError``) carrying ``field``, ``value``, and
        ``allowed=list(self.builders)``. No imports, no side effects — so the
        builders can fail fast at build time before any backend object exists.
        """
        if not isinstance(value, str) or value == '' or value not in self.builders:
            raise ConfigError(field=self.field, value=value, allowed=list(self.builders))
        return value

    def construct(self, value: str, ctor_kwargs: dict[str, Any]) -> Any:
        """Validate then build exactly the one backend bound to the matched value."""
        self.validate(value)
        return self.builders[value](ctor_kwargs)
