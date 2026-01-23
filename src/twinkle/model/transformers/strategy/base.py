# Copyright (c) ModelScope Contributors.
# Minimal compatibility layer for TrainStrategy imports.

from __future__ import annotations

class TrainStrategy:
    """
    Base training strategy (compatibility shim).

    Some branches expect `twinkle.model.transformers.strategy.base.TrainStrategy`.
    Minimal cookbook runs can proceed without a full strategy implementation.
    """
    name: str = "base"

    def __init__(self, *args, **kwargs):
        pass
