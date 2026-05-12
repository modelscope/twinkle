# Copyright (c) ModelScope Contributors. All rights reserved.
"""Abstract base class for experiment trackers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ExperimentTracker(ABC):
    """Base class for experiment tracking backends (SwanLab, W&B, etc.).

    Subclasses must implement :meth:`log`.  The optional methods
    :meth:`log_hyperparams` and :meth:`cleanup` have reasonable
    no-op defaults.
    """

    @abstractmethod
    def log(self, data: Dict[str, float], step: int) -> None:
        """Log a set of metric values.

        Args:
            data: Metric names mapped to numeric values.  The dict has
                already been normalised by :func:`clean_metrics` so
                values are guaranteed to be ``float``.
            step: The current training step.
        """

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Record hyperparameters (optional)."""

    def cleanup(self) -> None:
        """Flush pending data and release resources (optional)."""
