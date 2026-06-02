# Copyright (c) ModelScope Contributors. All rights reserved.
"""Experiment tracking dispatch for twinkle training metrics.

Usage::

    from twinkle.tracker import SwanLabTracker, register_tracker

    register_tracker(SwanLabTracker(project="my-project"))
    # training loop unchanged — dispatch happens automatically.

Or via environment variables (no code change)::

    TWINKLE_TRACKERS=swanlab SWANLAB_API_KEY=xxx python train.py
"""

import atexit
import logging
import os
from typing import Any, Dict, List, Optional

from twinkle.server.model.backends.common import clean_metrics
from .base import ExperimentTracker
from .swanlab import SwanLabTracker
from .wandb import WandbTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_trackers: List[ExperimentTracker] = []
_rank: int = 0
_hparams_dispatched: set = set()  # track which adapters have sent hyperparams

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_tracker(tracker: ExperimentTracker) -> None:
    """Register an experiment tracker.

    Multiple trackers can be registered — ``dispatch`` will send metric
    data to each one in order.  Trackers are cleaned up automatically on
    normal interpreter exit via ``atexit``.
    """
    _trackers.append(tracker)


def set_rank(rank: int) -> None:
    """Set the distributed rank for the current process.

    Called by ``twinkle.initialize()`` — not intended for direct use.
    Only the process with ``rank == 0`` dispatches metrics; all others
    are no-ops.
    """
    global _rank
    _rank = rank


def list_trackers() -> List[ExperimentTracker]:
    """Return a snapshot of currently registered trackers."""
    return list(_trackers)


def clear_trackers() -> None:
    """Call ``cleanup()`` on every registered tracker and clear the list.

    Registered automatically via ``atexit``; may also be called manually.
    """
    for t in _trackers:
        try:
            t.cleanup()
        except Exception:
            logger.warning('Tracker %s.cleanup() failed', type(t).__name__, exc_info=True)
    _trackers.clear()


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------


def dispatch(data: Dict[str, float], step: int) -> None:
    """Send computed metrics to all registered trackers.

    Metric values are normalized to ``float`` via :func:`clean_metrics`
    before dispatching.  Only the rank-0 process performs the dispatch;
    all other ranks return immediately with no overhead.

    Args:
        data: Raw metric dict (may contain strings, ints, floats).
        step: Current training step (``cur_step`` from optimizer group).
    """
    if not _trackers:
        return
    if _rank != 0:
        return

    cleaned = clean_metrics(data)
    if not cleaned:
        return

    for tracker in _trackers:
        try:
            tracker.log(cleaned, step=step)
        except Exception:
            logger.warning('Tracker %s.log() failed', type(tracker).__name__, exc_info=True)


def dispatch_hyperparams(params: Dict[str, Any], adapter_name: Optional[str] = None) -> None:
    """Send hyperparameters to all registered trackers (call once at training start).

    Idempotent per ``(adapter_name,)`` — repeated calls with the same
    *adapter_name* are silently ignored so that this can safely be called
    from ``calculate_metrics`` on its first invocation without
    flooding trackers with redundant config updates.

    Args:
        params: Flat or nested dict of hyperparameters (e.g. model config,
            training args, LoRA config).
        adapter_name: Optional adapter identifier.  If omitted, the params
            are dispatched unconditionally on every call.
    """
    if not _trackers or _rank != 0:
        return

    # Idempotency guard: only dispatch once per adapter
    if adapter_name is not None:
        if adapter_name in _hparams_dispatched:
            return
        _hparams_dispatched.add(adapter_name)

    for tracker in _trackers:
        try:
            tracker.log_hyperparams(params)
        except Exception:
            logger.warning('Tracker %s.log_hyperparams() failed', type(tracker).__name__, exc_info=True)


# ---------------------------------------------------------------------------
# Environment-variable auto-initialisation
# ---------------------------------------------------------------------------

_AUTO_INIT_DONE = False


def _auto_init_from_env() -> None:
    """Initialise trackers from environment variables (called once at import).

    Reads ``TWINKLE_TRACKERS`` (comma-separated, e.g. ``wandb,swanlab``)
    and backend-specific env vars, then registers matching tracker instances
    automatically.

    This lets users enable experiment tracking without *any* code change::

        TWINKLE_TRACKERS=wandb WANDB_PROJECT=my-project python train.py
    """
    global _AUTO_INIT_DONE
    if _AUTO_INIT_DONE:
        return
    _AUTO_INIT_DONE = True

    trackers_str = os.environ.get('TWINKLE_TRACKERS', '').strip()
    if not trackers_str:
        return

    project = os.environ.get('TWINKLE_TRACKER_PROJECT', 'twinkle-training')
    experiment_name = os.environ.get('TWINKLE_TRACKER_EXPERIMENT', None)

    for name in (t.strip().lower() for t in trackers_str.split(',') if t.strip()):
        try:
            if name == 'wandb':
                _trackers.append(
                    WandbTracker(
                        project=project,
                        experiment_name=experiment_name,
                        entity=os.environ.get('WANDB_ENTITY'),
                    ))
                logger.info('Auto-registered WandbTracker from TWINKLE_TRACKERS env var')
            elif name == 'swanlab':
                _trackers.append(
                    SwanLabTracker(
                        project=project,
                        experiment_name=experiment_name,
                        output_dir=os.environ.get('TWINKLE_OUTPUT_DIR'),
                    ))
                logger.info('Auto-registered SwanLabTracker from TWINKLE_TRACKERS env var')
            else:
                logger.warning('Unknown tracker backend in TWINKLE_TRACKERS: %s', name)
        except Exception:
            logger.warning("Failed to auto-init tracker '%s' from env", name, exc_info=True)


# Run auto-init once at import time (before user code or atexit runs)
_auto_init_from_env()

# ---------------------------------------------------------------------------
# At-exit cleanup
# ---------------------------------------------------------------------------
atexit.register(clear_trackers)
