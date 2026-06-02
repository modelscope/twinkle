# Copyright (c) ModelScope Contributors. All rights reserved.
"""Experiment tracking dispatch for twinkle training metrics.

Usage::

    from twinkle.tracker import SwanLabTracker, register_tracker

    # Global tracker — receives metrics from all adapters.
    register_tracker(SwanLabTracker(project="my-project"))

    # Per-adapter tracker — receives metrics only from a specific adapter.
    register_tracker(SwanLabTracker(project="adapter-a"), adapter_name="lora_a")

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
# Trackers that receive metrics from ALL adapters.
_global_trackers: List[ExperimentTracker] = []
# Trackers that receive metrics only from a specific adapter.
# Key: adapter_name. Value: list of trackers.
_adapter_trackers: Dict[str, List[ExperimentTracker]] = {}
_rank: int = 0
_hparams_dispatched: set = set()  # track which adapters have sent hyperparams

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_tracker(tracker: ExperimentTracker, adapter_name: Optional[str] = None) -> None:
    """Register an experiment tracker.

    Args:
        tracker: An ``ExperimentTracker`` instance.
        adapter_name: If provided, the tracker receives metrics only
            from the training loop of *adapter_name*.  If ``None``
            (default), the tracker receives metrics from **all** adapters.

    Multiple trackers can be registered — ``dispatch`` will send metric
    data to each one in order.  Trackers are cleaned up automatically on
    normal interpreter exit via ``atexit``.
    """
    if adapter_name is not None:
        _adapter_trackers.setdefault(adapter_name, []).append(tracker)
    else:
        _global_trackers.append(tracker)


def set_rank(rank: int) -> None:
    """Set the distributed rank for the current process.

    Called by ``twinkle.initialize()`` — not intended for direct use.
    Only the process with ``rank == 0`` dispatches metrics; all others
    are no-ops.
    """
    global _rank
    _rank = rank


def list_trackers(adapter_name: Optional[str] = None) -> List[ExperimentTracker]:
    """Return a snapshot of currently registered trackers.

    Args:
        adapter_name: If provided, returns only trackers registered
            for that specific adapter (plus global trackers).  If
            ``None``, returns all trackers.
    """
    result = list(_global_trackers)
    if adapter_name is not None:
        result.extend(_adapter_trackers.get(adapter_name, []))
    else:
        for ts in _adapter_trackers.values():
            result.extend(ts)
    return result


def clear_trackers() -> None:
    """Call ``cleanup()`` on every registered tracker and clear the list.

    Registered automatically via ``atexit``; may also be called manually.
    """
    all_trackers = list(_global_trackers)
    for ts in _adapter_trackers.values():
        all_trackers.extend(ts)
    for t in all_trackers:
        try:
            t.cleanup()
        except Exception:
            logger.warning('Tracker %s.cleanup() failed', type(t).__name__, exc_info=True)
    _global_trackers.clear()
    _adapter_trackers.clear()


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------


def _target_trackers(adapter_name: Optional[str] = None) -> List[ExperimentTracker]:
    """Resolve the list of trackers that should receive data for *adapter_name*.

    Global trackers always receive data.  If *adapter_name* is given,
    per-adapter trackers for that name also receive data.
    """
    result = list(_global_trackers)
    if adapter_name is not None:
        result.extend(_adapter_trackers.get(adapter_name, []))
    return result


def dispatch(data: Dict[str, float], step: int, adapter_name: Optional[str] = None) -> None:
    """Send computed metrics to registered trackers.

    Metric values are normalized to ``float`` via :func:`clean_metrics`
    before dispatching.  Only the rank-0 process performs the dispatch;
    all other ranks return immediately with no overhead.

    Args:
        data: Raw metric dict (may contain strings, ints, floats).
        step: Current training step (``cur_step`` from optimizer group).
        adapter_name: Optional adapter identifier.  If provided, metrics
            are sent to both global trackers and any trackers registered
            specifically for this adapter.
    """
    targets = _target_trackers(adapter_name)
    if not targets:
        return
    if _rank != 0:
        return

    cleaned = clean_metrics(data)
    if not cleaned:
        return

    for tracker in targets:
        try:
            tracker.log(cleaned, step=step)
        except Exception:
            logger.warning('Tracker %s.log() failed', type(tracker).__name__, exc_info=True)


def dispatch_hyperparams(params: Dict[str, Any], adapter_name: Optional[str] = None) -> None:
    """Send hyperparameters to registered trackers (call once at training start).

    Idempotent per ``(adapter_name,)`` — repeated calls with the same
    *adapter_name* are silently ignored so that this can safely be called
    from ``calculate_metrics`` on its first invocation without
    flooding trackers with redundant config updates.

    Args:
        params: Flat or nested dict of hyperparameters (e.g. model config,
            training args, LoRA config).
        adapter_name: Optional adapter identifier.  If omitted, the params
            are dispatched unconditionally to global trackers on every
            call.  If provided, dispatched to both global and per-adapter
            trackers, with idempotency guard.
    """
    targets = _target_trackers(adapter_name)
    if not targets or _rank != 0:
        return

    # Idempotency guard: only dispatch once per adapter
    if adapter_name is not None:
        if adapter_name in _hparams_dispatched:
            return
        _hparams_dispatched.add(adapter_name)

    for tracker in targets:
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
                _global_trackers.append(
                    WandbTracker(
                        project=project,
                        experiment_name=experiment_name,
                        entity=os.environ.get('WANDB_ENTITY'),
                    ))
                logger.info('Auto-registered WandbTracker from TWINKLE_TRACKERS env var')
            elif name == 'swanlab':
                _global_trackers.append(
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
