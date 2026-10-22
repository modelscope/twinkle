# Copyright (c) ModelScope Contributors. All rights reserved.
"""Monotonic counters -- the metrics half of the perf package."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator


class CounterRegistry:
    """Always-live monotonic counters, the complement of :class:`TimeRecorder`.

    Spans answer "where did the time go, inside which method, for which
    profiling window" -- they are windowed (a dump resets them) and gated
    (a disabled recorder is a zero-allocation fast path). Counters answer
    "how much has accumulated since the start" -- always live, strictly
    additive, never reset. The two semantics live in separate objects on
    purpose, so windowing and enable gates never have to be arbitrated
    against a control loop that reads a delta at every step.

    Usage::

        counters.add('rows', 1)
        with counters.time_span('train'):
            ...
        mark = counters.snapshot()
        ...  # a window passes
        deltas = {k: v - mark[k] for k, v in counters.snapshot().items()}

    A name nobody has fed reads as ``0.0`` -- an accumulator that has not
    started yet. The registry is process-local; aggregating worker-side
    counters onto a driver is a transport concern and stays outside this
    class.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[str, float] = {}

    def add(self, name: str, value: float) -> None:
        """Accumulate ``value`` into ``name`` -- the one mutation primitive."""
        with self._lock:
            self._values[name] = self._values.get(name, 0.0) + value

    def get(self, name: str) -> float:
        """Current total of ``name``; a never-fed name reads as ``0.0``."""
        with self._lock:
            return self._values.get(name, 0.0)

    def snapshot(self) -> Dict[str, float]:
        """A copy of all current totals, for window-delta reads."""
        with self._lock:
            return dict(self._values)

    @contextmanager
    def time_span(self, name: str) -> Iterator[None]:
        """Accumulate wall time into ``name``; exception paths still count."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)


#: Process-level default, mirroring ``default_recorder``. Unlike the recorder it
#: carries no enable gate: a few float additions per step are always affordable,
#: which is exactly why control loops can depend on it being there.
default_counters = CounterRegistry()
