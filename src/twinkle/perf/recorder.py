# Copyright (c) ModelScope Contributors. All rights reserved.
"""Wall-clock span recorder: a synchronous/async-agnostic timing primitive.

The recorder is deliberately framework-free (no torch import) so it can time ANY
callable, not just model code. Spans nest through a :mod:`contextvars` chain, so
the same ``@timed`` decorator / ``span()`` block produces a call tree whether it
runs in a plain thread or inside an ``await``-driven coroutine.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

__all__ = ['TimeRecorder', 'default_recorder', 'timed', 'span', 'enable_perf', 'disable_perf']

_NS_PER_SEC = 1e9

#: Source of ``_Span.sid``. Allocation happens at span construction, so a parent's
#: sid is always SMALLER than its children's -- the ordering offline timeline
#: analysis relies on when folding subtree self-times (see :mod:`.timeline`).
_SPAN_IDS = itertools.count()


class _Span:
    """One open timing interval, linked to its parent to form the nesting tree.

    ``child_ns`` accumulates the total duration of DIRECT children while this span
    is still open; ``self_ns`` (computed at close) is ``total - child_ns``. Storing
    a parent POINTER rather than a shared stack list is what makes concurrent
    contexts safe: each ``span()`` enter creates a fresh object and rebinds the
    ContextVar to it, so two coroutines never mutate one another's chain.

    ``sid`` (construction order, process-unique) and ``tid`` (owning thread id)
    make a closed span a self-contained event: the id pair is what the offline
    timeline rebuild keys its nesting edges and per-thread lanes on, since span
    NAMES repeat while identities do not.
    """

    __slots__ = ('name', 'parent', 'sid', 'tid', 'start_ns', 'child_ns')

    def __init__(self, name: str, parent: Optional['_Span'], start_ns: int, tid: int) -> None:
        self.name = name
        self.parent = parent
        self.sid = next(_SPAN_IDS)
        self.tid = tid
        self.start_ns = start_ns
        self.child_ns = 0


class TimeRecorder:
    """Aggregate per-name span timings into a summary and a metric-ready dict.

    Enable/disable is checked at CALL time (not decoration time), so decorating a
    function once and toggling the recorder later behaves correctly, and a disabled
    recorder takes a no-op fast path with no timing or allocation.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        # name -> [count, total_ns, self_ns]
        self._stats: Dict[str, list] = {}
        # (parent_name_or_None, name) -> [count, total_ns]
        self._edges: Dict[Tuple[Optional[str], str], list] = {}
        # The open-span chain for the CURRENT context; None when no span is active.
        self._current: ContextVar[Optional[_Span]] = ContextVar('twinkle_perf_span', default=None)
        self._first_start_ns: Optional[int] = None
        self._last_end_ns: Optional[int] = None
        # Append-only event stream for offline timeline analysis (see timeline.py):
        # one 8-tuple per CLOSED span, in close order. Shares the aggregate's window
        # semantics -- cleared by reset(), snapshotted (not consumed) by trace_events().
        self._events: List[Tuple[Optional[int], Optional[int], str, int, int, int, int, int]] = []

    # ------------------------------------------------------------------ control

    def __reduce__(self):
        """Pickle by reference to the process-wide recorder.

        The recorder holds a threading.Lock, so any ``@timed`` wrapper that closes
        over it is unpicklable. Ray serialises actor classes by value (the modified
        class is dynamic), and a single ``@timed`` method then makes the whole
        actor un-exportable ("cannot pickle '_thread.lock' object"). The recorder
        is process-global state, so a worker must time into ITS OWN default
        recorder rather than a stale copy.
        """
        return (_get_default_recorder, ())

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        """Drop all accumulated timings (e.g. between logging windows)."""
        with self._lock:
            self._stats.clear()
            self._edges.clear()
            self._events.clear()
            self._first_start_ns = None
            self._last_end_ns = None

    # ------------------------------------------------------------------ recording

    def span(self, name: str):
        """Time the wrapped block as ``name``; a no-op ``nullcontext`` when disabled.

        Dispatching here (rather than inside the generator) keeps the disabled path
        free of even a context-manager allocation, so a hot ``@timed`` function costs
        nothing when profiling is off -- the no-op fast path the module contract
        promises (mirrors rl-insight's ``_STATE.enabled`` short-circuit).
        """
        if not self._enabled:
            return nullcontext()
        return self._span(name)

    @contextmanager
    def _span(self, name: str) -> Generator[None, None, None]:
        """The recording span: nests under the active span in this context.

        Recorded on exit even if the body raised (the ``finally`` runs, capturing the
        partial duration); the exception is NOT suppressed.
        """
        parent = self._current.get()
        current = _Span(name, parent, time.perf_counter_ns(), threading.get_ident())
        token = self._current.set(current)
        try:
            yield
        finally:
            # Reset BEFORE recording so a raising body cannot leave a stale span on
            # the chain for subsequent code in this context.
            self._current.reset(token)
            self._record(current, time.perf_counter_ns())

    def _record(self, current: _Span, end_ns: int) -> None:
        total_ns = end_ns - current.start_ns
        # Fold this span's duration into its still-open parent's child total.
        if current.parent is not None:
            current.parent.child_ns += total_ns
        self_ns = total_ns - current.child_ns
        parent_name = current.parent.name if current.parent is not None else None
        parent_sid = current.parent.sid if current.parent is not None else None
        with self._lock:
            stat = self._stats.get(current.name)
            if stat is None:
                stat = [0, 0, 0]
                self._stats[current.name] = stat
            stat[0] += 1
            stat[1] += total_ns
            stat[2] += self_ns
            edge = self._edges.get((parent_name, current.name))
            if edge is None:
                edge = [0, 0]
                self._edges[(parent_name, current.name)] = edge
            edge[0] += 1
            edge[1] += total_ns
            self._events.append(
                (current.sid, parent_sid, current.name, current.tid, current.start_ns, end_ns, self_ns, total_ns))
            if self._first_start_ns is None or current.start_ns < self._first_start_ns:
                self._first_start_ns = current.start_ns
            if self._last_end_ns is None or end_ns > self._last_end_ns:
                self._last_end_ns = end_ns

    def timed(self, name: Optional[str] = None) -> Callable:
        """Decorator recording one span per call; usable bare or with a name.

        Supports ``@recorder.timed``, ``@recorder.timed('name')``, sync and async
        functions. A coroutine function stays a coroutine function (the wrapper is
        ``async def`` and awaits), so timing brackets the full await lifecycle.
        """
        # Bare usage: @recorder.timed  ->  name is actually the wrapped function.
        if callable(name):
            return self._wrap(name, None)

        def decorator(func: Callable) -> Callable:
            return self._wrap(func, name)

        return decorator

    def _wrap(self, func: Callable, name: Optional[str]) -> Callable:
        span_name = name or getattr(func, '__qualname__', repr(func))
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self._enabled:
                    return await func(*args, **kwargs)
                with self.span(span_name):
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not self._enabled:
                return func(*args, **kwargs)
            with self.span(span_name):
                return func(*args, **kwargs)

        return sync_wrapper

    # ------------------------------------------------------------------ output

    def trace_events(self) -> List[Tuple[Optional[int], Optional[int], str, int, int, int, int, int]]:
        """Snapshot the raw event stream: one tuple per CLOSED span, in close order.

        Each event is ``(sid, parent_sid, name, tid, start_ns, end_ns, self_ns,
        total_ns)`` -- identities and thread ids included, which is exactly what
        the offline timeline transforms (:mod:`.timeline`) need to rebuild the
        nesting tree, the per-thread lanes and the gaps between stations. Copy on
        read (NOT consumed): aggregates keep accumulating afterwards, so this
        pairs with ``reset()``'s window semantics exactly like ``summary()`` does.
        """
        with self._lock:
            return list(self._events)

    def _wall_ns(self) -> int:
        if self._first_start_ns is None or self._last_end_ns is None:
            return 0
        return self._last_end_ns - self._first_start_ns

    def bubble_s(self) -> float:
        """Idle seconds in the recorded window: wall time not covered by any root span.

        A root span (parent ``None``) is a top-level timed operation, so the union of the
        root spans is the window's busy time and ``wall - busy`` is the gap spent waiting --
        for data, for dispatch, for a peer rank. Clamped at 0 because concurrent roots can
        overlap and over-count busy. Complements each span's ``self_s`` (idle *inside* a
        span, between its children) with the idle *between* top-level spans.
        """
        with self._lock:
            wall_ns = self._wall_ns()
            root_busy_ns = sum(total for (parent, _name), (_count, total) in self._edges.items() if parent is None)
        return max(0, wall_ns - root_busy_ns) / _NS_PER_SEC

    def summary(self) -> Dict[str, Any]:
        """Structured aggregate: per-span timings plus the nesting tree.

        Snapshot under the lock so a concurrent span closing mid-read cannot tear
        the counters.
        """
        with self._lock:
            spans = {
                name: {
                    'count': count,
                    'total_s': total / _NS_PER_SEC,
                    'mean_s': (total / count / _NS_PER_SEC) if count else 0.0,
                    'self_s': self_ns / _NS_PER_SEC,
                }
                for name, (count, total, self_ns) in self._stats.items()
            }
            children: Dict[Optional[str], list] = {}
            for (parent_name, child_name), (count, total) in self._edges.items():
                children.setdefault(parent_name, []).append({
                    'name': child_name,
                    'count': count,
                    'total_s': total / _NS_PER_SEC,
                })
            wall_s = self._wall_ns() / _NS_PER_SEC

        def build(parent_name: Optional[str]) -> list:
            nodes = []
            for edge in sorted(children.get(parent_name, []), key=lambda e: -e['total_s']):
                node = dict(edge)
                node['children'] = build(edge['name'])
                nodes.append(node)
            return nodes

        return {'enabled': self._enabled, 'wall_s': wall_s, 'spans': spans, 'tree': build(None)}

    def flat_summary(self, prefix: str = 'perf/') -> Dict[str, float]:
        """Flatten the aggregate to ``dict[str, float]`` for merging into metrics.

        Keys are namespaced (``{prefix}{span}/{field}``) so they never collide with
        the consumer's own metric names; every value is a float so the dict can be
        merged directly into a step-metrics dict or a ``MetricRecord.values``.
        """
        out: Dict[str, float] = {}
        with self._lock:
            stats = dict(self._stats)
            wall_ns = self._wall_ns()
        out[f'{prefix}wall_s'] = wall_ns / _NS_PER_SEC
        for name, (count, total, self_ns) in stats.items():
            out[f'{prefix}{name}/count'] = float(count)
            out[f'{prefix}{name}/total_s'] = total / _NS_PER_SEC
            out[f'{prefix}{name}/mean_s'] = (total / count / _NS_PER_SEC) if count else 0.0
            out[f'{prefix}{name}/self_s'] = self_ns / _NS_PER_SEC
        return out

    def print_summary(self, *, sort: str = 'total', top: Optional[int] = None) -> None:
        """Print a human-readable table plus the nesting tree."""
        data = self.summary()
        print(f'twinkle perf summary (enabled={data["enabled"]}, wall {data["wall_s"]:.3f}s)')
        rows = sorted(
            data['spans'].items(),
            key=lambda kv: -kv[1]['total_s' if sort == 'total' else 'self_s'],
        )
        if top is not None:
            rows = rows[:top]
        print(f'{"name":<40}{"count":>8}{"total_s":>12}{"mean_s":>12}{"self_s":>12}')
        for name, s in rows:
            print(f'{name:<40}{s["count"]:>8}{s["total_s"]:>12.4f}{s["mean_s"]:>12.4f}{s["self_s"]:>12.4f}')
        print('call tree:')
        self._print_tree(data['tree'], depth=0)

    def _print_tree(self, nodes: list, depth: int) -> None:
        for node in nodes:
            indent = '  ' * depth
            print(f'{indent}- {node["name"]} (x{node["count"]}, {node["total_s"]:.4f}s)')
            self._print_tree(node['children'], depth + 1)


#: Process-wide recorder bound by the module-level :func:`timed` / :func:`span`.
#: Disabled by default so importing perf never changes existing behaviour.
default_recorder = TimeRecorder(enabled=False)


def _get_default_recorder() -> TimeRecorder:
    """Unpickling entry point for :meth:`TimeRecorder.__reduce__`."""
    return default_recorder

#: Decorator bound to :data:`default_recorder`; ``@timed`` or ``@timed('name')``.
timed = default_recorder.timed


@contextmanager
def span(name: str) -> Generator[None, None, None]:
    """Module-level ``with span('data'):`` bound to :data:`default_recorder`.

    Delegates to the recorder, which short-circuits to a ``nullcontext`` when the
    default recorder is off; the ``with`` below is transparent to that fast path.
    """
    with default_recorder.span(name):
        yield


def enable_perf() -> None:
    """Turn on the default recorder (module-level ``timed``/``span`` start timing)."""
    default_recorder.enable()


def disable_perf() -> None:
    """Turn off the default recorder; subsequent spans take the no-op fast path."""
    default_recorder.disable()
