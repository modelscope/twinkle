# Copyright (c) ModelScope Contributors. All rights reserved.
"""twinkle.perf -- a general-purpose performance observability component.

Four independent, low-intrusion parts:

* :class:`TimeRecorder` / :func:`timed` / :func:`span` -- a synchronous- and
  async-agnostic wall-clock span recorder that nests into an overall timing tree
  and summarizes to a printable report or a metric-ready ``dict[str, float]``.
  Pure Python (no torch), so it can time ANY callable. The raw event stream
  (:meth:`TimeRecorder.trace_events`) feeds offline timeline analysis
  (:func:`chrome_trace` / :func:`dag_report`).
* :class:`CounterRegistry` -- always-live monotonic counters, the metrics half
  complementing the span (trace) half above: ``add`` accumulates, ``snapshot``
  reads window deltas, never reset, no enable gate. For control loops that read
  a delta at every step -- spans are windowed and gated, counters are neither.
* :class:`ModuleProfiler` -- a forward-hook profiler that gates by step, selects
  target modules, and leaves *what to measure* open via a list of collector
  functions (built-ins: :func:`activation_stats` / :func:`param_stats`). Attaches
  to any ``nn.Module`` subclass.
* :mod:`twinkle.perf.timeline` -- stateless transforms of that event stream:
  Chrome Trace rendering and a text DAG / lane report. Consumed by whoever holds
  the window's events; the recorder itself only appends.

The recorder and profiler are disabled/no-op by default, so importing perf never
changes existing behaviour; the counter registry is always live by design (a few
float additions per step). None of the parts imports twinkle/swift business
modules or Ray: the
driver-side orchestration (:class:`Perf`) duck-types the per-rank handles, and the
process-local primitives (``Perf.enable_process`` / ``step_process`` / ``dump_process``
/ ``disable_process``) run in whatever process calls them and own the whole
recorder+profiler protocol. The only Ray crossing is :func:`twinkle.infra.remote_class`
wrapping those primitives so they execute on every rank -- perf itself stays
standalone and reusable without it.
"""

from .control import Perf
from .counters import CounterRegistry, default_counters
from .module_profiler import ModuleProfiler, activation_stats, build_module_profiler, param_stats
from .recorder import TimeRecorder, default_recorder, disable_perf, enable_perf, span, timed
from .stats import describe_shape, primary_tensor, tensor_moments, tensor_numel
from .timeline import chrome_trace, dag_report

__all__ = [
    'Perf',
    'TimeRecorder',
    'default_recorder',
    'timed',
    'span',
    'enable_perf',
    'disable_perf',
    'CounterRegistry',
    'default_counters',
    'ModuleProfiler',
    'activation_stats',
    'param_stats',
    'build_module_profiler',
    'chrome_trace',
    'dag_report',
    'primary_tensor',
    'tensor_moments',
    'tensor_numel',
    'describe_shape',
]
