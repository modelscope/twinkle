# Copyright (c) ModelScope Contributors. All rights reserved.
"""twinkle.perf -- a general-purpose performance profiling component.

Two independent, low-intrusion parts:

* :class:`TimeRecorder` / :func:`timed` / :func:`span` -- a synchronous- and
  async-agnostic wall-clock span recorder that nests into an overall timing tree
  and summarizes to a printable report or a metric-ready ``dict[str, float]``.
  Pure Python (no torch), so it can time ANY callable.
* :class:`ModuleProfiler` / :class:`ActivationStatsProfiler` -- a forward-hook
  profiler that gates by step, selects target modules, and leaves *what to
  measure* open via the :meth:`ModuleProfiler.collect` subclass hook. Attaches to
  any ``nn.Module`` subclass.

Both are disabled/no-op by default, so importing perf never changes existing
behaviour. Neither part imports twinkle/swift business modules; the only Ray
crossing (attaching to a worker-resident model) lives on the model base class as
``@remote_function`` handles, not here.
"""

from .module_profiler import ActivationStatsProfiler, ModuleProfiler, build_module_profiler
from .recorder import TimeRecorder, default_recorder, disable_perf, enable_perf, span, timed
from .stats import describe_shape, primary_tensor, tensor_moments, tensor_numel

__all__ = [
    'TimeRecorder',
    'default_recorder',
    'timed',
    'span',
    'enable_perf',
    'disable_perf',
    'ModuleProfiler',
    'ActivationStatsProfiler',
    'build_module_profiler',
    'primary_tensor',
    'tensor_moments',
    'tensor_numel',
    'describe_shape',
]
