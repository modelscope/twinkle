# Copyright (c) ModelScope Contributors. All rights reserved.
"""Forward-hook module profiler.

The base class fixes the *when* (step gating), the *who* (module selection) and
the *how* (hook registration / removal); the only thing a subclass configures is
*what to measure*, by overriding :meth:`ModuleProfiler.collect`. This keeps the
intrusive part (touching ``nn.Module`` forward) in one reviewed place while the
statistics stay open for extension.

Only ``torch.nn`` is depended on, so the profiler attaches to ANY ``nn.Module``
subclass, not just twinkle/transformers models. Collection failures are downgraded
to warnings and NEVER alter or interrupt the wrapped forward.
"""

from __future__ import annotations

import functools
import importlib
import re
import sys
import threading
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.nn as nn

from .stats import describe_shape, primary_tensor, tensor_moments, tensor_numel

__all__ = ['ModuleProfiler', 'ActivationStatsProfiler', 'build_module_profiler']

_NS_PER_MS = 1e6

#: A matcher deciding whether a module (by qualified name) is profiled.
Matcher = Callable[[str, nn.Module], bool]


def _resolve_dotted(spec: Any) -> Any:
    """Resolve ``'attr'`` (this module) or ``'pkg.mod:attr'`` (import) to an object.

    Strings are what can cross a Ray boundary; a bare name is looked up in this
    module (so built-in profilers are addressable by name), a ``module:attr`` form
    is imported (so user-defined subclasses/predicates are addressable). Fails
    loudly rather than silently falling back.
    """
    if not isinstance(spec, str):
        return spec
    if ':' in spec:
        module_name, _, attr = spec.partition(':')
        module = importlib.import_module(module_name)
    else:
        module = sys.modules[__name__]
        attr = spec
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ValueError(f'cannot resolve {spec!r}: module {module.__name__!r} has no attribute {attr!r}.') from None


def _build_matcher(select: Any) -> Matcher:
    """Turn a ``select`` spec into a ``(name, module) -> bool`` matcher.

    Accepted forms (string forms are the only ones that survive a Ray boundary):
      - ``None`` / ``'all'``            -> every module;
      - ``'type:ClassName'``            -> ``module.__class__.__name__ == ClassName``;
      - ``'pkg.mod:callable'``          -> an imported predicate ``(name, module) -> bool``;
      - any other string               -> regex ``re.search`` against the module name;
      - a class (local only)           -> ``isinstance``;
      - a list/tuple/set of names      -> exact name membership;
      - a callable (local only)        -> used directly.
    """
    if select is None or select == 'all':
        return lambda name, module: True
    if isinstance(select, str):
        if select.startswith('type:'):
            class_name = select[len('type:'):]
            return lambda name, module: module.__class__.__name__ == class_name
        if ':' in select:
            predicate = _resolve_dotted(select)
            if not callable(predicate):
                raise TypeError(f'select {select!r} did not resolve to a callable predicate.')
            return predicate
        pattern = re.compile(select)
        return lambda name, module: bool(pattern.search(name))
    if isinstance(select, type):
        return lambda name, module: isinstance(module, select)
    if isinstance(select, (list, tuple, set, frozenset)):
        names = set(select)
        return lambda name, module: name in names
    if callable(select):
        return select
    raise TypeError(f'unsupported select spec: {select!r}.')


class ModuleProfiler:
    """Profile selected modules' forward passes on gated steps.

    Args:
        every_n_steps: collect only when ``step % every_n_steps == 0``; other steps
            run a near-zero-cost pass-through (no tensor work).
        select: which modules to hook; see :func:`_build_matcher`.
        enabled: when False, no hooks are registered at all (``attach`` stores the
            model but registers nothing; ``enable`` registers lazily).
    """

    def __init__(self, *, every_n_steps: int = 1, select: Any = 'all', enabled: bool = True) -> None:
        if every_n_steps < 1:
            raise ValueError('every_n_steps must be >= 1.')
        self.every_n_steps = int(every_n_steps)
        self._matcher: Matcher = _build_matcher(select)
        self._enabled = enabled
        self._step = 0
        # A model may be a single module or a list of chunks (e.g. Megatron VPP), so
        # hooks are registered across every module in the list.
        self._models: Optional[List[nn.Module]] = None
        self._handles: List[Any] = []
        # id(module) -> stack of start_ns, so re-entrant forwards of one module nest correctly.
        self._open: Dict[int, List[int]] = {}
        self._lock = threading.Lock()
        # name -> metric_key -> [count, sum, min, max]
        self._agg: Dict[str, Dict[str, list]] = {}
        self._counts: Dict[str, int] = {}
        self._shapes: Dict[str, str] = {}

    # ------------------------------------------------------------------ control

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Turn collection on, registering hooks lazily if a model is attached."""
        self._enabled = True
        if self._models is not None and not self._handles:
            self._register()

    def disable(self) -> None:
        """Turn collection off and remove all hooks (truly zero hook overhead)."""
        self._enabled = False
        self._remove_handles()

    def advance_step(self) -> None:
        """Advance the step counter; the training loop calls this once per step.

        Also clears the re-entrancy stack: a step boundary means no forward is in
        flight, so any entry left there (a pre/post gate flip from an out-of-order or
        repeated ``advance_step``) is stale. Bounding ``_open`` to a single step keeps
        a misordered call from stranding start timestamps and skewing later times.
        """
        self._step += 1
        if self._open:
            self._open.clear()

    def attach(self, model: Any) -> 'ModuleProfiler':
        """Register forward hooks on the selected modules of ``model``.

        ``model`` is a single ``nn.Module`` or a list/tuple of them (Megatron keeps
        the model as a list of VPP chunks). Re-attaching detaches first, so the
        profiler never double-registers.
        """
        self.detach()
        self._models = list(model) if isinstance(model, (list, tuple)) else [model]
        if self._enabled:
            self._register()
        return self

    def detach(self) -> None:
        """Remove every hook and forget the model; safe to call repeatedly."""
        self._remove_handles()
        self._models = None
        self._open.clear()

    def _register(self) -> None:
        for model in self._models:
            for name, module in model.named_modules():
                if not self._matches(name, module):
                    continue
                self._handles.append(
                    module.register_forward_pre_hook(functools.partial(self._pre_hook, name), with_kwargs=True))
                self._handles.append(
                    module.register_forward_hook(functools.partial(self._post_hook, name), with_kwargs=True))

    def _remove_handles(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _matches(self, name: str, module: nn.Module) -> bool:
        # A user matcher that raises must not break attach; skip that module.
        try:
            return bool(self._matcher(name, module))
        except Exception as exc:  # observation failure, not a business error
            warnings.warn(f'[twinkle.perf] select matcher failed for {name!r}; skipping: {exc}', RuntimeWarning)
            return False

    def _collecting(self) -> bool:
        return self._enabled and (self._step % self.every_n_steps == 0)

    # ------------------------------------------------------------------ hooks

    def _pre_hook(self, name: str, module: nn.Module, args: Any, kwargs: Any) -> None:
        if not self._collecting():
            return None
        self._open.setdefault(id(module), []).append(time.perf_counter_ns())
        return None

    def _post_hook(self, name: str, module: nn.Module, args: Any, kwargs: Any, output: Any) -> None:
        # Always return None: a forward hook's return value would REPLACE the module
        # output, and profiling must never alter the forward result.
        if not self._collecting():
            return None
        forward_ms = self._pop_elapsed_ms(module)
        try:
            stats = self.collect(name, module, args, kwargs, output)
            shape = describe_shape(output)
        except Exception as exc:  # observation failure, not a business error
            warnings.warn(f'[twinkle.perf] collect failed for {name!r}; skipping stats: {exc}', RuntimeWarning)
            return None
        self._accumulate(name, forward_ms, stats, shape)
        return None

    def _pop_elapsed_ms(self, module: nn.Module) -> Optional[float]:
        stack = self._open.get(id(module))
        if not stack:
            return None
        start_ns = stack.pop()
        if not stack:
            self._open.pop(id(module), None)
        return (time.perf_counter_ns() - start_ns) / _NS_PER_MS

    def collect(self, name: str, module: nn.Module, args: Any, kwargs: Any, output: Any) -> Dict[str, float]:
        """Extension point: return the ``{key: float}`` stats to record for this module.

        The base records only the forward duration (measured by the hook pair);
        subclasses override this to add activation statistics, norms, ranks, etc.
        """
        return {}

    # ------------------------------------------------------------------ aggregation

    def _accumulate(self, name: str, forward_ms: Optional[float], stats: Dict[str, float], shape: str) -> None:
        with self._lock:
            per_key = self._agg.setdefault(name, {})

            def add(key: str, value: float) -> None:
                cell = per_key.get(key)
                if cell is None:
                    cell = [0, 0.0, None, None]
                    per_key[key] = cell
                cell[0] += 1
                cell[1] += value
                cell[2] = value if cell[2] is None else min(cell[2], value)
                cell[3] = value if cell[3] is None else max(cell[3], value)

            if forward_ms is not None:
                add('forward_ms', forward_ms)
            for key, value in stats.items():
                add(key, float(value))
            self._counts[name] = self._counts.get(name, 0) + 1
            self._shapes[name] = shape

    def reset(self) -> None:
        """Drop accumulated stats (keeps the step counter and attached hooks)."""
        with self._lock:
            self._agg.clear()
            self._counts.clear()
            self._shapes.clear()

    # ------------------------------------------------------------------ output

    def summary(self) -> Dict[str, Any]:
        """Structured per-module aggregate: count, last shape, and per-key mean/min/max."""
        with self._lock:
            modules = {}
            for name, count in self._counts.items():
                stats = {}
                for key, (n, total, lo, hi) in self._agg.get(name, {}).items():
                    stats[key] = {'mean': (total / n) if n else 0.0, 'min': lo, 'max': hi}
                modules[name] = {'count': count, 'shape': self._shapes.get(name, '-'), 'stats': stats}
        return {
            'enabled': self._enabled,
            'every_n_steps': self.every_n_steps,
            'step': self._step,
            'modules': modules,
        }

    def flat_summary(self, prefix: str = 'perf/module/') -> Dict[str, float]:
        """Flatten to ``dict[str, float]`` (per-module count + per-key mean) for metric merge."""
        out: Dict[str, float] = {}
        with self._lock:
            counts = dict(self._counts)
            agg = {name: dict(keys) for name, keys in self._agg.items()}
        for name, count in counts.items():
            out[f'{prefix}{name}/count'] = float(count)
            for key, (n, total, _lo, _hi) in agg.get(name, {}).items():
                out[f'{prefix}{name}/{key}/mean'] = (total / n) if n else 0.0
        return out

    def print_summary(self, *, sort: str = 'forward_ms', top: Optional[int] = None) -> None:
        """Print a per-module table sorted by the mean of ``sort`` (default forward time)."""
        data = self.summary()
        print(f'twinkle module profiler (enabled={data["enabled"]}, every_n_steps={data["every_n_steps"]}, '
              f'step={data["step"]})')

        def sort_key(item: Tuple[str, dict]) -> float:
            return -item[1]['stats'].get(sort, {}).get('mean', 0.0)

        rows = sorted(data['modules'].items(), key=sort_key)
        if top is not None:
            rows = rows[:top]
        print(f'{"module":<48}{"count":>7}{"fwd_ms":>10}{"out_l2":>12}{"shape":>16}')
        for name, info in rows:
            stats = info['stats']
            fwd = stats.get('forward_ms', {}).get('mean', 0.0)
            l2 = stats.get('out_l2', {}).get('mean', 0.0)
            print(f'{name:<48}{info["count"]:>7}{fwd:>10.3f}{l2:>12.4f}{info["shape"]:>16}')


class ActivationStatsProfiler(ModuleProfiler):
    """Default subclass: forward time (from the base) plus input/output activation stats."""

    def collect(self, name: str, module: nn.Module, args: Any, kwargs: Any, output: Any) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        out_tensor = primary_tensor(output)
        if out_tensor is not None:
            for key, value in tensor_moments(out_tensor).items():
                stats[f'out_{key}'] = value
        stats['out_numel'] = tensor_numel(output)
        stats['in_numel'] = tensor_numel(args[0]) if args else 0.0
        return stats


def build_module_profiler(spec: Optional[Dict[str, Any]]) -> ModuleProfiler:
    """Construct a profiler from a serializable spec dict (the Ray-boundary contract).

    ``spec['profiler']`` is a class name or ``'pkg.mod:Class'`` import string
    (default ``'ActivationStatsProfiler'``); every other key is forwarded to the
    constructor (``every_n_steps`` / ``select`` / ``enabled``). Kept free of live
    objects so it can be passed from a driver to a worker.
    """
    spec = dict(spec or {})
    profiler_spec = spec.pop('profiler', 'ActivationStatsProfiler')
    cls = _resolve_dotted(profiler_spec)
    if not (isinstance(cls, type) and issubclass(cls, ModuleProfiler)):
        raise TypeError(f'profiler {profiler_spec!r} did not resolve to a ModuleProfiler subclass.')
    return cls(**spec)
