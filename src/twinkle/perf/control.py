# Copyright (c) ModelScope Contributors. All rights reserved.
"""The control plane for :mod:`twinkle.perf`: one :class:`Perf` entry point over the recorder + profiler.

``Perf.initialize`` is the single opt-in call a run makes -- the perf counterpart to ``twinkle.initialize``.
Instrumentation lives at DEFINITION sites: twinkle/swift components carry ``@timed`` on their time-significant
methods, and every ``@remote_class`` actor is injected with a generic ``perf_on`` / ``perf_step`` / ``perf_dump``
/ ``perf_off`` control set (see :mod:`twinkle.infra`). Because ``@timed`` records into the process-local
``default_recorder`` and a remote method's body runs in the WORKER under Ray, timing lands on the rank that
actually executed -- per-worker, with no driver-side RPC overhead folded in. So ``initialize`` has only to turn
each actor's recorder on; the decorated methods then produce per-rank stage timings (and, where a model lives,
module stats) with no further per-loop code.

Aggregation rides the same injected handles: :meth:`Perf.collect_step_summary` fans ``perf_dump`` out to every
registered actor (``dispatch='all'``), reads each rank's flat summary back, and merges them under
``perf/<label>/rank<k>/`` so parallel imbalance stays visible instead of collapsing. Configuration is
explicit-kwargs-first with twinkle's ``TWINKLE_PERF*`` env contract as the fallback. This module imports neither
torch nor the model layer -- the only Ray crossing lives on the injected ``@remote_function`` handles, which it
duck-types.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .recorder import default_recorder, disable_perf, enable_perf

__all__ = ['Perf']

#: ``label -> actor handle``, filled by :meth:`Perf.initialize` and drained by :meth:`Perf.shutdown`. A module
#: global (like the recorder it drives): the actors are process-wide and the dev loops tick / collect without
#: threading the handles through, so there is no per-object state to hold.
_REGISTRY: Dict[str, Any] = {}

#: Namespace every perf key lives under; the per-actor merge swaps it for ``perf/<label>/``.
_PREFIX = 'perf/'


class Perf:
    """Process-wide performance profiling, driven by ONE :meth:`initialize` call.

    A namespace of static methods rather than an instance: the recorder, the module profiler and the actor
    registry are all process globals, so there is no per-object state to hold. Grouping the control surface
    here keeps "turn profiling on", "tick a step", "read a step's summary" and "turn it off" behind a single
    name instead of loose module-level functions a caller has to know the spelling of.
    """

    @staticmethod
    def initialize(actors: Any, *, enabled: Optional[bool] = None,
                   profiler: Optional[Dict[str, Any]] = None) -> bool:
        """Enable per-rank timing (and optionally the module profiler) across ``actors``; return whether perf is on.

        Args:
            actors: the components to profile -- a ``dict`` of ``label -> handle`` (preferred: the label
                namespaces its metrics), a single handle, or a list of handles (labelled by class name). Each
                handle is a ``@remote_class`` actor, so its injected ``perf_on`` reaches all of its worker ranks.
            enabled: turn timing on/off. ``None`` (the default) reads twinkle's ``TWINKLE_PERF`` env flag, so
                ``Perf.initialize(actors)`` is the zero-config path.
            profiler: the serializable module-profiler spec (``select`` / ``collectors`` must be
                strings, since it crosses the Ray boundary). This is an INDEPENDENT switch from
                ``enabled``: the forward hooks force device syncs and cost more than the timing
                decorators, so they stay OFF unless a spec is given here or via
                ``TWINKLE_PERF_SPEC`` / ``TWINKLE_PERF_EVERY_N``.

        Idempotent per actor: an already-registered handle is skipped, so a loop may call this at the top of every
        fit (e.g. across a resume) and profile the FIRST step rather than losing it to a lazy attach, and may
        register further actors later without disturbing the ones already live.
        """
        if enabled is None:
            enabled = Perf._enabled_from_env()
        if not enabled:
            return False
        if profiler is None:
            profiler = Perf._profiler_from_env()
        for label, handle in Perf._normalize_actors(actors).items():
            if _REGISTRY.get(label) is handle:
                continue
            handle.perf_on(profiler_spec=profiler)
            _REGISTRY[label] = handle
        if not default_recorder.enabled:
            enable_perf()
        return True

    @staticmethod
    def advance_step() -> None:
        """Tick every registered actor's profiler step counter (once per optimizer step).

        Non-blocking: ``perf_step`` runs on the same actor as the next ``forward_backward``, and Ray's per-actor
        FIFO ordering lands the increment before that forward reads the step, so no cross-actor sync -- and no
        per-step ``ray.get`` -- is needed. The recorder needs no tick; only the step-gated module profiler does,
        and ``perf_step`` is a no-op on actors that have no profiler attached.
        """
        for handle in _REGISTRY.values():
            handle.perf_step()

    @staticmethod
    def collect_step_summary() -> Dict[str, float]:
        """Return this window's merged per-rank perf dict (``{}`` when perf is off).

        Self-gating on perf being live, so a caller needs no separate flag: when off this returns ``{}`` and
        leaves the step record untouched. Otherwise it snapshots + resets the driver's own recorder (spans from
        ``@timed`` on driver-resident code, labelled ``driver``), then fans ``perf_dump`` out to every registered
        actor -- each rank returns its flat ``perf/stage/*`` (timings + bubble, and net-of-overhead wall where a
        profiler ran) and, where a model lives, ``perf/module/*`` (activation/param stats + overhead). Called on
        the logging cadence, so each summary is the window's aggregate.

        Under local mode there is a single process, so the ``driver`` read drains the one shared recorder and the
        actor dumps come back empty -- per-actor / per-rank separation is a Ray / torchrun property, and local
        mode honestly collapses to the single ``driver`` label.
        """
        if not default_recorder.enabled and not _REGISTRY:
            return {}
        driver_flat = Perf.dump_process()
        metrics = Perf.merge_actor_summary('driver', [driver_flat])
        for label, handle in _REGISTRY.items():
            metrics.update(Perf.merge_actor_summary(label, handle.perf_dump()))
        return metrics

    @staticmethod
    def shutdown() -> None:
        """Turn perf off across every registered actor, drop the profiler hooks, and clear the registry."""
        for handle in _REGISTRY.values():
            handle.perf_off()
        _REGISTRY.clear()
        disable_perf()

    # ------------------------------------------------------------------ process-local primitives
    # The four methods below run INSIDE one process -- the driver under local mode, a worker rank under
    # Ray -- and touch only that process's ``default_recorder`` and the profiler handed to them. They know
    # nothing about Ray or ``@remote_class``: ``@remote_class`` merely wraps them so they execute on every
    # rank (see :mod:`twinkle.infra`). Keeping the recorder+profiler orchestration and the ``perf/stage/*``,
    # ``perf/module/*`` key protocol here -- not in the transport layer -- is what makes perf a standalone
    # component: any caller can profile its own process/model with these, with no ``@remote_class`` at all.

    @staticmethod
    def enable_process(*, model: Any = None, profiler_spec: Optional[Dict[str, Any]] = None) -> Any:
        """Turn on this process's recorder; attach a module profiler to ``model`` when a spec is given.

        Returns the attached :class:`ModuleProfiler` (or ``None``) for the caller to keep and pass back to
        :meth:`step_process` / :meth:`dump_process` / :meth:`disable_process`. The profiler is returned rather
        than stashed in a process global because it is per-model state, and one process may host several
        models (local mode runs a policy and its reference side by side).
        """
        enable_perf()
        if profiler_spec is not None and model is not None:
            from .module_profiler import build_module_profiler
            return build_module_profiler(profiler_spec).attach(model)
        return None

    @staticmethod
    def step_process(profiler: Any = None) -> None:
        """Advance ``profiler``'s step counter (a no-op when no profiler is attached)."""
        if profiler is not None:
            profiler.advance_step()

    @staticmethod
    def dump_process(profiler: Any = None) -> Dict[str, float]:
        """Snapshot + reset this process's recorder into the flat ``perf/*`` dict.

        The single source of the perf key protocol: stage timings under ``perf/stage/`` plus the window's
        ``_bubble_s``; when a ``profiler`` is given, its module stats under ``perf/module/``, its ``_overhead_s``,
        and ``perf/stage/_net_wall_s`` (wall minus profiling overhead, so deep stats never inflate the very
        timing they exist to explain). Called both by the driver for its own recorder and by every rank's
        injected ``perf_dump``, so the two never drift.

        Both accumulators are RESET after the read, so each call returns exactly the window since the last
        one: the recorder (stage timings) and the profiler (module stats + overhead accounts) advance in lockstep.
        Without the profiler reset ``overhead_s`` would grow cumulatively while ``wall_s`` stays per-window,
        driving ``_net_wall_s`` to zero after a few windows and reporting lifetime means instead of the window's.
        """
        out = default_recorder.flat_summary(prefix=f'{_PREFIX}stage/')
        wall_s = out.get(f'{_PREFIX}stage/wall_s', 0.0)
        out[f'{_PREFIX}stage/_bubble_s'] = default_recorder.bubble_s()
        default_recorder.reset()
        if profiler is not None:
            overhead_s = profiler.overhead_s()
            out.update(profiler.flat_summary(prefix=f'{_PREFIX}module/'))
            out[f'{_PREFIX}module/_overhead_s'] = overhead_s
            out[f'{_PREFIX}stage/_net_wall_s'] = max(0.0, wall_s - overhead_s)
            profiler.reset()
        return out

    @staticmethod
    def disable_process(profiler: Any = None) -> None:
        """Detach ``profiler`` (if any) and turn off this process's recorder."""
        if profiler is not None:
            profiler.detach()
        disable_perf()

    @staticmethod
    def merge_actor_summary(label: str, summaries: Any, *, prefix: str = _PREFIX) -> Dict[str, float]:
        """Namespace one actor's per-rank flat summaries under ``perf/<label>/rank<k>/`` (pure).

        ``summaries`` is the ``collect='none'`` result of ``perf_dump``: one flat dict per rank, or a bare dict
        when the actor has a single rank. With more than one rank the keys are tagged ``rank<k>/`` so parallel
        imbalance stays visible; a single rank keeps the untagged form. Each worker/driver key already starts with
        ``perf/`` (``perf/stage/...``, ``perf/module/...``); that leading ``perf/`` is swapped for ``perf/<label>/``
        so different actors' metrics never collide. Every value is coerced to float so the dict merges straight
        into a step-metrics record.
        """
        if not isinstance(summaries, list):
            summaries = [summaries]
        multi_rank = len(summaries) > 1
        metrics: Dict[str, float] = {}
        for rank, summary in enumerate(summaries):
            rank_segment = f'rank{rank}/' if multi_rank else ''
            for key, value in (summary or {}).items():
                rest = key[len(prefix):] if key.startswith(prefix) else key
                metrics[f'{prefix}{label}/{rank_segment}{rest}'] = float(value)
        return metrics

    @staticmethod
    def _normalize_actors(actors: Any) -> Dict[str, Any]:
        """Coerce the ``actors`` argument to a ``label -> handle`` mapping.

        A dict is used as given (its keys are the labels); a single handle or a list is labelled by class name.
        Labels namespace the merged metrics, so a dict with meaningful keys is the readable choice when a run
        holds several actors of the same class (e.g. a policy and a reference model).
        """
        if actors is None:
            return {}
        if isinstance(actors, dict):
            return dict(actors)
        if isinstance(actors, (list, tuple)):
            return {type(handle).__name__: handle for handle in actors}
        return {type(actors).__name__: actors}

    @staticmethod
    def _enabled_from_env() -> bool:
        """The opt-in gate from twinkle's ``TWINKLE_PERF`` env flag (the fallback for ``enabled=None``)."""
        return os.environ.get('TWINKLE_PERF', '').strip().lower() in ('1', 'true', 'yes', 'on')

    @staticmethod
    def _profiler_from_env() -> Optional[Dict[str, Any]]:
        """Build the module-profiler spec from env, or ``None`` to leave the forward hooks OFF.

        ``None`` is the default (timing only): the hook profiler forces device syncs, so it is opt-in and
        independent of ``TWINKLE_PERF``. ``TWINKLE_PERF_SPEC`` (a JSON object) wins and is forwarded verbatim to
        ``build_module_profiler`` -- ``select`` / ``collectors`` must be strings there, since the spec crosses the
        Ray boundary, e.g. ``'{"every_n_steps": 20, "collectors": ["activation", "param"]}'``. Otherwise
        ``TWINKLE_PERF_EVERY_N`` alone turns the profiler on at that cadence (hooking every module with the
        built-in activation stats).
        """
        raw = os.environ.get('TWINKLE_PERF_SPEC', '').strip()
        if raw:
            spec = json.loads(raw)
            if not isinstance(spec, dict):
                raise ValueError('TWINKLE_PERF_SPEC must be a JSON object, e.g. \'{"every_n_steps": 20}\'.')
            return spec
        every_n = os.environ.get('TWINKLE_PERF_EVERY_N', '').strip()
        if every_n:
            return {'every_n_steps': int(every_n)}
        return None
