# Copyright (c) ModelScope Contributors. All rights reserved.
"""Algorithm-agnostic per-sample streaming control flow for online RL.

:class:`StreamingDriver` replaces the retired batch-granular deep-buffer skeleton with a per-sample one:
trajectories are
admitted ONE AT A TIME (each an independent sampler submission), completions are polled as-completed
into an in-process ready buffer, and the consumer trains WHATEVER IS READY instead of waiting for a
fixed batch to finish together. A slot freed by a completion is immediately backfilled from the prompt
stream, so the engine stays loaded and no straggler tail idles it -- at every staleness setting.

sync and async are not separate code paths here: they are points on the ``max_staleness`` knob of the
injected :class:`~twinkle_agentic.async_rl.context_manager.RLContextManager` (0 = synchronous: the
admission window closes while its samples train; 1 = one-step overlap; >1 = deep buffer). The driver
never branches on the regime.

Control-plane mapping (the load-bearing design decision):

* ONE PARTITION = ONE PUBLISH CYCLE = ``parameter_sync_step`` optimizer steps. A partition does NOT
  declare fixed members -- it is an admission WINDOW opened via ``request_rollout_partition`` (which is
  where the staleness gate lives: at most ``max_staleness + 1`` windows live at once) and closed by the
  publish that ends its cycle. Samples flow through the shared ready buffer in completion order, so a
  straggler never blocks a window: it lands in a later one and, if by then its version lag exceeds
  ``max_staleness``, the post-publish stale scan drops it.
* The POLICY VERSION bumps once per publish (``on_partition_trained``), never per optimizer step, so a
  sample's lag (``current_version - record.version``) is measured in publish cycles under
  ``parameter_sync_step = K`` exactly as ``max_staleness`` is defined. Windows are published oldest-live
  first, which satisfies ``on_partition_training_started``'s FIFO requirement because windows open (and
  therefore age) in step order.
* Backpressure is a sample-count cap, ``buffer_depth``: admission pauses while
  ``in_flight + ready >= buffer_depth``. The consumer sizes the cap so one window's fill
  (``groups_per_partition * num_generations``) is at least one full training pull -- otherwise the
  driver could deadlock with a buffer too shallow to assemble a batch while admission is capped.

Injected seams (the external contract a consumer wires up; the driver knows no algorithm):

* ``submit(prompt_idx, trajectory_idx, policy) -> handle``: admit ONE trajectory without blocking,
  pinned to ``policy`` (its ``adapter_path`` travels with the submission under snapshot publication;
  ``None`` under in-place).
* ``poll(handles) -> completed_handles``: non-blocking as-completed query over the in-flight handles
  (a subset of ``handles``; unknown values are ignored).
* ``collect(handle) -> sample``: fetch one completed trajectory's training sample.
* ``assembly_ready(buffer) -> Optional[List[BufferRecord]]``: the algorithm-specific pull rule --
  inspect the ready buffer and, when it holds at least one full training unit (GRPO/RFT: whole groups,
  since every sibling of a group is admitted together under one policy and therefore shares ONE
  version, so a group is complete or untrainable; PPO/GKD: individual samples), remove and return
  exactly the records of one batch (an exact multiple of the consumer's optimizer-step batch, so the
  exact-global-batch invariant holds); ``None`` when not ready. Records left over stay buffered.
* ``consume(records) -> Optional[int]``: score and train the pulled batch (the existing
  ``_train_rollout_batch`` half of a dev loop). Returns the number of OPTIMIZER STEPS it ran (a falsy
  or ``None`` return counts as one), which drives the publish cadence.
* ``cancel(handle)``: drop an in-flight generation (budget hit, drain, or error).
* ``publish``: the injected :class:`~twinkle_agentic.async_rl.weight_sync.WeightSyncStrategy`;
  ``publish(admission)`` returns the new version's pinning handle (a path, or ``None`` for in-place).
* ``group_key(prompt_idx) -> str`` (optional): the training-group identifier stamped on every
  trajectory of one prompt (default ``str(prompt_idx)``).
* ``prune()`` (optional): drop published artifacts no live version and no in-flight pin reference.
* ``initial_adapter_path()`` (optional): the version-0 pin (the pre-training adapter path for
  snapshot publication, ``None`` for in-place).
* ``reached_max()`` (optional): whether the optimizer-step budget is exhausted.
* ``sleep(seconds)`` (optional): idle wait injected for testability; defaults to ``time.sleep``.

Staleness enforcement: a record's version lag is bounded at TWO points, both counted in
``stats.dropped_stale`` (a dropped record's pin was already released at collect). (1) AT COLLECT: a
straggler can stay in flight across more than ``max_staleness`` publishes -- the post-publish scan never
sees the in-flight set -- so a completion is dropped the moment it is fetched if its lag already exceeds
the bound. This is what keeps a late arrival from ENTERING the buffer untrainable and being consumed
stale before the next scan could catch it. (2) AFTER EVERY PUBLISH: the buffer is scanned and each record
whose lag now exceeds ``max_staleness`` is dropped, covering records that aged past the bound while
buffered. Dropping is per-record but sound for group consumers: siblings share one version (admitted
back-to-back under one policy), so a stale group is dropped whole -- at collect and/or across a scan --
and ``assembly_ready`` never pulls an incomplete group, so no unusable record survives past one publish
cycle.

Drain: when the prompt stream is exhausted, admission stops, in-flight generations are polled to
completion (blocking via ``sleep`` between polls), the consumer keeps training the ready buffer until
``assembly_ready`` declines, and then whatever remains (an incomplete group, or a remainder below one
full batch) is dropped and counted -- there is no future batch to carry it into. Every window still
live is cleared. A budget hit (``reached_max``) or an error instead cancels every STILL-in-flight
submission (releasing its pin), and clears every window in the ``finally``, leaking nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .context_manager import RLContextManager
from .types import PartitionAdmission, RLContext, RolloutPolicy
from .weight_sync import WeightSyncStrategy

#: One in-flight trajectory: the sampler handle to poll/collect it by, the policy it was pinned to
#: (released once collected or cancelled, so its adapter path can then be pruned), and the training
#: group key it will carry into the ready buffer (every trajectory of one prompt shares it, see
#: :class:`BufferRecord`).
_Inflight = Tuple[Any, RolloutPolicy, str]


@dataclass(eq=False)
class BufferRecord:
    """One completed trajectory sitting in the driver's ready buffer.

    The unit ``assembly_ready`` inspects and pulls. ``version`` is the policy version the trajectory
    was admitted under (its lag vs the current version is what the post-publish stale scan and the
    consumer's metrics read); ``group_key`` identifies the training group the sample belongs to --
    every trajectory of one prompt EXPANSION shares it (all are admitted back-to-back under one policy,
    so all carry the same ``version``), which is what lets a group consumer require whole groups and the
    stale scan treat a stale group as dropped whole. ``sample`` is opaque to the driver (a dev
    ``RolloutSample``).

    A record carries NO pin: its version pin was acquired at admission and released the moment the
    trajectory was COLLECTED (see :meth:`StreamingDriver.run`), because the pin guards the SAMPLER-side
    adapter path and a collected sample is trained in-process. So leaving the buffer (consumed, stale-
    dropped or drained) is pure bookkeeping -- releasing again there would double-release and raise.
    """
    sample: Any
    version: int
    group_key: str


@dataclass
class DriverStats:
    """Lifetime counters of one :meth:`StreamingDriver.run` (the consumer's metrics source)."""
    admitted: int = 0               # trajectories submitted to the sampler
    collected: int = 0              # completions fetched from the sampler (a stale one is then dropped)
    consumed: int = 0               # buffered samples handed to ``consume``
    off_policy_consumed: int = 0    # consumed samples whose version lagged the training version
    dropped_stale: int = 0          # samples dropped for exceeding max_staleness (at collect or by a scan)
    dropped_incomplete: int = 0     # samples left in the buffer at drain (no future batch to join)
    train_pulls: int = 0            # ``consume`` calls that returned a positive step count
    optimizer_steps: int = 0        # optimizer steps reported by ``consume``
    publishes: int = 0              # weight publications (partition cycles closed)
    final_version: int = 0          # policy version at exit (= publishes)
    #: Wall-clock seconds the driver thread spent blocked inside ``consume`` (training). The generation
    #: itself runs on the (disaggregated) sampler in the background and is NOT measured here -- the
    #: algorithm-agnostic driver cannot see sampler GPU time without instrumenting the data plane.
    train_active_time: float = 0.0
    #: Wall-clock seconds spent in the idle ``sleep`` branch: trajectories in flight, none completed this
    #: pass, buffer not yet trainable -- i.e. the trainer starved for ready work (the trainer bubble).
    idle_time: float = 0.0
    #: Wall-clock seconds for the whole :meth:`StreamingDriver.run`, set in its ``finally`` (so it is
    #: defined on every exit path: drain, budget hit, or error). ``total >= train_active + idle``; the
    #: residual is the driver's own admit/poll/collect/publish overhead.
    total_time: float = 0.0


def _no_prune() -> None:
    """Default ``prune`` seam: in-place publication keeps no per-version artifacts to drop."""


def _never_max() -> bool:
    """Default ``reached_max`` seam: no optimizer-step budget, run until the prompt stream drains."""


def _no_initial_path() -> None:
    """Default ``initial_adapter_path`` seam: in-place has no version-0 adapter path to pin."""


class StreamingDriver:
    """Per-sample streaming control flow over an injected control plane and publication strategy.

    Construct with the control-plane objects (``ctx_mgr`` / ``ctx``), the publication strategy
    (``weight_sync``), the prompt stream (``prompt_stream``, yielding one prompt identifier at a time),
    the regime knobs (``max_staleness`` -- a mirror of ``ctx_mgr.max_staleness``, validated against it;
    ``num_generations`` trajectories per prompt; ``groups_per_partition`` = the groups one publish
    cycle is expected to train, used as the window's sizing declaration; ``parameter_sync_step`` = the
    publish cadence in optimizer steps; ``buffer_depth`` = the in-flight + ready backpressure cap in
    samples), and the seams. Then call :meth:`run` once; :attr:`stats` holds the lifetime counters
    afterwards.
    """

    def __init__(self,
                 *,
                 ctx_mgr: RLContextManager,
                 ctx: RLContext,
                 weight_sync: WeightSyncStrategy,
                 prompt_stream: Iterable[Any],
                 max_staleness: int,
                 num_generations: int,
                 groups_per_partition: int,
                 parameter_sync_step: int,
                 buffer_depth: int,
                 submit: Callable[[Any, int, RolloutPolicy], Any],
                 poll: Callable[[Sequence[Any]], Sequence[Any]],
                 collect: Callable[[Any], Any],
                 assembly_ready: Callable[[List[BufferRecord]], Optional[List[BufferRecord]]],
                 consume: Callable[[List[BufferRecord]], Optional[int]],
                 cancel: Callable[[Any], None],
                 group_key: Optional[Callable[[Any], str]] = None,
                 prune: Optional[Callable[[], None]] = None,
                 initial_adapter_path: Optional[Callable[[], Optional[str]]] = None,
                 reached_max: Optional[Callable[[], bool]] = None,
                 sleep: Optional[Callable[[float], None]] = None) -> None:
        # Fail loudly on a mis-wired seam: a non-callable injected here would otherwise surface deep
        # inside the run loop (or silently skip a lifecycle step), far from the construction that caused it.
        for name, fn in (('submit', submit), ('poll', poll), ('collect', collect),
                         ('assembly_ready', assembly_ready), ('consume', consume), ('cancel', cancel)):
            if not callable(fn):
                raise TypeError(f'StreamingDriver seam {name!r} must be callable, got {type(fn).__name__}')
        if not isinstance(weight_sync, WeightSyncStrategy):
            raise TypeError(
                f'StreamingDriver weight_sync must be a WeightSyncStrategy, got {type(weight_sync).__name__}')
        if not hasattr(ctx_mgr, 'request_rollout_partition'):
            raise TypeError(f'StreamingDriver ctx_mgr must be an RLContextManager, got {type(ctx_mgr).__name__}')
        if max_staleness < 0:
            raise ValueError(f'max_staleness must be non-negative, got {max_staleness}')
        if ctx_mgr.max_staleness != max_staleness:
            raise ValueError(f'max_staleness={max_staleness} disagrees with the injected control plane '
                             f'(ctx_mgr.max_staleness={ctx_mgr.max_staleness}): the gate lives on the control '
                             f'plane and the stale scan / backpressure sizing here must match it.')
        if num_generations <= 0:
            raise ValueError(f'num_generations must be positive, got {num_generations}')
        if groups_per_partition <= 0:
            raise ValueError(f'groups_per_partition must be positive, got {groups_per_partition}')
        if parameter_sync_step <= 0:
            raise ValueError(f'parameter_sync_step must be positive, got {parameter_sync_step}')
        if buffer_depth <= 0:
            raise ValueError(f'buffer_depth must be positive, got {buffer_depth}')
        self._ctx_mgr = ctx_mgr
        self._ctx = ctx
        self._weight_sync = weight_sync
        self._prompt_stream = prompt_stream
        self._max_staleness = max_staleness
        self._num_generations = num_generations
        self._groups_per_partition = groups_per_partition
        self._parameter_sync_step = parameter_sync_step
        self._buffer_depth = buffer_depth
        self._submit = submit
        self._poll = poll
        self._collect = collect
        self._assembly_ready = assembly_ready
        self._consume = consume
        self._cancel = cancel
        self._group_key = group_key if group_key is not None else str
        self._prune = prune if prune is not None else _no_prune
        self._initial_adapter_path = initial_adapter_path if initial_adapter_path is not None else _no_initial_path
        self._reached_max = reached_max if reached_max is not None else _never_max
        self._sleep = sleep if sleep is not None else time.sleep
        self.stats = DriverStats()
        #: Optimizer steps the most recent ``_train_pull`` consume reported (its return channel).
        self._last_pull_steps = 0

    # --- lifecycle --------------------------------------------------------------------------------------

    def run(self) -> None:
        """Stream the prompt set: admit per-sample, train whatever is ready, publish every K steps.

        Each pass: (1) publish phase -- when the optimizer-step count since the last publish reaches
        ``parameter_sync_step``, close the OLDEST live window (FIFO, as ``on_partition_training_started``
        requires): publish the trained weights through the strategy, bump the tracked version, prune
        unreferenced artifacts, and drop buffered samples whose version lag now exceeds
        ``max_staleness``; (2) admission -- open a fresh window while the staleness gate stays open,
        then admit trajectories one at a time (each prompt expanded into ``num_generations`` of them,
        back-to-back under one pinned policy) until the prompt stream ends or ``buffer_depth`` caps the
        in-flight + ready count; (3) collection -- one non-blocking poll, fetching every completion,
        releasing its pin, and moving it into the ready buffer unless its version lag already exceeds
        ``max_staleness`` (a straggler that aged past the bound while in flight is dropped, not buffered);
        (4) consumption -- when ``assembly_ready`` pulls a
        batch, train it and count the optimizer steps it reports. A pass with in-flight work but no
        completion and no ready batch sleeps briefly instead of spinning.

        Exits when the budget (``reached_max``) is hit, or one pass after the prompt stream is
        exhausted, the in-flight set has drained and ``assembly_ready`` declines the remaining buffer
        (the leftover -- an incomplete group or a sub-batch remainder -- is dropped in the ``finally``,
        counted as ``stats.dropped_incomplete``). The ``finally`` also cancels and unpins whatever is
        still generating and clears every live window, so a budget hit or an error leaks neither a pin
        nor a sampler submission nor a partition.
        """
        ctx_mgr, ctx = self._ctx_mgr, self._ctx
        # Version 0: pin the initial (pre-training) policy so the first admissions reference a real
        # version and the sampler generates from it, never an unpinned base model. None for in-place.
        initial_path = self._initial_adapter_path()
        ctx_mgr.register_context(ctx, adapter_path=initial_path, policy_version=0)
        _run_start = time.perf_counter()
        prompts = iter(self._prompt_stream)
        # The prompt currently being expanded into trajectories: (prompt_idx, trajectory_idx, group_key).
        # A backpressure cap can interrupt the expansion mid-prompt; the cursor resumes it (carrying the
        # SAME group_key) instead of dropping the remaining trajectories. The group_key is namespaced by a
        # per-expansion ordinal (group_seq) so the SAME prompt re-emitted in a later epoch is a DIFFERENT
        # group: its siblings are admitted under a later policy version and must never be assembled,
        # advantage-scored or stale-dropped together with the earlier epoch's group.
        partial: Optional[Tuple[Any, int, str]] = None
        pending: Optional[Any] = next(prompts, None)
        group_seq = 0
        # The in-flight set, keyed by sampler handle -- the SINGLE source of truth for what is still
        # generating. A handle is removed the instant it is collected, so the backpressure count, the
        # drain condition and the teardown below all see the CURRENT in-flight set, never a cumulative one.
        handles: Dict[Any, _Inflight] = {}
        ready: List[BufferRecord] = []
        stream_done = pending is None
        exhausted = False
        steps_in_cycle = 0
        try:
            while True:
                if exhausted or self._reached_max():
                    break

                # (1) Publish phase: close the oldest live window once its cycle's steps are trained.
                live_windows = self._live_windows()
                if steps_in_cycle >= self._parameter_sync_step and live_windows:
                    window = live_windows[0]
                    ctx_mgr.on_partition_training_started(window)
                    new_path = self._weight_sync.publish(window)
                    ctx_mgr.on_partition_trained(window, adapter_path=new_path)
                    ctx_mgr.on_partition_cleared(window)
                    self.stats.publishes += 1
                    steps_in_cycle = 0
                    self._drop_stale(ready)
                    # Drop published artifacts no live version, in-flight pin or buffered sample still
                    # references (runs AFTER the stale scan, which released the dropped pins).
                    self._prune()

                # (2) Admission: open a fresh window while the staleness gate is open, then admit
                #     per-sample until the stream ends or the backpressure cap is reached.
                if not stream_done:
                    self._open_window()
                    capacity = self._buffer_depth - len(handles) - len(ready)
                    while capacity > 0:
                        if partial is None:
                            if pending is None:
                                stream_done = True
                                break
                            # Resolve this expansion's group key ONCE (namespaced by the emission ordinal,
                            # see the cursor comment) so every sibling trajectory shares it and a prompt
                            # re-emitted in a later epoch never collides with its earlier group.
                            group_key = f'{self._group_key(pending)}#{group_seq}'
                            group_seq += 1
                            partial = (pending, 0, group_key)
                            pending = next(prompts, None)
                        prompt_idx, trajectory_idx, group_key = partial
                        policy = ctx_mgr.acquire_rollout_policy(ctx)
                        try:
                            handle = self._submit(prompt_idx, trajectory_idx, policy)
                        except BaseException:
                            # A failed submit must not leak the pin it just acquired.
                            ctx_mgr.release_rollout_policy(policy)
                            raise
                        handles[handle] = (handle, policy, group_key)
                        self.stats.admitted += 1
                        capacity -= 1
                        trajectory_idx += 1
                        partial = ((prompt_idx, trajectory_idx, group_key)
                                   if trajectory_idx < self._num_generations else None)
                    if stream_done:
                        # Bookkeeping only (no partition is requested after this point), but it keeps the
                        # control-plane status honest for an introspecting consumer.
                        ctx_mgr.on_dataset_exhausted(ctx)

                # (3) Collection: one non-blocking poll; every completion is fetched, its pin released,
                #     and -- unless it is already too stale to train -- moved into the ready buffer.
                progressed = False
                if handles:
                    # The version cannot change during collection (a publish is phase (1), never here),
                    # so read it once for the whole poll's staleness decision.
                    current = self._current_version()
                    for handle in self._poll(list(handles)):
                        entry = handles.pop(handle, None)
                        if entry is None:
                            continue  # a handle this driver does not know: ignore, never double-collect
                        _, policy, group_key = entry
                        try:
                            sample = self._collect(handle)
                        finally:
                            # The sampler is done with this trajectory, so its version pin is released
                            # HERE (acquire's contract is "pin while one sampler request is using it"): the
                            # pin guards the SAMPLER-side adapter path against pruning, and the collected
                            # sample is now in-process (training never reloads the behaviour adapter). This
                            # is the pin's ONLY release on the collected path -- releasing again when the
                            # record leaves the buffer would double-release and raise. In a finally so a
                            # collect failure leaks no pin either.
                            ctx_mgr.release_rollout_policy(policy)
                        self.stats.collected += 1
                        progressed = True
                        # A straggler can stay in flight across MORE than ``max_staleness`` publishes -- the
                        # post-publish stale scan (phase (1)) only sees the BUFFER, never the in-flight set
                        # -- so by the time it completes, its version lag may already exceed the bound. Drop
                        # it HERE instead of buffering it: a record must never ENTER the buffer untrainable,
                        # or -- if its group completes the same pass -- ``assembly_ready`` would pull it and
                        # ``consume`` would train it stale before the next scan could catch it. Its siblings
                        # (admitted back-to-back, same version) are dropped the same way, so the stale group
                        # is dropped WHOLE and never assembled. Counted with the scan's drops.
                        if current - policy.version > self._max_staleness:
                            self.stats.dropped_stale += 1
                            continue
                        ready.append(BufferRecord(sample=sample, version=policy.version, group_key=group_key))

                # (4) Consumption: train whatever the assembly rule says is ready (exact-global-batch
                #     pulls only; the remainder stays buffered for the next batch).
                records = self._assembly_ready(ready) if ready else None
                if records:
                    _t = time.perf_counter()
                    self._train_pull(records, ready)
                    self.stats.train_active_time += time.perf_counter() - _t
                    steps_in_cycle += self._last_pull_steps
                    progressed = True
                elif stream_done and partial is None and pending is None and not handles:
                    # The stream drained, nothing is generating and the buffer is not (or no longer)
                    # trainable: this run is over. Break at the top of the next pass; the finally then
                    # drops the unusable remainder and clears every live window. No trailing publish is
                    # needed -- training is done, so there is no further generation to publish to.
                    exhausted = True

                if not progressed and handles:
                    # Generations in flight, none completed, buffer not (yet) trainable: wait briefly
                    # instead of hot-spinning the poll.
                    _t = time.perf_counter()
                    self._sleep(0.01)
                    self.stats.idle_time += time.perf_counter() - _t
        finally:
            # Budget hit, drained, or an error: cancel and unpin everything STILL generating (uncollected
            # handles -- a collected one already released its pin at collect), drop the unusable buffer
            # remainder, and clear every live window so the control plane ends empty.
            for handle, (_, policy, _) in list(handles.items()):
                self._cancel(handle)
                ctx_mgr.release_rollout_policy(policy)
            handles.clear()
            # Buffered records were collected, so their pins are already released; only count the drop.
            self.stats.dropped_incomplete += len(ready)
            ready.clear()
            for window in self._live_windows():
                ctx_mgr.on_partition_cleared(window)
            self.stats.final_version = self._current_version()
            self.stats.total_time = time.perf_counter() - _run_start

    # --- internals ---------------------------------------------------------------------------------------

    def _train_pull(self, records: List[BufferRecord], ready: List[BufferRecord]) -> None:
        """Remove one pulled batch from the buffer, train it, and account the steps it reports.

        ``self._last_pull_steps`` carries the consume return back to the caller (0 when the consumer
        reports doing no optimizer step, which then also does not advance the publish cadence). No pin is
        released here: each record's version pin was already released when its trajectory was COLLECTED
        (the pin guards the SAMPLER-side adapter path, and training reads the samples in-process), so a
        record leaving the buffer is pure bookkeeping.
        """
        for record in records:
            ready.remove(record)
        steps = self._consume(records)
        steps = int(steps) if steps else 1
        self._last_pull_steps = steps
        if steps > 0:
            current = self._current_version()
            self.stats.train_pulls += 1
            self.stats.optimizer_steps += steps
            for record in records:
                self.stats.consumed += 1
                if current - record.version > 0:
                    self.stats.off_policy_consumed += 1

    def _live_windows(self) -> List[PartitionAdmission]:
        """The live admission windows (partitions) of this context, oldest step first (FIFO publish)."""
        return sorted(
            (a for a in self._ctx_mgr.list_live_partitions() if a.context.key == self._ctx.key),
            key=lambda a: a.step)

    def _open_window(self) -> None:
        """Open one fresh admission window if the staleness gate allows (bounded by ``max_staleness``).

        ``target_groups`` is the window's sizing DECLARATION (the groups one publish cycle trains,
        ``groups_per_partition``); membership is not fixed -- the ready buffer fills it in completion
        order, which is why a straggler never blocks a window and the FIFO publish order never waits on
        a specific sample.
        """
        self._ctx_mgr.request_rollout_partition(
            self._ctx, target_groups=self._groups_per_partition, num_generations=self._num_generations)

    def _current_version(self) -> int:
        return self._ctx_mgr.get_rollout_policy(self._ctx).version

    def _drop_stale(self, ready: List[BufferRecord]) -> None:
        """Post-publish stale scan: drop every BUFFERED sample whose version lag exceeds the bound.

        This is the SECOND of the two staleness guards -- the first drops a straggler AT COLLECT, before
        it can enter the buffer (see :meth:`run`); this one covers records that were within the bound when
        buffered and then aged past ``max_staleness`` as later publishes bumped the version. Per-record
        but group-sound: every trajectory of a group is admitted back-to-back under one pinned policy, so
        siblings share one version and a stale group is dropped whole across this scan. No pin is released
        here -- a collected record's pin was already released at collect.
        """
        current = self._current_version()
        kept: List[BufferRecord] = []
        for record in ready:
            if current - record.version > self._max_staleness:
                self.stats.dropped_stale += 1
            else:
                kept.append(record)
        ready[:] = kept
