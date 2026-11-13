# Copyright (c) ModelScope Contributors. All rights reserved.
"""The streaming driver: keeps episodes running and hands the trainer whole groups, batch by batch.

One driver per run. It reads the dataloader, turns each row into ``num_generations`` episodes and
steps them through the rollout; a finished episode is scored by every ORM, a closed PRM segment by
every PRM, and a group whose members are all scored becomes ready. ``__next__`` hands out
``batch_size`` rows of ready groups as ONE list of trajectories: collecting the batch and merging
the named score columns (``rewards`` / ``prm_segments`` fields plus the identity facts) onto the
rows is one locked section of row-level work -- the per-token response-contract validation ran
at its producer, the rollout's merge -- and the light batch facts ride :meth:`batch_summary` for
the consumer that just received the batch.

How far ahead of the trainer the driver runs is bounded by the policy version ``v`` (the checkpoint
manager's sync count): the rows admitted and not dropped never exceed
``(v + max_staleness + 1) * sync_interval * batch_size``. A group older than ``max_staleness``
versions when it is handed out is dropped.

All the waiting happens through completion callbacks: every step / score Future puts itself on one
queue when it finishes, and the driver's thread only reads that queue, so its cost does not grow
with the number of episodes in flight.
"""
from __future__ import annotations

import copy
import math
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from twinkle import remote_class, remote_function
from twinkle.data_format import user_data_get
from twinkle.processor import batch_summary
from twinkle.utils import get_logger
from twinkle_agentic.rollout.multi_turn import EpisodeError, RolloutSample

logger = get_logger()

__all__ = ['StreamingDriver', 'PRM_SEGMENT_TURNS']

#: The ``RolloutSample.extra`` key recording each PRM segment's end: the episode's ``turns`` when the
#: segment closed, one per column of ``prm_rewards``.
PRM_SEGMENT_TURNS = 'prm_segment_turns'

_BASELINE = 'baseline'


def _is_episode_error(exc: BaseException) -> bool:
    return isinstance(exc, EpisodeError) or isinstance(getattr(exc, 'cause', None), EpisodeError)


def _role_of(traj: Any) -> str:
    """A trajectory's role: the top-level field first, the legacy ``user_data`` spelling second.

    The group verdict returns baseline rows to the driver before they are submitted; those rows
    may carry either spelling, so both are read -- writing is top-level only.
    """
    role = traj.get('role')
    if role is None:
        role = user_data_get(traj.get('user_data'), 'role')
    return str(role or '')


@dataclass
class _Group:
    """One prompt row's episodes: the current attempt's members and the baseline episodes it waits for."""
    key: str
    row: Any
    attempt: int = 0
    #: The current attempt's member trajectories, aligned with ``members`` (a lost one is redone from here).
    source: List[Any] = field(default_factory=list)
    members: List['_Episode'] = field(default_factory=list)
    baselines: List['_Episode'] = field(default_factory=list)
    #: Members of the current attempt that will not produce a sample (failed, stopped, dropped).
    missing: int = 0


@dataclass
class _Episode:
    episode_id: str
    group: _Group
    role: str
    version: int
    traj: Any = None
    sample: Optional[RolloutSample] = None
    done: bool = False
    #: Set once the episode has no further work: scored (``finished``) or given up (``lost``).
    finished: bool = False
    lost: bool = False
    orm: Optional[List[Any]] = None
    orm_pending: int = 0
    prm: Optional[List[List[Any]]] = None
    prm_turns: List[int] = field(default_factory=list)
    prm_pending: int = 0
    #: The PRM segment the episode waits for before its next step (only with ``on_step``).
    held: Optional[int] = None


@remote_class(execute='first', max_concurrency=8)
class StreamingDriver:
    """Stream episodes through ``rollout`` and hand out scored groups; iterate it for training batches.

    Args:
        dataloader: yields batches of prompt rows (trajectories); read for ``num_epochs`` passes.
        rollout: the :class:`~twinkle_agentic.rollout.MultiTurnRollout` whose ``step`` advances an episode.
        batch_size: rows (trajectories) per batch; a multiple of ``num_generations``.
        num_generations: episodes per prompt row (one group).
        num_epochs: passes over the dataloader (a fraction still reads a whole pass; bound it with
            ``max_steps``).
        max_steps: batches to hand out before stopping (``None``: until the data ends).
        max_staleness: how many policy versions a handed-out group may lag behind.
        sync_interval: batches the trainer trains between two weight syncs.
        checkpoint_manager: its ``version()`` is the policy version; without one the version is the
            number of batches handed out divided by ``sync_interval`` and only paces admission (at
            most ``(max_staleness + 1) * sync_interval * batch_size`` rows ahead), nothing is dropped
            as stale.
        orms: outcome scorers, ``scorer([traj]) -> [score]``, run on every finished episode.
        orm_names: each ORM's name in the handed-out ``rewards.orm`` column; ``None`` names them
            by position (``orm_0`` ...).
        prms: process scorers, run on the trajectory so far at every closed segment.
        prm_names: each PRM's name in the handed-out ``prm_segments['scores']``; ``None`` names
            them by position (``prm_0`` ...).
        prm_segment: ``prm_segment(sample) -> bool``, whether the step just finished closes a PRM
            segment (default: every step; the last step always does).
        on_step: ``on_step(sample, prm_scores) -> bool``, called with a closed segment's scores;
            ``False`` stops the episode (it is dropped like a failed one).
        on_sample: ``on_sample(sample) -> Optional[bool]``, called on every scored episode;
            ``False`` drops it.
        on_group: ``on_group(row, samples, attempt) -> list[traj]``, called on every complete group:
            an empty list accepts it; trajectories to run are returned built from ``row``
            (``user_data.sampling_params`` overrides the sampling, ``user_data.role='baseline'``
            marks a baseline). Returning non-baseline trajectories replaces the group with them (the
            new group is offered to ``on_group`` again); once replacing is no longer allowed, or when
            only baselines are returned, the baselines run and their ORM rows are attached to the
            group's samples (``RolloutSample.baseline``) once they are scored.
        max_group_regenerations: how many times a group may be replaced (by ``on_group`` or to make
            up for missing members) before it is accepted as is, or dropped if incomplete.
        cluster_adapter: rebalances replicas after each version change (``adjust``), lends the
            trainer's devices while it waits for a batch (``lend``) and takes them back before the
            batch is handed out (``reclaim``).
    """

    #: Stop after this many episode failures in a row.
    MAX_CONSECUTIVE_FAILURES = 32
    #: ... or when more than ``MAX_FAILURE_RATE`` of the last ``FAILURE_WINDOW`` episodes failed.
    FAILURE_WINDOW = 256
    MAX_FAILURE_RATE = 0.5

    def __init__(self,
                 dataloader: Any,
                 rollout: Any,
                 *,
                 batch_size: int,
                 num_generations: int = 1,
                 num_epochs: float = 1.0,
                 max_steps: Optional[int] = None,
                 max_staleness: int = 0,
                 sync_interval: int = 1,
                 checkpoint_manager: Any = None,
                 orms: Sequence[Any] = (),
                 orm_names: Optional[Sequence[str]] = None,
                 prms: Sequence[Any] = (),
                 prm_names: Optional[Sequence[str]] = None,
                 prm_segment: Optional[Callable[[RolloutSample], bool]] = None,
                 on_step: Optional[Callable[[RolloutSample, List[Any]], bool]] = None,
                 on_sample: Optional[Callable[[RolloutSample], Optional[bool]]] = None,
                 on_group: Optional[Callable[[Any, List[RolloutSample], int], List[Any]]] = None,
                 max_group_regenerations: int = 0,
                 cluster_adapter: Any = None):
        if num_generations < 1 or batch_size < 1 or batch_size % num_generations:
            raise ValueError(f'batch_size ({batch_size}) must be a positive multiple of num_generations '
                             f'({num_generations}): a batch holds whole groups.')
        if num_epochs <= 0:
            raise ValueError(f'num_epochs must be > 0, got {num_epochs}.')
        if max_staleness < 0 or sync_interval < 1:
            raise ValueError(f'max_staleness must be >= 0 and sync_interval >= 1, got {max_staleness} / '
                             f'{sync_interval}.')
        if on_step is not None and not prms:
            raise ValueError('on_step reads PRM scores; pass prms= or drop on_step.')
        self.orms = list(orms)
        self.orm_names = list(orm_names) if orm_names is not None else [f'orm_{i}' for i in range(len(self.orms))]
        if len(self.orm_names) != len(self.orms):
            raise ValueError(f'orm_names ({len(self.orm_names)}) must name every ORM ({len(self.orms)}).')
        self.prms = list(prms)
        self.prm_names = list(prm_names) if prm_names is not None else [f'prm_{i}' for i in range(len(self.prms))]
        if len(self.prm_names) != len(self.prms):
            raise ValueError(f'prm_names ({len(self.prm_names)}) must name every PRM ({len(self.prms)}).')
        self.dataloader = dataloader
        self.rollout = rollout
        self.batch_size = batch_size
        self.num_generations = num_generations
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.max_staleness = max_staleness
        self.sync_interval = sync_interval
        self.checkpoint_manager = checkpoint_manager
        self.prm_segment = prm_segment
        self.on_step = on_step
        self.on_sample = on_sample
        self.on_group = on_group
        self.max_group_regenerations = max_group_regenerations
        self.cluster_adapter = cluster_adapter

        self._lock = threading.Condition()
        self._events: 'queue.Queue[Tuple[Any, ...]]' = queue.Queue()
        self._ready: deque = deque()
        self._ready_rows = 0
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._exhausted = False
        self._error: Optional[BaseException] = None
        self._rows: Optional[Iterator[Any]] = None
        self._prompt_seq = 0
        self._groups: Dict[str, _Group] = {}
        #: Member rows admitted and not dropped (baselines excluded): what the version bound caps.
        self._live = 0
        self._inflight = 0
        self._handed = 0
        self._version = 0
        self._base_version = 0
        self._failures: deque = deque(maxlen=self.FAILURE_WINDOW)
        self._consecutive_failures = 0
        self._counters: Dict[str, int] = {
            'admitted': 0, 'dropped_stale': 0, 'dropped_sample': 0, 'dropped_group': 0, 'episode_errors': 0,
            'stopped': 0, 'regenerated': 0
        }
        self._adjusting = False
        self._wait_time = 0.0
        self._returned_at: Optional[float] = None
        self._idle_ratio = 0.0
        self._source_id = uuid.uuid4().hex
        self._iter_started = False
        self._iter_handed: Optional[int] = None
        self._yields = 0
        self._last_batch_index: Optional[int] = None
        self._last_summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ iteration

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=True)
    def __iter__(self):
        """Iterate the driver once: yields the handed-out trajectory lists, batch by batch."""
        if self._iter_started:
            raise RuntimeError('StreamingDriver serves ONE iteration per run; a second one would re-read '
                               'the dataloader from an unrelated position. Wrap the single iterator instead.')
        self._iter_started = True
        with self._lock:
            self._iter_handed = self._handed
        self._yields = 0
        return self._batches()

    def _batches(self) -> Iterator[List[Any]]:
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            yield batch

    def _next_batch(self) -> Optional[List[Any]]:
        """Wait for ``batch_size`` rows of complete, fresh groups; ``None`` once nothing more will come."""
        self._start()
        started = time.monotonic()
        if self._returned_at is not None:
            busy = started - self._returned_at
            self._idle_ratio = self._wait_time / max(self._wait_time + busy, 1e-9)
        # the trainer may have synced just before asking: judge staleness against that version
        version = self._read_version()
        with self._lock:
            self._version = max(self._version, version)
            short = self._ready_rows < self.batch_size
        # the trainer waits from here on: its devices may be lent until the batch is ready
        lending = None
        if self.cluster_adapter is not None and short:
            lending = threading.Thread(target=self._lend, name='twinkle-cluster-lend', daemon=True)
            lending.start()
        try:
            with self._lock:
                while True:
                    if self._error is not None:
                        raise self._error
                    if self.max_steps is not None and self._handed >= self.max_steps:
                        return None
                    self._drop_stale()
                    if self._ready_rows >= self.batch_size:
                        break
                    if self._exhausted and self._inflight == 0:
                        return None
                    self._lock.wait(0.5)
        finally:
            if self.cluster_adapter is not None:
                if lending is not None:
                    lending.join()
                self.cluster_adapter.reclaim()
        # ONE locked section: collect the batch AND merge the named score columns onto the rows.
        # Both halves are row-level work -- a shallow dict copy, a few assignments, length reads;
        # the per-token response-contract validation already ran at its producer (the rollout's
        # merge), so nothing here scales with the tokens and the event thread is never held up.
        # Collecting, merging and counting move together: a failure inside raises before the
        # delivery counter moves, so the batch is never half-published.
        with self._lock:
            self._drop_stale()
            trajs: List[Any] = []
            version_spans: List[int] = []
            while len(trajs) < self.batch_size and self._ready:
                for episode in self._ready.popleft():
                    episode.sample.version_span = self._version - episode.sample.policy_version
                    merged = dict(episode.traj)
                    merged.update(self._score_update(episode))
                    episode.traj = merged
                    trajs.append(merged)
                    version_spans.append(episode.sample.version_span)
            self._ready_rows -= len(trajs)
            self._last_batch_index = self._iter_handed + self._yields
            self._handed += 1
            self._yields += 1
            self._lock.notify_all()
        self._last_summary = dict(batch_summary(trajs), version_spans=version_spans)
        self._returned_at = time.monotonic()
        self._wait_time = self._returned_at - started
        return trajs

    def _score_update(self, episode: _Episode) -> Dict[str, Any]:
        """One row's update: the named score columns and the identity facts, off the private bookkeeping.

        ``rewards.orm`` / ``rewards.baseline`` map scorer name -> float-or-None; a missing judge
        verdict stays ``None`` (never a silent 0), an unparseable one keeps its nan. The rollout's
        own ``rewards.env`` rides along untouched. ``prm_segments`` carries each segment's end turn
        and every scorer's raw segment scores -- the facts the PRM fusion broadcasts from.
        """
        sample = episode.sample
        rewards = dict(episode.traj.get('rewards') or {})
        if sample.baseline is not None:
            rows = sample.baseline if (sample.baseline and isinstance(sample.baseline[0], (list, tuple))) \
                else [sample.baseline]
            if len(rows) != 1:
                raise RuntimeError(f'episode {episode.episode_id} carries {len(rows)} baseline rows; the named '
                                   'baseline column carries one. Return a single baseline per group.')
            rewards['baseline'] = {name: (None if value is None else float(value))
                                   for name, value in zip(self.orm_names, rows[0])}
        update: Dict[str, Any] = {
            'episode_id': episode.episode_id,
            'group_key': episode.group.key,
            'role': episode.role or '',
            'policy_version': episode.version,
            'batch_id': f'{self._source_id}:{self._iter_handed + self._yields}',
            'rewards': rewards,
        }
        if episode.prm is not None:
            update['prm_segments'] = {
                'end_turns': [int(turns) for turns in episode.prm_turns],
                'scores': {name: [(None if value is None else float(value)) for value in column]
                           for name, column in zip(self.prm_names, episode.prm)},
            }
        return update

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def batch_summary(self, batch_index: int) -> Dict[str, Any]:
        """The light summary of batch ``batch_index``, as captured when it was handed out.

        One iteration, one consumer: only the CURRENT delivery's summary is kept -- earlier
        batches' are gone, so read it while the batch is still the latest one. ``batch_index``
        must therefore name the latest delivery (the batches handed out before this iteration
        began plus the yields so far, minus the one in flight); anything else is a misread,
        reported, not replayed.
        """
        with self._lock:
            if self._last_batch_index is None:
                raise RuntimeError('batch_summary reads a handed-out batch; iterate the driver first.')
            if batch_index != self._last_batch_index:
                raise ValueError(f'batch_summary(batch_index={batch_index}) names a batch other than the latest '
                                 f'delivery ({self._last_batch_index}); only the latest batch has a live summary.')
            return dict(self._last_summary or {})

    def _drop_stale(self) -> None:
        """Drop ready groups more than ``max_staleness`` versions behind (under the lock).

        Without a checkpoint manager the policy never changes, so nothing is stale: the version then
        only paces admission.
        """
        if self.checkpoint_manager is None:
            return
        kept = deque()
        for members in self._ready:
            if self._version - min(episode.version for episode in members) > self.max_staleness:
                self._ready_rows -= len(members)
                self._live -= len(members)
                self._counters['dropped_stale'] += len(members)
            else:
                kept.append(members)
        self._ready = kept

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def stats(self) -> Dict[str, Any]:
        """Counters of the run so far (small; for logging)."""
        with self._lock:
            return dict(self._counters, version=self._version, handed=self._handed, inflight=self._inflight,
                        live=self._live, ready_rows=self._ready_rows, train_idle_ratio=self._idle_ratio)

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def state_dict(self) -> Dict[str, Any]:
        """What a resumed run restores: the batches handed out and the prompt rows drawn.

        TODO: resuming re-reads the dataloader from its own saved position, which is ahead of the rows
        that were drawn but not yet trained (in flight or buffered); up to one buffer of prompts,
        ``(max_staleness + 1) * sync_interval * batch_size`` rows, is skipped. Keeping them would mean
        saving the in-flight episodes, which live in the rollout and env sessions.
        """
        with self._lock:
            return {'handed': self._handed, 'prompts_drawn': self._prompt_seq}

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def close(self) -> None:
        """Stop admitting, stop the episodes in flight and end the background thread."""
        with self._lock:
            self._closed = True
            episodes = [episode for group in self._groups.values()
                        for episode in group.members + group.baselines if not episode.finished]
            self._lock.notify_all()
        for episode in episodes:
            if not episode.done:
                try:
                    self.rollout.stop(episode.traj or {}, episode_id=episode.episode_id)
                except Exception as exc:  # noqa: BLE001 -- best effort on the way out
                    logger.warning(f'StreamingDriver.close: stopping {episode.episode_id} failed: {exc}')
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ------------------------------------------------------------------ background loop

    def _start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._version = self._read_version()
            # the bound counts versions from here: a manager synced before the run starts above 0
            self._base_version = self._version
            self._thread = threading.Thread(target=self._run, name='twinkle-streaming-driver', daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            # _next_batch also advances self._version, so version changes are tracked here on their own
            adjusted = self._version
            while True:
                with self._lock:
                    if self._closed:
                        return
                version = self._read_version()
                with self._lock:
                    self._version = max(self._version, version)
                    self._admit()
                    self._lock.notify_all()
                if version != adjusted:
                    adjusted = version
                    self._maybe_adjust()
                try:
                    event = self._events.get(timeout=0.1)
                except queue.Empty:
                    continue
                while event is not None:
                    self._handle(*event)
                    try:
                        event = self._events.get_nowait()
                    except queue.Empty:
                        event = None
        except BaseException as exc:  # noqa
            logger.error(f'StreamingDriver stopped: {type(exc).__name__}: {exc}')
            with self._lock:
                self._error = exc
                self._lock.notify_all()

    def _read_version(self) -> int:
        if self.checkpoint_manager is not None:
            return int(self.checkpoint_manager.version())
        return self._handed // self.sync_interval

    def _maybe_adjust(self) -> None:
        """Hand ``adjust`` to a thread of its own; skipped while the previous one still runs."""
        if self.cluster_adapter is None or self._adjusting:
            return
        self._adjusting = True

        def _adjust():
            try:
                self.cluster_adapter.adjust()
            except Exception as exc:  # noqa: BLE001 -- rebalancing is an optimisation, never fatal
                logger.warning(f'cluster adapter adjust failed: {type(exc).__name__}: {exc}')
            finally:
                self._adjusting = False

        threading.Thread(target=_adjust, name='twinkle-cluster-adjust', daemon=True).start()

    def _lend(self) -> None:
        """Run ``lend`` while the trainer waits; ``reclaim`` joins this thread before undoing it."""
        try:
            self.cluster_adapter.lend(self._idle_ratio)
        except Exception as exc:  # noqa: BLE001 -- lending is an optimisation, never fatal
            logger.warning(f'cluster adapter lend failed: {type(exc).__name__}: {exc}')

    # ------------------------------------------------------------------ admission

    def _capacity(self) -> int:
        return (self._version - self._base_version + self.max_staleness + 1) * self.sync_interval * self.batch_size

    def _admit(self) -> None:
        """Open groups while the admitted-and-alive rows stay within the version bound (under the lock)."""
        if self.max_steps is not None and self._handed >= self.max_steps:
            self._exhausted = True
        while not self._exhausted and self._live + self.num_generations <= self._capacity():
            row = self._next_row()
            if row is None:
                self._exhausted = True
                return
            key = str(self._prompt_seq)
            self._prompt_seq += 1
            group = _Group(key=key, row=row)
            self._groups[key] = group
            self._start_attempt(group, [copy.deepcopy(row) for _ in range(self.num_generations)])

    def _next_row(self) -> Any:
        # TODO: rows are drawn here, ahead of training; see state_dict for what a resume skips.
        if self._rows is None:
            self._rows = self._iter_rows()
        return next(self._rows, None)

    def _iter_rows(self) -> Iterator[Any]:
        for epoch in range(max(1, math.ceil(self.num_epochs))):
            set_epoch = getattr(self.dataloader, 'set_epoch', None)
            if set_epoch is not None:
                set_epoch(epoch)
            for batch in self.dataloader:
                if isinstance(batch, (list, tuple)):
                    yield from batch
                else:
                    yield batch

    def _start_attempt(self, group: _Group, trajectories: List[Any]) -> None:
        """Submit one attempt's trajectories: the non-baseline ones replace the members (under the lock)."""
        members = [traj for traj in trajectories if _role_of(traj) != _BASELINE]
        baselines = [traj for traj in trajectories if _role_of(traj) == _BASELINE]
        if members:
            group.source = members
            group.members = []
            group.missing = 0
            self._live += len(members)
            self._counters['admitted'] += len(members)
        group.baselines = []
        for traj in members:
            group.members.append(self._submit(group, traj, ''))
        for traj in baselines:
            group.baselines.append(self._submit(group, traj, _BASELINE))

    def _submit(self, group: _Group, traj: Any, role: str) -> _Episode:
        episode = _Episode(episode_id=uuid.uuid4().hex, group=group, role=role, version=self._version)
        # The identity travels on top of the trajectory: episode / group / role / the admission
        # version. The rollout reads the episode id to route its steps back to this episode.
        traj = dict(traj)
        traj.update(episode_id=episode.episode_id, group_key=group.key, role=role,
                    policy_version=episode.version)
        self._inflight += 1
        self._step(episode, traj)
        return episode

    def _step(self, episode: _Episode, traj: Any) -> None:
        future = self.rollout.step(traj, episode_id=episode.episode_id)
        future.add_done_callback(lambda f, episode=episode: self._events.put(('step', episode, None, f)))

    # ------------------------------------------------------------------ completions

    def _handle(self, kind: str, episode: _Episode, index: Optional[int], future: Any) -> None:
        exc = future.exception()
        with self._lock:
            if self._closed:
                return
            if episode.lost:
                if kind == 'step' and exc is None and not future.result()[1].done:
                    # dropped while this step ran: release its rollout state now
                    self.rollout.stop(future.result()[0], episode_id=episode.episode_id)
                return
            if exc is not None and not (kind == 'step' and _is_episode_error(exc)):
                raise exc
            if kind == 'step':
                if exc is not None:
                    self._fail(episode, exc)
                else:
                    self._on_step(episode, *future.result())
            elif kind == 'orm':
                episode.orm[index] = future.result()[0]
                episode.orm_pending -= 1
                self._maybe_finish(episode)
            else:
                segment, scorer = index
                episode.prm[scorer][segment] = future.result()[0]
                episode.prm_pending -= 1
                if episode.held == segment and not any(
                        episode.prm[k][segment] is _PENDING for k in range(len(self.prms))):
                    self._release(episode, segment)
                self._maybe_finish(episode)
            self._lock.notify_all()

    def _on_step(self, episode: _Episode, traj: Any, sample: RolloutSample) -> None:
        episode.traj = traj
        episode.sample = sample
        episode.done = bool(sample.done)
        if self.prms and episode.role != _BASELINE and (episode.done or self.prm_segment is None
                                                         or self.prm_segment(sample)):
            segment = self._score_prm(episode)
            if self.on_step is not None and not episode.done:
                episode.held = segment
                return
        if episode.done:
            self._consecutive_failures = 0
            self._failures.append(False)
            self._score_orm(episode)
        else:
            self._step(episode, traj)

    def _score_prm(self, episode: _Episode) -> int:
        if episode.prm is None:
            episode.prm = [[] for _ in self.prms]
        segment = len(episode.prm_turns)
        episode.prm_turns.append(int(episode.sample.turns))
        for scorer, prm in enumerate(self.prms):
            episode.prm[scorer].append(_PENDING)
            episode.prm_pending += 1
            future = prm([episode.traj])
            future.add_done_callback(
                lambda f, episode=episode, key=(segment, scorer): self._events.put(('prm', episode, key, f)))
        return segment

    def _release(self, episode: _Episode, segment: int) -> None:
        """A held episode's segment scores are in: ask ``on_step`` whether it goes on."""
        episode.held = None
        if self.on_step(episode.sample, [column[segment] for column in episode.prm]):
            self._step(episode, episode.traj)
            return
        self._counters['stopped'] += 1
        self.rollout.stop(episode.traj, episode_id=episode.episode_id)
        self._lose(episode)

    def _score_orm(self, episode: _Episode) -> None:
        episode.orm = [_PENDING] * len(self.orms)
        episode.orm_pending = len(self.orms)
        for index, orm in enumerate(self.orms):
            future = orm([episode.traj])
            future.add_done_callback(
                lambda f, episode=episode, index=index: self._events.put(('orm', episode, index, f)))
        self._maybe_finish(episode)

    def _fail(self, episode: _Episode, exc: BaseException) -> None:
        self._counters['episode_errors'] += 1
        self._consecutive_failures += 1
        self._failures.append(True)
        logger.warning(f'episode {episode.episode_id} skipped: {exc}')
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES or (
                len(self._failures) == self.FAILURE_WINDOW
                and sum(self._failures) > self.MAX_FAILURE_RATE * self.FAILURE_WINDOW):
            raise RuntimeError(f'too many episodes fail ({self._consecutive_failures} in a row, '
                               f'{sum(self._failures)} of the last {len(self._failures)}); last: {exc}') from exc
        self._lose(episode)

    def _lose(self, episode: _Episode) -> None:
        """An episode that will produce no sample: its group is short one member (under the lock)."""
        episode.lost = True
        self._inflight -= 1
        group = episode.group
        if episode.role == _BASELINE:
            self._drop_group(group)
            return
        group.missing += 1
        self._live -= 1
        self._settle(group)

    def _maybe_finish(self, episode: _Episode) -> None:
        if not episode.done or episode.finished or episode.lost or episode.orm_pending or episode.prm_pending:
            return
        sample = episode.sample
        sample.prompt_id = episode.group.key
        sample.group_key = episode.group.key
        sample.episode_id = episode.episode_id
        sample.role = episode.role
        sample.policy_version = episode.version
        sample.orm_reward = list(episode.orm)
        if episode.prm is not None:
            sample.prm_rewards = [list(column) for column in episode.prm]
            sample.extra = dict(sample.extra or {}, **{PRM_SEGMENT_TURNS: list(episode.prm_turns)})
        if episode.role != _BASELINE and self.on_sample is not None and self.on_sample(sample) is False:
            self._counters['dropped_sample'] += 1
            self._lose(episode)
            return
        if self.orms:
            # the named ORM column lands on the row HERE, at finish, a plain local write (no
            # publish): the group verdict reads it to rule on admission before any hand-out.
            rewards = dict(episode.traj.get('rewards') or {})
            rewards['orm'] = {name: (None if value is None else float(value))
                               for name, value in zip(self.orm_names, episode.orm or ())}
            episode.traj = dict(episode.traj, rewards=rewards)
        episode.finished = True
        self._inflight -= 1
        self._settle(episode.group)

    # ------------------------------------------------------------------ groups

    def _settle(self, group: _Group) -> None:
        """Act on a group whose current attempt has no work left (under the lock)."""
        if group.key not in self._groups:
            return
        if any(not (e.finished or e.lost) for e in group.members + group.baselines):
            return
        if group.missing:
            if group.attempt >= self.max_group_regenerations:
                self._drop_group(group)
                return
            group.attempt += 1
            self._counters['regenerated'] += 1
            # the lost members are redone from their trajectories; the scored ones stay
            self._live += group.missing
            self._counters['admitted'] += group.missing
            group.missing = 0
            for index, episode in enumerate(group.members):
                if episode.lost:
                    group.members[index] = self._submit(group, copy.deepcopy(group.source[index]), '')
            return
        if group.baselines:
            rows = [list(e.sample.orm_reward) for e in group.baselines]
            for episode in group.members:
                episode.sample.baseline = rows[0] if len(rows) == 1 else rows
            self._accept(group)
            return
        if self.on_group is not None:
            trajectories = list(self.on_group(group.row, [e.traj for e in group.members], group.attempt) or [])
            regenerate = [t for t in trajectories if _role_of(t) != _BASELINE]
            if regenerate and len(regenerate) != self.num_generations:
                raise ValueError(f'on_group returned {len(regenerate)} trajectories to replace a group of '
                                 f'{self.num_generations}; a group is always num_generations rows.')
            if regenerate and group.attempt < self.max_group_regenerations:
                # baselines returned alongside wait: the replaced group is offered to on_group again
                self._live -= len(group.members)
                self._counters['regenerated'] += 1
                group.attempt += 1
                self._start_attempt(group, regenerate)
                return
            baselines = [t for t in trajectories if _role_of(t) == _BASELINE]
            if baselines:
                self._start_attempt(group, baselines)
                return
        self._accept(group)

    def _accept(self, group: _Group) -> None:
        del self._groups[group.key]
        self._ready.append(list(group.members))
        self._ready_rows += len(group.members)

    def _drop_group(self, group: _Group) -> None:
        """Give the group up: its counted rows leave the bound, its unfinished episodes stop (under the lock)."""
        if self._groups.pop(group.key, None) is None:
            return
        self._counters['dropped_group'] += 1
        for episode in group.members + group.baselines:
            if episode.lost:
                continue
            episode.lost = True
            if episode.role != _BASELINE:
                self._live -= 1
            if not episode.finished:
                self._inflight -= 1
                if episode.held is not None:
                    # waiting for its PRM scores, no step in flight; a stepping one is stopped when it returns
                    self.rollout.stop(episode.traj, episode_id=episode.episode_id)


#: Placeholder for a score whose Future has not completed yet.
_PENDING = object()
