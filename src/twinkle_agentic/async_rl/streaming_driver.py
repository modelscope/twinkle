# Copyright (c) ModelScope Contributors. All rights reserved.
"""The streaming driver: keeps episodes running and hands the trainer whole groups, batch by batch.

One driver per run. It reads the dataloader, turns each row into ``num_generations`` episodes and
steps them through the rollout; a finished episode is scored by every ORM, a closed PRM segment by
every PRM, and a group whose members are all scored becomes ready. ``__next__`` hands out
``batch_size`` rows of ready groups as ``(rows, facts)``: the rows are the rollout's own artefacts,
handed out exactly as they travel (a reference under ray -- the driver never opens one; the
identity fields were stamped on them at admission), and the light facts (row lengths, the named
score columns aligned row for row, version spans) ride the SAME yield, so the consumer never
calls back for them. The scores land on ``facts['scores']`` -- one row-shaped entry per trajectory
(``rewards.orm`` / ``rewards.baseline``, optionally ``prm_segments``) -- and meet the rows again
inside the advantage worker, which receives both.

Admission runs a bounded window ahead of the batches already trained or skipped. Weight syncs and
requests for the next batch can both advance the admission window. Only policy versions determine
whether a group is older than ``max_staleness`` at delivery.

All the waiting happens through completion callbacks: every step / score Future puts itself on one
queue when it finishes, and the driver's thread only reads that queue, so its cost does not grow
with the number of episodes in flight.
"""
from __future__ import annotations

import copy
import math
import queue
import random
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from twinkle import remote_class, remote_function
from twinkle.data_format import user_data_get
from twinkle.utils import get_logger
from twinkle_agentic.rollout.multi_turn import EpisodeError, RolloutStatus

logger = get_logger()

__all__ = ['StreamingDriver', 'PRM_SEGMENT_TURNS']

#: The ``RolloutStatus.extra`` key recording each PRM segment's end: the episode's ``turns`` when the
#: segment closed, one per column of ``prm_rewards``.
PRM_SEGMENT_TURNS = 'prm_segment_turns'

_BASELINE = 'baseline'


def _record_driver_probe(event: str, episode_id: str = '', **fields: Any) -> None:
    """Append one driver event per call when ``TWINKLE_DRIVER_PROBE`` names a directory.

    The rollout-side dispatch probe shows an episode's opening step returning and then no
    continuation ever being submitted. This records the other half of that hand-off: whether the
    step future resolved, whether its completion reached the event queue, and whether the loop
    that drains the queue is still turning. Without it the two possible causes -- a future that
    never resolves and a queue that is never drained -- are indistinguishable from outside.
    """
    import json
    import os
    root = os.environ.get('TWINKLE_DRIVER_PROBE')
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'driver.{os.getpid()}.jsonl')
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'t': time.monotonic(), 'event': event, 'episode_id': episode_id,
                                 **fields}, default=str) + '\n')


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


def _sample_facts(samples: Sequence[RolloutStatus], scores: Sequence[Dict[str, Any]],
                   version_spans: Sequence[int]) -> Dict[str, Any]:
    """The delivery facts of one batch, off the samples -- never the trajectories.

    The rows are never opened for this (under ray they are references): the lengths /
    truncation / turns / identity facts come off the light sample records, and the named score
    columns arrive row-aligned in ``scores`` -- the one payload-adjacent lane, which the advantage
    estimator and the reward metrics consume without materialising a trajectory.
    """
    prompt_lengths: List[int] = []
    response_lengths: List[int] = []
    for sample in samples:
        tokens = sample.token_lengths
        # None means the episode never produced an encoded frame (an empty prompt); a delivered
        # batch is finished, so the lengths are final.
        prompt_lengths.append(tokens[0] if tokens else 0)
        response_lengths.append(tokens[1] if tokens else 0)
    truncated = [bool(sample.truncated) for sample in samples]
    return {
        'num_rows': len(samples),
        'episode_ids': [sample.episode_id for sample in samples],
        'prompt_lengths': prompt_lengths,
        'response_lengths': response_lengths,
        'response_length_min': min(response_lengths) if response_lengths else 0,
        'response_length_max': max(response_lengths) if response_lengths else 0,
        'response_length_sum': sum(response_lengths),
        'policy_versions': sorted({sample.policy_version for sample in samples}),
        'truncated': truncated,
        'num_truncated': sum(truncated),
        'turns': [sample.turns for sample in samples],
        'version_spans': list(version_spans),
        'scores': list(scores),
    }


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
    sample: Optional[RolloutStatus] = None
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
    #: The episode's training rows as the last step returned them (one, or one per collected
    #: segment); an unfinished episode holds none.
    deliver: List[Any] = field(default_factory=list)
    #: The trajectory this episode was submitted with, kept so a no-slot retry can resubmit the same
    #: member slot without touching group_attempt or the quality-regeneration budget.
    submit_traj: Any = None
    #: No-slot retry state (scheduling delay, not failure): when to retry next, the wall-clock deadline
    #: after which the group is dropped, and how many no-slot returns this member has accumulated.
    no_slot_retry_at: Optional[float] = None
    no_slot_deadline: Optional[float] = None
    no_slot_attempts: int = 0
    #: Whether this episode currently contributes 1 to ``_inflight``. True from submission until it
    #: finishes, is lost, or returns no-slot (the call returned, the member waits to retry). Lets
    #: ``_drop_group`` decrement exactly the episodes still counted, so a no-slot member waiting retry
    #: is not decremented a second time.
    counted_inflight: bool = False


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
        orm_names: each ORM's name in the handed-out ``facts['scores']`` rows' ``rewards.orm``
            column; ``None`` names them by position (``orm_0`` ...).
        prms: process scorers, run on the trajectory so far at every closed segment.
        prm_names: each PRM's name in the handed-out ``prm_segments['scores']``; ``None`` names
            them by position (``prm_0`` ...).
        prm_segment: ``prm_segment(sample, traj) -> bool``, whether the step just finished closes a
            PRM segment (default: every step; the last step always does). ``traj`` is the
            trajectory as the driver holds it -- the in-process dict on local direct calls, the
            stashed object reference under ray (the driver never materializes it itself; a
            content-based segmentation resolves it at its own cost, e.g. one ``ray.get`` per
            step).
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
            group's samples (``RolloutStatus.baseline``) once they are scored.
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
    #: One ``adjust`` call still running past this many seconds is reported as stuck: it hung
    #: inside the adapter holding its lock, so rebalancing is down until the call returns.
    ADJUST_STUCK_TIMEOUT = 300.0
    #: No-slot retry backoff: exponential from NO_SLOT_BASE_DELAY, capped at NO_SLOT_MAX_DELAY, plus
    #: uniform jitter in [0, NO_SLOT_JITTER] so many groups hitting a full pool at once do not retry in
    #: lockstep (a sentinel storm). The wall-clock ceiling is the per-instance max_no_slot_wait.
    NO_SLOT_BASE_DELAY = 0.5
    NO_SLOT_MAX_DELAY = 15.0
    NO_SLOT_JITTER = 0.5

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
                 prm_segment: Optional[Callable[[RolloutStatus, Any], bool]] = None,
                 on_step: Optional[Callable[[RolloutStatus, List[Any]], bool]] = None,
                 on_sample: Optional[Callable[[RolloutStatus], Optional[bool]]] = None,
                 on_group: Optional[Callable[[Any, List[RolloutStatus], int], List[Any]]] = None,
                 max_group_regenerations: int = 0,
                 cluster_adapter: Any = None,
                 max_no_slot_wait: float = 1800.0):
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
        #: Wall-clock ceiling on how long one member may keep retrying a full session pool before its
        #: group is loudly dropped. Scheduling delay is not quality regeneration: it never spends
        #: max_group_regenerations or increments group_attempt, but it must still be bounded.
        self.max_no_slot_wait = max_no_slot_wait

        self._lock = threading.Condition()
        self._close_lock = threading.Lock()
        self._close_future: Future | None = None
        self._close_waiting_on = ''
        self._closing_episodes: dict[str, _Episode] = {}
        self._pending: dict[Any, tuple[str, _Episode]] = {}
        self._events: 'queue.Queue[Tuple[Any, ...]]' = queue.Queue()
        self._ready: deque = deque()
        self._stale_groups: deque[_Group] = deque()
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
        self._consumed_batches = 0
        self._version = 0
        self._base_version = 0
        self._failures: deque = deque(maxlen=self.FAILURE_WINDOW)
        self._consecutive_failures = 0
        self._counters: Dict[str, int] = {
            'admitted': 0, 'dropped_stale': 0, 'dropped_sample': 0, 'dropped_group': 0, 'episode_errors': 0,
            'stopped': 0, 'regenerated': 0, 'no_slot': 0, 'no_slot_dropped': 0
        }
        self._adjusting = False
        #: monotonic stamps: when the running ``adjust`` started, and when its being stuck was last reported.
        self._adjust_started = 0.0
        self._adjust_reported = 0.0
        self._wait_time = 0.0
        #: the last batch's wait share spent joining the lend thread and reclaiming (excluded from
        #: ``_idle_ratio`` so the machinery's own cost never reads as a sampler bottleneck)
        self._reclaim_cost = 0.0
        #: the lend->reclaim cycle now in flight (the thread, until the reclaim in ``_next_batch``'s
        #: finally completes): while it runs the adapter's lock is legitimately held
        self._lending = None
        self._returned_at: Optional[float] = None
        self._idle_ratio = 0.0
        self._iter_started = False

    # ------------------------------------------------------------------ iteration

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=(0,))
    def __iter__(self):
        """Iterate the driver once: ``(rows, facts)`` per batch, delivered together."""
        if self._iter_started:
            raise RuntimeError('StreamingDriver serves ONE iteration per run; a second one would re-read '
                               'the dataloader from an unrelated position. Wrap the single iterator instead.')
        self._iter_started = True
        return self._batches()

    def _batches(self) -> Iterator[List[Any]]:
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            yield batch

    def _next_batch(self) -> Optional[tuple]:
        """Wait for ``batch_size`` rows of complete, fresh groups; ``None`` once nothing more will come.

        A delivery is ``(rows, facts)``: the rows exactly as they travel (a reference under ray,
        a dict in-process -- never opened here), and the light per-batch facts (row lengths, the
        row-aligned score column, version spans) built in the same locked section.
        """
        self._start()
        started = time.monotonic()
        waited = 0.0
        if self._returned_at is not None:
            busy = started - self._returned_at
            # only the sampler-caused wait counts: the machinery's own reclaim is not a reason to lend
            waited = max(self._wait_time - self._reclaim_cost, 0.0)
            self._idle_ratio = waited / max(waited + busy, 1e-9)
        # the trainer may have synced just before asking: judge staleness against that version
        version = self._read_version()
        with self._lock:
            if self._closed:
                return None
            self._version = max(self._version, version)
            # The next pull acknowledges the last delivery, even if the trainer skipped its update.
            self._consumed_batches = self._handed
            short = self._ready_rows < self.batch_size
        # the trainer waits from here on: its devices may be lent until the batch is ready
        lending = None
        if self.cluster_adapter is not None and short:
            lending = threading.Thread(target=self._lend, args=(waited,), name='twinkle-cluster-lend', daemon=True)
            lending.start()
        #: mirrors ``lending`` from its start until the reclaim below completes -- an ``adjust``
        #: queued behind that cycle is not stuck, so the watchdog stays quiet while it runs
        self._lending = lending
        try:
            with self._lock:
                if not self._wait_for_batch():
                    return None
        finally:
            if self.cluster_adapter is not None:
                settle = time.monotonic()
                if lending is not None:
                    lending.join()
                self.cluster_adapter.reclaim()
                self._reclaim_cost = time.monotonic() - settle
                self._lending = None
        # ONE locked section: collect the batch and build its light facts. The rows are handed
        # out exactly as they travel (a reference under ray, a dict in-process) -- the driver
        # never opens one; the score columns ride the facts instead, aligned row for row, and
        # the identity fields were stamped on the rows at admission. An episode delivered as
        # several rows (collected segments) contributes each of them: its episode-level sample
        # and score entry repeat per row, aligned. Nothing here scales with
        # the tokens, so the event thread is never held up; collecting and counting move
        # together, so the batch is never half-published.
        with self._lock:
            # A policy sync during reclaim can invalidate the batch we waited for.
            # Wait for its replacement before consuming a training step.
            if not self._wait_for_batch():
                return None
            if self.checkpoint_manager is not None:
                # Same-window backstop the old whole-group min(version) drop provided: after _drop_stale
                # every ready member is within max_staleness of the current version, so a group's own
                # span cannot exceed it. A violation means a member slipped through staleness handling.
                for members in self._ready:
                    versions = [episode.version for episode in members]
                    if versions and max(versions) - min(versions) > self.max_staleness:
                        raise RuntimeError(
                            f'ready group spans {max(versions) - min(versions)} policy versions '
                            f'({min(versions)}..{max(versions)}), more than max_staleness='
                            f'{self.max_staleness}; per-member staleness left an over-age member.')
            trajs: List[Any] = []
            samples: List[RolloutStatus] = []
            scores: List[Dict[str, Any]] = []
            version_spans: List[int] = []
            while len(trajs) < self.batch_size and self._ready:
                for episode in self._ready.popleft():
                    episode.sample.version_span = self._version - episode.sample.policy_version
                    score = self._score_update(episode)
                    lengths = (episode.sample.extra or {}).get('row_token_lengths')
                    if len(episode.deliver) == 1 or not lengths:
                        # The identity path: one row carries the episode's own facts verbatim.
                        row_samples = [episode.sample] * len(episode.deliver)
                    else:
                        if len(lengths) != len(episode.deliver):
                            raise RuntimeError(
                                f'episode {episode.sample.episode_id!r}: {len(episode.deliver)} delivered rows '
                                f'but {len(lengths)} row lengths on its status.')
                        # A collected episode's rows carry their own length facts: the episode
                        # totals would repeat on every row and weigh the metrics by row count.
                        extra = {key: value for key, value in (episode.sample.extra or {}).items()
                                 if key != 'row_token_lengths'}
                        row_samples = [
                            replace(episode.sample, token_lengths=tuple(map(int, pair)),
                                    extra=dict(extra, segment_index=index))
                            for index, pair in enumerate(lengths)]
                    for row, row_sample in zip(episode.deliver, row_samples):
                        trajs.append(row)
                        samples.append(row_sample)
                        scores.append(score)
                        version_spans.append(row_sample.version_span)
            self._ready_rows -= len(trajs)
            self._handed += 1
            self._lock.notify_all()
        self._returned_at = time.monotonic()
        self._wait_time = self._returned_at - started
        return trajs, _sample_facts(samples, scores, version_spans)

    def _wait_for_batch(self) -> bool:
        """Wait under the condition lock, rechecking staleness before delivery."""
        while not self._closed:
            if self._error is not None:
                raise self._error
            if self.max_steps is not None and self._handed >= self.max_steps:
                return False
            self._drop_stale()
            if self._ready_rows >= self.batch_size:
                return True
            if self._exhausted and self._inflight == 0 and not self._groups:
                return False
            self._lock.wait(0.5)
        return False

    def _score_update(self, episode: _Episode) -> Dict[str, Any]:
        """One row's light score entry, off the private bookkeeping -- never written onto the row.

        The entry mirrors the row's column contract (``rewards.orm`` / ``rewards.baseline`` map
        scorer name -> float-or-None; ``prm_segments`` carries each segment's end turn and every
        scorer's raw segment scores) but rides the delivery facts instead of the trajectory: the
        driver holds the row as a reference under ray and never opens it, so the scores meet the
        row again inside the advantage worker, which receives both. A missing judge verdict stays
        ``None`` (never a silent 0), an unparseable one keeps its nan. The rollout's own
        ``rewards.env`` stays on the row, where its producer wrote it.
        """
        sample = episode.sample
        rewards: Dict[str, Any] = {
            'orm': {name: (None if value is None else float(value))
                    for name, value in zip(self.orm_names, episode.orm or ())},
        }
        if sample.baseline is not None:
            rows = sample.baseline if (sample.baseline and isinstance(sample.baseline[0], (list, tuple))) \
                else [sample.baseline]
            if len(rows) != 1:
                raise RuntimeError(f'episode {episode.episode_id} carries {len(rows)} baseline rows; the named '
                                   'baseline column carries one. Return a single baseline per group.')
            rewards['baseline'] = {name: (None if value is None else float(value))
                                   for name, value in zip(self.orm_names, rows[0])}
        score: Dict[str, Any] = {'rewards': rewards}
        if episode.prm is not None:
            score['prm_segments'] = {
                'end_turns': [int(turns) for turns in episode.prm_turns],
                'scores': {name: [(None if value is None else float(value)) for value in column]
                           for name, column in zip(self.prm_names, episode.prm)},
            }
        return score

    def _drop_stale(self) -> None:
        """Retire over-age members of ready groups, per member (under the lock).

        Without a checkpoint manager the policy never changes, so nothing is stale: the version then
        only paces admission. A member more than ``max_staleness`` versions behind is lost back into its
        group's ``missing`` and the group returns to ``_groups`` for ``_settle`` to regenerate exactly
        those slots -- the fresh members are kept, not buried with the stale ones. ``_accept`` asserts the
        group's version span stays within ``max_staleness``, the same-window backstop the old whole-group
        ``min(version)`` drop used to provide.
        """
        if self.checkpoint_manager is None:
            return
        kept = deque()
        for members in self._ready:
            stale = [episode for episode in members if self._version - episode.version > self.max_staleness]
            if not stale:
                kept.append(members)
                continue
            group = members[0].group
            self._ready_rows -= sum(len(episode.deliver) for episode in members)
            self._groups[group.key] = group
            for episode in stale:
                episode.lost = True
                episode.counted_inflight = False
                group.missing += 1
                self._live -= 1
                self._counters['dropped_stale'] += 1
            # A policy change noticed by the trainer can trigger regeneration.
            # Only the background thread submits and records its new futures.
            self._stale_groups.append(group)
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
    def close(self, timeout: float = 30.0) -> None:
        """Wait up to timeout seconds for cleanup; retry close to confirm settlement."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('close timeout must be finite and positive')
        with self._close_lock:
            drain = self._close_future
            if drain is None or (drain.done() and drain.exception() is not None):
                self._closed = True
                self._close_waiting_on = 'driver thread'
                drain = self._close_future = Future()
                # The worker keeps ownership if the caller times out or is interrupted.
                threading.Thread(target=self._drain_on_close, args=(drain,),
                                 name='twinkle-streaming-close', daemon=True).start()
        try:
            drain.result(timeout=timeout)
        except TimeoutError:
            if drain.done():
                drain.result()
            else:
                raise TimeoutError(f'StreamingDriver.close timed out waiting for {self._close_waiting_on}') from None

    def _drain_on_close(self, drain: Future) -> None:
        try:
            self._close_rollouts()
        except BaseException as error:
            drain.set_exception(error)
        else:
            drain.set_result(None)

    def _close_rollouts(self) -> None:
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join()

        episodes = self._closing_episodes
        episodes.update((episode.episode_id, episode) for group in self._groups.values()
                        for episode in group.members + group.baselines if not episode.done)
        # A dropped group can still have a pending call that owns its session.
        while self._pending:
            future = next(iter(self._pending))
            kind, episode = self._pending[future]
            episode_id = episode.episode_id
            self._close_waiting_on = f'{kind} for episode {episode_id}'
            if kind in ('step', 'stop'):
                episodes[episode_id] = episode
            try:
                result = future.result()
            except Exception:
                # Failed session operations are followed by stop once the call settles.
                pass
            else:
                if kind == 'stop':
                    episode.done = True
                elif kind == 'step':
                    rows, sample = result
                    episode.done = sample.done
                    if rows:
                        episode.traj = rows[0]
                if episode.done:
                    episodes.pop(episode_id, None)
            del self._pending[future]

        errors = []
        for episode_id, episode in list(episodes.items()):
            self._close_waiting_on = f'stop for episode {episode_id}'
            try:
                future = self.rollout.stop(episode.traj or episode.submit_traj or {}, episode_id=episode_id)
                if future is not None:
                    future.result()
            except Exception as error:
                error.add_note(f'Stopping rollout episode {episode_id}')
                errors.append(error)
            else:
                episode.done = True
                del episodes[episode_id]
        if errors:
            raise ExceptionGroup('could not close rollout episodes', errors)
        with self._lock:
            self._groups.clear()
            self._ready.clear()
            self._stale_groups.clear()
            self._ready_rows = self._live = self._inflight = 0
        while not self._events.empty():
            self._events.get_nowait()

    # ------------------------------------------------------------------ background loop

    def _start(self) -> None:
        with self._lock:
            if self._closed or self._thread is not None:
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
                version = self._read_version()
                with self._lock:
                    if self._closed:
                        return
                    self._version = max(self._version, version)
                    while self._stale_groups:
                        self._settle(self._stale_groups.popleft())
                    self._admit()
                    self._retry_no_slot_due()
                    self._lock.notify_all()
                _record_driver_probe('admit', live=self._live, capacity=self._capacity(),
                                     qsize=self._events.qsize(), version=version)
                if version != adjusted:
                    adjusted = version
                    self._maybe_adjust()
                try:
                    event = self._events.get(timeout=0.1)
                except queue.Empty:
                    continue
                _record_driver_probe('event_dequeued', event[1].episode_id, kind=event[0])
                while event is not None:
                    self._handle(*event)
                    try:
                        event = self._events.get_nowait()
                    except queue.Empty:
                        event = None
        except BaseException as exc:  # noqa
            _record_driver_probe('run_stopped', error=f'{type(exc).__name__}: {exc}')
            logger.error(f'StreamingDriver stopped: {type(exc).__name__}: {exc}')
            with self._lock:
                self._error = exc
                self._lock.notify_all()

    def _read_version(self) -> int:
        if self.checkpoint_manager is not None:
            return int(self.checkpoint_manager.version())
        return self._handed // self.sync_interval

    def _maybe_adjust(self) -> None:
        """Hand ``adjust`` to a thread of its own; skipped while the previous one still runs.

        One still running past ``ADJUST_STUCK_TIMEOUT`` is only reported, not replaced: it holds
        the adapter's lock (hung, or warming a large set of weights), so a new call would just
        queue behind it; if it ever returns, the ``finally`` below re-arms the next version
        change. A lend->reclaim cycle in flight holds the same lock legitimately -- an ``adjust``
        queued behind one is not reported.
        """
        if self.cluster_adapter is None:
            return
        if self._adjusting:
            if self._lending is not None:
                return  # a lend->reclaim cycle legitimately holds the adapter lock: queueing is not stuck
            now = time.monotonic()
            stuck = now - self._adjust_started
            if stuck >= self.ADJUST_STUCK_TIMEOUT and now - self._adjust_reported >= self.ADJUST_STUCK_TIMEOUT:
                self._adjust_reported = now
                logger.warning(f'cluster adapter adjust has been running for {stuck:.0f}s '
                               '(a large weight warm takes this long too and a stuck RPC looks '
                               'the same); rebalancing is paused until it returns')
            return
        self._adjusting = True
        self._adjust_started = time.monotonic()

        def _adjust():
            try:
                self.cluster_adapter.adjust()
            except Exception as exc:  # noqa: BLE001 -- rebalancing is an optimisation, never fatal
                logger.warning(f'cluster adapter adjust failed: {type(exc).__name__}: {exc}')
            finally:
                self._adjusting = False

        threading.Thread(target=_adjust, name='twinkle-cluster-adjust', daemon=True).start()

    def _lend(self, waited: float) -> None:
        """Run ``lend`` while the trainer waits; ``reclaim`` joins this thread before undoing it."""
        try:
            self.cluster_adapter.lend(self._idle_ratio, waited=waited)
        except Exception as exc:  # noqa: BLE001 -- lending is an optimisation, never fatal
            logger.warning(f'cluster adapter lend failed: {type(exc).__name__}: {exc}')

    # ------------------------------------------------------------------ admission

    def _capacity(self) -> int:
        synced_batches = (self._version - self._base_version) * self.sync_interval
        ahead_batches = (self.max_staleness + 1) * self.sync_interval
        return (max(synced_batches, self._consumed_batches) + ahead_batches) * self.batch_size

    def _admit(self) -> None:
        """Open groups within the version window and remaining-batch budget (under the lock)."""
        remaining_groups = None
        if self.max_steps is not None and self._handed >= self.max_steps:
            self._exhausted = True
        elif self.max_steps is not None:
            # Each successful episode delivers at least one row, including when segments are collected.
            remaining_groups = (self.max_steps - self._handed) * (self.batch_size // self.num_generations)
        while not self._closed and not self._exhausted and self._live + self.num_generations <= self._capacity():
            # A group with missing members keeps its reservation until the group itself is dropped.
            if remaining_groups is not None and len(self._ready) + len(self._groups) >= remaining_groups:
                return
            row = self._next_row()
            if self._closed:
                return
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
        for index, traj in enumerate(members):
            # Positional sample identity within the group: the rollout derives a per-request seed
            # from (group_key, sample_index) so a group's num_generations samples differ. A constant
            # seed would collapse them to one identical trajectory and zero the group's advantage.
            traj['sample_index'] = index
            group.members.append(self._submit(group, traj, ''))
        for traj in baselines:
            group.baselines.append(self._submit(group, traj, _BASELINE))

    def _submit(self, group: _Group, traj: Any, role: str) -> _Episode:
        episode = _Episode(episode_id=uuid.uuid4().hex, group=group, role=role, version=self._version)
        # The identity travels on top of the trajectory: episode / group / role / the admission
        # version. The rollout reads the episode id to route its steps back to this episode.
        traj = dict(traj)
        traj.update(episode_id=episode.episode_id, group_key=group.key, role=role,
                    policy_version=episode.version, group_attempt=group.attempt)
        episode.submit_traj = traj
        self._inflight += 1
        episode.counted_inflight = True
        self._step(episode, traj)
        return episode

    def _step(self, episode: _Episode, traj: Any) -> None:
        # session_wait=0: the admission loop never blocks on a full session pool -- a no-slot status
        # comes back along the return channel and the member is retried later (see _no_slot).
        future = self.rollout.step(traj, episode_id=episode.episode_id, session_wait=0)
        _record_driver_probe('step_submitted', episode.episode_id)

        self._track(future, 'step', episode)

    def _track(self, future, kind, episode, index=None):
        self._pending[future] = (kind, episode)

        def done(result):
            if self._closed:
                return
            if kind == 'step':
                _record_driver_probe('step_future_done', episode.episode_id)
            self._events.put((kind, episode, index, result))

        future.add_done_callback(done)

    def _stop(self, episode, trajectory):
        future = self.rollout.stop(trajectory, episode_id=episode.episode_id)
        if future is not None:
            self._track(future, 'stop', episode)

    # ------------------------------------------------------------------ completions

    def _handle(self, kind: str, episode: _Episode, index: Optional[int], future: Any) -> None:
        _record_driver_probe('handle', episode.episode_id, kind=kind)
        exc = future.exception()
        with self._lock:
            if self._closed:
                return
            if kind == 'stop' and exc is not None:
                raise exc
            self._pending.pop(future, None)
            if kind == 'stop':
                return
            if episode.lost:
                if kind == 'step' and exc is None and not future.result()[1].done:
                    # dropped while this step ran: release its rollout state now
                    self._stop(episode, future.result()[0][0])
                return
            if exc is not None and not (kind == 'step' and _is_episode_error(exc)):
                raise exc
            if kind == 'step':
                if exc is not None:
                    self._fail(episode, exc)
                else:
                    rows, sample = future.result()
                    if sample.no_slot:
                        # Short-circuit BEFORE _on_step: a no-slot return must not write traj/sample/done,
                        # must not score a PRM segment (a bogus end_turn would break _process_row), and
                        # must not advance the turn. It is a scheduling delay, handled by retry.
                        self._no_slot(episode)
                    else:
                        self._on_step(episode, rows, sample)
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

    def _on_step(self, episode: _Episode, trajs: Any, sample: RolloutStatus) -> None:
        episode.traj = trajs[0]
        episode.sample = sample
        episode.done = bool(sample.done)
        if episode.done:
            episode.deliver = list(trajs)
        if self.prms and episode.role != _BASELINE and (episode.done or self.prm_segment is None
                                                         or self.prm_segment(sample, trajs[0])):
            segment = self._score_prm(episode)
            if self.on_step is not None and not episode.done:
                episode.held = segment
                return
        if episode.done:
            self._consecutive_failures = 0
            self._failures.append(False)
            self._score_orm(episode)
        else:
            self._step(episode, trajs[0])

    def _score_prm(self, episode: _Episode) -> int:
        if episode.prm is None:
            episode.prm = [[] for _ in self.prms]
        segment = len(episode.prm_turns)
        episode.prm_turns.append(int(episode.sample.turns))
        for scorer, prm in enumerate(self.prms):
            episode.prm[scorer].append(_PENDING)
            episode.prm_pending += 1
            future = prm([episode.traj])
            self._track(future, 'prm', episode, (segment, scorer))
        return segment

    def _release(self, episode: _Episode, segment: int) -> None:
        """A held episode's segment scores are in: ask ``on_step`` whether it goes on."""
        episode.held = None
        if self.on_step(episode.sample, [column[segment] for column in episode.prm]):
            self._step(episode, episode.traj)
            return
        self._counters['stopped'] += 1
        self._stop(episode, episode.traj)
        self._lose(episode)

    def _score_orm(self, episode: _Episode) -> None:
        episode.orm = [_PENDING] * len(self.orms)
        episode.orm_pending = len(self.orms)
        for index, orm in enumerate(self.orms):
            future = orm([episode.traj])
            self._track(future, 'orm', episode, index)
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
        episode.counted_inflight = False
        group = episode.group
        if episode.role == _BASELINE:
            self._drop_group(group)
            return
        group.missing += 1
        self._live -= 1
        self._settle(group)

    def _no_slot(self, episode: _Episode) -> None:
        """A step found no free session permit (under the lock): retry this member later, not a failure.

        Only ``_inflight`` moves -- the call returned, so the episode is no longer counted; ``missing``
        and ``_live`` stay put because the member still owns its slot and will produce a sample, and
        ``group_attempt`` is untouched because a scheduling delay is not a quality regeneration. The
        episode stays neither finished nor lost, so ``_settle``'s all-done gate keeps the group waiting
        (siblings keep running, finished samples are kept) until the retry succeeds or the deadline drops it.
        """
        self._inflight -= 1
        episode.counted_inflight = False
        self._counters['no_slot'] += 1
        now = time.monotonic()
        if episode.no_slot_deadline is None:
            episode.no_slot_deadline = now + self.max_no_slot_wait
        episode.no_slot_attempts += 1
        delay = min(self.NO_SLOT_BASE_DELAY * (2 ** (episode.no_slot_attempts - 1)), self.NO_SLOT_MAX_DELAY)
        episode.no_slot_retry_at = now + delay + random.uniform(0, self.NO_SLOT_JITTER)

    def _retry_no_slot_due(self) -> None:
        """Resubmit members whose no-slot backoff has elapsed (under the lock, called from ``_run``).

        Each retry replaces only its own member/baseline slot with a fresh episode submitted from the
        same trajectory at the same group attempt -- never the ``_settle`` loop that redoes ALL lost
        members. The carried attempt count and deadline keep the exponential backoff and the wall-clock
        ceiling monotonic across retries.
        """
        now = time.monotonic()
        for group in list(self._groups.values()):
            if group.key not in self._groups:
                continue
            for slots in (group.members, group.baselines):
                for index, episode in enumerate(slots):
                    if episode.no_slot_retry_at is None or episode.no_slot_retry_at > now:
                        continue
                    if episode.no_slot_deadline is not None and now > episode.no_slot_deadline:
                        self._counters['no_slot_dropped'] += 1
                        logger.warning(f'group {group.key} dropped: member {episode.episode_id} could not get '
                                       f'a session slot within {self.max_no_slot_wait:.0f}s '
                                       f'({episode.no_slot_attempts} no-slot retries)')
                        self._drop_group(group)
                        break
                    episode.no_slot_retry_at = None
                    fresh = self._submit(group, copy.deepcopy(episode.submit_traj), episode.role)
                    fresh.no_slot_attempts = episode.no_slot_attempts
                    fresh.no_slot_deadline = episode.no_slot_deadline
                    slots[index] = fresh
                if group.key not in self._groups:
                    break

    def _maybe_finish(self, episode: _Episode) -> None:
        if not episode.done or episode.finished or episode.lost or episode.orm_pending or episode.prm_pending:
            return
        # The trajectory itself stays exactly as it travels (a reference under ray, the rollout's
        # stash channel): it is only ever passed on to remote calls (the forwards, the advantage,
        # the filter) and handed out at delivery -- the driver never opens it. The scores and the
        # identity the recipe reads live on the sample and the delivery facts instead.
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
        episode.finished = True
        self._inflight -= 1
        episode.counted_inflight = False
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
            # the verdict reads the samples (light, local: the scores are already stamped on
            # them); the trajectories themselves are references under ray and never opened here.
            trajectories = list(self.on_group(group.row, [e.sample for e in group.members], group.attempt) or [])
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
        self._ready_rows += sum(len(episode.deliver) for episode in group.members)

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
            if episode.counted_inflight:
                self._inflight -= 1
                episode.counted_inflight = False
                if episode.held is not None:
                    # waiting for its PRM scores, no step in flight; a stepping one is stopped when it returns
                    self._stop(episode, episode.traj)


#: Placeholder for a score whose Future has not completed yet.
_PENDING = object()
