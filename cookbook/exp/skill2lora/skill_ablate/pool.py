# Copyright (c) ModelScope Contributors. All rights reserved.
"""SamplePool: accumulate training samples across chunks and emit fixed-size batches.

Ray-infra size rule (skill_quality_analysis.md #11-12): training must drop_last to a
TRAIN_DP multiple and must NEVER pad new sequences (overfitting guard). The actual
drop_last happens inside v2's ``_train_step``; this pool's only job is "accumulate until a
full batch is available, then hand exactly one batch to ``_train_step``". Batch size is a
TRAIN_DP multiple (default 16 = sft_batch_size), so the batch always divides evenly.

Two modes:
- plain (balanced=False): a single FIFO queue; ready when it holds >= batch_size; a draw
  pops the oldest ``batch_size`` samples and keeps the remainder pooled for next time.
- balanced (balanced=True): separate positive/negative queues for the improve-skill+SFT
  1:1 requirement (#15b/#18b). A batch is half positives + half negatives; ready when BOTH
  halves are available. After every chunk the caller invokes ``rebalance()``: the majority
  side is down-sampled to the minority side and the surplus is DISCARDED immediately
  (#18b "以少的一侧为准下采样多的一侧，其余丢弃不积压") — no stale majority backlog can
  accumulate, so early-policy easy positives never train dozens of chunks later.
  ``max_pool`` remains as a safety cap only (drop oldest).

This module is dependency-free (pure stdlib) and unit-testable without torch / a GPU.
"""
from collections import deque
from typing import Any, Deque, Dict, List, Optional

POS, NEG = 'pos', 'neg'


class SamplePool:
    def __init__(self, batch_size: int = 16, balanced: bool = False,
                 max_pool: Optional[int] = None):
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        if balanced and batch_size % 2 != 0:
            raise ValueError('balanced pool needs an even batch_size (half pos + half neg)')
        self.batch_size = batch_size
        self.balanced = balanced
        self.max_pool = max_pool
        self._q: Deque[Dict[str, Any]] = deque()                 # plain mode
        self._pos: Deque[Dict[str, Any]] = deque()               # balanced mode
        self._neg: Deque[Dict[str, Any]] = deque()
        self._added = 0
        self._emitted = 0

    # -- ingest -------------------------------------------------------------------------
    def add(self, sample: Dict[str, Any], label: str = NEG) -> None:
        self._added += 1
        if not self.balanced:
            self._q.append(sample)
            self._trim(self._q)
            return
        if label == POS:
            self._pos.append(sample)
            self._trim(self._pos)
        elif label == NEG:
            self._neg.append(sample)
            self._trim(self._neg)
        else:
            raise ValueError(f'label must be {POS!r} or {NEG!r}, got {label!r}')

    def add_many(self, samples: List[Dict[str, Any]], label: str = NEG) -> None:
        for s in samples:
            self.add(s, label)

    def _trim(self, q: Deque[Dict[str, Any]]) -> None:
        if self.max_pool is not None:
            while len(q) > self.max_pool:
                q.popleft()  # drop oldest to bound memory / avoid stale majority backlog

    def rebalance(self) -> int:
        """Balanced mode: down-sample the majority queue to the minority size, discarding
        the NEWEST surplus (this chunk's excess intake — the #18b "其余丢弃不积压" rule).
        Called once per chunk; since intake is re-balanced every chunk, both queues stay
        equal-length and no side ever backlogs. Returns the number of discarded samples.
        No-op in plain mode."""
        if not self.balanced:
            return 0
        target = min(len(self._pos), len(self._neg))
        dropped = 0
        for q in (self._pos, self._neg):
            while len(q) > target:
                q.pop()  # newest first: the surplus was added this chunk
                dropped += 1
        return dropped

    # -- state --------------------------------------------------------------------------
    def ready(self) -> bool:
        if not self.balanced:
            return len(self._q) >= self.batch_size
        half = self.batch_size // 2
        return len(self._pos) >= half and len(self._neg) >= half

    def sizes(self) -> Dict[str, int]:
        if not self.balanced:
            return {'pool': len(self._q)}
        return {'pos': len(self._pos), 'neg': len(self._neg)}

    @property
    def total_added(self) -> int:
        return self._added

    @property
    def total_emitted(self) -> int:
        return self._emitted

    # -- draw ---------------------------------------------------------------------------
    def draw(self) -> List[Dict[str, Any]]:
        """Pop exactly one batch; raises if not ready. Remainder stays pooled."""
        if not self.ready():
            raise RuntimeError('draw() called while pool not ready; guard with ready()')
        if not self.balanced:
            batch = [self._q.popleft() for _ in range(self.batch_size)]
        else:
            half = self.batch_size // 2
            batch = [self._pos.popleft() for _ in range(half)]
            batch += [self._neg.popleft() for _ in range(half)]
        self._emitted += len(batch)
        return batch

    def draw_all_ready(self) -> List[List[Dict[str, Any]]]:
        """Pop as many full batches as currently available (0 or more)."""
        out = []
        while self.ready():
            out.append(self.draw())
        return out
