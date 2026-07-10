# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-granularity score aggregation.

Two scorers operate at deliberately different granularities:

- :class:`HardScorer` is **per-round** — tool-call validity / execution /
  protocol are facts about a single assistant turn, cheap and deterministic.
- :class:`RubricVerifier` is **per-segment** — soft quality (sub-goal progress,
  reasoning soundness, no redundant calls) needs multi-round context.

This module bridges the two without forcing either onto the other's grain:

    rounds ── HardScorer (per round) ──► h_1..h_R
                                          │  aggregate over the rounds in a segment
    segment ── RubricVerifier (whole) ──► rubric_scalar
                                          │  fuse(hard_agg, rubric_scalar)
                                          ▼
                                       segment score ──► aggregate ──► trajectory score

It also implements the **short-circuit gate**: when a segment's hard score is
extreme (e.g. every tool call failed, no final answer), the soft rubric chain
is skipped entirely — saving the long/expensive LLM path exactly when the
answer is already decided.

The functions here are pure: pass callables/detail objects, get scores back.
They do not import HardScorer/RubricVerifier, so they stay easily testable and
decoupled (the orchestrating TrajectoryScorer will wire real scorers in later).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from twinkle_agentic.segment.base import Segmenter

NUM_LEVELS = 5

# aggregation strategies for combining a list of scalars in [0, 1]
_REDUCERS: Dict[str, Callable[[Sequence[float]], float]] = {
    'mean': lambda xs: sum(xs) / len(xs),
    'min': min,
    'max': max,
    'median': lambda xs: statistics.median(xs),
    # geometric mean: harsher than mean, one bad round drags the whole segment
    'gmean': lambda xs: (statistics.geometric_mean([max(1e-6, x) for x in xs])),
}


def _reduce(xs: Sequence[float], how: str) -> float:
    xs = [x for x in xs if x is not None]
    if not xs:
        return 0.0
    fn = _REDUCERS.get(how)
    if fn is None:
        raise ValueError(f'unknown reducer {how!r}; choose from {list(_REDUCERS)}')
    return float(fn(xs))


def scalar_to_level(scalar: float, num_levels: int = NUM_LEVELS) -> int:
    scalar = min(1.0, max(0.0, scalar))
    return min(num_levels - 1, max(0, int(round(scalar * (num_levels - 1)))))


# ---------------------------------------------------------------------------
# score containers
# ---------------------------------------------------------------------------
@dataclass
class RoundScore:
    """Per-round hard score."""
    index: int                    # round index within the segment
    hard_scalar: float
    gated: bool = False
    detail: Any = None            # optional HardScoreDetail


@dataclass
class SegmentScore:
    """Fused per-segment score."""
    index: int
    scalar: float                 # final fused score in [0, 1]
    level: int
    hard_scalar: float            # aggregated hard score over the segment's rounds
    rubric_scalar: Optional[float]  # None if the soft chain was short-circuited
    short_circuited: bool
    n_rounds: int
    rounds: List[RoundScore] = field(default_factory=list)
    detail: Any = None            # optional rubric ScoreDetail


@dataclass
class TrajectoryScore:
    """Trajectory-level score aggregated over segments."""
    scalar: float
    level: int
    segments: List[SegmentScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# round -> segment
# ---------------------------------------------------------------------------
def split_segment_into_rounds(segment: dict) -> List[dict]:
    """Split a segment sub-trajectory into per-round sub-trajectories.

    Reuses :meth:`Segmenter.split_turns`. The segment's preamble (system + the
    first user message) is carried onto every round so a per-round HardScorer
    still sees tools/context. Each returned dict is a valid sub-trajectory
    (``messages`` + ``tools``/``user_data`` when present).
    """
    messages = list(segment.get('messages', []) or [])
    preamble, start = Segmenter._split_preamble(messages)
    turns = Segmenter.split_turns(messages, start)
    rounds: List[dict] = []
    for t in turns:
        # skip pure 'user' boundary turns: nothing tool-verifiable there
        if t.role_kind == 'user':
            continue
        r: Dict[str, Any] = {'messages': list(preamble) + list(t.messages)}
        if segment.get('tools'):
            r['tools'] = list(segment['tools'])
        if segment.get('user_data'):
            r['user_data'] = list(segment['user_data'])
        rounds.append(r)
    return rounds


def aggregate_hard_over_rounds(
    round_scores: Sequence[RoundScore],
    *,
    how: str = 'gmean',
) -> float:
    """Aggregate per-round hard scalars into one segment-level hard scalar.

    Default ``gmean`` (geometric mean) is intentionally harsher than plain mean:
    a single badly-formed round meaningfully drags the segment, which matches
    the intuition that one hallucinated/failed tool call hurts the sub-task.
    """
    if not round_scores:
        return 1.0  # no rounds to fault -> neutral (e.g. a text-only segment)
    return _reduce([r.hard_scalar for r in round_scores], how)


# ---------------------------------------------------------------------------
# fusion (hard x soft) with short-circuit
# ---------------------------------------------------------------------------
def fuse_segment(
    index: int,
    round_scores: Sequence[RoundScore],
    rubric_fn: Optional[Callable[[], Any]] = None,
    *,
    hard_agg: str = 'gmean',
    fusion: str = 'product',
    hard_floor: float = 0.25,
    hard_ceil_skip: Optional[float] = None,
    num_levels: int = NUM_LEVELS,
) -> SegmentScore:
    """Fuse per-round hard scores with a (lazily computed) segment rubric score.

    Args:
        index: segment index.
        round_scores: per-round hard scores (already computed; cheap/code-only).
        rubric_fn: zero-arg callable returning a rubric ScoreDetail (something
            with a ``.scalar`` attribute) OR a float. Called ONLY when the soft
            chain is not short-circuited — this is what saves the long LLM path.
        hard_agg: reducer for per-round hard scores ('gmean'|'mean'|'min'|...).
        fusion: how to combine hard_agg and rubric:
            'product'  -> hard_agg * rubric   (hard acts as a floor/gatekeeper)
            'hard_soft_blend' -> product, but when hard is high blend rubric toward
                a floor so all-pass tool segments are not one-shot vetoed
            'min'      -> min(hard_agg, rubric)
            'mean'     -> (hard_agg + rubric)/2
            'hard_only'-> ignore rubric entirely
        hard_floor: if aggregated hard score < this, SHORT-CIRCUIT: skip rubric,
            segment score = hard_agg (the answer is already decided as bad).
        hard_ceil_skip: (optional) if aggregated hard score >= this, also skip
            rubric and use hard_agg. Set None to disable. Useful for
            trivially-good tool-only segments where soft quality adds little.
        num_levels: level discretization.
    """
    hard_agg_val = aggregate_hard_over_rounds(round_scores, how=hard_agg)

    short = False
    rubric_scalar: Optional[float] = None

    if fusion == 'hard_only' or rubric_fn is None:
        scalar = hard_agg_val
        short = True
    elif hard_agg_val < hard_floor:
        # bad hard signal -> don't waste the soft chain
        scalar = hard_agg_val
        short = True
    elif hard_ceil_skip is not None and hard_agg_val >= hard_ceil_skip:
        scalar = hard_agg_val
        short = True
    else:
        rubric_scalar = _rubric_scalar(rubric_fn())
        scalar = _combine(hard_agg_val, rubric_scalar, fusion)

    return SegmentScore(
        index=index,
        scalar=scalar,
        level=scalar_to_level(scalar, num_levels),
        hard_scalar=hard_agg_val,
        rubric_scalar=rubric_scalar,
        short_circuited=short,
        n_rounds=len(round_scores),
        rounds=list(round_scores),
    )


def _rubric_scalar(result: Any) -> float:
    if result is None:
        return 0.0
    if isinstance(result, (int, float)):
        return float(result)
    scalar = getattr(result, 'scalar', None)
    return float(scalar) if scalar is not None else 0.0


def _combine(hard: float, soft: float, fusion: str) -> float:
    if fusion == 'product':
        return hard * soft
    if fusion == 'hard_soft_blend':
        # When hard checks are strong (tool/format all pass), a harsh rubric on a
        # long agent trace must not one-shot veto the segment (product → ~0.08).
        if hard >= 0.9:
            mix = 0.55 * soft + 0.45
        elif hard >= 0.75:
            mix = 0.75 * soft + 0.25
        else:
            mix = soft
        return hard * mix
    if fusion == 'min':
        return min(hard, soft)
    if fusion == 'mean':
        return (hard + soft) / 2.0
    raise ValueError(f'unknown fusion {fusion!r}')


# ---------------------------------------------------------------------------
# segment -> trajectory
# ---------------------------------------------------------------------------
def aggregate_trajectory(
    segment_scores: Sequence[SegmentScore],
    *,
    how: str = 'mean',
    weight_by_rounds: bool = True,
    num_levels: int = NUM_LEVELS,
) -> TrajectoryScore:
    """Aggregate segment scores into a trajectory score.

    Args:
        how: reducer when ``weight_by_rounds`` is False.
        weight_by_rounds: when True, weight each segment by its round count so
            longer sub-tasks count proportionally (ignores ``how``, uses a
            round-weighted mean). Text-only segments count as weight 1.
    """
    if not segment_scores:
        return TrajectoryScore(scalar=0.0, level=0, segments=[])
    if weight_by_rounds:
        num = 0.0
        den = 0.0
        for s in segment_scores:
            w = max(1, s.n_rounds)
            num += s.scalar * w
            den += w
        scalar = num / den if den else 0.0
    else:
        scalar = _reduce([s.scalar for s in segment_scores], how)
    return TrajectoryScore(
        scalar=scalar,
        level=scalar_to_level(scalar, num_levels),
        segments=list(segment_scores),
    )
