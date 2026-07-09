# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-round / per-segment / per-trajectory scoring — tag only, never drop (AUDIT D7).

This preprocessor wires the existing ``segment`` + ``verifier`` + ``aggregation``
infrastructure into the cleaning pipeline. It is a **mapper**: it never removes a
row, it only writes scores into ``user_data`` (see :mod:`label_schema`, A5). A
downstream tail filter (``TrajectoryOutcomeFilter``, D6) reads those labels and
decides what to drop — so scoring and filtering stay decoupled and the pipeline
remains a linear list (no DAG, no filter↔verifier code coupling).

Flow per trajectory::

    Segmenter(traj) ─► segments
      for each segment:
        split_segment_into_rounds ─► rounds
        HardScorer(round) ─► RoundScore              (per-round hard scalar)
        fuse_segment(round_scores, rubric_fn) ─► SegmentScore
          └ rubric_fn lazily calls RubricVerifier ONLY when not short-circuited
      aggregate_trajectory(segment_scores) ─► TrajectoryScore

Labels written (all JSON-packed, PyArrow-stable):
    round_scores, round_gated, segment_scores, traj_score, traj_level, score_meta.

The ``RubricVerifier`` is optional: when no sampler/teacher is available the soft
chain returns an empty score and fusion falls back to the hard signal, so the
scorer still produces useful per-round hard scores with zero LLM calls.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger

from . import label_schema as L

logger = get_logger()


class TrajectoryScorer(Preprocessor):
    """Score every trajectory and write the scores into ``user_data`` (never drops).

    Args:
        segmenter: a :class:`~twinkle_agentic.segment.base.Segmenter`. Defaults to
            structural ``TurnSegmenter('cluster')`` (LLM-free).
        hard_scorer: a :class:`~twinkle_agentic.verifier.HardScorer` (per-round,
            deterministic). Defaults to a plain ``HardScorer()``.
        rubric_verifier: optional :class:`~twinkle_agentic.verifier.RubricVerifier`
            (per-segment, soft/LLM). If ``None``, only hard scores are used.
        hard_agg / fusion / hard_floor / hard_ceil_skip: passed to
            :func:`~twinkle_agentic.verifier.fuse_segment`.
        traj_agg / weight_by_rounds: passed to
            :func:`~twinkle_agentic.verifier.aggregate_trajectory`.
        write_round_detail: also store per-check breakdown into ``score_meta``.
    """

    def __init__(
        self,
        segmenter: Optional[Any] = None,
        hard_scorer: Optional[Any] = None,
        rubric_verifier: Optional[Any] = None,
        *,
        hard_agg: str = 'gmean',
        fusion: str = 'product',
        hard_floor: float = 0.25,
        hard_ceil_skip: Optional[float] = None,
        traj_agg: str = 'mean',
        weight_by_rounds: bool = True,
        write_round_detail: bool = False,
        calibrate: bool = True,
        disagree_margin: float = 0.34,
    ):
        # Lazy imports keep the module importable even if verifier/segment deps
        # are heavy; construction still fails loudly if the packages are absent.
        from twinkle_agentic.segment import TurnSegmenter
        from twinkle_agentic.verifier import HardScorer

        self.segmenter = segmenter if segmenter is not None else TurnSegmenter('cluster')
        self.hard_scorer = hard_scorer if hard_scorer is not None else HardScorer()
        self.rubric_verifier = rubric_verifier
        self.hard_agg = hard_agg
        self.fusion = fusion
        self.hard_floor = float(hard_floor)
        self.hard_ceil_skip = hard_ceil_skip
        self.traj_agg = traj_agg
        self.weight_by_rounds = bool(weight_by_rounds)
        self.write_round_detail = bool(write_round_detail)
        # D7c: self-evolving calibration (no human alignment).
        self.calibrate = bool(calibrate)
        self.disagree_margin = float(disagree_margin)

    # ------------------------------------------------------------------
    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                out.append(self._score_row(row))
            except Exception as e:  # scoring must never break the pipeline
                logger.warning(f'[TrajectoryScorer] scoring failed, row left unscored: {e}')
                out.append(row)
        return out, []  # mapper: never drops

    # ------------------------------------------------------------------
    def _score_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        from twinkle_agentic.verifier import (aggregate_trajectory, fuse_segment,
                                              split_segment_into_rounds)
        from twinkle_agentic.verifier.aggregation import RoundScore

        messages = row.get('messages')
        if not isinstance(messages, list) or not messages:
            return row

        trajectory = {'messages': messages}
        if row.get('tools'):
            trajectory['tools'] = row['tools']

        segments = self.segmenter.segment(trajectory)
        if not segments:
            return row

        query = self._infer_query(messages)
        all_round_scalars: List[float] = []
        all_round_gated: List[bool] = []
        segment_scalars: List[float] = []
        segment_confidence: List[float] = []
        segment_scores = []

        for s_idx, segment in enumerate(segments):
            rounds = split_segment_into_rounds(segment)
            round_scores = []
            for r_idx, rnd in enumerate(rounds):
                detail = self.hard_scorer.score_detail(rnd)
                round_scores.append(RoundScore(
                    index=r_idx,
                    hard_scalar=detail.scalar,
                    gated=detail.gated,
                    detail=detail if self.write_round_detail else None,
                ))
                all_round_scalars.append(detail.scalar)
                all_round_gated.append(detail.gated)

            rubric_fn = self._make_rubric_fn(segment, query, round_scores)
            seg_score = fuse_segment(
                s_idx, round_scores, rubric_fn,
                hard_agg=self.hard_agg, fusion=self.fusion,
                hard_floor=self.hard_floor, hard_ceil_skip=self.hard_ceil_skip,
            )
            # carry the rubric ScoreDetail (stashed by rubric_fn) for confidence
            seg_score.detail = segment.pop('_last_rubric', None)
            segment_scores.append(seg_score)
            segment_scalars.append(seg_score.scalar)
            segment_confidence.append(self._segment_confidence(seg_score))

        traj = aggregate_trajectory(
            segment_scores, how=self.traj_agg, weight_by_rounds=self.weight_by_rounds)

        labels: Dict[str, Any] = {
            L.KEY_ROUND_SCORES: [round(x, 6) for x in all_round_scalars],
            L.KEY_ROUND_GATED: all_round_gated,
            L.KEY_SEGMENT_SCORES: [round(x, 6) for x in segment_scalars],
            L.KEY_TRAJ_SCORE: round(traj.scalar, 6),
            L.KEY_TRAJ_LEVEL: traj.level,
        }
        if self.calibrate:
            labels[L.KEY_SEGMENT_CONFIDENCE] = [round(c, 6) for c in segment_confidence]
            labels[L.KEY_TRAJ_CONFIDENCE] = round(
                sum(segment_confidence) / len(segment_confidence), 6) if segment_confidence else 1.0
        if self.write_round_detail:
            labels[L.KEY_SCORE_META] = {
                'n_segments': len(segment_scores),
                'short_circuited': [s.short_circuited for s in segment_scores],
                'segment_hard': [round(s.hard_scalar, 6) for s in segment_scores],
                'segment_rubric': [
                    None if s.rubric_scalar is None else round(s.rubric_scalar, 6)
                    for s in segment_scores
                ],
            }
        return L.set_labels(row, labels)

    # ------------------------------------------------------------------
    def _make_rubric_fn(self, segment: dict, query: str, round_scores):
        """Return a zero-arg callable for the soft chain, or None if unavailable.

        ``fuse_segment`` only invokes this when the segment is NOT short-circuited,
        so the expensive LLM path runs exactly when the hard signal is inconclusive.

        D7c objective→subjective correction: score once, and if the rubric verdict
        disagrees with the deterministic hard signal beyond ``disagree_margin``,
        re-score with the objective evidence folded into the transcript so the
        judge revises WITH the hard facts in view. The last ``ScoreDetail`` is
        stashed on the segment (``_last_rubric``) for confidence estimation.
        """
        rv = self.rubric_verifier
        if rv is None:
            return None
        from twinkle_agentic.verifier.aggregation import aggregate_hard_over_rounds
        hard_agg_val = aggregate_hard_over_rounds(round_scores, how=self.hard_agg)

        def _fn():
            detail = rv.score_detail(segment, query=query)
            if self.calibrate and detail is not None and getattr(detail, 'scalar', None) is not None:
                if abs(detail.scalar - hard_agg_val) >= self.disagree_margin:
                    evidence = (f'Deterministic tool/answer checks scored this '
                                f'segment {hard_agg_val:.2f} out of 1.0. Reconcile '
                                f'your assessment with this objective evidence.')
                    revised = rv.score_detail(segment, query=query, extra_context=evidence)
                    if revised is not None:
                        detail = revised
            segment['_last_rubric'] = detail
            return detail

        return _fn

    # ------------------------------------------------------------------
    def _segment_confidence(self, seg_score) -> float:
        """Self-evolving confidence in a segment score (no human labels).

        Combines three automatic signals:
        1. hard↔rubric agreement — 1 minus their absolute gap (objective anchoring);
        2. voting stability — fewer escalated votes ⇒ the judge was decisive;
        3. decisiveness — distance of the fused score from the ambiguous 0.5 band.
        Short-circuited (hard-only) segments are highly confident by construction.
        """
        if seg_score.short_circuited or seg_score.rubric_scalar is None:
            return 1.0
        agree = 1.0 - min(1.0, abs(seg_score.hard_scalar - seg_score.rubric_scalar))
        detail = getattr(seg_score, 'detail', None)
        n_votes = getattr(detail, 'n_votes', 1) or 1
        max_votes = getattr(self.rubric_verifier, 'max_votes', 1) or 1
        stability = 1.0 if max_votes <= 1 else 1.0 - (n_votes - 1) / max(1, max_votes - 1)
        decisive = min(1.0, abs(seg_score.scalar - 0.5) * 2.0)
        return max(0.0, min(1.0, 0.5 * agree + 0.3 * stability + 0.2 * decisive))

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_query(messages: List[dict]) -> str:
        for m in messages:
            if isinstance(m, dict) and m.get('role') == 'user':
                c = m.get('content')
                if isinstance(c, str) and c.strip():
                    return c.strip()
        return '(no explicit query)'
