# Copyright (c) ModelScope Contributors. All rights reserved.
"""Active-learning pre-selection for the expensive rubric pass.

Motivation
----------
At high daily volume it is neither affordable to LLM-label every trajectory nor
smart to sample at random (most rows are unremarkable). This mapper assigns each
row a **cheap, fully deterministic** ``value_score`` — an estimate of how much an
expensive rubric/LLM pass would *learn* from it — so a downstream gate can send
only the top fraction to the LLM. Self-evolution needs few, well-chosen samples.

The score is a weighted blend of three LLM-free signals (all in ``[0, 1]``):

- **uncertainty** — how close the deterministic hard signal is to undecided.
  A row the hard checks already call clearly good (all 1.0) or clearly bad
  (all 0.0) teaches the LLM little; rows near the boundary, or with internal
  disagreement across rounds, are where a rubric pass pays off most.
- **difficulty** — structural complexity (rounds, tool calls, distinct tools,
  segments), log-compressed so a few giant traces don't dominate. Long agentic
  traces carry more signal than single-turn chit-chat.
- **error_signal** — deterministic failure evidence (gated rounds, failed
  tool execution / termination / repetition checks). Mistakes are valuable
  learning material for self-evolution (negative / correction examples).

Two-pass usage (see ``TrajectoryScorer`` gate)
----------------------------------------------
1. Run this mapper over the full stream (parallel, no global state) to stamp
   ``value_score`` on every row.
2. After ``map`` completes, in the *single* driver process call
   :func:`select_top_for_rubric` to flip ``selected_for_rubric=True`` on the
   global top ``select_frac``. Only those rows spend an LLM call.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger

from . import label_schema as L
from .utils import normalize_tool_calls

logger = get_logger()


def _log_norm(x: float, cap: float) -> float:
    """Log-compress a count into [0, 1], saturating at ``cap``."""
    if x <= 0:
        return 0.0
    return min(1.0, math.log1p(x) / math.log1p(cap))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pstdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


class ValueSelector(Preprocessor):
    """Stamp a deterministic ``value_score`` on every row (never drops).

    Args:
        hard_scorer: a :class:`~twinkle_agentic.verifier.HardScorer` (reused for
            the per-round hard scalars that feed ``uncertainty`` / ``error``).
            Defaults to a plain ``HardScorer()``. No LLM is ever called.
        segmenter: a segmenter for round splitting. Defaults to
            ``TurnSegmenter('cluster')`` (LLM-free, same as TrajectoryScorer).
        w_uncertainty / w_difficulty / w_error: blend weights (need not sum to 1;
            normalized internally).
        rounds_cap / toolcalls_cap / tools_cap / segments_cap: saturation caps
            for the difficulty sub-signals.
        write_meta: also store the per-component breakdown under ``value_meta``.
    """

    def __init__(
        self,
        hard_scorer: Optional[Any] = None,
        segmenter: Optional[Any] = None,
        *,
        w_uncertainty: float = 0.45,
        w_difficulty: float = 0.30,
        w_error: float = 0.25,
        rounds_cap: int = 20,
        toolcalls_cap: int = 15,
        tools_cap: int = 6,
        segments_cap: int = 8,
        write_meta: bool = True,
    ):
        from twinkle_agentic.segment import TurnSegmenter
        from twinkle_agentic.verifier import HardScorer

        self.hard_scorer = hard_scorer if hard_scorer is not None else HardScorer()
        self.segmenter = segmenter if segmenter is not None else TurnSegmenter('cluster')
        total = w_uncertainty + w_difficulty + w_error
        if total <= 0:
            raise ValueError('at least one value weight must be > 0')
        self.w_uncertainty = w_uncertainty / total
        self.w_difficulty = w_difficulty / total
        self.w_error = w_error / total
        self.rounds_cap = int(rounds_cap)
        self.toolcalls_cap = int(toolcalls_cap)
        self.tools_cap = int(tools_cap)
        self.segments_cap = int(segments_cap)
        self.write_meta = bool(write_meta)

    # ------------------------------------------------------------------
    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out = []
        for row in rows:
            try:
                out.append(self._score_row(row))
            except Exception as e:  # never break the pipeline on a bad row
                logger.warning(f'[ValueSelector] scoring failed, value=0: {e}')
                out.append(L.set_label(row, L.KEY_VALUE_SCORE, 0.0))
        return out, []  # mapper: never drops

    # ------------------------------------------------------------------
    def _score_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        from twinkle_agentic.verifier import split_segment_into_rounds

        messages = row.get('messages')
        if not isinstance(messages, list) or not messages:
            return L.set_label(row, L.KEY_VALUE_SCORE, 0.0)

        trajectory = {'messages': messages}
        if row.get('tools'):
            trajectory['tools'] = row['tools']

        segments = self.segmenter.segment(trajectory) or []

        round_scalars: List[float] = []
        any_gated = False
        soft_fail = 0.0  # worst non-critical hard-check miss across rounds
        for segment in segments:
            for rnd in split_segment_into_rounds(segment):
                detail = self.hard_scorer.score_detail(rnd)
                round_scalars.append(detail.scalar)
                if detail.gated:
                    any_gated = True
                soft_fail = max(soft_fail, self._soft_fail(detail))

        uncertainty = self._uncertainty(round_scalars)
        difficulty = self._difficulty(messages, segments)
        error_signal = self._error_signal(any_gated, soft_fail)

        value = (self.w_uncertainty * uncertainty
                 + self.w_difficulty * difficulty
                 + self.w_error * error_signal)
        value = max(0.0, min(1.0, value))

        updates: Dict[str, Any] = {L.KEY_VALUE_SCORE: round(value, 6)}
        if self.write_meta:
            updates[L.KEY_VALUE_META] = {
                'uncertainty': round(uncertainty, 4),
                'difficulty': round(difficulty, 4),
                'error': round(error_signal, 4),
                'n_rounds': len(round_scalars),
            }
        return L.set_labels(row, updates)

    # ------------------------------------------------------------------
    # signal components
    # ------------------------------------------------------------------
    @staticmethod
    def _uncertainty(round_scalars: List[float]) -> float:
        """High when the hard signal is undecided OR rounds disagree."""
        if not round_scalars:
            return 0.0
        mean_hard = _mean(round_scalars)
        central = 1.0 - abs(2.0 * mean_hard - 1.0)          # peak at 0.5
        disagreement = min(1.0, 2.0 * _pstdev(round_scalars))  # spread across rounds
        return max(central, disagreement)

    def _difficulty(self, messages: List[dict], segments: List[dict]) -> float:
        n_rounds = sum(1 for m in messages
                       if isinstance(m, dict) and m.get('role') == 'assistant')
        n_toolcalls = 0
        tool_names = set()
        for m in messages:
            if not isinstance(m, dict) or m.get('role') != 'assistant':
                continue
            for tc in (normalize_tool_calls(m) or []):
                n_toolcalls += 1
                fn = (tc.get('function') or {}) if isinstance(tc, dict) else {}
                name = fn.get('name') if isinstance(fn, dict) else None
                if name:
                    tool_names.add(name)
        n_segments = len(segments)
        return _mean([
            _log_norm(n_rounds, self.rounds_cap),
            _log_norm(n_toolcalls, self.toolcalls_cap),
            _log_norm(len(tool_names), self.tools_cap),
            _log_norm(n_segments, self.segments_cap),
        ])

    @staticmethod
    def _soft_fail(detail: Any) -> float:
        """Worst miss among informative non-critical checks in a round (0..1)."""
        watch = {'tool_executed', 'clean_termination', 'no_repeated_calls',
                 'protocol_pairing', 'final_answer'}
        worst = 0.0
        for c in getattr(detail, 'checks', None) or []:
            if getattr(c, 'name', None) in watch:
                worst = max(worst, 1.0 - float(getattr(c, 'score', 1.0)))
        return worst

    @staticmethod
    def _error_signal(any_gated: bool, soft_fail: float) -> float:
        """Deterministic evidence the model made a mistake worth studying."""
        if any_gated:
            return 1.0
        return soft_fail


# ---------------------------------------------------------------------------
# global top-fraction selection (driver process, after Dataset.map)
# ---------------------------------------------------------------------------
def select_top_for_rubric(
    dataset,
    *,
    select_frac: float = 0.1,
    min_select: int = 0,
    max_select: Optional[int] = None,
    value_key: str = L.KEY_VALUE_SCORE,
    selected_key: str = L.KEY_SELECTED_FOR_RUBRIC,
):
    """Flip ``selected_for_rubric`` on the global top-``select_frac`` by value.

    Must run in the single driver process AFTER ``Dataset.map`` (the top
    fraction is a global order that per-shard workers cannot compute). Returns
    ``(dataset, n_selected)``; the dataset is mutated via a lightweight map.

    Ties at the cutoff are all included (selection is by a value threshold), so
    the realized count can slightly exceed ``select_frac * N``.
    """
    hf = dataset.dataset
    n = len(hf)
    if n == 0:
        return dataset, 0

    def _val(row) -> float:
        v = L.get_label(row, value_key, 0.0)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    values = sorted((_val(hf[i]) for i in range(n)), reverse=True)
    k = int(round(select_frac * n))
    if min_select:
        k = max(k, min_select)
    if max_select is not None:
        k = min(k, max_select)
    k = max(0, min(k, n))
    if k == 0:
        threshold = float('inf')
    else:
        threshold = values[k - 1]

    def _mark(batch):
        rows = Preprocessor.map_col_to_row(batch)
        out = [L.set_label(r, selected_key, _val(r) >= threshold) for r in rows]
        return Preprocessor.map_row_to_col(out, keys=list(batch.keys()))

    marked = hf.map(_mark, batched=True, load_from_cache_file=False,
                    remove_columns=list(hf.column_names))
    # Write back to BOTH views so the next Dataset.map sees the marks: twinkle's
    # Dataset.map operates on self.datasets[key] (not self.dataset), so updating
    # only self.dataset would silently drop selected_for_rubric before pass 2.
    dataset.dataset = marked
    datasets = getattr(dataset, 'datasets', None)
    if isinstance(datasets, dict):
        for k in list(datasets.keys()):
            if datasets[k] is hf or len(datasets) == 1:
                datasets[k] = marked

    n_selected = sum(1 for i in range(len(marked))
                     if L.get_label(marked[i], selected_key, False))
    logger.info(f'[ValueSelector] selected {n_selected}/{n} rows for rubric '
                f'(frac={select_frac}, threshold={threshold:.4f})')
    return dataset, n_selected
