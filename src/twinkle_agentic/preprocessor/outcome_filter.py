# Copyright (c) ModelScope Contributors. All rights reserved.
"""Drop failed / dead-end trajectories by reading scores — pure tag reader (AUDIT D6).

This filter does **not** compute anything. It reads the scores that
:class:`TrajectoryScorer` (D7) already wrote into ``user_data`` and drops rows
whose trajectory score / safety score fall below configurable thresholds. Because
it only consumes labels, the dependency on the scorer is a *data* dependency
(scorer writes ``traj_score``, this reads it) enforced simply by pipeline order —
no module import of the verifier, no DAG.

Thresholds are meant to be **set by default and then tuned against the observed
score distribution** (self-evolving framework: no human-labeled calibration set).
A row with no score label is kept by default (fail-open) so that placing this
filter before the scorer, or scoring being disabled, never silently drops data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger

from . import label_schema as L

logger = get_logger()


class TrajectoryOutcomeFilter(Preprocessor):
    """Drop trajectories whose written scores fall below thresholds (reads only).

    Args:
        min_traj_score: drop if ``traj_score`` < this. ``None`` disables.
        min_safety_score: drop if ``safety_score`` < this. ``None`` disables.
        drop_unsafe_flag: drop if ``safety_unsafe`` is True. Default True.
        require_score: if True, rows with no ``traj_score`` label are DROPPED
            (fail-closed); default False keeps them (fail-open) so a mis-ordered
            or scorer-disabled pipeline never silently deletes data.
    """

    def __init__(
        self,
        *,
        min_traj_score: float = 0.25,
        min_safety_score: float = 0.5,
        drop_unsafe_flag: bool = True,
        require_score: bool = False,
    ):
        self.min_traj_score = min_traj_score
        self.min_safety_score = min_safety_score
        self.drop_unsafe_flag = bool(drop_unsafe_flag)
        self.require_score = bool(require_score)

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for row in rows:
            reason = self._drop_reason(row)
            if reason is None:
                kept.append(row)
            else:
                dropped.append(dict(row, drop_reason=reason))
        return kept, dropped

    def _drop_reason(self, row: Dict[str, Any]):
        traj = L.get_label(row, L.KEY_TRAJ_SCORE, None)
        if traj is None:
            if self.require_score:
                return 'no_score'
            # fail-open: unscored rows pass through
        elif self.min_traj_score is not None and float(traj) < self.min_traj_score:
            return 'low_traj_score'

        if self.drop_unsafe_flag and bool(L.get_label(row, L.KEY_SAFETY_UNSAFE, False)):
            return 'unsafe'
        safety = L.get_label(row, L.KEY_SAFETY_SCORE, None)
        if safety is not None and self.min_safety_score is not None \
                and float(safety) < self.min_safety_score:
            return 'low_safety_score'
        return None
