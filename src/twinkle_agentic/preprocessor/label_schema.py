# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unified ``user_data`` label envelope (AUDIT A5).

All scoring / safety / provenance annotations produced by the pipeline are
written into a trajectory's ``user_data`` as ``(key, pack_value(value))`` pairs.
This is the single data contract that lets us decouple *tagging* (mappers that
never drop) from *filtering* (a tail filter that only reads tags), so the whole
pipeline stays a linear ``QualityPreprocessor`` list — no DAG, no cross-module
imports between a filter and the verifier it depends on.

PyArrow hard constraint
-----------------------
``user_data`` MUST be a ``List[Tuple[str, str]]`` (see
``twinkle/data_format/trajectory.py``). We NEVER put a bare ``dict`` in a row
column: HF ``datasets``' PyArrow backend cannot stably serialize
heterogeneous / nested dicts. Structured values are JSON-encoded to a single
string via :func:`pack_value`; on read :func:`user_data_get` JSON-decodes them.

Keep this module dependency-light: it only knows the *keys* and thin get/set
helpers, so both preprocessors and (optionally) other modules can share it
without pulling in verifier/segment code.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from twinkle.data_format import pack_value, user_data_get

# ---------------------------------------------------------------------------
# Canonical label keys
# ---------------------------------------------------------------------------
# Per-round hard scores, aligned to assistant/round order within the trajectory.
# Value: List[float] in [0, 1].
KEY_ROUND_SCORES = 'round_scores'
# Per-round gated flags (a critical hard check zeroed the round). Value: List[bool].
KEY_ROUND_GATED = 'round_gated'

# Per-segment fused scores. Value: List[float] in [0, 1].
KEY_SEGMENT_SCORES = 'segment_scores'
# Per-segment score confidence (D7c calibration). Value: List[float] in [0, 1].
KEY_SEGMENT_CONFIDENCE = 'segment_confidence'

# Whole-trajectory fused score in [0, 1] and its discrete level.
KEY_TRAJ_SCORE = 'traj_score'
KEY_TRAJ_LEVEL = 'traj_level'
# Aggregate confidence for the trajectory score (D7c). Value: float in [0, 1].
KEY_TRAJ_CONFIDENCE = 'traj_confidence'

# Safety score in [0, 1] (D8, higher = safer) + boolean unsafe flag.
KEY_SAFETY_SCORE = 'safety_score'
KEY_SAFETY_UNSAFE = 'safety_unsafe'

# Provenance blob (D10): dict-like value JSON-encoded (source/teacher/student/ts).
KEY_PROVENANCE = 'provenance'

# Free-form scoring metadata (short-circuit stats, per-check breakdown, etc.).
KEY_SCORE_META = 'score_meta'

# Active-learning pre-selection (ValueSelector): a cheap, LLM-free "how worth an
# expensive rubric pass is this row" score in [0, 1], its per-component
# breakdown, and the boolean gate the rubric stage reads to decide whether to
# spend an LLM call on this row (top-fraction by value_score).
KEY_VALUE_SCORE = 'value_score'
KEY_VALUE_META = 'value_meta'
KEY_SELECTED_FOR_RUBRIC = 'selected_for_rubric'

# Persisted rubric diagnosis for rubric-scored rows: a per-segment verification
# chain (rubric text + per-criterion verdict/reason/fix + raw model output +
# query/segment_text). This is the SFT corpus for distilling a PRM / error-checker
# LoRA — store it so training never has to re-run the (expensive) teacher.
# Value: List[dict], one entry per rubric-scored segment (see TrajectoryScorer).
KEY_RUBRIC_DIAGNOSIS = 'rubric_diagnosis'


# ---------------------------------------------------------------------------
# thin get / set helpers over the (key, pack_value) envelope
# ---------------------------------------------------------------------------
def get_user_data(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return the row's ``user_data`` as a list (never a dict), defaulting to []."""
    ud = row.get('user_data')
    if ud is None:
        return []
    if isinstance(ud, list):
        return ud
    # Be forgiving of a stray dict (e.g. hand-authored rows) — flatten to pairs.
    if isinstance(ud, dict):
        return [(k, v if isinstance(v, str) else pack_value(v)) for k, v in ud.items()]
    return []


def get_label(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Read+JSON-decode the first label matching ``key`` from ``row['user_data']``."""
    return user_data_get(get_user_data(row), key, default)


def set_labels(row: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied row with ``updates`` merged into ``user_data``.

    Existing entries for the same keys are replaced (last-write-wins), preserving
    the original order for untouched keys. Values are packed with :func:`pack_value`
    so the column stays ``List[Tuple[str, str]]`` (PyArrow-stable).
    """
    if not updates:
        return row
    existing = get_user_data(row)
    replace = set(updates.keys())
    merged: List[Tuple[str, str]] = [(k, v) for (k, v) in existing if k not in replace]
    for k, v in updates.items():
        merged.append((k, pack_value(v)))
    new_row = dict(row)
    new_row['user_data'] = merged
    return new_row


def set_label(row: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """Convenience: set a single label."""
    return set_labels(row, {key: value})
