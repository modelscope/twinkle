"""Intent-keyed rubric library (DESIGN follow-up: stabilize rubric scoring).

Rubric *generation* is flexible but high-variance: for template-like intents
(tool_call / code / math) the model re-invents slightly different criteria every
call, which is the main source of score jitter and occasional task-type
misreads. This module supplies two levels of stabilization, both keyed by the
intent vocabulary in :mod:`twinkle_agentic.preprocessor.intents`:

- ``INTENT_BASE_RUBRICS``  — half-fixed **skeletons**: a small, stable core of
  criteria that is PREPENDED to the distilled rubric. The generator still adds
  task-specific criteria on top, so flexibility is preserved while the shared
  core makes scores comparable across similar segments. This is the DEFAULT
  policy (does NOT sacrifice flexibility).
- ``INTENT_FIXED_RUBRICS`` — fully-fixed rubrics per intent (no generation).
  Maximum stability, minimum flexibility; opt-in for callers that want it.

Each criterion is written to match the grader prompt conventions:
- starts with "The response" / "The agent",
- [Hard Rule] for objectively checkable constraints, [Principle] for quality,
- scoped to what is observable INSIDE one segment (never assumes later steps).

Criteria are deliberately GENERIC (no entities/values) so a single skeleton
generalizes across all segments of that intent.
"""
from typing import Dict, List

from .rubric_verifier import RubricItem

# Re-export intent constants so callers wire the library without importing the
# heavier classifier module.
from ..preprocessor.intents import (INTENT_CODE, INTENT_MATH,  # noqa: F401
                                     INTENT_TOOL_CALL)


def _h(text: str) -> RubricItem:
    return RubricItem(text=text, is_hard=True)


def _p(text: str) -> RubricItem:
    return RubricItem(text=text, is_hard=False)


# --------------------------------------------------------------------------- #
# Half-fixed skeletons (DEFAULT). Kept intentionally short (2-3 items) so the
# distilled generator still supplies the bulk of task-specific coverage.
# --------------------------------------------------------------------------- #
_TOOL_CALL_SKELETON: List[RubricItem] = [
    _h('The agent emits tool calls whose arguments are valid, complete JSON '
       'matching the tool schema'),
    _h('The agent selects tools appropriate to the sub-goal and does not invent '
       'unavailable tools or arguments'),
    _p('The agent uses each tool result to advance the sub-goal without '
       'redundant or repeated identical calls'),
]

_CODE_SKELETON: List[RubricItem] = [
    _h('The response produces code that is syntactically well-formed and '
       'self-consistent within the segment'),
    _p('The response addresses the stated coding sub-goal with correct, relevant '
       'logic rather than placeholder or off-topic code'),
    _p('The response avoids obvious defects (undefined names, wrong API usage) '
       'visible within the segment'),
]

_MATH_SKELETON: List[RubricItem] = [
    _h('The response performs each mathematical step correctly with no '
       'arithmetic or algebraic error visible in the segment'),
    _p('The response follows a valid, coherent solution path toward the '
       'sub-goal without unjustified leaps'),
    _p('The response states intermediate/final results clearly and consistently '
       'with the work shown'),
]

INTENT_BASE_RUBRICS: Dict[str, List[RubricItem]] = {
    INTENT_TOOL_CALL: _TOOL_CALL_SKELETON,
    INTENT_CODE: _CODE_SKELETON,
    INTENT_MATH: _MATH_SKELETON,
}


# --------------------------------------------------------------------------- #
# Fully-fixed rubrics (opt-in). Same criteria plus a couple more so the fixed
# set is self-sufficient without any generation.
# --------------------------------------------------------------------------- #
INTENT_FIXED_RUBRICS: Dict[str, List[RubricItem]] = {
    INTENT_TOOL_CALL: _TOOL_CALL_SKELETON + [
        _p('The agent grounds its next action in the actual tool output rather '
           'than hallucinating results'),
    ],
    INTENT_CODE: _CODE_SKELETON + [
        _p('The response explains or structures the code enough to be usable in '
           'the surrounding task context'),
    ],
    INTENT_MATH: _MATH_SKELETON + [
        _p('The response keeps units, signs and notation consistent throughout '
           'the segment'),
    ],
}


def default_intent_base_rubrics() -> Dict[str, List[RubricItem]]:
    """The recommended half-fixed policy (flexible + stabilized)."""
    return {k: list(v) for k, v in INTENT_BASE_RUBRICS.items()}


def default_intent_fixed_rubrics() -> Dict[str, List[RubricItem]]:
    """The opt-in fully-fixed policy (max stability, min flexibility)."""
    return {k: list(v) for k, v in INTENT_FIXED_RUBRICS.items()}
