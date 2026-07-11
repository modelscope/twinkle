# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rubric-based verifier for a single (pre-segmented) trajectory segment.

Design follows the OpenRubrics -> RubricARROW line of work, adapted to this
repo's progressive-distillation setup:

1. **Two LLM stages, both distilled via ``llm_backup``**
   - *Rubric generation*: given the segment, produce a small set of scoring
     criteria, each tagged ``[Hard Rule]`` or ``[Principle]``.
   - *Rubric scoring*: given the segment + rubric, emit a per-criterion
     verdict. Scores are aggregated ARROW-style into one pointwise scalar.

2. **Code-level hard verification does NOT go through the LLM.**
   Tool-call success/failure, argument JSON validity and call formatting are
   checked deterministically (free + un-hackable) and blended in as a
   "gatekeeper" floor on the final score.

3. **Cost-aware scoring**: a single scoring pass yields a soft margin. Only
   when the judge is uncertain (|margin| small) do we escalate to majority
   voting, aggregating with a median / trimmed-mean (robust to outliers).

The public ``__call__`` returns an ``int`` in ``[0, NUM_LEVELS)`` per the
:class:`Verifier` contract. ``score_detail`` exposes the continuous score and
breakdown for callers that want the raw signal (e.g. an RL reward).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from twinkle_agentic.utils.llm_backup import llm_backup

from .base import Verifier

if TYPE_CHECKING:
    from twinkle.data_format import SamplingParams  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_GEN_SYSTEM = """\
You write evaluation rubrics for a single segment of an AI agent trajectory. \
The segment may contain reasoning, tool calls and tool results.

Produce a SHORT list of scoring criteria that discriminate a good segment from \
a bad one. Each criterion:
- starts with "The response" or "The agent",
- is checkable and non-overlapping (no two criteria testing the same thing),
- ends with a tag: [Hard Rule] for objectively verifiable constraints \
(tool actually called, argument schema valid, required output present) or \
[Principle] for softer quality (reasoning soundness, sub-goal progress, no \
redundant calls).

Scope discipline (critical — avoid over-strict, mismatched rubrics):
- This is ONE SEGMENT, possibly the MIDDLE of a longer task. Only write criteria \
about behavior that is OBSERVABLE INSIDE THIS SEGMENT. Do NOT invent criteria \
about a final deliverable, later steps, or task completion that this segment is \
not expected to reach (e.g. "registers the component", "updates the entry point").
- Infer the task type ONLY from what the segment actually does. Do NOT assume it \
is an "implement a feature" task unless the segment clearly shows that. When the \
segment only reads/inspects/answers, judge reading/answering quality, not delivery.
- Reasoning shown inside <think>...</think> (or <thinking>) is internal scratch \
work. Never write a criterion that penalizes the mere presence of such reasoning, \
and do NOT let it count against "output only X" style constraints.

Rules:
- Output {min_n}-{max_n} criteria, as FEW as needed to cover the key axes.
- Do NOT reference specific entities/values from THIS segment; keep criteria \
generalizable to similar segments.
- Output ONLY a numbered list, one criterion per line, nothing else.
"""

_GEN_USER = """\
## Task / query (context)
{query}

## Segment to build a rubric for
{segment}

Now output the numbered rubric list."""

_SCORE_SYSTEM = """\
You are a strict rubric grader for one segment of an agent trajectory.

You are given a rubric (numbered criteria, each tagged [Hard Rule] or \
[Principle]) and the segment. For EACH criterion output one line:

    <index>: PASS   or   <index>: FAIL

Judge every criterion independently and literally. A [Hard Rule] fails unless \
it is unambiguously satisfied.

Grading discipline:
- Judge ONLY what is observable in THIS segment; if a criterion asks about a \
step/deliverable this segment was not meant to reach, do not FAIL it for that \
alone — grade it satisfied when the in-segment behavior is correct.
- Content inside <think>...</think> (or <thinking>) is internal reasoning, not \
user-facing output. For "output only X / no extra text" style criteria, ignore \
such reasoning blocks; judge the actual response payload.

Output only the verdict lines, in order, then stop. Do not add explanations."""

_SCORE_USER = """\
## Task / query (context)
{query}

## Rubric
{rubric}

## Segment
{segment}

Now output one PASS/FAIL line per criterion, in order."""


# --- diagnostic mode: single call yields verdict + reason together ---------
# Used to distil an "on-the-fly error checker" LoRA (DESIGN §11.6). Unlike the
# terse scorer above, this asks for a COMPLETE verification chain over EVERY
# criterion (both pass and fail) so the distilled LoRA learns to also emit
# "checked, all good, continue" — not only to nitpick. Verdict and reason are
# produced in ONE pass so they can never disagree.
_DIAG_SYSTEM = """\
You are a process error checker for one segment of an agent trajectory. You are \
given a rubric (numbered criteria, each tagged [Hard Rule] or [Principle]) and \
the segment. Walk through EVERY criterion in order and, for each, decide PASS or \
FAIL and briefly justify it grounded in the segment.

Output STRICT JSON (no prose outside it) with this shape:
{
  "items": [
    {"index": 1, "verdict": "PASS", "reason": "<why, grounded in the segment>",
     "fix": ""},
    {"index": 2, "verdict": "FAIL", "reason": "<what is wrong and where>",
     "fix": "<one concrete, actionable correction>"}
  ],
  "overall": "OK" | "ISSUES",
  "summary": "<one sentence: 'no process errors, continue' OR the key issue(s)>"
}

Rules:
- Judge every criterion independently and literally; a [Hard Rule] is FAIL \
unless unambiguously satisfied.
- Judge ONLY what is observable in THIS segment; do not FAIL a criterion merely \
because a later step/deliverable it references is outside this segment's scope.
- Content inside <think>...</think> (or <thinking>) is internal reasoning, not \
user-facing output; ignore it for "output only X" style criteria.
- For PASS items, leave "fix" as "". For FAIL items, "fix" must be a concrete \
correction (e.g. add the missing argument, redo step k).
- Keep every "reason" and "fix" clear and concise — one short sentence each, \
stating only the essential point; do not restate the criterion, quote the segment \
at length, or add filler.
- "overall" is "OK" only if NO criterion is FAIL.
- Output only the JSON object."""

_DIAG_USER = """\
## Task / query (context)
{query}

## Rubric
{rubric}

## Segment
{segment}

Now output the diagnostic JSON object."""


# ---------------------------------------------------------------------------
# Data holders
# ---------------------------------------------------------------------------
_VERDICT_RE = re.compile(r'^\s*(\d+)\s*[:.)]\s*(pass|fail|true|false|yes|no|1|0)\b',
                         re.IGNORECASE)


@dataclass
class RubricItem:
    text: str
    is_hard: bool


@dataclass
class ScoreDetail:
    """Full breakdown behind the final integer level."""
    level: int
    scalar: float                       # continuous pointwise score in [0, 1]
    llm_scalar: float                   # LLM (principle+softhard) component in [0, 1]
    hard_pass_rate: float               # code-verified hard-rule pass rate in [0, 1]
    gated: bool                         # True if code gatekeeper capped the score
    n_votes: int                        # scoring passes actually spent
    rubric: List[RubricItem] = field(default_factory=list)
    per_item_pass_rate: List[float] = field(default_factory=list)


@dataclass
class DiagnosisItem:
    """Per-criterion diagnostic verdict with its justification."""
    index: int
    verdict: bool                       # True == PASS
    reason: str = ''
    fix: str = ''                       # concrete correction, only for FAIL


@dataclass
class DiagnoseDetail:
    """A complete verification chain over one segment (DESIGN §11.6).

    Produced in a single LLM call so verdict and reason are always consistent.
    Covers EVERY criterion (pass and fail) so a distilled checker learns to emit
    "checked, no error, continue" as well as concrete fault localisation.
    """
    scalar: float                       # aggregated pointwise score in [0, 1]
    overall_ok: bool                    # True == no criterion failed
    summary: str                        # one-line human-readable conclusion
    items: List[DiagnosisItem] = field(default_factory=list)
    rubric: List[RubricItem] = field(default_factory=list)
    raw: str = ''                       # raw model output (for SFT targets)
    query: str = ''                     # task/query context (for SFT inputs)
    segment_text: str = ''              # rendered segment (for SFT inputs)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
class RubricVerifier(Verifier):
    """Score one trajectory segment on a 0..NUM_LEVELS-1 scale via auto rubrics.

    Args:
        sampler: Student model sampler (local inference). If ``None`` the
            verifier still works but every LLM call is served by the teacher
            API through ``llm_backup`` (useful before a student exists).
        model_path: Model identifier (bookkeeping only).
        sampling_params: Default sampling params for LLM calls.
        gen_lora_path: LoRA adapter for the rubric-generator student.
        score_lora_path: LoRA adapter for the rubric-scorer student.
        min_rubrics / max_rubrics: Target rubric-count window per segment.
        hard_weight / principle_weight: Aggregation weights (ARROW uses 3 / 1).
        margin_threshold: |margin| below which we escalate to voting.
        max_votes: Voting cap for uncertain segments (odd recommended).
        gate_floor_ratio: If code-verified hard rules fail, the final scalar is
            capped at ``hard_pass_rate`` (gatekeeper). Set to 1.0 to hard-cap,
            0.0 to disable gating.
    """

    def __init__(
        self,
        sampler: Optional['Sampler'] = None,
        *,
        model_path: str = '',
        sampling_params: Optional['SamplingParams'] = None,
        gen_lora_path: Optional[str] = None,
        score_lora_path: Optional[str] = None,
        min_rubrics: int = 5,
        max_rubrics: int = 8,
        hard_weight: float = 3.0,
        principle_weight: float = 1.0,
        margin_threshold: float = 0.25,
        max_votes: int = 5,
        gate: bool = True,
        fixed_rubric: Optional[List['RubricItem']] = None,
        base_rubric: Optional[List['RubricItem']] = None,
        intent_rubrics: Optional[Dict[str, List['RubricItem']]] = None,
        intent_base_rubrics: Optional[Dict[str, List['RubricItem']]] = None,
        max_segment_chars: int = 14_000,
        long_segment_chars: int = 8_000,
        min_votes_long: int = 3,
        long_margin_threshold: float = 0.18,
        max_votes_long: int = 3,
        min_votes_high: int = 3,
        high_score_threshold: float = 0.85,
        diag_max_tokens: int = 2048,
    ):
        if max_rubrics < min_rubrics:
            raise ValueError('max_rubrics must be >= min_rubrics')
        if min_rubrics < 1:
            raise ValueError('min_rubrics must be >= 1')
        if hard_weight <= 0 or principle_weight <= 0:
            raise ValueError('weights must be > 0')
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError('margin_threshold must be in [0, 1]')
        if max_votes < 1:
            raise ValueError('max_votes must be >= 1')

        self.sampler = sampler
        self.model_path = model_path
        self.sampling_params = sampling_params
        self.gen_lora_path = gen_lora_path or None
        self.score_lora_path = score_lora_path or None
        self.min_rubrics = int(min_rubrics)
        self.max_rubrics = int(max_rubrics)
        self.hard_weight = float(hard_weight)
        self.principle_weight = float(principle_weight)
        self.margin_threshold = float(margin_threshold)
        self.max_votes = int(max_votes)
        self.gate = bool(gate)
        self.max_segment_chars = int(max_segment_chars)
        self.long_segment_chars = int(long_segment_chars)
        self.min_votes_long = max(1, int(min_votes_long))
        self.long_margin_threshold = float(long_margin_threshold)
        self.max_votes_long = max(1, int(max_votes_long))
        # High-confidence band: force at least this many votes when the first
        # pass lands >= high_score_threshold, so 4/4-looking "level 4" segments
        # are not decided by a single lucky sample (reduces high-band variance).
        self.min_votes_high = max(1, int(min_votes_high))
        self.high_score_threshold = float(high_score_threshold)
        # Diagnosis emits a full per-criterion (verdict+reason+fix) JSON; it needs
        # a far larger token budget than terse scoring or it truncates mid-JSON.
        self.diag_max_tokens = max(256, int(diag_max_tokens))
        # When provided, skip stage-1 rubric generation and score against these
        # fixed criteria (e.g. a safety rubric — AUDIT D8).
        self.fixed_rubric: Optional[List['RubricItem']] = list(fixed_rubric) if fixed_rubric else None
        # Skeleton criteria PREPENDED to every distilled rubric (half-fixed mode,
        # DESIGN follow-up): stabilizes cross-segment comparability while still
        # letting stage-1 add task-specific criteria. Ignored when fixed_rubric set.
        self.base_rubric: Optional[List['RubricItem']] = list(base_rubric) if base_rubric else None
        # Intent-aware routing: per-intent fully-fixed rubrics (highest priority)
        # and per-intent half-fixed skeletons. Keys are intent strings (intents.py).
        self.intent_rubrics: Optional[Dict[str, List['RubricItem']]] = (
            {k: list(v) for k, v in intent_rubrics.items()} if intent_rubrics else None)
        self.intent_base_rubrics: Optional[Dict[str, List['RubricItem']]] = (
            {k: list(v) for k, v in intent_base_rubrics.items()} if intent_base_rubrics else None)

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def __call__(self, trajectory: dict, **kwargs) -> int:
        return self.score_detail(trajectory, **kwargs).level

    def score_detail(self, trajectory: dict, *, query: Optional[str] = None,
                     sampling_params: Any = None,
                     extra_context: Optional[str] = None,
                     intent: Optional[str] = None) -> ScoreDetail:
        query = query or self._infer_query(trajectory)
        segment_text = self._trim_segment_for_llm(self._render_segment(trajectory))
        # D7c: fold an objective finding into the scored transcript so the judge
        # re-scores WITH the hard evidence in view (objective corrects subjective).
        if extra_context:
            segment_text = f'{segment_text}\n\n[OBJECTIVE EVIDENCE]\n{extra_context}'

        # --- code-level hard verification (free, un-hackable) ---
        hard_pass_rate, has_hard = self._code_hard_checks(trajectory)

        # No LLM (no student sampler AND no teacher API): skip both LLM stages and
        # fall back to the deterministic code signal, instead of letting the
        # llm_backup teacher path raise a missing-credentials error.
        if not self._llm_available():
            scalar = hard_pass_rate if has_hard else 0.0
            return ScoreDetail(
                level=self._to_level(scalar), scalar=scalar, llm_scalar=0.0,
                hard_pass_rate=hard_pass_rate if has_hard else 1.0,
                gated=False, n_votes=0, rubric=[],
            )

        # --- stage 1: rubric (fixed if configured, else distilled generation) ---
        rubric = self._build_rubric(query, segment_text, sampling_params, intent=intent)
        if not rubric:
            # No usable rubric: fall back to the code signal alone.
            scalar = hard_pass_rate if has_hard else 0.0
            return ScoreDetail(
                level=self._to_level(scalar), scalar=scalar, llm_scalar=0.0,
                hard_pass_rate=hard_pass_rate if has_hard else 1.0,
                gated=False, n_votes=0, rubric=[],
            )

        # --- stage 2: rubric scoring with margin-adaptive voting ---
        per_item_rate, n_votes = self._score_with_voting(
            query, segment_text, rubric, sampling_params)

        llm_scalar = self._aggregate(rubric, per_item_rate)

        # --- gatekeeper: code-verified hard failures cap the score ---
        scalar = llm_scalar
        gated = False
        if self.gate and has_hard and hard_pass_rate < 1.0:
            capped = min(llm_scalar, hard_pass_rate)
            gated = capped < llm_scalar
            scalar = capped

        return ScoreDetail(
            level=self._to_level(scalar),
            scalar=scalar,
            llm_scalar=llm_scalar,
            hard_pass_rate=hard_pass_rate if has_hard else 1.0,
            gated=gated,
            n_votes=n_votes,
            rubric=rubric,
            per_item_pass_rate=per_item_rate,
        )

    def diagnose(self, trajectory: dict, *, query: Optional[str] = None,
                 sampling_params: Any = None,
                 intent: Optional[str] = None) -> DiagnoseDetail:
        """Produce a COMPLETE verification chain over the segment (DESIGN §11.6).

        Unlike :meth:`score_detail` (terse PASS/FAIL, tuned to be cheap), this
        emits, in a SINGLE llm_backup-distilled call, a per-criterion verdict
        *with* its reason and (on FAIL) a concrete fix, plus an overall verdict.
        The single call keeps verdict and reason mutually consistent, and it
        covers passing criteria too so a distilled checker learns to say
        "checked, no error, continue" — not only to nitpick.

        Every call flows through ``llm_backup``: the (student, teacher, match)
        pairs it records are exactly the SFT corpus for the error-checker LoRA.
        Store all of them (both OK and ISSUES segments); balancing is a
        training-time sampling concern, not a collection-time one.
        """
        query = query or self._infer_query(trajectory)
        segment_text = self._trim_segment_for_llm(self._render_segment(trajectory))

        if not self._llm_available():
            # No LLM: fall back to the deterministic code signal only.
            hard_pass_rate, has_hard = self._code_hard_checks(trajectory)
            scalar = hard_pass_rate if has_hard else 1.0
            return DiagnoseDetail(
                scalar=scalar, overall_ok=scalar >= 1.0,
                summary='no LLM available; code-hard signal only',
                items=[], rubric=[], raw='', query=query, segment_text=segment_text)

        # Reuse the same rubric machinery as scoring (fixed / half-fixed / gen).
        rubric = self._build_rubric(query, segment_text, sampling_params, intent=intent)
        if not rubric:
            hard_pass_rate, has_hard = self._code_hard_checks(trajectory)
            scalar = hard_pass_rate if has_hard else 1.0
            return DiagnoseDetail(
                scalar=scalar, overall_ok=scalar >= 1.0,
                summary='no usable rubric; code-hard signal only',
                items=[], rubric=rubric, raw='', query=query, segment_text=segment_text)

        rubric_block = self._render_rubric(rubric)
        rubric_key = _short_hash(rubric_block)
        raw = self._diagnose_once(
            trajectory=self._diagnose_trajectory(query, rubric_block, segment_text),
            sampling_params=self._diagnose_sampling_params(sampling_params, temperature=0.0),
            query=query, rubric_key=rubric_key)

        items, overall_ok, summary = self._parse_diagnosis(raw, len(rubric))
        # Blend deterministic hard checks in as a gatekeeper floor, mirroring
        # score_detail so the diagnostic scalar is comparable to the scoring one.
        per_item_rate = [1.0 if it.verdict else 0.0 for it in items]
        llm_scalar = self._aggregate(rubric, per_item_rate) if per_item_rate else 0.0
        hard_pass_rate, has_hard = self._code_hard_checks(trajectory)
        scalar = llm_scalar
        if self.gate and has_hard and hard_pass_rate < 1.0:
            scalar = min(llm_scalar, hard_pass_rate)
        return DiagnoseDetail(
            scalar=scalar, overall_ok=overall_ok, summary=summary,
            items=items, rubric=rubric, raw=raw,
            query=query, segment_text=segment_text)

    # ------------------------------------------------------------------
    # stage 1: rubric assembly (fixed | half-fixed skeleton + distilled | gen)
    # ------------------------------------------------------------------
    def _build_rubric(self, query, segment_text, sampling_params,
                      intent: Optional[str] = None) -> List[RubricItem]:
        """Return the rubric to score against.

        Selection order (intent-aware routing, DESIGN follow-up):
        1. ``intent_rubrics[intent]`` set -> fully fixed for this intent (most
           stable; template-like tasks such as tool_call / code / math).
        2. ``fixed_rubric`` set           -> global fixed rubric, verbatim.
        3. else                           -> distilled generation, optionally
           PREPENDED with a fixed skeleton: ``intent_base_rubrics[intent]`` if
           present, else the global ``base_rubric`` (half-fixed). Skeleton gives
           cross-segment comparability; the generated tail adds task-specific
           coverage. Duplicate criteria (same normalized text) drop, skeleton wins.
        """
        if intent and self.intent_rubrics and intent in self.intent_rubrics:
            return list(self.intent_rubrics[intent])
        if self.fixed_rubric is not None:
            return list(self.fixed_rubric)

        skeleton: Optional[List[RubricItem]] = None
        if intent and self.intent_base_rubrics and intent in self.intent_base_rubrics:
            skeleton = self.intent_base_rubrics[intent]
        elif self.base_rubric:
            skeleton = self.base_rubric

        gen_min = self.min_rubrics
        gen_max = self.max_rubrics
        if skeleton:
            # leave room for the skeleton so the total stays in the count window
            gen_min = max(1, self.min_rubrics - len(skeleton))
            gen_max = max(gen_min, self.max_rubrics - len(skeleton))
        raw_rubric = self._gen_rubric(
            trajectory=self._gen_trajectory(query, segment_text, gen_min, gen_max),
            sampling_params=self._gen_sampling_params(sampling_params),
            query=query)
        generated = self._parse_rubric(raw_rubric)
        if not skeleton:
            return generated
        merged = list(skeleton)
        seen = {_norm_criterion(it.text) for it in merged}
        for it in generated:
            key = _norm_criterion(it.text)
            if key and key not in seen:
                seen.add(key)
                merged.append(it)
        return merged

    @llm_backup(key_params=['query'], comparator=lambda a, b: _rubric_similar(a, b))
    def _gen_rubric(self, trajectory, sampling_params, query: str = None) -> str:
        return self._sample_text(trajectory, sampling_params, self.gen_lora_path)

    # ------------------------------------------------------------------
    # stage 2: rubric scoring (student, distilled via llm_backup)
    # ------------------------------------------------------------------
    # Note: the scoring pass is distilled on the *verdict pattern*; the
    # comparator matches on binned pass-rate so student/teacher agree when
    # their PASS/FAIL vectors are close (not byte-identical).
    @llm_backup(key_params=['query', 'rubric_key'],
                comparator=lambda a, b: _verdicts_close(a, b))
    def _score_once(self, trajectory, sampling_params, query: str = None,
                    rubric_key: str = '') -> str:
        return self._sample_text(trajectory, sampling_params, self.score_lora_path)

    # ------------------------------------------------------------------
    # diagnostic pass (student, distilled via llm_backup) — DESIGN §11.6
    # ------------------------------------------------------------------
    # Distilled on the full (verdict + reason) chain. Consistency is checked on
    # the per-criterion verdict vector (same idea as scoring), not on the free
    # text of the reasons — two valid reasons for the same verdict should match.
    @llm_backup(key_params=['query', 'rubric_key'],
                comparator=lambda a, b: _diag_verdicts_close(a, b))
    def _diagnose_once(self, trajectory, sampling_params, query: str = None,
                       rubric_key: str = '') -> str:
        return self._sample_text(trajectory, sampling_params, self.score_lora_path)

    def _score_with_voting(self, query, segment_text, rubric, sampling_params
                           ) -> Tuple[List[float], int]:
        n = len(rubric)
        rubric_block = self._render_rubric(rubric)
        rubric_key = _short_hash(rubric_block)
        score_traj = self._score_trajectory(query, rubric_block, segment_text)

        margin_thr = self.margin_threshold
        vote_cap = self.max_votes
        min_votes = 1
        if len(segment_text) >= self.long_segment_chars:
            margin_thr = self.long_margin_threshold
            min_votes = self.min_votes_long
            vote_cap = min(vote_cap, self.max_votes_long)

        # First (cheap) pass.
        votes: List[List[bool]] = []
        first = self._score_once(
            trajectory=score_traj,
            sampling_params=self._score_sampling_params(sampling_params, temperature=0.0),
            query=query, rubric_key=rubric_key)
        votes.append(self._parse_verdicts(first, n))

        # High-confidence band: a lone pass that looks like "all good" (>= the
        # high threshold) still gets re-sampled, so top-band scores are not
        # decided by one lucky draw. Raise the required vote depth accordingly.
        rate = self._vote_rates(votes)
        first_scalar = self._aggregate(rubric, rate)
        if first_scalar >= self.high_score_threshold:
            min_votes = max(min_votes, min(self.min_votes_high, vote_cap))

        # Escalate when uncertain OR when the vote-depth floor is not yet met.
        if vote_cap > 1 and (self._is_uncertain(rate, margin_thr) or len(votes) < min_votes):
            sp = self._score_sampling_params(sampling_params, temperature=0.7)
            while len(votes) < vote_cap:
                extra = self._score_once(
                    trajectory=score_traj, sampling_params=sp,
                    query=query, rubric_key=rubric_key)
                votes.append(self._parse_verdicts(extra, n))
                rate = self._vote_rates(votes)
                if len(votes) >= min_votes and not self._is_uncertain(rate, margin_thr):
                    break
        return rate, len(votes)

    def _trim_segment_for_llm(self, text: str) -> str:
        cap = self.max_segment_chars
        if len(text) <= cap:
            return text
        head = cap // 2 - 96
        tail = cap // 2 - 96
        omitted = len(text) - head - tail
        return (f'{text[:head]}\n\n[... {omitted} chars omitted for rubric scoring ...]\n\n'
                f'{text[-tail:]}')

    # ------------------------------------------------------------------
    # code-level hard verification (no LLM)
    # ------------------------------------------------------------------
    @staticmethod
    def _code_hard_checks(trajectory: dict) -> Tuple[float, bool]:
        """Deterministic tool-call checks -> (pass_rate, has_any_hard_signal).

        Each assistant tool_call contributes checks:
        - arguments parse as JSON (schema-ish validity),
        - a following tool message exists and is non-empty / non-ERROR.
        Returns pass_rate over all such checks; has_hard=False when the
        segment has no tool calls (nothing to code-verify).
        """
        msgs = trajectory.get('messages', []) or []
        n_msgs = len(msgs)
        checks: List[bool] = []
        for i, m in enumerate(msgs):
            if m.get('role') != 'assistant':
                continue
            tool_calls = m.get('tool_calls') or []
            for tc in tool_calls:
                fn = (tc.get('function') or {}) if isinstance(tc, dict) else {}
                args = fn.get('arguments', '')
                # 1) argument validity
                checks.append(_is_valid_json_args(args))
                # 2) execution success: find the matching/following tool message
                ok = False
                j = i + 1
                while j < n_msgs and msgs[j].get('role') == 'tool':
                    content = msgs[j].get('content') or ''
                    text = content if isinstance(content, str) else str(content)
                    if text.strip() and not text.lstrip().startswith('ERROR'):
                        ok = True
                        break
                    j += 1
                checks.append(ok)
        if not checks:
            return 1.0, False
        return sum(1 for c in checks if c) / len(checks), True

    # ------------------------------------------------------------------
    # aggregation & mapping
    # ------------------------------------------------------------------
    def _aggregate(self, rubric: List[RubricItem], per_item_rate: List[float]) -> float:
        """Weighted mean of per-criterion pass-rates (ARROW-style, Hard>Principle)."""
        num = 0.0
        den = 0.0
        for item, rate in zip(rubric, per_item_rate):
            w = self.hard_weight if item.is_hard else self.principle_weight
            num += w * rate
            den += w
        return num / den if den else 0.0

    def _to_level(self, scalar: float) -> int:
        scalar = min(1.0, max(0.0, scalar))
        # Map [0,1] onto {0..NUM_LEVELS-1} with even-width bins.
        level = int(round(scalar * (self.NUM_LEVELS - 1)))
        return min(self.NUM_LEVELS - 1, max(0, level))

    def _is_uncertain(self, per_item_rate: Sequence[float],
                      margin_threshold: Optional[float] = None) -> bool:
        """A segment is uncertain if any criterion sits near the 0.5 boundary."""
        thr = self.margin_threshold if margin_threshold is None else margin_threshold
        if not per_item_rate:
            return False
        # distance of the aggregate margin from a confident 0/1 verdict
        for r in per_item_rate:
            if abs(r - 0.5) * 2.0 < thr:
                return True
        return False

    @staticmethod
    def _vote_rates(votes: List[List[bool]]) -> List[float]:
        """Per-criterion PASS rate across votes (robust: median-of-means style).

        For each criterion we average the boolean verdicts across votes. This
        equals majority-vote direction while keeping a soft rate for
        aggregation. Outlier passes/fails wash out as votes accumulate.
        """
        if not votes:
            return []
        n = max(len(v) for v in votes)
        rates: List[float] = []
        for k in range(n):
            col = [v[k] for v in votes if k < len(v)]
            if not col:
                rates.append(0.0)
                continue
            rates.append(sum(1 for c in col if c) / len(col))
        return rates

    # ------------------------------------------------------------------
    # LLM sampling plumbing (mirrors Summarizer)
    # ------------------------------------------------------------------
    def _llm_available(self) -> bool:
        """True if a student sampler exists or a teacher API is configured.

        Mirrors the env vars ``llm_backup`` uses for its teacher; when neither a
        student nor a teacher is present we must not attempt an LLM call (it would
        raise a missing-credentials error inside the llm_backup teacher path).
        """
        if self.sampler is not None:
            return True
        return bool(os.environ.get('LLM_BACKUP_API_KEY')
                    or os.environ.get('OPENAI_API_KEY')
                    or os.environ.get('LLM_BACKUP_BASE_URL'))

    def _sample_text(self, trajectory, sampling_params, lora_path) -> str:
        if self.sampler is None:
            return ''
        sample_kwargs: dict[str, Any] = {'sampling_params': sampling_params}
        if lora_path is None:
            sample_kwargs['use_base_model'] = True
        else:
            sample_kwargs['adapter_path'] = lora_path
        responses = self.sampler.sample([trajectory], **sample_kwargs)
        resp = list(responses)[0] if responses else None
        if resp is None:
            return ''
        seqs = getattr(resp, 'sequences', None) or []
        return (getattr(seqs[0], 'decoded', None) or '') if seqs else ''

    def _gen_trajectory(self, query: str, segment_text: str,
                        min_n: Optional[int] = None, max_n: Optional[int] = None) -> dict:
        user = _fill(_GEN_USER, query=query, segment=segment_text)
        system = _fill(_GEN_SYSTEM,
                       min_n=self.min_rubrics if min_n is None else min_n,
                       max_n=self.max_rubrics if max_n is None else max_n)
        return {'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]}

    def _score_trajectory(self, query: str, rubric_block: str, segment_text: str) -> dict:
        user = _fill(_SCORE_USER, query=query, rubric=rubric_block, segment=segment_text)
        return {'messages': [
            {'role': 'system', 'content': _SCORE_SYSTEM},
            {'role': 'user', 'content': user},
        ]}

    def _diagnose_trajectory(self, query: str, rubric_block: str, segment_text: str) -> dict:
        user = _fill(_DIAG_USER, query=query, rubric=rubric_block, segment=segment_text)
        return {'messages': [
            {'role': 'system', 'content': _DIAG_SYSTEM},
            {'role': 'user', 'content': user},
        ]}

    def _gen_sampling_params(self, override):
        if override is not None:
            return override
        if self.sampling_params is not None:
            return self.sampling_params
        from twinkle.data_format.sampling import SamplingParams
        return SamplingParams(temperature=0.3, max_tokens=512)

    def _score_sampling_params(self, override, *, temperature: float):
        if override is not None:
            return override
        from twinkle.data_format.sampling import SamplingParams
        return SamplingParams(temperature=temperature, max_tokens=256)

    def _diagnose_sampling_params(self, override, *, temperature: float):
        """Token budget for the diagnostic pass.

        Scoring emits terse PASS/FAIL lines (256 tokens is plenty), but the
        diagnosis emits a full JSON object with a per-criterion reason AND fix
        for EVERY rubric item. With ~7 criteria that easily exceeds 256 tokens
        and the JSON gets truncated mid-string (unparsable -> all-FAIL fallback,
        useless as SFT data). Give it a much larger budget, scaled by rubric size
        and overridable via ``RUBRIC_DIAG_MAX_TOKENS``.
        """
        if override is not None:
            return override
        from twinkle.data_format.sampling import SamplingParams
        cap = int(os.environ.get('RUBRIC_DIAG_MAX_TOKENS', str(self.diag_max_tokens)))
        return SamplingParams(temperature=temperature, max_tokens=cap)

    # ------------------------------------------------------------------
    # rendering / parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_query(trajectory: dict) -> str:
        for msg in trajectory.get('messages', []) or []:
            if msg.get('role') == 'user':
                c = msg.get('content')
                if isinstance(c, str) and c.strip():
                    return c.strip()
        return '(no explicit query)'

    @staticmethod
    def _render_segment(trajectory: dict) -> str:
        """Flatten a segment's messages into a readable transcript."""
        lines: List[str] = []
        for m in trajectory.get('messages', []) or []:
            role = m.get('role', '?')
            if role == 'system':
                continue
            content = m.get('content')
            if isinstance(content, list):
                content = '\n'.join(
                    p.get('text', '') for p in content
                    if isinstance(p, dict) and p.get('type') == 'text')
            content = content or ''
            tool_calls = m.get('tool_calls') or []
            if tool_calls:
                calls = '; '.join(
                    f"{(tc.get('function') or {}).get('name', '?')}"
                    f"({(tc.get('function') or {}).get('arguments', '')})"
                    for tc in tool_calls if isinstance(tc, dict))
                lines.append(f'[{role}] {content}\n  tool_calls: {calls}'.rstrip())
            else:
                lines.append(f'[{role}] {content}'.rstrip())
        return '\n'.join(lines).strip()

    _TAG_HARD_RE = re.compile(r'\[\s*hard\s*rule\s*\]', re.IGNORECASE)
    _TAG_PRIN_RE = re.compile(r'\[\s*principle\s*\]', re.IGNORECASE)
    _NUM_LINE_RE = re.compile(r'^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$')

    @classmethod
    def _parse_rubric(cls, raw: str) -> List[RubricItem]:
        items: List[RubricItem] = []
        for line in (raw or '').splitlines():
            m = cls._NUM_LINE_RE.match(line)
            text = (m.group(1) if m else line).strip()
            if not text:
                continue
            is_hard = bool(cls._TAG_HARD_RE.search(text))
            is_prin = bool(cls._TAG_PRIN_RE.search(text))
            if not (is_hard or is_prin):
                # Untagged line that isn't clearly a criterion -> skip noise.
                if not m:
                    continue
                is_hard = False  # default to principle
            clean = cls._TAG_HARD_RE.sub('', cls._TAG_PRIN_RE.sub('', text)).strip(' .')
            if clean:
                items.append(RubricItem(text=clean, is_hard=is_hard))
        return items

    @staticmethod
    def _render_rubric(rubric: List[RubricItem]) -> str:
        return '\n'.join(
            f'{i + 1}. {it.text} [{"Hard Rule" if it.is_hard else "Principle"}]'
            for i, it in enumerate(rubric))

    @staticmethod
    def _parse_verdicts(raw: str, n: int) -> List[bool]:
        """Parse '<i>: PASS/FAIL' lines into a length-n boolean vector.

        Missing verdicts default to FAIL (conservative for hard rules).
        """
        verdicts = [False] * n
        for line in (raw or '').splitlines():
            m = _VERDICT_RE.match(line)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                verdicts[idx] = m.group(2).lower() in ('pass', 'true', 'yes', '1')
        return verdicts

    @classmethod
    def _parse_diagnosis(cls, raw: str, n: int
                         ) -> Tuple[List[DiagnosisItem], bool, str]:
        """Parse the diagnostic JSON into (items, overall_ok, summary).

        Robust to models that wrap JSON in code fences or add stray prose. Falls
        back to the PASS/FAIL line parser when JSON is unrecoverable, so a
        malformed diagnostic still yields usable verdicts (missing -> FAIL).
        """
        obj = _extract_json_obj(raw)
        entries: List[dict] = []
        if isinstance(obj, dict) and isinstance(obj.get('items'), list):
            entries = [e for e in obj['items'] if isinstance(e, dict)]
        if not entries:
            # The full JSON did not parse (commonly a truncated response): salvage
            # every COMPLETE ``{...}`` item object so partial diagnoses stay usable
            # instead of degrading to an all-FAIL, reason-less vector.
            entries = _salvage_diag_items(raw)

        items: List[DiagnosisItem] = []
        for i, entry in enumerate(entries):
            try:
                idx = int(entry.get('index', i + 1))
            except (TypeError, ValueError):
                idx = i + 1
            verdict = str(entry.get('verdict', '')).strip().lower() in (
                'pass', 'true', 'yes', '1', 'ok')
            items.append(DiagnosisItem(
                index=idx, verdict=verdict,
                reason=str(entry.get('reason', '') or '').strip(),
                fix=str(entry.get('fix', '') or '').strip()))
        if not items:
            # Last resort: the terse verdict-line parser (missing -> FAIL).
            verdicts = cls._parse_verdicts(raw, n)
            items = [DiagnosisItem(index=i + 1, verdict=v)
                     for i, v in enumerate(verdicts)]

        overall_ok = all(it.verdict for it in items) if items else False
        summary = ''
        if isinstance(obj, dict):
            summary = str(obj.get('summary', '') or '').strip()
            overall_raw = str(obj.get('overall', '') or '').strip().lower()
            if overall_raw in ('ok', 'pass', 'good'):
                # Trust an explicit OK only if no item contradicts it.
                overall_ok = overall_ok and True
            elif overall_raw in ('issues', 'issue', 'fail', 'bad'):
                overall_ok = False
        if not summary:
            summary = ('no process errors, continue' if overall_ok
                       else 'process issues found')
        return items, overall_ok, summary


# ---------------------------------------------------------------------------
# module-level helpers (comparators etc.)
# ---------------------------------------------------------------------------
def _fill(template: str, **kw) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace('{' + k + '}', str(v))
    return out


def _is_valid_json_args(args: Any) -> bool:
    if isinstance(args, dict):
        return True
    if not isinstance(args, str):
        return False
    s = args.strip()
    if not s:
        return True  # a no-arg call is valid
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _short_hash(text: str) -> str:
    import hashlib
    return hashlib.md5((text or '').encode()).hexdigest()[:12]


_NORM_RE = re.compile(r'[^a-z0-9]+')


def _norm_criterion(text: str) -> str:
    """Normalize criterion text for dedup (lowercase, alnum-only, first 12 words)."""
    words = _NORM_RE.sub(' ', (text or '').lower()).split()
    return ' '.join(words[:12])


_TAG_ANY_RE = re.compile(r'\[\s*(hard\s*rule|principle)\s*\]', re.IGNORECASE)


def _rubric_similar(a: str, b: str) -> bool:
    """Comparator for rubric generation: rubrics rarely match verbatim, so we
    compare on *shape* — similar criterion count and similar hard/principle mix.
    This is a proxy; downstream consistency filtering is the real quality gate.
    """
    ca = _TAG_ANY_RE.findall(a or '')
    cb = _TAG_ANY_RE.findall(b or '')
    na, nb = len(ca), len(cb)
    if na == 0 and nb == 0:
        return True
    if na == 0 or nb == 0:
        return False
    # count within +/-2 and hard-ratio within 0.34
    if abs(na - nb) > 2:
        return False
    hard_a = sum(1 for t in ca if t.lower().startswith('hard')) / na
    hard_b = sum(1 for t in cb if t.lower().startswith('hard')) / nb
    return abs(hard_a - hard_b) <= 0.34


def _parse_rate(raw: str) -> Optional[float]:
    pos = 0
    total = 0
    for line in (raw or '').splitlines():
        m = _VERDICT_RE.match(line)
        if not m:
            continue
        total += 1
        if m.group(2).lower() in ('pass', 'true', 'yes', '1'):
            pos += 1
    if total == 0:
        return None
    return pos / total


def _verdicts_close(a: str, b: str, tol: float = 0.25) -> bool:
    """Comparator for rubric scoring: student/teacher agree when their overall
    PASS rate is within ``tol`` (binned agreement, not byte-identical text)."""
    ra, rb = _parse_rate(a), _parse_rate(b)
    if ra is None or rb is None:
        return (a or '').strip() == (b or '').strip()
    return abs(ra - rb) <= tol


def _salvage_diag_items(raw: str) -> List[dict]:
    """Recover complete diagnosis item objects from a (possibly truncated) reply.

    Scans for balanced ``{...}`` spans (string-aware, so braces inside a reason
    like ``\\subsubsection*{...}`` don't corrupt the depth count) and json-parses
    each object that carries a ``verdict`` key. A response cut off mid-stream
    still yields every item emitted before the cut, so the diagnosis keeps its
    reasons/fixes instead of collapsing to an all-FAIL, reason-less vector.
    """
    if not raw:
        return []
    out: List[dict] = []
    stack: List[int] = []          # start index of each open brace, by depth
    in_str = False
    escaped = False
    for i, ch in enumerate(raw):
        if in_str:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            start = stack.pop()
            frag = raw[start:i + 1]
            # Only leaf-ish item objects carry a verdict; the outer envelope
            # ({"items": [...]}) usually never closes when truncated anyway.
            if '"verdict"' in frag and '"items"' not in frag:
                try:
                    obj = json.loads(frag)
                    if isinstance(obj, dict):
                        out.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
    return out


def _extract_json_obj(raw: str) -> Optional[dict]:
    """Best-effort extraction of the first JSON object from a model response.

    Handles bare JSON, ```json fenced blocks, and JSON embedded in prose.
    """
    if not raw:
        return None
    s = raw.strip()
    # Strip a leading/trailing code fence if present.
    if s.startswith('```'):
        s = re.sub(r'^```[a-zA-Z]*\s*', '', s)
        s = re.sub(r'\s*```$', '', s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to the widest {...} span.
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _diag_rate(raw: str) -> Optional[float]:
    """Overall PASS rate from a diagnostic JSON response (for the comparator)."""
    obj = _extract_json_obj(raw)
    if isinstance(obj, dict) and isinstance(obj.get('items'), list):
        verdicts = [str(e.get('verdict', '')).strip().lower() in
                    ('pass', 'true', 'yes', '1', 'ok')
                    for e in obj['items'] if isinstance(e, dict)]
        if verdicts:
            return sum(1 for v in verdicts if v) / len(verdicts)
    return _parse_rate(raw)


def _diag_verdicts_close(a: str, b: str, tol: float = 0.25) -> bool:
    """Comparator for the diagnostic pass: student/teacher agree when their
    overall PASS rate is within ``tol``. Reasons are free text, so we match on
    the verdict vector (two valid phrasings of the same verdict count as a
    match), not on byte-identical explanations."""
    ra, rb = _diag_rate(a), _diag_rate(b)
    if ra is None or rb is None:
        return (a or '').strip() == (b or '').strip()
    return abs(ra - rb) <= tol
