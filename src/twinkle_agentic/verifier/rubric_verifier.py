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
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

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
it is unambiguously satisfied. Output only the verdict lines, in order, then \
stop. Do not add explanations."""

_SCORE_USER = """\
## Task / query (context)
{query}

## Rubric
{rubric}

## Segment
{segment}

Now output one PASS/FAIL line per criterion, in order."""


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
        # When provided, skip stage-1 rubric generation and score against these
        # fixed criteria (e.g. a safety rubric — AUDIT D8).
        self.fixed_rubric: Optional[List['RubricItem']] = list(fixed_rubric) if fixed_rubric else None

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def __call__(self, trajectory: dict, **kwargs) -> int:
        return self.score_detail(trajectory, **kwargs).level

    def score_detail(self, trajectory: dict, *, query: Optional[str] = None,
                     sampling_params: Any = None,
                     extra_context: Optional[str] = None) -> ScoreDetail:
        query = query or self._infer_query(trajectory)
        segment_text = self._render_segment(trajectory)
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
        if self.fixed_rubric is not None:
            rubric = list(self.fixed_rubric)
        else:
            raw_rubric = self._gen_rubric(
                trajectory=self._gen_trajectory(query, segment_text),
                sampling_params=self._gen_sampling_params(sampling_params),
                query=query,
            )
            rubric = self._parse_rubric(raw_rubric)
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

    # ------------------------------------------------------------------
    # stage 1: rubric generation (student, distilled via llm_backup)
    # ------------------------------------------------------------------
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

    def _score_with_voting(self, query, segment_text, rubric, sampling_params
                           ) -> Tuple[List[float], int]:
        n = len(rubric)
        rubric_block = self._render_rubric(rubric)
        rubric_key = _short_hash(rubric_block)
        score_traj = self._score_trajectory(query, rubric_block, segment_text)

        # First (cheap) pass.
        votes: List[List[bool]] = []
        first = self._score_once(
            trajectory=score_traj,
            sampling_params=self._score_sampling_params(sampling_params, temperature=0.0),
            query=query, rubric_key=rubric_key)
        votes.append(self._parse_verdicts(first, n))

        # Decide whether to escalate: uncertainty = closeness of pass-rate to 0.5.
        rate = self._vote_rates(votes)
        if self._is_uncertain(rate) and self.max_votes > 1:
            sp = self._score_sampling_params(sampling_params, temperature=0.7)
            # Escalate up to max_votes; early-stop once verdicts stabilize.
            while len(votes) < self.max_votes:
                extra = self._score_once(
                    trajectory=score_traj, sampling_params=sp,
                    query=query, rubric_key=rubric_key)
                votes.append(self._parse_verdicts(extra, n))
                rate = self._vote_rates(votes)
                if not self._is_uncertain(rate):
                    break
        return rate, len(votes)

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

    def _is_uncertain(self, per_item_rate: Sequence[float]) -> bool:
        """A segment is uncertain if any criterion sits near the 0.5 boundary."""
        if not per_item_rate:
            return False
        # distance of the aggregate margin from a confident 0/1 verdict
        for r in per_item_rate:
            if abs(r - 0.5) * 2.0 < self.margin_threshold:
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
        import os
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

    def _gen_trajectory(self, query: str, segment_text: str) -> dict:
        user = _fill(_GEN_USER, query=query, segment=segment_text)
        system = _fill(_GEN_SYSTEM, min_n=self.min_rubrics, max_n=self.max_rubrics)
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
