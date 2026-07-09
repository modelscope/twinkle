# Copyright (c) ModelScope Contributors. All rights reserved.
"""Domain-specific deterministic checks for :class:`HardScorer`.

These are LLM-free, dependency-free (stdlib ``ast``/``re``/``json`` only) and
reuse the answer-extraction / F1 helpers already in ``reward/f1.py``. They are
factories: call them with config and get back a plain ``CheckFn`` that plugs
into ``HardScorer(checks=[...])``.

Coverage (initial, no sandbox):
- output format: ``\\boxed{}`` / fenced code block / parseable JSON present
- numeric equivalence: lightweight fraction/decimal/percent normalization
- reference match: F1/EM vs ``ground_truth`` (reuses ``_f1_score``)
- code syntax: ``ast.parse`` on the last fenced block (stdlib, does NOT run)
- instruction constraints: length / keyword must-include / must-exclude / lang
- degeneration: empty / too-short / repetitive final answer

Sandbox-based math (``math-verify``/sympy) and code execution (unit tests) can
be added later as additional CheckFns without touching HardScorer.
"""
from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence

from twinkle_agentic.reward.f1 import _extract_final_answer
from twinkle_agentic.reward.f1 import _f1_score as _f1_score_stemmed

from .hard_scorer import CheckResult, TrajectoryView

if TYPE_CHECKING:
    from .hard_scorer import CheckFn  # noqa: F401


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
_CODE_FENCE_RE = re.compile(r'```([a-zA-Z0-9_+-]*)\s*\n(.*?)```', re.DOTALL)
_BOXED_RE = re.compile(r'\\boxed\s*\{')
_NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:/\d+)?%?')


def _ground_truths(trajectory: dict) -> List[str]:
    """Read ground_truth values from user_data (same convention as F1Reward)."""
    out: List[str] = []
    for entry in trajectory.get('user_data', []) or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] == 'ground_truth':
            v = entry[1]
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
            if isinstance(v, (list, tuple)):
                out.extend(str(x) for x in v if x)
            elif v:
                out.append(str(v))
    return out


def _last_code_block(text: str) -> Optional[str]:
    matches = _CODE_FENCE_RE.findall(text or '')
    if not matches:
        return None
    return matches[-1][1]


def _to_number(token: str) -> Optional[float]:
    """Normalize a numeric token: fraction 'a/b', percent 'x%', or decimal."""
    if token is None:
        return None
    s = str(token).strip().replace(',', '').replace('$', '').replace(' ', '')
    if not s:
        return None
    percent = s.endswith('%')
    if percent:
        s = s[:-1]
    try:
        if '/' in s:
            num, den = s.split('/', 1)
            val = float(num) / float(den)
        else:
            val = float(s)
    except (ValueError, ZeroDivisionError):
        return None
    return val / 100.0 if percent else val


def _numbers_in(text: str) -> List[float]:
    vals = []
    for tok in _NUMBER_RE.findall(text or ''):
        v = _to_number(tok)
        if v is not None:
            vals.append(v)
    return vals


_PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
_ARTICLE_RE = re.compile(r'\b(a|an|the)\b')


def _f1_score(prediction: str, gold: str):
    """F1/EM with graceful fallback when nltk (used by f1.py stemming) is
    unavailable — degrade to a stemmer-free token F1 instead of crashing."""
    try:
        return _f1_score_stemmed(prediction, gold)
    except ImportError:
        pass
    from collections import Counter
    norm = lambda s: _ARTICLE_RE.sub(  # noqa: E731
        ' ', _PUNCT_RE.sub('', (s or '').lower())).split()
    p_tok, g_tok = norm(prediction), norm(gold)
    if not p_tok or not g_tok:
        em = float(p_tok == g_tok)
        return em, em
    em = float(p_tok == g_tok)
    common = Counter(p_tok) & Counter(g_tok)
    same = sum(common.values())
    if same == 0:
        return 0.0, em
    prec, rec = same / len(p_tok), same / len(g_tok)
    return 2 * prec * rec / (prec + rec), em


# ---------------------------------------------------------------------------
# format / structure
# ---------------------------------------------------------------------------
def check_output_format(fmt: str, *, weight: float = 1.5, critical: bool = True
                        ) -> 'CheckFn':
    """Require a specific output artifact in the final answer.

    Args:
        fmt: one of ``'boxed'`` (\\boxed{...}), ``'code'`` (fenced block),
            ``'json'`` (a parseable JSON object/array anywhere in the answer).
    """
    fmt = fmt.lower()
    if fmt not in ('boxed', 'code', 'json'):
        raise ValueError("fmt must be 'boxed', 'code' or 'json'")

    def _check(view: TrajectoryView) -> CheckResult:
        text = view.last_assistant_text()
        if fmt == 'boxed':
            ok = bool(_extract_final_answer(text)) or bool(_BOXED_RE.search(text))
            detail = 'boxed present' if ok else 'no \\boxed{}'
        elif fmt == 'code':
            ok = _last_code_block(text) is not None
            detail = 'code block present' if ok else 'no code block'
        else:  # json
            block = _last_code_block(text) or text
            ok = _has_parseable_json(block)
            detail = 'json parseable' if ok else 'no parseable json'
        return CheckResult(f'format_{fmt}', 1.0 if ok else 0.0, weight,
                           critical=critical, n=1, detail=detail)

    return _check


def _has_parseable_json(text: str) -> bool:
    s = (text or '').strip()
    if not s:
        return False
    # try whole-string first, then first {...}/[...] span
    for candidate in (s, _first_bracket_span(s)):
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return True
        except (json.JSONDecodeError, ValueError):
            continue
    return False


def _first_bracket_span(s: str) -> Optional[str]:
    starts = [i for i, c in enumerate(s) if c in '{[']
    if not starts:
        return None
    i = starts[0]
    open_c = s[i]
    close_c = '}' if open_c == '{' else ']'
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_c:
            depth += 1
        elif s[j] == close_c:
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return None


# ---------------------------------------------------------------------------
# numeric equivalence (lightweight, no sympy)
# ---------------------------------------------------------------------------
def check_numeric_equiv(*, tol: float = 1e-6, weight: float = 2.0,
                        critical: bool = False) -> 'CheckFn':
    """Compare the extracted final number(s) against ground_truth numerically.

    Handles fractions / decimals / percentages. For symbolic equivalence,
    swap this for a sympy/math-verify CheckFn later. Neutral pass when there
    is no numeric ground truth to compare against.
    """
    def _check(view: TrajectoryView) -> CheckResult:
        golds = _ground_truths(view.trajectory)
        gold_nums = [n for g in golds for n in _numbers_in(g)]
        if not gold_nums:
            return CheckResult('numeric_equiv', 1.0, weight, critical=False, n=0,
                               detail='no numeric ground truth')
        text = view.last_assistant_text()
        boxed = _extract_final_answer(text)
        pred_nums = _numbers_in(boxed) if boxed else _numbers_in(text)
        if not pred_nums:
            return CheckResult('numeric_equiv', 0.0, weight, critical=critical, n=1,
                               detail='no number in answer')
        # match if any predicted number equals any gold (last pred preferred)
        target = gold_nums[-1]
        ok = any(abs(p - target) <= tol + tol * abs(target) for p in pred_nums)
        return CheckResult('numeric_equiv', 1.0 if ok else 0.0, weight,
                           critical=critical, n=1,
                           detail=f'pred~{pred_nums[-1]} vs gold~{target}')

    return _check


# ---------------------------------------------------------------------------
# reference match (reuse f1.py)
# ---------------------------------------------------------------------------
def check_answer_match(*, threshold: float = 0.6, weight: float = 2.0,
                       critical: bool = False, use_em: bool = False) -> 'CheckFn':
    """F1/EM of the extracted answer vs ground_truth (reuses ``_f1_score``).

    Score is the max F1 over gold answers (or EM when ``use_em``); pass/fail is
    F1 >= threshold. Neutral pass when there is no ground truth.
    """
    def _check(view: TrajectoryView) -> CheckResult:
        golds = _ground_truths(view.trajectory)
        if not golds:
            return CheckResult('answer_match', 1.0, weight, critical=False, n=0,
                               detail='no ground truth')
        text = view.last_assistant_text()
        boxed = _extract_final_answer(text)
        pred = boxed or text
        scored = [_f1_score(pred, g) for g in golds]
        best_f1 = max(f for f, _ in scored)
        best_em = max(e for _, e in scored)
        # Containment fallback: when the answer isn't boxed, a short gold that
        # appears verbatim in the answer counts as a hit (robust to preamble
        # like "The answer is Paris.").
        contained = False
        if not boxed:
            low = text.lower()
            contained = any(g.strip() and g.lower() in low and len(g.split()) <= 6
                            for g in golds)
        if contained:
            best_f1 = max(best_f1, 1.0)
            best_em = max(best_em, 1.0)
        val = best_em if use_em else best_f1
        return CheckResult('answer_match', val, weight, critical=critical, n=1,
                           detail=f'f1={best_f1:.2f} em={best_em:.0f}'
                                  + (' contained' if contained else '')
                                  + ('' if val >= threshold else ' <thr'))

    return _check


# ---------------------------------------------------------------------------
# code syntax (stdlib ast, does NOT execute)
# ---------------------------------------------------------------------------
def check_code_parses(*, language: str = 'python', weight: float = 1.5,
                      critical: bool = False) -> 'CheckFn':
    """Last fenced code block must parse (Python only, via stdlib ``ast``).

    This validates *syntax* without a sandbox and without running anything.
    Non-Python blocks are a neutral pass (we can't cheaply verify them here).
    """
    def _check(view: TrajectoryView) -> CheckResult:
        text = view.last_assistant_text()
        block = _last_code_block(text)
        if block is None:
            return CheckResult('code_parses', 0.0, weight, critical=critical, n=1,
                               detail='no code block')
        if language.lower() != 'python':
            return CheckResult('code_parses', 1.0, weight, critical=False, n=0,
                               detail=f'{language} not statically checked')
        try:
            ast.parse(block)
            return CheckResult('code_parses', 1.0, weight, critical=critical, n=1,
                               detail='parses')
        except SyntaxError as e:
            return CheckResult('code_parses', 0.0, weight, critical=critical, n=1,
                               detail=f'SyntaxError: {e.msg}')

    return _check


# ---------------------------------------------------------------------------
# instruction constraints (IFEval-style, pure code)
# ---------------------------------------------------------------------------
def check_instruction_constraints(
    *,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    must_include: Optional[Sequence[str]] = None,
    must_exclude: Optional[Sequence[str]] = None,
    match_source_language: bool = False,
    weight: float = 1.0,
    critical: bool = False,
) -> 'CheckFn':
    """Verify code-checkable instruction-following constraints on the answer.

    Score is the fraction of active sub-constraints satisfied.
    """
    must_include = list(must_include or [])
    must_exclude = list(must_exclude or [])

    def _check(view: TrajectoryView) -> CheckResult:
        text = view.last_assistant_text()
        words = text.split()
        n_words = len(words)
        checks: List[bool] = []
        notes: List[str] = []

        if min_words is not None:
            ok = n_words >= min_words
            checks.append(ok)
            if not ok:
                notes.append(f'words<{min_words}')
        if max_words is not None:
            ok = n_words <= max_words
            checks.append(ok)
            if not ok:
                notes.append(f'words>{max_words}')
        low = text.lower()
        for kw in must_include:
            ok = kw.lower() in low
            checks.append(ok)
            if not ok:
                notes.append(f'missing:{kw}')
        for kw in must_exclude:
            ok = kw.lower() not in low
            checks.append(ok)
            if not ok:
                notes.append(f'forbidden:{kw}')
        if match_source_language:
            ok = _language_matches(view)
            checks.append(ok)
            if not ok:
                notes.append('lang-mismatch')

        if not checks:
            return CheckResult('instruction_constraints', 1.0, weight,
                               critical=False, n=0, detail='no active constraints')
        score = sum(1 for c in checks if c) / len(checks)
        return CheckResult('instruction_constraints', score, weight,
                           critical=critical, n=len(checks),
                           detail=', '.join(notes) or 'all satisfied')

    return _check


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
              or '\u3040' <= c <= '\u30ff'
              or '\uac00' <= c <= '\ud7a3')
    return cjk / len(text)


def _language_matches(view: TrajectoryView) -> bool:
    """Cheap heuristic: answer's CJK-ness matches the first user message's."""
    user_text = ''
    for m in view.messages:
        if m.get('role') == 'user':
            user_text = view.text_of(m)
            break
    ans = view.last_assistant_text()
    if not user_text or not ans:
        return True
    return abs(_cjk_ratio(user_text[:400]) - _cjk_ratio(ans[:400])) < 0.3


# ---------------------------------------------------------------------------
# degeneration
# ---------------------------------------------------------------------------
def check_not_degenerate(*, min_chars: int = 1, max_repeat_ratio: float = 0.5,
                         ngram: int = 8, weight: float = 1.0,
                         critical: bool = False) -> 'CheckFn':
    """Fail on empty / trivially short / highly repetitive final answers."""
    def _check(view: TrajectoryView) -> CheckResult:
        text = view.last_assistant_text().strip()
        if len(text) < min_chars:
            return CheckResult('not_degenerate', 0.0, weight, critical=critical,
                               n=1, detail='too short/empty')
        rep = _repetition_ratio(text, ngram)
        ok = rep <= max_repeat_ratio
        return CheckResult('not_degenerate', 1.0 if ok else 0.0, weight,
                           critical=critical, n=1,
                           detail=f'repeat={rep:.2f}' + ('' if ok else ' >thr'))

    return _check


def _repetition_ratio(text: str, ngram: int) -> float:
    if _cjk_ratio(text[:500]) > 0.3:
        tokens = [c for c in text if not c.isspace()]
    else:
        tokens = text.split()
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


# Convenience presets keyed by domain, for use with a router later.
def default_checks_for(domain: str) -> List['CheckFn']:
    """Return a reasonable initial check bundle for a domain (no sandbox)."""
    domain = (domain or '').lower()
    if domain == 'math':
        return [check_output_format('boxed', critical=False),
                check_numeric_equiv(),
                check_not_degenerate()]
    if domain == 'code':
        return [check_output_format('code', critical=False),
                check_code_parses(),
                check_not_degenerate()]
    if domain in ('factual_qa', 'factual', 'qa'):
        return [check_answer_match(),
                check_not_degenerate()]
    if domain in ('open_qa', 'open', 'writing'):
        return [check_instruction_constraints(),
                check_not_degenerate()]
    return [check_not_degenerate()]
