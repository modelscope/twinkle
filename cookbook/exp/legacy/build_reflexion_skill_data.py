"""Offline builder for reflexion skill RFT data (self-contained, cached).

Runs the SAME pipeline as the online trainer -- base greedy solve -> rubric
process-check (view A) -> skill-gen -> leak filter -> with-skill greedy pass ->
group-relative GRPO advantage -- but never updates the skill model. It emits
``skill_dataset.jsonl`` (trainer-schema training records), ``gen_records.jsonl``
(full per-problem traces) and ``eval_holdout.jsonl`` (the fixed holdout).

The expensive base rollouts and rubric diagnoses are cached to disk (one jsonl
each, keyed by an md5 of their inputs) so a re-run skips them entirely.

8 GPUs: ranks 0-3 skill_sampler (vLLM tp1 dp4), ranks 4-7 base_sampler. Leak /
rubric use the backup teacher API: set LLM_BACKUP_API_KEY / LLM_BACKUP_BASE_URL /
LLM_BACKUP_MODEL.

Launch:
    LLM_BACKUP_API_KEY=... python cookbook/exp/embedding/build_reflexion_skill_data.py \
        --total-problems 3200 --base-success-frac 0.3
"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.sampler import vLLMSampler
from twinkle_agentic.verifier import LeakVerifier, RubricVerifier
from twinkle_agentic.verifier.rubric_verifier import RubricItem

logger = get_logger()

MODEL_ID = os.environ.get('GEN_MODEL_ID', 'Qwen/Qwen3-4B')
GPU_MEM = float(os.environ.get('GEN_GPU_MEM', 0.8))
GEN_TEMPERATURE = float(os.environ.get('GEN_TEMPERATURE', 0.6))
GEN_TOP_P = float(os.environ.get('GEN_TOP_P', 0.95))
AOPS_DATASET_ID = os.environ.get('AOPS_DATASET_ID', 'AI-MO/aops')
MATH_DATASET_ID = os.environ.get('MATH_DATASET_ID', 'modelscope/competition_math')


# ===========================================================================
# Block A -- boxed extraction + answer grading
# ===========================================================================
_BOXED_RE = re.compile(r'\\boxed\s*\{')


def extract_boxed(text: str) -> Optional[str]:
    """Last ``\\boxed{...}`` content, brace-balanced."""
    if not text:
        return None
    last = None
    for m in _BOXED_RE.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            depth += (text[i] == '{') - (text[i] == '}')
            i += 1
        if depth == 0:
            last = text[m.end():i - 1].strip()
    return last


def normalize_answer(ans: str) -> str:
    if not ans:
        return ''
    s = str(ans).strip()
    m = re.match(r'^\\?(?:textbf|text|mathrm|mathbf|mathbb)?\{?\(?([A-E])\)?\}?$', s)
    if m:
        return m.group(1)
    s = s.strip('$').strip().replace('−', '-')
    s = re.sub(r'\\(?:text|mathrm|mathbf|textbf|operatorname)\{([^}]*)\}', r'\1', s)
    s = s.replace(r'\displaystyle', '')
    s = re.sub(r'\\(?:left|right)[.()\[\]{}|]', '', s)
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    s = s.replace(r'\,', '').replace(r'\;', '').replace(r'\!', '')
    s = re.sub(r'\\(?:quad|qquad|\s)', '', s)
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'\\sqrt([0-9A-Za-z])', r'\\sqrt{\1}', s)
    s = re.sub(r'\\frac\{([^{}]+)\}([^{}\\])', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\\frac([^{\\])([^{\\])', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\{[a-zA-Z]+\}$', '', s)
    s = re.sub(r'\^\\circ|\^\{\\circ\}|°|\\circ', 'deg', s)
    s = s.replace(r'\minus{}', '-').replace(r'\minus', '-')

    def _frac_to_slash(mt):
        text = mt.group(0)
        pos = text.index('{') + 1
        depth, num_start = 1, pos
        while depth > 0:
            depth += (text[pos] == '{') - (text[pos] == '}')
            pos += 1
        numer = text[num_start:pos - 1]
        pos += 1
        den_start, depth = pos, 1
        while depth > 0:
            depth += (text[pos] == '{') - (text[pos] == '}')
            pos += 1
        return f'({numer})/({text[den_start:pos - 1]})'

    s = re.sub(r'\\frac\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', _frac_to_slash, s)
    s = re.sub(r'(?<!\w)(\d+)/(\d+)(?!\w)', r'(\1)/(\2)', s)
    return s


def _try_numeric_equal(a: str, b: str) -> bool:
    try:
        va, vb = float(a.replace('(', '').replace(')', '')), float(b.replace('(', '').replace(')', ''))
        return abs(va - vb) < 1e-9 * max(1, abs(va), abs(vb))
    except (ValueError, ZeroDivisionError):
        pass
    frac_re = re.compile(r'^\(([^)]+)\)/\(([^)]+)\)$')

    def _eval_frac(s):
        m = frac_re.match(s)
        if m:
            try:
                return float(m.group(1)) / float(m.group(2))
            except (ValueError, ZeroDivisionError):
                pass
        return None

    va, vb = _eval_frac(a), _eval_frac(b)
    return va is not None and vb is not None and abs(va - vb) < 1e-9 * max(1, abs(va), abs(vb))


_MCQ_REF_RE = re.compile(
    r'^\\(?:textbf|text|mathrm|mathbf)\{\(?([A-E])\)?\s*\}\s*(.+)$'
    r'|^\(?([A-E])\)\s+(.+)$')
_VAR_PREFIX_RE = re.compile(r'^(?:[A-Za-z](?:\([^)]*\))?|\([^)]*\))\s*=\s*(.+)$', re.DOTALL)


def _split_mcq(ans: str):
    s = ans.strip()
    m = _MCQ_REF_RE.match(s)
    if m:
        return (m.group(1) or m.group(3)), ((m.group(2) or m.group(4) or '').strip() or None)
    bl = re.match(r'^\\?(?:textbf|text|mathrm|mathbf|mathbb)?\{?\(?([A-E])\)?\}?$', s)
    return (bl.group(1), None) if bl else (None, s or None)


def _strip_var_prefix(ans: str) -> str:
    m = _VAR_PREFIX_RE.match((ans or '').strip())
    return m.group(1).strip() if m else (ans or '')


def _normalize_tuple(ans: str) -> str:
    return re.sub(r'[\s()\[\]{}\\]', '', ans or '')


def _math_verify_equal(predicted: str, reference: str) -> bool:
    try:
        from math_verify import parse, verify
        gold = parse(r'\boxed{' + reference + '}', parsing_timeout=1)
        pred = parse(r'\boxed{' + predicted + '}', parsing_timeout=1)
        return bool(gold and pred and verify(gold, pred, timeout_seconds=1))
    except Exception:
        return False


def answers_match(predicted: str, reference: str) -> bool:
    if not predicted or not reference:
        return False
    norm_p, norm_r = normalize_answer(predicted), normalize_answer(reference)
    if norm_p == norm_r or norm_p.lower() == norm_r.lower() or _try_numeric_equal(norm_p, norm_r):
        return True
    stripped_p = normalize_answer(_strip_var_prefix(predicted))
    stripped_r = normalize_answer(_strip_var_prefix(reference))
    if stripped_p and stripped_r:
        if (stripped_p == stripped_r or stripped_p.lower() == stripped_r.lower()
                or _try_numeric_equal(stripped_p, stripped_r)):
            return True
    p_letter, p_value = _split_mcq(predicted)
    r_letter, r_value = _split_mcq(reference)
    if p_letter and r_letter and p_letter == r_letter:
        return True
    if (p_letter and p_value is None) and (r_letter and r_value is None):
        return False
    p_val = normalize_answer(p_value) if p_value else norm_p
    r_val = normalize_answer(r_value) if r_value else norm_r
    if p_val and r_val and (p_val == r_val or p_val.lower() == r_val.lower() or _try_numeric_equal(p_val, r_val)):
        return True
    for left, right in ((norm_p, norm_r), (stripped_p, stripped_r)):
        tl, tr = _normalize_tuple(left), _normalize_tuple(right)
        if ',' in tl and tl == tr:
            return True
    if '=' in norm_r and '=' not in norm_p:
        if any(part and (part == norm_p or _try_numeric_equal(part, norm_p)) for part in norm_r.split('=')):
            return True
    if '=' in norm_p and '=' not in norm_r:
        if any(part and (part == norm_r or _try_numeric_equal(part, norm_r)) for part in norm_p.split('=')):
            return True
    return _math_verify_equal(stripped_p or norm_p, stripped_r or norm_r)


# ===========================================================================
# Block B -- prompts, skill parsing, batched sampling
# ===========================================================================
DIRECT_SYSTEM = (
    'You are an expert competition mathematician. Solve the following problem '
    'step by step. Provide your final answer inside \\boxed{}.')

# Appended to solve turns: box BOTH the letter and value of an MCQ so the model
# never loops deciding which form to box.
MCQ_INSTRUCTION = (
    '\n\nNote: If the problem is multiple-choice (it lists options such as '
    '(A), (B), (C), ...), put BOTH the option letter and its value in the box, '
    'e.g. \\boxed{(B) 21}. Otherwise, box the value directly. Decide the answer '
    'format once and do not deliberate over which form to box.')

_SKILL_SOLVE_PREFIX = (
    DIRECT_SYSTEM + '\n\n'
    'Before you start, keep these reminders in mind to avoid common mistakes on this '
    'type of problem:\n')
_SKILL_SOLVE_SUFFIX = '\nApply them where relevant, but rely on your own reasoning to reach the answer.'


def build_direct_prompt(problem: str) -> Dict[str, Any]:
    return {'messages': [{'role': 'system', 'content': DIRECT_SYSTEM},
                         {'role': 'user', 'content': problem}]}


def build_skill_solve_prompt(problem: str, skill: str) -> Dict[str, Any]:
    # Concatenation (not .format): DIRECT_SYSTEM/skill contain literal braces.
    return {'messages': [
        {'role': 'system', 'content': _SKILL_SOLVE_PREFIX + skill + _SKILL_SOLVE_SUFFIX},
        {'role': 'user', 'content': problem + MCQ_INSTRUCTION}]}


# -- skill-gen prompts (view A: problem + rubric findings; view B: query only) --
SKILL_GEN_SYSTEM = (
    'You are a mathematics coach. You are shown a competition problem together with an '
    'automated process-check of an earlier solver attempt at it -- which solution '
    'criteria the attempt passed or failed, and suggested fixes for the failures. You '
    'do NOT see the attempt itself, only this check. Treat the check as privileged '
    'training scaffolding: study it together with the problem, identify the '
    'problem-visible features that make each useful flagged failure relevant, then '
    'rephrase those lessons as self-contained reusable skills. The goal is not to '
    'continue from the check, cite it, or hide it silently; the goal is to turn it into '
    'a problem-triggered reasoning pattern a query-only solver could reproduce later.\n\n'
    'Good skills name the observable trigger, the method worth reaching for, the '
    'pitfall to watch, and a quick verification habit. Prefer formulations like '
    '"When a configuration has ...", "Before setting up ...", or "Check whether ..." '
    'over references to the process-check, failed criteria, or the earlier attempt. '
    "These tips are advisory: they will be placed in a solver's system prompt as gentle "
    'reminders before it works through a SIMILAR problem on its own, without seeing '
    'this process-check. So keep them general and transferable rather than a worked '
    'solution to this exact problem, and do not state its specific intermediate values '
    'or final answer. Think briefly first, then give your tips as a markdown bullet '
    'list wrapped in <skills> and </skills>, like the example below.')

SKILL_GEN_SYSTEM_Q = (
    'You are a mathematics coach. You are shown ONE competition problem and nothing '
    'else — no solution and no attempt. Think about what approach this KIND of problem '
    'calls for and where solvers tend to slip, then distil a few reusable tips.\n\n'
    "These tips are advisory: they will be placed in a solver's system prompt as gentle "
    'reminders before it works through a SIMILAR problem on its own. So keep them '
    'general and transferable — the method worth reaching for, the pitfall to watch and '
    'a quick check, and the discipline to settle on a final answer — rather than a '
    'worked solution to this exact problem, and without stating its specific '
    'intermediate values or its final answer. Think briefly first, then give your tips '
    'as a markdown bullet list wrapped in <skills> and </skills>, like the example below.')

SKILL_GEN_USER_Q = (
    'Problem:\n{problem}\n\n'
    'Now reason about this TYPE of problem, then output the skills bullet list.')

SKILL_GEN_USER_RUBRIC = (
    'Problem:\n{problem}\n\n'
    'Process check of an earlier attempt (automated rubric verifier -- treat as '
    'evidence, not gospel; PASS/FAIL per criterion with suggested fixes for failures):\n'
    '{diagnosis}\n\n'
    'Now output a self-contained skills bullet list. Each bullet should still be useful '
    'if the process check were removed: connect any useful flagged failure to '
    'problem-visible features, general methods, and quick checks rather than citing the '
    'rubric or the earlier attempt.')

_EX_PROBLEM = 'Simplify $\\sqrt{72} + \\sqrt{18}$ and give the result.'
_EX_SKILLS = (
    '<skills>\n'
    '- Rewrite each square root by factoring its radicand into a perfect square times '
    'a remainder, then move the perfect-square factor outside.\n'
    '- Avoid the classic trap $\\sqrt{a}+\\sqrt{b}\\ne\\sqrt{a+b}$; only combine terms '
    'sharing the same simplest radical, and sanity-check by estimating each root.\n'
    '- Procedure: simplify every radical, group like radical terms, add their '
    'coefficients, then reduce to simplest form.\n'
    '- Once the expression is in simplest form, commit to that single result as the '
    'final answer rather than re-checking indefinitely.\n'
    '</skills>')


def _skillgen_messages(problem: str, view: str, diagnosis: str) -> List[Dict[str, Any]]:
    """Single source of truth for the skill-gen prompt. View A with a localisable
    failure uses problem + rubric findings; view B -- or a view-A problem whose rubric
    flagged nothing (no ``[FAIL]``) -- degrades to the query-only prompt."""
    if view == 'B' or '[FAIL]' not in (diagnosis or ''):
        return [{'role': 'system', 'content': SKILL_GEN_SYSTEM_Q},
                {'role': 'user', 'content': SKILL_GEN_USER_Q.format(problem=_EX_PROBLEM)},
                {'role': 'assistant', 'content': _EX_SKILLS},
                {'role': 'user', 'content': SKILL_GEN_USER_Q.format(problem=problem)}]
    return [{'role': 'system', 'content': SKILL_GEN_SYSTEM},
            {'role': 'user', 'content': SKILL_GEN_USER_Q.format(problem=_EX_PROBLEM)},
            {'role': 'assistant', 'content': _EX_SKILLS},
            {'role': 'user', 'content': SKILL_GEN_USER_RUBRIC.format(problem=problem, diagnosis=diagnosis)}]


def _assign_view(problem: str, args: argparse.Namespace) -> str:
    h = int(hashlib.md5(f'{args.seed}:{problem}'.encode('utf-8')).hexdigest(), 16)
    return 'B' if (h % 100000) / 100000.0 < args.view_b_frac else 'A'


def _view_prompt(r: Dict[str, Any]) -> Dict[str, Any]:
    return {'messages': _skillgen_messages(r['problem'], r['_view'], r.get('_rubric_diag', ''))}


_BULLET_RE = re.compile(r'(?m)^\s*(?:[-*]|\d+[.)])\s')
_META_RE = re.compile(
    r'\b(the student|the solver|the attempt|this attempt|the trace|the response|'
    r'in the (?:attempt|trace|response|solution)|as (?:shown|seen|noted) above|'
    r'the (?:above|previous|earlier)|my (?:reasoning|analysis)|i (?:think|need|will))\b',
    re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


def _clean_text(decoded: Optional[str]) -> str:
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').rstrip()


def _is_clean_block(block: str) -> bool:
    """Pure bullet list (every non-empty line a bullet) with no meta/trajectory ref."""
    lines = [ln for ln in (l.strip() for l in block.splitlines()) if ln]
    if not lines or not all(_BULLET_RE.match(ln) for ln in lines):
        return False
    return _META_RE.search(block) is None


def _extract_skills_block(text: str) -> Optional[str]:
    """Clean ``<skills>...</skills>`` block, or None. Requires ``</think>`` (skill-gen
    runs thinking ON); reads only the answer after the last one, so a mid-reasoning draft
    or a demo echo can never be mistaken for the answer."""
    low = text.lower()
    end_think = low.rfind('</think>')
    if end_think < 0:
        return None
    answer = text[end_think + len('</think>'):]
    low_a = answer.lower()
    s = low_a.find('<skills>')
    if s < 0:
        return None
    inner = s + len('<skills>')
    e = low_a.find('</skills>', inner)
    block = (answer[inner:e] if e >= 0 else answer[inner:]).strip()
    block = re.sub(r'</?(?:skills|think)>', '', block, flags=re.IGNORECASE).strip()
    return block if _is_clean_block(block) else None


def _parse_seq(seq, gold: str) -> Dict[str, Any]:
    """Grade one sampled sequence into a rollout record."""
    text = _clean_text(getattr(seq, 'decoded', '') or '')
    pred = extract_boxed(text)
    correct = bool(pred) and answers_match(pred, gold)
    terminated = getattr(seq, 'stop_reason', None) != 'length'
    return {'pred': pred, 'correct': correct, 'terminated': terminated,
            'passed': bool(correct and terminated),
            'stop_reason': getattr(seq, 'stop_reason', None),
            'gen_tokens': len(getattr(seq, 'tokens', None) or []), 'text': text}


def _run_samples(sampler, prompts: List[Any], num_samples: int, max_tokens: int,
                 gen_dp: int, temperature: Optional[float] = None,
                 top_p: Optional[float] = None, top_k: Optional[int] = None) -> List[List[Any]]:
    """One batched sampler call -> per-prompt list of raw sequences. vLLM dp needs
    batch len >= dp, so pad the tail and slice back."""
    if not prompts:
        return []
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=GEN_TEMPERATURE if temperature is None else temperature,
        top_p=GEN_TOP_P if top_p is None else top_p,
        num_samples=num_samples, **({} if top_k is None else {'top_k': top_k}))
    padded = prompts
    if gen_dp > 1 and 0 < len(prompts) < gen_dp:
        padded = prompts + [copy.deepcopy(prompts[-1]) for _ in range(gen_dp - len(prompts))]
    responses = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in responses]


# ===========================================================================
# Block C -- data loading via twinkle.Dataset + numeric filtering
# ===========================================================================
def load_problems(dataset: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Load boxed-answer problems as ``{problem, reference_answer, level?}`` via
    twinkle.Dataset (ModelScope hub), sampled to ``n`` (0 = all)."""
    ds_id = AOPS_DATASET_ID if dataset == 'aops' else MATH_DATASET_ID
    rows = Dataset(DatasetMeta(dataset_id=f'ms://{ds_id}', split='train')).dataset
    out: List[Dict[str, Any]] = []
    for row in rows:
        if dataset == 'aops' and not (row.get('metadata') or {}).get('boxed'):
            continue
        ref = extract_boxed(row.get('solution', ''))
        if not ref:
            continue
        rec = {'problem': row['problem'], 'reference_answer': ref}
        if row.get('level'):
            rec['level'] = row['level']
        out.append(rec)
    logger.info(f'[data] {dataset}: {len(out)} boxed problems')
    rng = np.random.RandomState(seed)
    rng.shuffle(out)
    return out[:n] if (n and n < len(out)) else out


_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _norm_num_text(num: str) -> str:
    try:
        f = float(num)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return str(num).strip()


def _numeric_value(raw: Any) -> Optional[str]:
    """Collapse an answer to a single int/decimal/fraction, or None."""
    if raw is None:
        return None
    s = str(raw).strip().strip('$').strip()
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    s = re.sub(r'\\!|\\,|\\;|\\ |\\left|\\right|\s', '', s)
    for pat in (r'\\frac\{(-?\d+)\}\{(-?\d+)\}', r'(-?\d+)/(-?\d+)'):
        m = re.fullmatch(pat, s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return _norm_num_text(str(a / b)) if b else None
    return _norm_num_text(s) if _NUM_RE.fullmatch(s) else None


def _load_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Load, numeric-filter, shuffle, then split a fixed eval holdout off the front."""
    # Load all when filtering or splitting (else the eval holdout could starve train).
    load_n = 0 if (args.numeric_only or args.eval_size > 0) else args.n
    records = load_problems(args.dataset, load_n, args.seed)
    raw_n, dropped = len(records), 0
    if args.numeric_only:
        kept = []
        for r in records:
            ref = _numeric_value(r.get('reference_answer'))
            if ref is None:
                dropped += 1
                continue
            kept.append({**r, 'reference_answer': ref})
        records = kept
    np.random.RandomState(args.seed).shuffle(records)
    eval_n = min(args.eval_size, len(records)) if args.eval_size > 0 else 0
    eval_records = [dict(r) for r in records[:eval_n]]
    pool = records[eval_n:]
    train_n = args.n if args.n > 0 else len(pool)
    train_records = [dict(r) for r in pool[:train_n]]
    overlap = {r['problem'] for r in train_records} & {r['problem'] for r in eval_records}
    if overlap:
        raise ValueError(f'eval/train overlap: {len(overlap)} problems')
    stats = {'raw_loaded': raw_n, 'numeric_dropped': dropped,
             'train_records': len(train_records), 'eval_records': len(eval_records)}
    return train_records, eval_records, stats


# ===========================================================================
# Block D -- disk cache, problem pool, baseline rollout, rubric check
# ===========================================================================
class DiskCache:
    """Append-only jsonl kv cache (md5 key -> value). Loads on init, appends on put.
    Disabled instances (``enabled=False``) always miss and never write."""

    def __init__(self, path: str, enabled: bool = True):
        self.path, self.enabled = path, enabled
        self._mem: Dict[str, Any] = {}
        self._fh = None
        if not enabled:
            return
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._mem[row['key']] = row['value']
        self._fh = open(path, 'a', encoding='utf-8')

    @staticmethod
    def key_for(*parts: str) -> str:
        return hashlib.md5('\x1f'.join(parts).encode('utf-8')).hexdigest()

    def __contains__(self, key: str) -> bool:
        return key in self._mem

    def get(self, key: str) -> Any:
        return self._mem.get(key)

    def put(self, key: str, value: Any) -> None:
        self._mem[key] = value
        if self._fh is not None:
            self._fh.write(json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '\n')
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


class ProblemPool:
    """Cyclic draw source. Each full pass reshuffles with ``seed + epoch``; the initial
    pass keeps the loader order. ``draw(k)`` returns k distinct problems (k << pool)."""

    def __init__(self, records: List[Dict[str, Any]], seed: int):
        self._records = list(records)
        self._seed, self._cursor, self.epoch = seed, 0, 0

    def draw(self, k: int) -> List[Dict[str, Any]]:
        out, seen = [], set()
        while len(out) < k:
            if self._cursor >= len(self._records):
                self.epoch += 1
                np.random.RandomState(self._seed + self.epoch).shuffle(self._records)
                self._cursor = 0
            r = self._records[self._cursor]
            self._cursor += 1
            if id(r) not in seen:
                seen.add(id(r))
                out.append(r)
        return out


def _empty_roll() -> Dict[str, Any]:
    return {'pred': '', 'correct': False, 'terminated': False, 'passed': False,
            'stop_reason': 'empty', 'gen_tokens': 0, 'text': ''}


def _apply_baseline(r: Dict[str, Any], roll: Dict[str, Any]) -> None:
    """Attach a greedy baseline roll and reset per-chunk working state."""
    r['_baseline_rolls'], r['_cands'], r['_init'] = [roll], [], [roll]
    r['_failed'] = not roll['correct']
    r['_baseline_pass'] = 1.0 if roll['correct'] else 0.0
    r['_hard'] = True  # process every problem; group variance selects (SEAM-style)


def baseline_rollout(base_sampler, problems: List[Dict[str, Any]], base_dp: int,
                     args: argparse.Namespace, cache: DiskCache) -> int:
    """Phase 1: base solves each problem greedily once (T=0, M=1), disk-cached by
    problem text. The base is frozen + greedy so the cache is exact. Returns the number
    of fresh (cache-miss) rollouts."""
    todo = [r for r in problems if DiskCache.key_for(r['problem']) not in cache]
    if todo:
        out = _run_samples(base_sampler, [build_direct_prompt(r['problem']) for r in todo],
                           1, args.max_tokens, base_dp, temperature=0.0)
        for r, seqs in zip(todo, out):
            roll = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
            cache.put(DiskCache.key_for(r['problem']), roll)
    for r in problems:
        _apply_baseline(r, cache.get(DiskCache.key_for(r['problem'])))
    return len(todo)


# -- rubric process-check (view A): teacher diagnoses the base's attempt --
_RFT_DIAG_SYSTEM = """\
You are a process error checker for a math solution attempt. You are given a math
problem, a rubric, and one attempted solution segment. Decide PASS or FAIL for each
criterion and explain only the process error type.

Output STRICT JSON (no prose outside it) with this shape:
{
  "items": [
    {"index": 1, "verdict": "PASS", "reason": "<why the process satisfies it>",
     "fix": ""},
    {"index": 2, "verdict": "FAIL", "reason": "<what process step is wrong>",
     "fix": "<method-level correction, without computing the corrected result>"}
  ],
  "overall": "OK" | "ISSUES",
  "summary": "<one sentence naming the process issue, not the answer>"
}

Rules:
- Judge every criterion independently and literally; a [Hard Rule] is FAIL unless
  unambiguously satisfied.
- Judge ONLY what is observable in THIS segment.
- Content inside <think>...</think> (or <thinking>) is internal reasoning, not
  user-facing output; ignore it for "output only X" style criteria.
- For PASS items, leave "fix" as "".
- For FAIL items, "reason", "fix", and "summary" must describe only the flawed
  step, theorem, arithmetic operation, case split, or verification habit.
- NEVER state the correct final answer, corrected final expression, option letter,
  graph/choice label, or any exact value that the answer should become.
- NEVER write phrases like "the correct answer is", "which gives", "yielding",
  "should be <value>", "Option <letter>", or "Graph <letter>".
- If a fix would require naming a corrected value, replace it with a method-level
  instruction such as "redo that computation carefully" or "apply the theorem with
  the correct quantities".
- Keep every "reason" and "fix" clear and concise — one short sentence each.
- "overall" is "OK" only if NO criterion is FAIL.
- Output only the JSON object."""

_RFT_DIAG_USER = """\
## Task / query (context)
{query}

## Rubric
{rubric}

## Segment
{segment}

Now output the diagnostic JSON object."""

_MATH_RUBRIC = [
    ('The reasoning contains no arithmetic or algebraic error', True),
    ('Each step follows logically from the previous ones', True),
    ('No formula or theorem is misstated or misapplied', True),
    ('The approach is on track to answer the actual question asked', False),
    ('No step contradicts an earlier established fact', False),
]


class _RftRubricVerifier(RubricVerifier):
    def _diagnose_trajectory(self, query: str, rubric_block: str, segment_text: str) -> dict:
        return {'messages': [
            {'role': 'system', 'content': _RFT_DIAG_SYSTEM},
            {'role': 'user', 'content': _RFT_DIAG_USER.format(
                query=query, rubric=rubric_block, segment=segment_text)}]}


def build_rubric_checker() -> Optional[RubricVerifier]:
    """Fixed math-process rubric verifier (teacher-served). None if no LLM backup env."""
    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('LLM_BACKUP_BASE_URL')
            or os.environ.get('OPENAI_API_KEY')):
        return None
    return _RftRubricVerifier(
        fixed_rubric=[RubricItem(t, is_hard=h) for t, h in _MATH_RUBRIC], gate=True)


def _format_diagnosis(detail) -> str:
    """One line per criterion (PASS/FAIL + reason + fix) then a summary."""
    rub = detail.rubric
    lines = []
    for it in detail.items:
        text = rub[it.index - 1].text if 0 < it.index <= len(rub) else f'criterion {it.index}'
        if it.verdict:
            lines.append(f'- [PASS] {text}')
        else:
            tail = f': {it.reason}' if it.reason else ''
            tail += f' (fix: {it.fix})' if it.fix else ''
            lines.append(f'- [FAIL] {text}{tail}')
    if detail.summary:
        lines.append(f'Summary: {detail.summary}')
    return '\n'.join(lines)


def diagnose_views(checker, hard: List[Dict[str, Any]], args: argparse.Namespace,
                   cache: DiskCache) -> None:
    """Rubric-check every view-A problem's greedy attempt in parallel (disk-cached by
    problem + attempt), stashing the formatted findings on ``r['_rubric_diag']``."""
    targets = [r for r in hard if r.get('_view') == 'A']
    if not checker or not targets:
        return

    def _key(r: Dict[str, Any]) -> str:
        return DiskCache.key_for(r['problem'], r.get('_init', [{}])[0].get('text', ''))

    pending = []
    for r in targets:
        key = _key(r)
        if key in cache:
            r['_rubric_diag'] = cache.get(key)
        else:
            pending.append((r, key))
    if not pending:
        return

    def _run(item):
        r, key = item
        seg = {'messages': [{'role': 'user', 'content': r['problem']},
                            {'role': 'assistant', 'content': r['_init'][0]['text']}]}
        try:
            return r, key, _format_diagnosis(checker.diagnose(seg, query=r['problem']))
        except Exception as exc:  # teacher hiccup -> no-diagnosis prompt (not cached)
            logger.warning(f'[rubric] diagnose error: {exc}')
            return r, key, None

    workers = max(1, min(args.rubric_workers, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r, key, diag in ex.map(_run, pending):
            r['_rubric_diag'] = diag or ''
            if diag is not None:
                cache.put(key, diag)


# ===========================================================================
# Block E -- chunk draw, pipeline, record building
# ===========================================================================
def _baseline_class(r: Dict[str, Any]) -> str:
    """success | fail_loop (out of length / never terminated) | fail_wrong."""
    roll = r['_init'][0]
    if roll['correct']:
        return 'success'
    return 'fail_loop' if (roll['stop_reason'] == 'length' or not roll['terminated']) else 'fail_wrong'


def _select_balanced(buckets: Dict[str, List[Dict[str, Any]]], n_success: int,
                     n_fail: int, n_fail_loop: int) -> List[Dict[str, Any]]:
    """Pick n_fail base-fails (toward n_fail_loop loop-fails, best-effort) + n_success
    base-successes; top up any shortfall from leftovers."""
    loop, wrong, succ = buckets['fail_loop'], buckets['fail_wrong'], buckets['success']
    take_loop = min(n_fail_loop, len(loop))
    take_wrong = min(n_fail - take_loop, len(wrong))
    take_loop = min(n_fail - take_wrong, len(loop))
    sel = loop[:take_loop] + wrong[:take_wrong] + succ[:n_success]
    target = n_success + n_fail
    if len(sel) < target:
        used = {id(x) for x in sel}
        sel += [x for b in (loop, wrong, succ) for x in b if id(x) not in used][:target - len(sel)]
    return sel


def draw_chunk(pool: ProblemPool, base_sampler, base_dp: int, args: argparse.Namespace,
               cache: DiskCache) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Draw one chunk, running baseline rollout (Phase 1) on every drawn problem. With
    ``--balance``, keep drawing+baselining until the target base fail:success mix is
    reachable (or the budget is hit), then select a balanced subset."""
    if not args.balance:
        chunk = pool.draw(args.chunk_size)
        fresh = baseline_rollout(base_sampler, chunk, base_dp, args, cache)
        return chunk, {'enabled': False, 'n_drawn': len(chunk), 'n_baseline_fresh': fresh}

    n_success = max(0, min(args.chunk_size, round(args.chunk_size * args.balance_success_frac)))
    n_fail = args.chunk_size - n_success
    n_fail_loop = round(n_fail * args.balance_loop_frac)
    buckets: Dict[str, List[Dict[str, Any]]] = {'success': [], 'fail_loop': [], 'fail_wrong': []}
    budget, n_drawn, n_fresh, seen = args.chunk_size * args.balance_max_draws_mult, 0, 0, set()
    while n_drawn < budget:
        if (len(buckets['success']) >= n_success
                and len(buckets['fail_loop']) + len(buckets['fail_wrong']) >= n_fail):
            break
        batch = pool.draw(args.chunk_size)
        n_fresh += baseline_rollout(base_sampler, batch, base_dp, args, cache)
        n_drawn += len(batch)
        for r in batch:
            if id(r) not in seen:
                seen.add(id(r))
                buckets[_baseline_class(r)].append(r)

    reached = (len(buckets['success']) >= n_success
               and len(buckets['fail_loop']) + len(buckets['fail_wrong']) >= n_fail)
    chunk = _select_balanced(buckets, n_success, n_fail, n_fail_loop)
    sel_success = sum(1 for r in chunk if not r['_failed'])
    stats = {
        'enabled': True, 'n_drawn': n_drawn, 'n_baseline_fresh': n_fresh, 'n_selected': len(chunk),
        'target_success': n_success, 'target_fail': n_fail, 'target_fail_loop': n_fail_loop,
        'selected_success': sel_success, 'selected_fail': len(chunk) - sel_success,
        'selected_fail_loop': sum(1 for r in chunk if _baseline_class(r) == 'fail_loop'),
        'selected_fail_wrong': sum(1 for r in chunk if _baseline_class(r) == 'fail_wrong'),
        'selected_success_frac': (sel_success / len(chunk)) if chunk else 0.0,
        'budget_hit': not reached,
    }
    return chunk, stats


def _assign_advantages(hard: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Group-relative advantage over each problem's scored candidates using the greedy
    binary reward R in {0,1}: ``A = (R - mean) / (std + eps)``. std==0 groups (all solve
    / all fail) get advantage 0 and no gradient -- GRPO's variance selects informative
    problems, so no explicit difficulty gate is needed."""
    eps = 1e-6
    for r in hard:
        for c in r['_cands']:
            c['advantage'], c['grpo_adv'], c['kept'] = 0.0, 0.0, False
        cs = ([c for c in r['_cands'] if c.get('reward') is not None] if args.format_in_reward
              else [c for c in r['_cands'] if c['leaked'] is False and c.get('reward') is not None])
        if len(cs) < 2:
            continue
        rewards = [c['reward'] for c in cs]
        mean_r = sum(rewards) / len(rewards)
        std = (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
        if std < 1e-9:
            continue
        for c in cs:
            adv = (c['reward'] - mean_r) / (std + eps)
            c['advantage'], c['grpo_adv'], c['kept'] = adv, adv, c['reward'] > mean_r


def process_chunk(base_sampler, skill_sampler, leak: LeakVerifier, chunk: List[Dict[str, Any]],
                  ci: int, base_dp: int, skill_dp: int, args: argparse.Namespace,
                  checker, rubric_cache: DiskCache
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """view assign -> rubric-check (view A) -> skill-gen -> leak-filter -> with-skill
    greedy pass -> GRPO advantages. ``chunk`` arrives already baselined by draw_chunk."""
    hard = chunk

    # Phase 2: view routing + view-A rubric check (view B is query-only, no rubric).
    for r in hard:
        r['_view'], r['_rubric_diag'] = _assign_view(r['problem'], args), ''
    diagnose_views(checker, hard, args, rubric_cache)

    # Phase 3: skill-gen (thinking ON), per-view prompt; re-sample empties.
    flat: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    pending = list(hard)
    for _ in range(args.skill_retries + 1):
        if not pending:
            break
        sg_out = _run_samples(skill_sampler, [_view_prompt(r) for r in pending],
                              args.n_skills, args.skill_max_tokens, skill_dp,
                              temperature=args.skill_gen_temperature,
                              top_p=args.skill_gen_top_p, top_k=args.skill_gen_top_k)
        still = []
        for r, seqs in zip(pending, sg_out):
            got = False
            for s in seqs:
                resp = _clean_text(getattr(s, 'decoded', '') or '')
                block = _extract_skills_block(resp)
                cand = {'skills': block or '', 'response': resp, 'parseable': bool(block),
                        'view': r['_view'], 'leaked': None, 'leak_reason': '',
                        'leak_source': '', 'with_pass': None, 'reward': None, 'rolls': []}
                r['_cands'].append(cand)
                if block:
                    flat.append((r, cand))
                    got = True
            if not got:
                still.append(r)
        pending = still

    # Phase 4: leak filter (view A only; view B is query-only -> treated clean, SEAM-like).
    for r, c in flat:
        if r.get('_view') != 'A':
            c['leaked'], c['leak_reason'], c['leak_source'] = False, '', 'skipped_viewB'
    flat_a = [(r, c) for r, c in flat if r.get('_view') == 'A']
    if flat_a:
        details = leak.leak_batch(
            [{'content': c['skills'], 'query': r['problem'], 'reference': r['reference_answer']}
             for r, c in flat_a], max_workers=args.leak_workers)
        for (r, c), d in zip(flat_a, details):
            c['leaked'], c['leak_reason'], c['leak_source'] = bool(d.leaked), d.reason, d.source

    # Phase 5: with-skill greedy pass (T=0, M=1) on clean candidates. Reward = correct,
    # absolute (no baseline subtraction); the group mean in Phase 6 is the only baseline.
    clean = [(r, c) for r, c in flat if c['leaked'] is False]
    if clean:
        ws_out = _run_samples(base_sampler,
                              [build_skill_solve_prompt(r['problem'], c['skills']) for r, c in clean],
                              1, args.max_tokens, base_dp, temperature=0.0)
        for (r, c), seqs in zip(clean, ws_out):
            c['rolls'] = [_parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()]
            c['with_pass'] = 1.0 if c['rolls'][0]['correct'] else 0.0
            c['reward'] = c['with_pass']
    if args.format_in_reward:  # unparseable/leaked candidates score 0 and still join the group
        for r in hard:
            for c in r['_cands']:
                if c['reward'] is None:
                    c['reward'] = 0.0

    # Phase 6: group-relative GRPO advantage.
    _assign_advantages(hard, args)
    return ([_full_record(r, ci) for r in chunk], _chunk_summary(chunk, ci, args),
            _group_records(chunk, args))


def _roll(x: Dict[str, Any]) -> Dict[str, Any]:
    return {k: x[k] for k in ('pred', 'correct', 'terminated', 'passed',
                              'stop_reason', 'gen_tokens', 'text')}


def _is_trainable(c: Dict[str, Any], args: argparse.Namespace) -> bool:
    """A candidate reaches the GRPO update iff its advantage is non-zero (and, without
    --format-in-reward, is also clean and scored)."""
    adv_nz = abs(c.get('advantage') or 0.0) > 1e-9
    if args.format_in_reward:
        return adv_nz
    return c['leaked'] is False and c.get('with_pass') is not None and adv_nz


def _full_record(r: Dict[str, Any], ci: int) -> Dict[str, Any]:
    """Complete per-problem trace: init attempt, baseline, and all candidates."""
    init = r['_init'][0]
    return {
        'record_type': 'problem', 'chunk': ci, 'problem': r['problem'],
        'reference_answer': r['reference_answer'], 'level': r.get('level', ''),
        'failed_first_try': r['_failed'],
        'init_attempt': {'text': init['text'], 'pred': init['pred'], 'correct': init['correct'],
                         'terminated': init['terminated'], 'stop_reason': init['stop_reason'],
                         'gen_tokens': init['gen_tokens']},
        'baseline_pass': r['_baseline_pass'], 'is_hard': r['_hard'],
        'view': r.get('_view', ''), 'rubric_diag': r.get('_rubric_diag', ''),
        'baseline_rolls': [_roll(x) for x in r['_baseline_rolls']],
        'candidates': [{
            'skills': c['skills'], 'response': c['response'], 'parseable': c['parseable'],
            'leaked': c['leaked'], 'leak_reason': c['leak_reason'], 'leak_source': c['leak_source'],
            'with_pass': c['with_pass'], 'reward': c.get('reward'), 'advantage': c.get('advantage'),
            'grpo_adv': c.get('grpo_adv'), 'kept': c.get('kept'),
            'rolls': [_roll(x) for x in c['rolls']],
        } for c in r['_cands']],
    }


def _view_stats(hard: List[Dict[str, Any]], view: str) -> Dict[str, Any]:
    hv = [r for r in hard if r.get('_view') == view]
    cands = [c for r in hv for c in r['_cands'] if c['parseable']]
    clean = [c for c in cands if c['leaked'] is False]
    adopted = sum(1 for r in hv
                  if any(c['leaked'] is False and abs(c.get('advantage') or 0.0) > 1e-9
                         for c in r['_cands']))
    return {'n_hard': len(hv), 'n_candidates_parseable': len(cands), 'n_clean': len(clean),
            'n_adopted_problems': adopted, 'adoption_rate': (adopted / len(hv)) if hv else 0.0}


def _chunk_summary(chunk: List[Dict[str, Any]], ci: int, args: argparse.Namespace) -> Dict[str, Any]:
    hard = [r for r in chunk if r['_hard']]
    all_cands = [c for r in chunk for c in r['_cands']]
    cands = [c for c in all_cands if c['parseable']]
    scored = [c for c in cands if c['with_pass'] is not None]
    ws_rolls = [x for c in scored for x in c['rolls']]
    train_cands = [c for c in all_cands if _is_trainable(c, args)]
    fail_cands = [c for r in chunk if r['_failed'] for c in r['_cands']]
    base_acc = (sum(r['_baseline_pass'] for r in hard) / len(hard)) if hard else 0.0
    ws_acc = (sum(c['with_pass'] for c in scored) / len(scored)) if scored else 0.0
    abs_adv = lambda cs: sum(abs(c.get('advantage') or 0.0) for c in cs)
    total_abs = abs_adv(all_cands)
    return {
        'record_type': 'summary', 'chunk': ci, 'n': len(chunk),
        'n_failed_first_try': sum(1 for r in chunk if r['_failed']), 'n_hard': len(hard),
        'n_generated': len(all_cands), 'n_candidates_parseable': len(cands),
        'n_unparseable': len(all_cands) - len(cands),
        'n_leaked': sum(1 for c in cands if c['leaked']),
        'n_clean': sum(1 for c in cands if c['leaked'] is False),
        'n_reward_pos': sum(1 for c in scored if c['reward']), 'n_train_samples': len(train_cands),
        'n_train_from_fail': sum(1 for c in fail_cands if _is_trainable(c, args)),
        'abs_adv_from_fail_frac': (abs_adv(fail_cands) / total_abs) if total_abs > 0 else 0.0,
        'avg_baseline_pass_on_hard': base_acc, 'avg_withskill_pass': ws_acc,
        'avg_lift': ws_acc - base_acc,
        'termination_rate_withskill': (sum(1 for x in ws_rolls if x['terminated']) / len(ws_rolls)) if ws_rolls else 0.0,
        'view_A': _view_stats(hard, 'A'), 'view_B': _view_stats(hard, 'B'),
    }


def _group_records(chunk: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """GRPO training records: every trainable candidate with its view + rubric diagnosis
    (the prompt is rebuilt from those by ``_skillgen_messages``, no trajectory stored)."""
    out = []
    for r in chunk:
        if not r['_hard']:
            continue
        for c in r['_cands']:
            if _is_trainable(c, args):
                out.append({
                    'problem': r['problem'], 'reference_answer': r['reference_answer'],
                    'view': r.get('_view', 'A'), 'diagnosis': r.get('_rubric_diag', ''),
                    'response': c['response'], 'skills': c['skills'],
                    'advantage': c['advantage'], 'grpo_adv': c['grpo_adv'], 'kept': c['kept'],
                    'reward': c['reward'], 'with_pass': c['with_pass']})
    return out


# ===========================================================================
# Block F -- samplers, args, main
# ===========================================================================
def init_samplers(args: argparse.Namespace):
    """8 GPUs: ranks 0-3 skill_sampler, ranks 4-7 base_sampler (both vLLM tp1 dp4)."""
    twinkle.initialize(mode='ray', nproc_per_node=8, lazy_collect=False, groups=[
        DeviceGroup(name='skill_sampler', ranks=list(range(0, 4)), device_type='GPU'),
        DeviceGroup(name='base_sampler', ranks=list(range(4, 8)), device_type='GPU')])
    samplers = []
    for group in ('skill_sampler', 'base_sampler'):
        s = vLLMSampler(
            model_id=MODEL_ID,
            engine_args={'gpu_memory_utilization': GPU_MEM,
                         'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
            device_mesh=DeviceMesh.from_sizes(world_size=4, dp_size=4), remote_group=group)
        s.set_template('Template', model_id=MODEL_ID, enable_thinking=True,
                       max_length=args.max_model_len)
        samplers.append(s)
    return samplers[1], samplers[0], 4, 4  # base_sampler, skill_sampler, base_dp, skill_dp


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--total-problems', type=int, default=3200,
                   help='Final number of problems selected into generated chunks.')
    p.add_argument('--base-success-frac', type=float, default=0.3,
                   help='Target fraction of selected problems the frozen base solves.')
    p.add_argument('--output-dir', default='./output/reflexion_skill_data')
    p.add_argument('--cache-dir', default='',
                   help='Baseline/rubric cache dir (default <output-dir>/cache).')
    p.add_argument('--no-cache', action='store_true', help='Disable disk cache read/write.')
    p.add_argument('--overwrite', action='store_true', help='Replace existing output jsonl.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    p.add_argument('--n', type=int, default=0,
                          help='Raw train-pool size; 0 derives it from --total-problems.')
    p.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval-size', type=int, default=128)
    p.add_argument('--chunk-size', type=int, default=16)
    p.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--balance-loop-frac', type=float, default=0.5)
    p.add_argument('--balance-max-draws-mult', type=int, default=8)
    p.add_argument('--n-skills', type=int, default=8)
    p.add_argument('--view-b-frac', type=float, default=0.5)
    p.add_argument('--skill-retries', type=int, default=2)
    p.add_argument('--skill-gen-temperature', type=float, default=1.0)
    p.add_argument('--skill-gen-top-p', type=float, default=1.0)
    p.add_argument('--skill-gen-top-k', type=int, default=-1)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--skill-max-tokens', type=int, default=8192)
    p.add_argument('--leak-workers', type=int, default=16)
    p.add_argument('--rubric-workers', type=int, default=16)
    p.add_argument('--format-in-reward', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.total_problems <= 0 or args.chunk_size <= 0:
        raise ValueError('--total-problems and --chunk-size must be positive')
    if not 0.0 <= args.base_success_frac <= 1.0:
        raise ValueError('--base-success-frac must be in [0, 1]')
    args.chunks = math.ceil(args.total_problems / args.chunk_size)
    args.balance_success_frac = args.base_success_frac
    if args.n <= 0:
        args.n = max(args.total_problems + args.eval_size, math.ceil(args.total_problems * 1.5))
    return args


def _write(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = _build_args()
    records, eval_records, data_stats = _load_records(args)
    if not records:
        raise ValueError(f'loaded 0 {args.dataset} problems')
    if len(records) < args.chunk_size:
        raise ValueError(f'--chunk-size ({args.chunk_size}) exceeds loaded ({len(records)})')

    os.makedirs(args.output_dir, exist_ok=True)
    data_path = os.path.join(args.output_dir, 'skill_dataset.jsonl')
    gen_path = os.path.join(args.output_dir, 'gen_records.jsonl')
    eval_path = os.path.join(args.output_dir, 'eval_holdout.jsonl')
    for path in (data_path, gen_path, eval_path):
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f'{path} exists; pass --overwrite to replace it')

    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('OPENAI_API_KEY')):
        sys.stderr.write('[build] WARNING: no LLM backup env; leak/rubric checks degrade\n')

    base_sampler, skill_sampler, base_dp, skill_dp = init_samplers(args)
    leak = LeakVerifier(sampler=None, answer_only=True)
    checker = build_rubric_checker()
    pool = ProblemPool(records, args.seed)

    cache_dir = args.cache_dir or os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    use_cache = not args.no_cache
    baseline_cache = DiskCache(os.path.join(cache_dir, 'baseline.jsonl'), use_cache)
    rubric_cache = DiskCache(os.path.join(cache_dir, 'rubric.jsonl'), use_cache)

    cfg = {
        'record_type': 'config', 'mode': 'offline_data_build', 'model': MODEL_ID,
        'dataset': args.dataset, 'n': len(records), 'eval_n': len(eval_records),
        'total_problems': args.total_problems, 'seed': args.seed, 'numeric_only': args.numeric_only,
        'raw_loaded': data_stats['raw_loaded'], 'numeric_dropped': data_stats['numeric_dropped'],
        'chunks': args.chunks, 'chunk_size': args.chunk_size, 'n_skills': args.n_skills,
        'view_b_frac': args.view_b_frac, 'balance': args.balance,
        'base_success_frac': args.base_success_frac, 'balance_success_frac': args.balance_success_frac,
        'reward': 'greedy_binary(correct)', 'advantage': 'group_relative',
        'format_in_reward': args.format_in_reward, 'cache': use_cache,
        'rubric_check': 'fixed_math_5crit(viewA)' if checker else 'disabled',
        'started': int(time.time()),
    }
    total_groups, selected = 0, 0
    with open(gen_path, 'w', encoding='utf-8') as gen_f, \
            open(data_path, 'w', encoding='utf-8') as data_f, \
            open(eval_path, 'w', encoding='utf-8') as eval_f:
        for handle in (gen_f, data_f, eval_f):
            _write(handle, cfg)
        for rec in eval_records:
            _write(eval_f, {'record_type': 'eval_holdout', **rec})
        eval_f.flush()

        full_chunk_size = args.chunk_size
        for ci in range(args.chunks):
            remaining = args.total_problems - selected
            if remaining <= 0:
                break
            args.chunk_size = min(full_chunk_size, remaining)  # last chunk may be short
            chunk, balance = draw_chunk(pool, base_sampler, base_dp, args, baseline_cache)
            full, summary, groups = process_chunk(
                    base_sampler, skill_sampler, leak, chunk, ci, base_dp, skill_dp,
                    args, checker, rubric_cache)
            summary['balance'] = balance
            for rec in full:
                _write(gen_f, rec)
            _write(gen_f, summary)
            gen_f.flush()
            for row in groups:
                _write(data_f, {'chunk': ci, **row})
            data_f.flush()
            total_groups += len(groups)
            selected += len(chunk)
            sys.stderr.write(
                f'[build] g{ci}: problems={selected}/{args.total_problems} '
                f'train={len(groups)} total={total_groups} '
                f'acc={summary["avg_baseline_pass_on_hard"]:.2f}->{summary["avg_withskill_pass"]:.2f} '
                f'lift={summary["avg_lift"]:+.3f}\n')

    baseline_cache.close()
    rubric_cache.close()
    sys.stderr.write(f'[build] done: {total_groups} train records -> {data_path}; trace -> {gen_path}\n')


if __name__ == '__main__':
    main()
