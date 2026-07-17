"""Online GRPO RFT for the reflexion skill generator (self-contained, cached).

Trains an INDEPENDENT skill model to write reusable skills that, injected into a
FROZEN base solver's system prompt, raise its accuracy. The base is never trained;
it only produces the reward. Per chunk: base greedy solve -> rubric process-check
(view A) -> skill-gen (thinking ON, N candidates) -> leak filter -> with-skill
greedy pass -> group-relative advantage -> ONE on-policy GRPO step -> sync weights.
Reward is deterministic (T=0, M=1) binary correctness; A = (R - mean) / (std + eps)
within each problem-group, so std=0 groups give no gradient (GRPO variance selects).

Each problem is routed to EXACTLY ONE view: A = problem + rubric findings, B =
query only (deployment form). Skill-gen trains only the final <skills> turn.

Base greedy rollouts and rubric diagnoses are cached to disk (jsonl, md5-keyed) so
restarts skip them; skill-gen is on-policy and never cached.

8 GPUs default for high-memory cards: rank 0 trains the actor, rank 1 hosts a
frozen reference model, ranks 2-3 skill_sampler (synced), ranks 4-7 base_sampler
(frozen). Override TRAIN_GPUS / REF_GPUS / SKILL_SAMPLER_GPUS / BASE_SAMPLER_GPUS
for other layouts.
Leak / rubric use the backup teacher API: set LLM_BACKUP_API_KEY / LLM_BACKUP_BASE_URL /
LLM_BACKUP_MODEL.

Launch:
    LLM_BACKUP_API_KEY=... python cookbook/exp/embedding/train_reflexion_skill.py \
        --dataset aops --n 5000 --chunk-size 16 --lr 6e-6
"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.patch.no_split_modules import NoSplitModulesPatch
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle_agentic.verifier import RubricVerifier
from twinkle_agentic.verifier.rubric_verifier import RubricItem

logger = get_logger()

try:
    import swanlab
except ImportError:
    swanlab = None

MODEL_ID = os.environ.get('GEN_MODEL_ID', 'Qwen/Qwen3-4B')
GPU_MEM = float(os.environ.get('GEN_GPU_MEM', 0.8))
GEN_TEMPERATURE = float(os.environ.get('GEN_TEMPERATURE', 0.6))
GEN_TOP_P = float(os.environ.get('GEN_TOP_P', 0.95))
AOPS_DATASET_ID = os.environ.get('AOPS_DATASET_ID', 'AI-MO/aops')
MATH_DATASET_ID = os.environ.get('MATH_DATASET_ID', 'modelscope/competition_math')

# GPU layout: train actor (synced to skill_sampler) + frozen ref model + skill_sampler + base_sampler.
# High-memory cards can keep Qwen3-4B actor/ref as one GPU each and spend more GPUs
# on vLLM data-parallel sampling. The base side is heavier here because every
# clean skill candidate is re-solved by the frozen base model, plus balance/eval baselines.
TRAIN_GPUS = int(os.environ.get('TRAIN_GPUS', 2))
REF_GPUS = int(os.environ.get('REF_GPUS', 2))
SKILL_SAMPLER_GPUS = int(os.environ.get('SKILL_SAMPLER_GPUS', 2))
BASE_SAMPLER_GPUS = int(os.environ.get('BASE_SAMPLER_GPUS', 2))
NUM_GPUS = TRAIN_GPUS + REF_GPUS + SKILL_SAMPLER_GPUS + BASE_SAMPLER_GPUS
TRAIN_FSDP = int(os.environ.get('TRAIN_FSDP', min(1, TRAIN_GPUS)))
REF_FSDP = int(os.environ.get('REF_FSDP', min(1, REF_GPUS)))
if min(TRAIN_GPUS, REF_GPUS, SKILL_SAMPLER_GPUS, BASE_SAMPLER_GPUS, TRAIN_FSDP, REF_FSDP) < 1:
    raise ValueError('TRAIN_GPUS, REF_GPUS, SKILL_SAMPLER_GPUS, BASE_SAMPLER_GPUS, TRAIN_FSDP and REF_FSDP must all be >= 1')
if TRAIN_GPUS % TRAIN_FSDP != 0:
    raise ValueError(f'TRAIN_GPUS ({TRAIN_GPUS}) must be divisible by TRAIN_FSDP ({TRAIN_FSDP})')
if REF_GPUS % REF_FSDP != 0:
    raise ValueError(f'REF_GPUS ({REF_GPUS}) must be divisible by REF_FSDP ({REF_FSDP})')
TRAIN_DP = TRAIN_GPUS // TRAIN_FSDP
REF_DP = REF_GPUS // REF_FSDP


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

_SKILL_SOLVE_PREFIX = (
    DIRECT_SYSTEM + '\n\n'
    'Before you start, keep these reminders in mind to avoid common mistakes on this '
    'type of problem:\n')
_SKILL_SOLVE_SUFFIX = '\nApply them where relevant, but rely on your own reasoning to reach the answer.'


def build_direct_prompt(problem: str) -> Dict[str, Any]:
    return {'messages': [{'role': 'system', 'content': DIRECT_SYSTEM},
                         {'role': 'user', 'content': problem}]}


def build_skill_solve_prompt(problem: str, skill: str) -> Dict[str, Any]:
    skill = (skill or '').strip()
    if not skill:
        return build_direct_prompt(problem)
    # Concatenation (not .format): DIRECT_SYSTEM/skill contain literal braces.
    return {'messages': [
        {'role': 'system', 'content': _SKILL_SOLVE_PREFIX + skill + _SKILL_SOLVE_SUFFIX},
        {'role': 'user', 'content': problem}]}


# -- skill-gen prompts (view A: problem + rubric source; view B: query only) --
SKILL_GEN_SYSTEM = (
    'You are a math guidance writer. You are given a target problem and an automated '
    'process-check from a related problem. Use it as context to write reusable guidance '
    'for this and similar problems.\n'
    'Output only a non-empty <skills>...</skills> block.')

SKILL_GEN_SYSTEM_Q = (
    'You are a math guidance writer. Given the problem below, write reusable guidance '
    'for this and similar problems.\n'
    'Output only a non-empty <skills>...</skills> block.')

SKILL_GEN_USER_Q = (
    'Problem:\n{problem}\n\n'
    '<skills>')

SKILL_GEN_USER_RUBRIC = (
    'Target problem:\n{problem}\n\n'
    'Problem used for the process check:\n{rubric_problem}\n\n'
    'Process check:\n'
    '{diagnosis}\n\n'
    '<skills>')


def _skillgen_messages(problem: str, view: str, diagnosis: str,
                       rubric_problem: str = '') -> List[Dict[str, Any]]:
    """Single source of truth for the skill-gen prompt (used at BOTH generation and
    training so they never diverge). View A with a localisable failure uses the target
    problem plus the rubric source problem and findings; view B -- or a view-A problem
    whose rubric flagged nothing (no ``[FAIL]``) -- degrades to the query-only prompt."""
    if view == 'B' or '[FAIL]' not in (diagnosis or ''):
        return [{'role': 'system', 'content': SKILL_GEN_SYSTEM_Q},
                {'role': 'user', 'content': SKILL_GEN_USER_Q.format(problem=problem)}]
    rubric_problem = rubric_problem or problem
    return [{'role': 'system', 'content': SKILL_GEN_SYSTEM},
            {'role': 'user', 'content': SKILL_GEN_USER_RUBRIC.format(
                problem=problem, rubric_problem=rubric_problem, diagnosis=diagnosis)}]


def _assign_view(problem: str, args: argparse.Namespace) -> str:
    h = int(hashlib.md5(f'{args.seed}:{problem}'.encode('utf-8')).hexdigest(), 16)
    return 'B' if (h % 100000) / 100000.0 < args.view_b_frac else 'A'


def _view_prompt(r: Dict[str, Any]) -> Dict[str, Any]:
    return {'messages': _skillgen_messages(
        r['problem'], r['_view'], r.get('_rubric_diag', ''), r.get('_rubric_src', ''))}


_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


def _clean_text(decoded: Optional[str]) -> str:
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').rstrip()


def _extract_skills_block(text: str) -> Optional[str]:
    """Non-empty ``<skills>...</skills>`` block, or None. Requires ``</think>`` (skill-gen
    runs thinking ON); reads only the answer after the last one, so a mid-reasoning draft
    or a demo echo can never be mistaken for the answer. No format/wording gate beyond
    that -- the reward (does the frozen executor solve a similar problem with this skill?)
    is what judges skill quality (SEAM-style), so we do not lexically second-guess it."""
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
    if e < 0:
        return None
    block = answer[inner:e].strip()
    block = re.sub(r'</?(?:skills|think)>', '', block, flags=re.IGNORECASE).strip()
    return block or None


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
def _boxed_batch(rows: Dict[str, List[Any]], dataset: str) -> Dict[str, List[Any]]:
    """Batched (columnar) mapper for ``Dataset.map``: annotate each row with its boxed
    ``reference_answer`` and a ``_keep`` flag (aops also needs the ``boxed`` metadata)."""
    sols = rows['solution']
    metas = rows.get('metadata', [None] * len(sols))
    refs = [extract_boxed(s or '') for s in sols]
    keep = [bool(ref) and (dataset != 'aops' or bool((meta or {}).get('boxed')))
            for ref, meta in zip(refs, metas)]
    return {**rows, 'reference_answer': refs, '_keep': keep}


def load_problems(dataset: str, n: int, seed: int, num_proc: int = 0) -> List[Dict[str, Any]]:
    """Load boxed-answer problems as ``{problem, reference_answer, level?}`` via
    twinkle.Dataset (ModelScope hub), sampled to ``n`` (0 = all). Boxed extraction (regex
    + brace scan over every solution) is parallelised by ``Dataset.map``/``.filter``;
    ``num_proc`` defaults to all cores (set 1 to force serial)."""
    ds_id = AOPS_DATASET_ID if dataset == 'aops' else MATH_DATASET_ID
    ds = Dataset(DatasetMeta(dataset_id=f'ms://{ds_id}', split='train'))
    nproc = num_proc if num_proc > 0 else min(32, os.cpu_count() or 1)
    ds.map(lambda rows: _boxed_batch(rows, dataset), num_proc=nproc)
    ds.filter(lambda row: row['_keep'], num_proc=nproc)
    has_level = 'level' in ds.dataset.column_names
    out = [{'problem': row['problem'], 'reference_answer': row['reference_answer'],
            **({'level': row['level']} if has_level and row.get('level') else {})}
           for row in ds.dataset]
    logger.info(f'[data] {dataset}: {len(out)} boxed problems ({nproc} procs)')
    rng = np.random.RandomState(seed)
    rng.shuffle(out)
    return out[:n] if (n and n < len(out)) else out


# ---------------------------------------------------------------------------
# Bag-of-words neighbour pairing (cross-problem rubric transfer, --xproblem-rubric)
# ---------------------------------------------------------------------------
# Common English + math-scaffolding words that carry no problem-type signal. Kept small
# and deterministic on purpose (no external stopword list): what survives is the domain
# vocabulary ('triangle', 'prime', 'polynomial', ...) that actually defines the type.
_BOW_STOP = frozenset("""
a an the of to in on at for and or but if is are be was were been being this that these those
with without into onto from by as it its their his her our your my we you they he she them
find compute determine calculate evaluate solve show prove given let suppose consider assume
what which when where how many much value values number numbers expression form terms term
such that then than so if only when each every all any some both one two three four five six
seven eight nine ten first second third last non over under about above below between
problem answer result equal equals sum difference product total following there here have has
had do does did can could will would should may might must not no yes if then else
""".split())

_WORD_RE = re.compile(r'[a-z]+')


def _stem(w: str) -> str:
    """Crude suffix stripper so 'prime'/'primes', 'triangle'/'triangles' collapse to one
    type token. Not linguistically correct -- just enough to merge the common plural/gerund
    variants that otherwise split a type's vocabulary and starve the df filter."""
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'                     # properties -> property
    if len(w) > 4 and w.endswith('es') and w[-3] in 'sxzh':
        return w[:-2]                           # boxes -> box (keep primes -> prime below)
    for suf in ('ing', 'ed', 's'):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def _tokenize(problem: str) -> List[str]:
    """Deterministic bag-of-words tokens for type matching: lowercase alphabetic words
    (numbers dropped -- they are instance detail, not type), minus generic stopwords, then
    stemmed, so only domain terms remain. Latex control words (frac, sqrt, ...) survive."""
    return [_stem(w) for w in _WORD_RE.findall((problem or '').lower())
            if len(w) > 2 and w not in _BOW_STOP]


class BagOfWordsIndex:
    """TF-IDF cosine nearest-neighbour over problem statements. Sparse dict vectors +
    an inverted index, so ``nearest`` scores only problems sharing a term (near-linear in
    practice, no dense NxN). Deterministic; ties break on lower index for reproducibility.

    Answer-aware: ``nearest`` skips the same index, near-verbatim duplicates (cosine
    >= ``sim_max``) and -- crucially for anti-leak -- any candidate whose answer equals the
    query's, so a neighbour rubric can never hand over the query's own answer."""

    def __init__(self, problems: List[str], answers: Optional[List[str]] = None,
                 min_df: int = 2, max_df_frac: float = 0.5):
        self._toks = [_tokenize(p) for p in problems]
        self._ans = [(_numeric_value(a) or (str(a).strip() if a else '')) for a in answers] \
            if answers is not None else [''] * len(problems)
        n = len(self._toks)
        df: Dict[str, int] = {}
        for toks in self._toks:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        max_df = max(min_df, int(max_df_frac * n))
        self._idf = {w: math.log((n + 1) / (c + 1)) + 1.0
                     for w, c in df.items() if min_df <= c <= max_df}
        self._vecs: List[Dict[str, float]] = [self._vectorize(t) for t in self._toks]
        self._inverted: Dict[str, List[int]] = {}
        for i, v in enumerate(self._vecs):
            for w in v:
                self._inverted.setdefault(w, []).append(i)

    def _vectorize(self, toks: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for w in toks:
            if w in self._idf:
                tf[w] = tf.get(w, 0.0) + 1.0
        vec = {w: c * self._idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in vec.values()))
        return {w: x / norm for w, x in vec.items()} if norm > 0 else {}

    def nearest(self, i: int, sim_max: float = 0.98) -> Tuple[int, float]:
        """(index, cosine) of the most similar *distinct* problem: not i, not a duplicate
        (cosine < ``sim_max``) and not the same answer. (-1, 0.0) if none qualifies."""
        vi = self._vecs[i]
        if not vi:
            return -1, 0.0
        ai = self._ans[i]
        scores: Dict[int, float] = {}
        for w, xi in vi.items():
            for j in self._inverted.get(w, ()):
                if j != i:
                    scores[j] = scores.get(j, 0.0) + xi * self._vecs[j].get(w, 0.0)
        best_j, best_s = -1, 0.0
        for j, s in scores.items():
            if s >= sim_max or (ai and self._ans[j] == ai):
                continue
            if s > best_s or (s == best_s and (best_j < 0 or j < best_j)):
                best_j, best_s = j, s
        return best_j, best_s


def build_pairs(records: List[Dict[str, Any]], n: int, seed: int, sim_max: float = 0.98
                ) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[str, float]]]:
    """Single-pass cross-problem pairing over the whole pool (one index build).

    Returns ``(subset, neighbour_map)`` where ``subset`` is the ``n`` problems with the
    strongest qualifying neighbour (dense same-type pairs; n<=0/n>=len keeps all, shuffled)
    and ``neighbour_map[q] = (p, cosine)`` gives each kept problem its analogue P. P is drawn
    from the *full* pool (richer analogues) but never a duplicate or same-answer problem, so
    P's rubric can transfer method without ever leaking Q's answer."""
    index = BagOfWordsIndex([r['problem'] for r in records],
                            [str(r.get('reference_answer', '')) for r in records])
    nbr = [index.nearest(i, sim_max) for i in range(len(records))]
    order = sorted(range(len(records)), key=lambda i: (-nbr[i][1], i))
    keep = order[:n] if (0 < n < len(records)) else list(range(len(records)))
    rng = np.random.RandomState(seed)
    rng.shuffle(keep)
    subset = [records[i] for i in keep]
    neighbour_map = {records[i]['problem']: (records[nbr[i][0]]['problem'], nbr[i][1])
                     for i in keep if nbr[i][0] >= 0}
    return subset, neighbour_map


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


def _answer_leaked(skill: str, reference: str) -> bool:
    """Deterministic answer-leak check (replaces the LLM LeakVerifier). Flags ONLY the one
    real way a skill can game the deterministic reward: writing the final answer verbatim.
    On our numeric-only data the answer is a single number, so we require it as a
    standalone token (digit boundaries) -- that avoids matching an intermediate value like
    '3' inside '36'. Everything else (methods, plans, pitfalls) is left to the reward, the
    way SEAM/POPE handle it (no content audit). Non-numeric answers -> not flagged."""
    if not skill:
        return False
    for cand in {_numeric_value(reference), (str(reference).strip() or None)}:
        if cand and re.search(r'(?<![\d.])' + re.escape(cand) + r'(?![\d.])', skill):
            return True
    return False


def _load_records(args: argparse.Namespace
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                             Dict[str, Tuple[str, float]], Dict[str, str], Dict[str, int]]:
    """Load, numeric-filter, shuffle, split a fixed eval holdout, and (when --xproblem-rubric)
    select a same-type-dense train subset with its neighbour map -- all in one pass.
    Also returns ``pool_answers`` (every candidate neighbour P's true answer) so P's baseline
    can be graded/cached correctly even when P is not itself a training problem."""
    # Load all when filtering or splitting (else the eval holdout could starve train).
    load_n = 0 if (args.numeric_only or args.eval_size > 0) else args.n
    records = load_problems(args.dataset, load_n, args.seed)
    raw_n, dropped = len(records), 0
    if args.numeric_only:
        records = [{**r, 'reference_answer': v}
                   for r, v in ((r, _numeric_value(r.get('reference_answer'))) for r in records)
                   if v is not None]
        dropped = raw_n - len(records)
    np.random.RandomState(args.seed).shuffle(records)
    eval_n = min(args.eval_size, len(records)) if args.eval_size > 0 else 0
    eval_records = [dict(r) for r in records[:eval_n]]
    pool = records[eval_n:]
    train_n = args.n if args.n > 0 else len(pool)
    # Cross-problem rubric: pick the train_n problems with the strongest qualifying neighbour
    # (dense same-type pairs) and build {Q -> (P, sim)} in one index pass. Otherwise keep the
    # first train_n (already shuffled) with no neighbours.
    if args.xproblem_rubric:
        subset, neighbor_map = build_pairs(pool, train_n, args.seed)
        train_records = [dict(r) for r in subset]
        pool_answers = {r['problem']: str(r.get('reference_answer', '')) for r in pool}
    else:
        train_records, neighbor_map, pool_answers = [dict(r) for r in pool[:train_n]], {}, {}
    if {r['problem'] for r in train_records} & {r['problem'] for r in eval_records}:
        raise ValueError('eval/train overlap detected')
    stats = {'raw_loaded': raw_n, 'numeric_dropped': dropped,
             'train_records': len(train_records), 'eval_records': len(eval_records)}
    return train_records, eval_records, neighbor_map, pool_answers, stats


# ===========================================================================
# Block D -- disk cache, problem pool, baseline rollout, rubric check
# ===========================================================================
class DiskCache:
    """Append-only jsonl kv cache (md5 key -> value). Loads on init, appends on put.
    Disabled instances always miss and never write."""

    def __init__(self, path: str, enabled: bool = True):
        self._mem: Dict[str, Any] = {}
        self._fh = None
        self._lock = threading.Lock()  # base baseline is prefetched on a background thread
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
        with self._lock:
            return key in self._mem

    def get(self, key: str) -> Any:
        with self._lock:
            return self._mem.get(key)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._mem[key] = value
            if self._fh is not None:
                self._fh.write(json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '\n')
                self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


class _LockedSampler:
    """Serialises every ``.sample()`` on a sampler behind one lock. The base sampler is
    shared by the main thread (with-skill scoring, eval) and the baseline-prefetch thread;
    ``sample`` is a slice_dp/flatten remote call whose collect is NOT safe to interleave
    across two callers, so concurrent calls could mis-join sequences. The lock keeps base
    calls serial (prefetch still overlaps the skill-gen phase, which uses skill_sampler)."""

    def __init__(self, sampler):
        self._sampler = sampler
        self._lock = threading.Lock()

    def sample(self, *args, **kwargs):
        with self._lock:
            return self._sampler.sample(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._sampler, name)


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

    def peek(self, k: int) -> List[Dict[str, Any]]:
        """The next k distinct problems draw() would return, WITHOUT advancing state
        (no cursor move, no reshuffle). Used to prefetch their baseline into base_cache
        while the current chunk trains -- a wrong guess (cross-epoch reshuffle) only
        misses the cache, never corrupts the draw."""
        out, seen, cur = [], set(), self._cursor
        recs = self._records
        while len(out) < k and cur < len(recs):  # stop at epoch edge; don't simulate reshuffle
            r = recs[cur]
            cur += 1
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
    """Base solves each problem greedily once (T=0, M=1), disk-cached by problem text.
    The base is frozen + greedy so a cache hit is exact. Returns fresh (miss) count."""
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


def diagnose_views(checker, problems: List[Dict[str, Any]], args: argparse.Namespace,
                   cache: DiskCache) -> None:
    """Rubric-check every view-A problem's greedy attempt in parallel (disk-cached by
    problem + attempt), stashing the formatted findings on ``r['_rubric_diag']``."""
    targets = [r for r in problems if r.get('_view') == 'A']
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
# Block E -- chunk draw, generation pipeline, record building
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
    """Draw one chunk, baselining every drawn problem. With ``--balance``, keep
    drawing+baselining until the target base fail:success mix is reachable (or the budget
    is hit), then select a balanced subset."""
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
    binary reward R in {0,1}: ``A = (R - mean) / (std + eps)``. std==0 groups get no
    gradient -- GRPO's variance selects informative problems (no explicit difficulty gate)."""
    eps = 1e-6
    for r in hard:
        for c in r['_cands']:
            c['advantage'], c['grpo_adv'], c['kept'] = 0.0, 0.0, False
        cs = [c for c in r['_cands'] if c.get('reward') is not None]
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


def apply_neighbor_rubric(base_sampler, chunk: List[Dict[str, Any]],
                          neighbor_map: Dict[str, Tuple[str, float]],
                          pool_answers: Dict[str, str], base_dp: int,
                          args: argparse.Namespace, checker,
                          base_cache: DiskCache, rubric_cache: DiskCache) -> None:
    """Cross-problem rubric (--xproblem-rubric): for every view-A problem Q, replace its own
    rubric with the rubric of its bag-of-words neighbour P (Q keeps being the solved/scored
    problem). P is baselined + diagnosed here (both disk-cached) as a stub carrying P's REAL
    answer, so P's baseline grades correctly and legitimately shares the baseline cache with
    P-as-training-problem. P's findings are copied onto Q with the neighbour text + similarity
    for audit. A P that can't be diagnosed -> Q degrades to view B. Since P's answer differs
    from Q's (guaranteed at pairing time), any answer P's rubric leaks is useless for Q."""
    targets = [r for r in chunk if r.get('_view') == 'A' and neighbor_map.get(r['problem'])]
    if not targets:
        return
    stubs, by_problem = [], {}
    for r in targets:
        p, _ = neighbor_map[r['problem']]
        if p not in by_problem:
            stub = {'problem': p, 'reference_answer': pool_answers.get(p, ''), '_view': 'A'}
            by_problem[p] = stub
            stubs.append(stub)
    baseline_rollout(base_sampler, stubs, base_dp, args, base_cache)
    diagnose_views(checker, stubs, args, rubric_cache)
    for r in targets:
        p, sim = neighbor_map[r['problem']]
        r['_rubric_diag'] = by_problem[p].get('_rubric_diag', '')
        r['_rubric_src'], r['_neighbor_sim'] = p, sim


def process_chunk(base_sampler, skill_sampler, chunk: List[Dict[str, Any]],
                  ci: int, base_dp: int, skill_dp: int, args: argparse.Namespace,
                  checker, rubric_cache: DiskCache, base_cache: DiskCache = None,
                  neighbor_map: Optional[Dict[str, Tuple[str, float]]] = None,
                  pool_answers: Optional[Dict[str, str]] = None
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """view assign -> rubric-check (view A) -> skill-gen -> leak-filter -> with-skill
    greedy pass -> GRPO advantages. ``chunk`` arrives already baselined by draw_chunk.
    With --xproblem-rubric, view A gets its NEIGHBOUR's rubric (see apply_neighbor_rubric)."""
    hard = chunk
    for r in hard:
        r['_view'], r['_rubric_diag'] = _assign_view(r['problem'], args), ''
    if args.xproblem_rubric and neighbor_map:
        apply_neighbor_rubric(base_sampler, hard, neighbor_map, pool_answers or {}, base_dp,
                              args, checker, base_cache, rubric_cache)
    else:
        diagnose_views(checker, hard, args, rubric_cache)

    # skill-gen (thinking ON), per-view prompt; re-sample problems with no clean candidate.
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
                        'leak_source': '', 'with_pass': None, 'reward': None, 'rolls': [],
                        'skillgen_stop': getattr(s, 'stop_reason', None),
                        'skillgen_tokens': len(getattr(s, 'tokens', None) or [])}
                r['_cands'].append(cand)
                if block:
                    flat.append((r, cand))
                    got = True
            if not got:
                still.append(r)
        pending = still

    # leak audit: deterministic verbatim-answer check only (no LLM). This is observability
    # only: it records leak metrics for swanlab/jsonl, but does not block scoring, reward,
    # advantage assignment, or training sample selection.
    for r, c in flat:
        leaked = _answer_leaked(c['skills'], r['reference_answer'])
        c['leaked'] = leaked
        c['leak_reason'] = 'answer_verbatim' if leaked else ''
        c['leak_source'] = 'deterministic'

    # with-skill greedy pass (T=0, M=1); reward = correct, absolute (group mean is baseline).
    scored_inputs = flat
    if scored_inputs:
        ws_out = _run_samples(base_sampler,
                              [build_skill_solve_prompt(r['problem'], c['skills']) for r, c in scored_inputs],
                              1, args.max_tokens, base_dp, temperature=0.0)
        for (r, c), seqs in zip(scored_inputs, ws_out):
            c['rolls'] = [_parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()]
            c['with_pass'] = 1.0 if c['rolls'][0]['correct'] else 0.0
            c['reward'] = c['with_pass']
    if args.format_in_reward:  # unparseable candidates score 0 and still join the group
        for r in hard:
            for c in r['_cands']:
                if c['reward'] is None:
                    c['reward'] = 0.0

    _assign_advantages(hard, args)
    return ([_full_record(r, ci) for r in chunk], _chunk_summary(chunk, ci, args),
            _group_records(chunk, args))


def _roll(x: Dict[str, Any]) -> Dict[str, Any]:
    return {k: x[k] for k in ('pred', 'correct', 'terminated', 'passed',
                              'stop_reason', 'gen_tokens', 'text')}


def _is_trainable(c: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Reaches the GRPO update iff advantage is non-zero. Leak flags are audit-only."""
    adv_nz = abs(c.get('advantage') or 0.0) > 1e-9
    if args.format_in_reward:
        return adv_nz
    return c.get('with_pass') is not None and adv_nz


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
        # xproblem audit: which neighbour's rubric this problem borrowed, and how similar.
        'rubric_src': r.get('_rubric_src', ''), 'neighbor_sim': r.get('_neighbor_sim'),
        'baseline_rolls': [_roll(x) for x in r['_baseline_rolls']],
        'candidates': [{
            'skills': c['skills'], 'response': c['response'], 'parseable': c['parseable'],
            'skillgen_stop': c.get('skillgen_stop'), 'skillgen_tokens': c.get('skillgen_tokens'),
            'leaked': c['leaked'], 'leak_reason': c['leak_reason'], 'leak_source': c['leak_source'],
            'with_pass': c['with_pass'], 'reward': c.get('reward'), 'advantage': c.get('advantage'),
            'grpo_adv': c.get('grpo_adv'), 'kept': c.get('kept'),
            'rolls': [_roll(x) for x in c['rolls']],
        } for c in r['_cands']],
    }


def _view_stats(problems: List[Dict[str, Any]], view: str) -> Dict[str, Any]:
    pv = [r for r in problems if r.get('_view') == view]
    cands = [c for r in pv for c in r['_cands'] if c['parseable']]
    clean = [c for c in cands if c['leaked'] is False]
    adopted = sum(1 for r in pv
                  if any(abs(c.get('advantage') or 0.0) > 1e-9 for c in r['_cands']))
    return {'n': len(pv), 'n_candidates_parseable': len(cands), 'n_clean': len(clean),
            'n_adopted_problems': adopted, 'adoption_rate': (adopted / len(pv)) if pv else 0.0}


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _signal_stats(problems: List[Dict[str, Any]]) -> Dict[str, float]:
    """The heart of 'is there a learning signal': per problem, the scored candidates form a
    GRPO group. A group with zero reward variance (all skills solve, or none do -- the
    hard-problem dead zone) gives NO gradient. Tracks that fraction plus reward level and
    within-group variance so a collapse (all-0 or all-1) is visible immediately."""
    group_vars, all_rewards, zero_grad, groups = [], [], 0, 0
    for r in problems:
        rewards = [c['reward'] for c in r['_cands'] if c.get('reward') is not None]
        if len(rewards) < 2:
            continue
        groups += 1
        all_rewards.extend(rewards)
        v = _std(rewards)
        group_vars.append(v)
        if v < 1e-9:  # every skill got the same reward -> GRPO skips this problem
            zero_grad += 1
    return {'n_groups': groups, 'zero_grad_frac': (zero_grad / groups) if groups else 0.0,
            'reward_mean': _mean(all_rewards), 'reward_std': _std(all_rewards),
            'group_reward_std_mean': _mean(group_vars)}


def _chunk_summary(chunk: List[Dict[str, Any]], ci: int, args: argparse.Namespace) -> Dict[str, Any]:
    all_cands = [c for r in chunk for c in r['_cands']]
    cands = [c for c in all_cands if c['parseable']]
    scored = [c for c in cands if c['with_pass'] is not None]
    clean = [c for c in cands if c['leaked'] is False]
    ws_rolls = [x for c in scored for x in c['rolls']]
    base_acc = (sum(r['_baseline_pass'] for r in chunk) / len(chunk)) if chunk else 0.0
    ws_acc = _mean([c['with_pass'] for c in scored])
    # base failure taxonomy (you asked whether skills fail because the base loops out of length)
    classes = [_baseline_class(r) for r in chunk]
    n_fail = sum(1 for c in classes if c != 'success')
    skill_tokens = [c.get('skillgen_tokens') or 0 for c in cands]  # skill-gen response length
    trunc = sum(1 for r in chunk for c in r['_cands']
                for x in c['rolls'] if x['stop_reason'] == 'length')
    return {
        'record_type': 'summary', 'chunk': ci, 'n': len(chunk),
        'n_failed_first_try': sum(1 for r in chunk if r['_failed']),
        'n_generated': len(all_cands), 'n_candidates_parseable': len(cands),
        'n_unparseable': len(all_cands) - len(cands),
        'parse_rate': (len(cands) / len(all_cands)) if all_cands else 0.0,
        'n_leaked': sum(1 for c in cands if c['leaked']), 'n_clean': len(clean),
        'leak_rate': (sum(1 for c in cands if c['leaked']) / len(cands)) if cands else 0.0,
        'n_reward_pos': sum(1 for c in scored if c['reward']),
        'n_train_samples': sum(1 for c in all_cands if _is_trainable(c, args)),
        'signal': _signal_stats(chunk),
        'fail_loop_frac': (sum(1 for c in classes if c == 'fail_loop') / n_fail) if n_fail else 0.0,
        'fail_wrong_frac': (sum(1 for c in classes if c == 'fail_wrong') / n_fail) if n_fail else 0.0,
        'skill_tokens_mean': _mean(skill_tokens),
        'withskill_trunc_frac': (trunc / len(ws_rolls)) if ws_rolls else 0.0,
        'avg_baseline_pass': base_acc, 'avg_withskill_pass': ws_acc,
        'avg_lift': ws_acc - base_acc,
        'termination_rate_withskill': _mean([1.0 if x['terminated'] else 0.0 for x in ws_rolls]),
        'view_A': _view_stats(chunk, 'A'), 'view_B': _view_stats(chunk, 'B'),
        **_xproblem_stats(chunk, args),
    }


def _xproblem_stats(chunk: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """Cross-problem pairing health: of the view-A problems, how many actually got a
    neighbour's rubric (pair_rate) and how similar those neighbours were. Empty when off."""
    if not args.xproblem_rubric:
        return {}
    view_a = [r for r in chunk if r.get('_view') == 'A']
    paired = [r for r in view_a if r.get('_rubric_src')]
    return {'xproblem': {
        'n_view_a': len(view_a), 'n_paired': len(paired),
        'pair_rate': (len(paired) / len(view_a)) if view_a else 0.0,
        'neighbor_sim_mean': _mean([r.get('_neighbor_sim', 0.0) for r in paired])}}


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
                    'rubric_src': r.get('_rubric_src', ''),
                    'response': c['response'], 'skills': c['skills'],
                    'skillgen_stop': c.get('skillgen_stop'),
                    'advantage': c['advantage'], 'grpo_adv': c['grpo_adv'], 'kept': c['kept'],
                    'reward': c['reward'], 'with_pass': c['with_pass']})
    return out


# ===========================================================================
# Block G -- online GRPO training
# ===========================================================================
def _is_num(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _train_trajectory(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Training sample = the exact skill-gen prompt (rebuilt by ``_skillgen_messages`` so
    train/inference match) + the generated (think + skills) response. ``key_rounds``
    selects the final assistant turn; Template masks the prompt and trains the whole
    response (the key-round prefix already excludes the prompt-provided <think>)."""
    msgs = _skillgen_messages(
        rec['problem'], rec.get('view', 'A'), rec.get('diagnosis', ''), rec.get('rubric_src', ''))
    return {'messages': msgs + [{'role': 'assistant', 'content': rec['response']}],
            'user_data': {'key_rounds': [len(msgs)]}}


def _train_chunk(skill_model, ref_model, ckpt: CheckpointEngineManager, samples: List[Dict[str, Any]],
                 args: argparse.Namespace) -> Dict[str, Any]:
    """On-policy GRPO update over one chunk, then sync weights. Micro-batches of
    ``sft_batch_size`` accumulate gradients; ONE optimizer step is taken per PPO
    mini-batch of ``ppo_mini_batch_size`` samples (0 -> a single step over the whole
    chunk, the original behaviour). A frozen reference model provides ref_logps for the
    SEAM-style KL penalty.

    Multi-step correctness: with more than one step over the SAME rollout, later
    mini-batches see an already-updated policy, so we FREEZE the sampling-policy
    ``old_logps`` (recomputed once, before any step) and let GRPOLoss form the PPO ratio
    against them. A single step keeps ``old_logps=None`` (ratio==1, pure on-policy).
    The batch is padded to a multiple of ``sft_batch_size`` with advantage-0 copies that
    contribute no policy gradient."""
    trajs = [_train_trajectory(rec) for rec in samples]
    advs = [float(rec['advantage']) for rec in samples]
    rem = (-len(trajs)) % args.sft_batch_size
    if rem:
        trajs += [trajs[-1]] * rem
        advs += [0.0] * rem

    n, sft = len(trajs), args.sft_batch_size
    mini = args.ppo_mini_batch_size if args.ppo_mini_batch_size > 0 else n
    mini = max(sft, (mini // sft) * sft)  # align to a whole number of micro-batches
    multi_step = mini < n

    # Freeze ref_logps (frozen model) and -- only when we take multiple steps -- the
    # sampling-policy old_logps, BOTH before any optimizer step touches the weights. With
    # a single step old_logps stays None so GRPOLoss uses ratio==1 (pure on-policy).
    micro_ref, micro_old = [], []
    for i in range(0, n, sft):
        mb = trajs[i:i + sft]
        micro_ref.append(ref_model.forward_only(inputs=mb).get('logps'))
        micro_old.append(skill_model.forward_only(inputs=mb).get('logps') if multi_step else None)

    micro, n_steps = 0, 0
    for ms in range(0, n, mini):
        for i in range(ms, min(ms + mini, n), sft):
            k = i // sft
            skill_model.forward_backward(inputs=trajs[i:i + sft],
                                         advantages=advs[i:i + sft],
                                         old_logps=micro_old[k],
                                         ref_logps=micro_ref[k])
            micro += 1
        skill_model.clip_grad_and_step()
        n_steps += 1
    ckpt.sync_weights(merge_and_sync=True)
    metric = skill_model.calculate_metric(is_training=True)
    return {'n_samples': len(samples), 'n_steps': n_steps, 'n_micro_batches': micro,
            'metric': {k: (float(v) if _is_num(v) else v) for k, v in (metric or {}).items()}}


# ===========================================================================
# Block H -- fixed-holdout eval + metric formatting
# ===========================================================================
def run_greedy_eval(base_sampler, skill_sampler, eval_records: List[Dict[str, Any]],
                    ci: int, rounds: int, base_dp: int, skill_dp: int, args: argparse.Namespace,
                    base_cache: DiskCache
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
    """SEAM ``val-core/math/acc/mean@1`` on the fixed holdout: ONE greedy skill (T=0) per
    problem into ONE greedy base solve (T=0). Eval ALWAYS uses view B (query-only, the
    deployment form: no rubric, since rubric needs an online teacher unavailable at deploy);
    no leak filter (acc scores correctness alone). Baseline reuses the disk cache."""
    baseline_rollout(base_sampler, eval_records, base_dp, args, base_cache)
    for r in eval_records:
        r['_view'], r['_rubric_diag'] = 'B', ''
    sg_out = _run_samples(skill_sampler, [_view_prompt(r) for r in eval_records],
                          1, args.skill_max_tokens, skill_dp, temperature=0.0)
    skills = [(_extract_skills_block(_clean_text(getattr(seqs[0], 'decoded', '') or '')) or '',
               _clean_text(getattr(seqs[0], 'decoded', '') or '')) if seqs else ('', '')
              for seqs in sg_out]
    ws_out = _run_samples(base_sampler,
                          [build_skill_solve_prompt(r['problem'], sk) for r, (sk, _) in zip(eval_records, skills)],
                          1, args.max_tokens, base_dp, temperature=0.0)
    recs = []
    for r, (sk, sresp), seqs in zip(eval_records, skills, ws_out):
        roll = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
        recs.append({
            'record_type': 'eval_problem', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
            'problem': r['problem'], 'reference_answer': r['reference_answer'],
            'view': r['_view'], 'rubric_diag': r.get('_rubric_diag', ''),
            'baseline_pass': r['_baseline_pass'], 'skill': sk, 'skill_parseable': bool(sk),
            'skill_response': sresp, 'withskill_pred': roll['pred'],
            'withskill_correct': roll['correct'], 'withskill_terminated': roll['terminated'],
            'withskill_stop_reason': roll['stop_reason'], 'withskill_text': roll['text'],
        })
    acc = lambda rs: sum(1 for x in rs if x['withskill_correct']) / len(rs) if rs else 0.0
    ws = acc(recs)  # all view B (deployment form)
    base = sum(x['baseline_pass'] for x in recs) / len(recs) if recs else 0.0
    fmt = (sum(1 for x in recs if x['skill_parseable']) / len(recs)) if recs else 0.0
    term = (sum(1 for x in recs if x['withskill_terminated']) / len(recs)) if recs else 0.0
    summary = {'record_type': 'eval_summary', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
               'n': len(recs), 'view': 'B', 'acc_mean1': ws,
               'baseline_acc_mean1': base, 'lift_mean1': ws - base,
               'format_mean1': fmt, 'term_mean1': term}
    metrics = {'core/math/acc/mean@1': ws, 'core/math/baseline_acc/mean@1': base,
               'core/math/lift/mean@1': ws - base, 'core/math/format/mean@1': fmt,
               'core/math/term/mean@1': term}
    return recs, summary, metrics


def _trend_line(hist: List[Dict[str, float]], window: int, rounds_done: int) -> Optional[str]:
    """Contrast the FIRST vs the most recent ``window`` chunks -- if RFT works, adoption
    and lift on recent (fresh) chunks exceed the early baseline."""
    if len(hist) < 2 * window:
        return None
    base, rec = hist[:window], hist[-window:]
    m = lambda xs, k: sum(h[k] for h in xs) / len(xs)
    return (f'[trend] first {window} vs last {window} | '
            f'adopt A {m(base,"aA"):.2f}->{m(rec,"aA"):.2f} B {m(base,"aB"):.2f}->{m(rec,"aB"):.2f} | '
            f'lift {m(base,"lift"):+.3f}->{m(rec,"lift"):+.3f} | '
            f'0grad {m(base,"zero_grad"):.2f}->{m(rec,"zero_grad"):.2f} | '
            f'pos/chunk {m(base,"pos"):.1f}->{m(rec,"pos"):.1f} | rounds={rounds_done}')


def _swan_metrics(summary: Dict[str, Any], log: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Flat swanlab dict. ``signal/*`` is the primary health group (is GRPO getting a
    gradient at all); ``acc/*`` is the effect; the rest diagnose why. acc/adopt/skill are
    only logged when the chunk produced a scored group, so idle chunks don't dip charts."""
    sig = summary['signal']
    d: Dict[str, float] = {
        # --- signal: the FIRST thing to watch (no variance -> no learning) ---
        'signal/zero_grad_frac': sig['zero_grad_frac'], 'signal/n_groups': sig['n_groups'],
        'signal/reward_mean': sig['reward_mean'], 'signal/reward_std': sig['reward_std'],
        'signal/group_reward_std_mean': sig['group_reward_std_mean'],
        'signal/n_train_samples': summary['n_train_samples'],
        'signal/n_reward_pos': summary['n_reward_pos'],
        # --- skill format / leak health ---
        'skill/parse_rate': summary['parse_rate'], 'skill/tokens_mean': summary['skill_tokens_mean'],
        'leak/rate': summary['leak_rate'], 'leak/n': summary['n_leaked'],
        # --- base failure taxonomy (loop-out-of-length vs plain wrong) ---
        'fail/loop_frac': summary['fail_loop_frac'], 'fail/wrong_frac': summary['fail_wrong_frac'],
        'fail/frac_first_try': (summary['n_failed_first_try'] / summary['n']) if summary['n'] else 0.0,
    }
    bal = summary.get('balance') or {}
    if bal.get('enabled'):
        d.update({'balance/n_drawn': bal['n_drawn'], 'balance/n_baseline_fresh': bal['n_baseline_fresh'],
                  'balance/selected_success_frac': bal['selected_success_frac']})
    xp = summary.get('xproblem') or {}
    if xp:
        d.update({'xproblem/pair_rate': xp['pair_rate'], 'xproblem/neighbor_sim_mean': xp['neighbor_sim_mean']})
    if sig['n_groups'] > 0:
        d.update({'acc/baseline_pass': summary['avg_baseline_pass'],
                  'acc/withskill_pass': summary['avg_withskill_pass'], 'acc/lift': summary['avg_lift'],
                  'adopt/A': summary['view_A']['adoption_rate'],
                  'adopt/B': summary['view_B']['adoption_rate'],
                  'term/withskill': summary['termination_rate_withskill'],
                  'term/withskill_trunc_frac': summary['withskill_trunc_frac']})
    if log:
        d['train/n_steps'] = log['n_steps']
        d['train/n_micro_batches'] = log['n_micro_batches']
        for k, v in (log.get('metric') or {}).items():
            if not _is_num(v):
                continue
            if k.startswith('learning rate'):
                if 'group 1' in k:
                    d['train/lr'] = float(v)
            else:
                d[f'train/{k.replace(" ", "_")}'] = float(v)
    return d


def _view_a_rubric_leak_metrics(chunk: List[Dict[str, Any]],
                                pool_answers: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Swanlab-only audit for answer leakage in view-A rubric text. This never changes
    rewards, advantages, filtering, or training records."""
    view_a = [r for r in chunk if r.get('_view') == 'A']
    with_diag = [r for r in view_a if r.get('_rubric_diag')]
    target_leaks = sum(1 for r in with_diag
                       if _answer_leaked(r.get('_rubric_diag', ''), r.get('reference_answer', '')))
    source_leaks = 0
    pool_answers = pool_answers or {}
    for r in with_diag:
        src = r.get('_rubric_src')
        src_ref = pool_answers.get(src, '') if src else r.get('reference_answer', '')
        if _answer_leaked(r.get('_rubric_diag', ''), src_ref):
            source_leaks += 1
    n = len(with_diag)
    return {
        'rubric_leak/n_view_a': float(len(view_a)),
        'rubric_leak/n_checked': float(n),
        'rubric_leak/target_answer_n': float(target_leaks),
        'rubric_leak/target_answer_rate': (target_leaks / n) if n else 0.0,
        'rubric_leak/source_answer_n': float(source_leaks),
        'rubric_leak/source_answer_rate': (source_leaks / n) if n else 0.0,
    }


# ===========================================================================
# Block F -- components, args, main
# ===========================================================================
def init_components(args: argparse.Namespace):
    """Default 8-GPU layout: rank 0 trains the actor, rank 1 hosts a frozen ref model,
    2-3 skill_sampler (synced), 4-7 base_sampler (frozen). Returns
    (skill_model, ref_model, skill_sampler, base_sampler, ckpt, skill_dp, base_dp)."""
    r0, r1 = TRAIN_GPUS, TRAIN_GPUS + REF_GPUS
    r2, r3 = r1 + SKILL_SAMPLER_GPUS, NUM_GPUS
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, lazy_collect=False, groups=[
        DeviceGroup(name='train', ranks=list(range(0, r0)), device_type='GPU'),
        DeviceGroup(name='ref', ranks=list(range(r0, r1)), device_type='GPU'),
        DeviceGroup(name='skill_sampler', ranks=list(range(r1, r2)), device_type='GPU'),
        DeviceGroup(name='base_sampler', ranks=list(range(r2, r3)), device_type='GPU')])

    train_mesh = DeviceMesh.from_sizes(world_size=TRAIN_GPUS, dp_size=TRAIN_DP, fsdp_size=TRAIN_FSDP)
    skill_model = TransformersModel(model_id=MODEL_ID, device_mesh=train_mesh, remote_group='train',
                                    ddp_config={'find_unused_parameters': False})
    skill_model.apply_patch(NoSplitModulesPatch({'Qwen3DecoderLayer'}))
    skill_model.set_template(Template, model_id=MODEL_ID, enable_thinking=True,
                             max_length=args.max_model_len, truncation_strategy='delete')
    skill_model.set_processor(InputProcessor, padding_free=False)
    skill_model.set_loss('GRPOLoss', epsilon=args.grpo_epsilon, beta=args.kl_beta)
    skill_model.set_optimizer('AdamW', lr=args.lr)
    skill_model.set_lr_scheduler('CosineWarmupScheduler', num_warmup_steps=10,
                                 num_training_steps=args.max_train_rounds)

    ref_mesh = DeviceMesh.from_sizes(world_size=REF_GPUS, dp_size=REF_DP, fsdp_size=REF_FSDP)
    ref_model = TransformersModel(model_id=MODEL_ID, device_mesh=ref_mesh, remote_group='ref',
                                  ddp_config={'find_unused_parameters': False})
    ref_model.apply_patch(NoSplitModulesPatch({'Qwen3DecoderLayer'}))
    ref_model.set_template(Template, model_id=MODEL_ID, enable_thinking=True,
                           max_length=args.max_model_len, truncation_strategy='delete')
    ref_model.set_processor(InputProcessor, padding_free=False)
    ref_model.set_loss('GRPOLoss', epsilon=args.grpo_epsilon)

    def _sampler(group, world):
        s = vLLMSampler(model_id=MODEL_ID,
                        engine_args={'gpu_memory_utilization': GPU_MEM,
                                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
                        device_mesh=DeviceMesh.from_sizes(world_size=world, dp_size=world),
                        remote_group=group)
        s.set_template(Template, model_id=MODEL_ID, enable_thinking=True, max_length=args.max_model_len)
        return s

    skill_sampler = _sampler('skill_sampler', SKILL_SAMPLER_GPUS)
    # base_sampler is shared by the main thread and the baseline-prefetch thread -> serialise.
    base_sampler = _LockedSampler(_sampler('base_sampler', BASE_SAMPLER_GPUS))
    ckpt = CheckpointEngineManager(model=skill_model, sampler=skill_sampler)
    return skill_model, ref_model, skill_sampler, base_sampler, ckpt, SKILL_SAMPLER_GPUS, BASE_SAMPLER_GPUS


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    p.add_argument('--n', type=int, default=2000, help='Problems loaded into the draw pool.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval-size', type=int, default=128, help='Fixed holdout size (0 disables).')
    p.add_argument('--eval-every', type=int, default=5, help='Run holdout eval every N chunks.')
    p.add_argument('--chunk-size', type=int, default=16)
    p.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--balance-success-frac', type=float, default=0.4,
                   help='Target fraction of the chunk the base solves (rest are base-fail).')
    p.add_argument('--balance-loop-frac', type=float, default=0.5)
    p.add_argument('--balance-max-draws-mult', type=int, default=8)
    p.add_argument('--n-skills', type=int, default=8)
    p.add_argument('--view-b-frac', type=float, default=0.5)
    p.add_argument('--xproblem-rubric', action=argparse.BooleanOptionalAction, default=True,
                   help='View A uses its bag-of-words NEIGHBOUR problem\'s rubric (transfer '
                        'test; kills answer leakage since the neighbour\'s answer differs). '
                        'On (default); --no-xproblem-rubric = each problem uses its own rubric '
                        '(may leak).')
    p.add_argument('--skill-retries', type=int, default=2)
    p.add_argument('--skill-gen-temperature', type=float, default=1.0)
    p.add_argument('--skill-gen-top-p', type=float, default=1.0)
    p.add_argument('--skill-gen-top-k', type=int, default=-1)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--skill-max-tokens', type=int, default=8192)
    p.add_argument('--rubric-workers', type=int, default=16)
    p.add_argument('--sft-batch-size', type=int, default=8,
                   help='Driver micro-batch (gradient-accumulation unit); multiple of train dp.')
    p.add_argument('--ppo-mini-batch-size', type=int, default=0,
                   help='Samples per optimizer step (SEAM-style PPO mini-batch). 0 = one step '
                        'over the whole chunk (on-policy, ratio==1). When >0 and smaller than '
                        'the trainable count, multiple steps are taken over the same rollout and '
                        'old_logps are frozen so the PPO ratio/clip stays valid. Rounded down to '
                        'a multiple of --sft-batch-size.')
    p.add_argument('--grpo-epsilon', type=float, default=0.2)
    p.add_argument('--kl-beta', type=float, default=0.001,
                   help='SEAM-style reference KL coefficient for GRPOLoss.')
    p.add_argument('--format-in-reward', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--lr', type=float, default=6e-6)
    p.add_argument('--max-train-rounds', type=int, default=1500)
    p.add_argument('--save-rounds', type=int, default=200)
    p.add_argument('--trend-every', type=int, default=10)
    p.add_argument('--output-dir', default='./output/reflexion_skill_rft')
    p.add_argument('--cache-dir', default='', help='Baseline/rubric cache dir (default <output-dir>/cache).')
    p.add_argument('--no-cache', action='store_true', help='Disable disk cache read/write.')
    p.add_argument('--prefetch-baseline', action=argparse.BooleanOptionalAction, default=True,
                   help='Prefetch next chunk base baseline on a background thread (overlaps '
                        'with skill-gen; base_sampler is frozen so it never blocks the trainer).')
    p.add_argument('--swanlab-project', default='twinkle')
    p.add_argument('--swanlab-exp', default='')
    args = p.parse_args()
    if args.sft_batch_size % TRAIN_DP != 0:
        raise ValueError(f'--sft-batch-size ({args.sft_batch_size}) must be a multiple of train dp ({TRAIN_DP})')
    if args.chunk_size < 1:
        raise ValueError('--chunk-size must be >= 1')
    args.balance_success_frac = max(0.0, min(1.0, args.balance_success_frac))
    return args


def _write(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = _build_args()
    records, eval_records, neighbor_map, pool_answers, data_stats = _load_records(args)
    if len(records) < args.chunk_size:
        raise ValueError(f'--chunk-size ({args.chunk_size}) exceeds loaded ({len(records)}); raise --n')

    os.makedirs(args.output_dir, exist_ok=True)
    gen_path = os.path.join(args.output_dir, 'gen_records.jsonl')
    eval_path = os.path.join(args.output_dir, 'eval_records.jsonl')
    data_path = os.path.join(args.output_dir, 'skill_dataset.jsonl')
    train_log_path = os.path.join(args.output_dir, 'train_log.jsonl')

    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('OPENAI_API_KEY')):
        sys.stderr.write('[rft] WARNING: no LLM backup env; view-A rubric check disabled '
                         '(leak filter is deterministic, unaffected)\n')

    use_swan = swanlab is not None and os.environ.get('SWANLAB_MODE') != 'disabled'
    if use_swan:
        swanlab.init(project=args.swanlab_project, experiment_name=(args.swanlab_exp or None),
                     config={'model': MODEL_ID, 'dataset': args.dataset, 'n': len(records),
                             'eval_n': len(eval_records), 'n_skills': args.n_skills,
                             'view_b_frac': args.view_b_frac, 'balance': args.balance,
                             'balance_success_frac': args.balance_success_frac,
                             'skill_gen_temp': args.skill_gen_temperature,
                             'grpo_epsilon': args.grpo_epsilon, 'kl_beta': args.kl_beta,
                             'lr': args.lr})

    skill_model, ref_model, skill_sampler, base_sampler, ckpt, skill_dp, base_dp = init_components(args)
    checker = build_rubric_checker()
    if checker is None:
        sys.stderr.write('[rft] no LLM backup env -> view-A rubric process-check DISABLED\n')

    cache_dir = args.cache_dir or os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    use_cache = not args.no_cache
    base_cache = DiskCache(os.path.join(cache_dir, 'baseline.jsonl'), use_cache)
    eval_base_cache = DiskCache(os.path.join(cache_dir, 'eval_baseline.jsonl'), use_cache)
    rubric_cache = DiskCache(os.path.join(cache_dir, 'rubric.jsonl'), use_cache)
    if args.xproblem_rubric:
        sys.stderr.write(f'[rft] xproblem-rubric ON: {len(neighbor_map)} bag-of-words pairs\n')

    cfg = {'record_type': 'config', 'model': MODEL_ID, 'dataset': args.dataset,
           'n': len(records), 'eval_n': len(eval_records), 'seed': args.seed,
           'numeric_only': args.numeric_only, 'raw_loaded': data_stats['raw_loaded'],
           'numeric_dropped': data_stats['numeric_dropped'], 'eval_every': args.eval_every,
           'n_skills': args.n_skills, 'view_b_frac': args.view_b_frac,
           'skill_retries': args.skill_retries, 'balance': args.balance,
           'balance_success_frac': args.balance_success_frac,
           'skill_gen_temp': args.skill_gen_temperature, 'reward': 'greedy_binary(correct)',
           'advantage': 'group_relative', 'format_in_reward': args.format_in_reward, 'cache': use_cache,
           'rubric_check': 'fixed_math_5crit(viewA)' if checker else 'disabled',
           'xproblem_rubric': args.xproblem_rubric,
           'grpo_epsilon': args.grpo_epsilon, 'kl_beta': args.kl_beta, 'lr': args.lr,
           'train_gpus': TRAIN_GPUS, 'ref_gpus': REF_GPUS, 'ref_fsdp': REF_FSDP,
           'train_fsdp': TRAIN_FSDP, 'train_dp': TRAIN_DP,
           'skill_sampler_gpus': SKILL_SAMPLER_GPUS, 'base_sampler_gpus': BASE_SAMPLER_GPUS,
           'max_train_rounds': args.max_train_rounds, 'started': int(time.time())}
    sys.stderr.write(f'[rft] raw={data_stats["raw_loaded"]} numeric_drop={data_stats["numeric_dropped"]} '
                     f'train={len(records)} eval={len(eval_records)} {args.dataset}; '
                     f'train_gpus={TRAIN_GPUS} ref_gpus={REF_GPUS} train_fsdp={TRAIN_FSDP} '
                     f'train_dp={TRAIN_DP} skill_dp={skill_dp} base_dp={base_dp}\n')

    hist: List[Dict[str, float]] = []
    rounds = 0
    pool = ProblemPool(records, args.seed)
    with open(gen_path, 'w', encoding='utf-8') as gen_f, \
            open(eval_path, 'w', encoding='utf-8') as eval_f, \
            open(data_path, 'w', encoding='utf-8') as data_f, \
            open(train_log_path, 'w', encoding='utf-8') as tlog:
        for f in (gen_f, eval_f, data_f, tlog):
            _write(f, cfg)
        gstep = 0
        # base baseline (frozen sampler, disk-cached, no weight-sync) is prefetched on a
        # background thread while the current chunk generates: the skill-gen phase uses
        # skill_sampler, so prefetching the NEXT chunk's baseline on base_sampler overlaps
        # it and fills base_cache so the next draw_chunk hits it. base_sampler is wrapped in
        # _LockedSampler, so the prefetch and this chunk's with-skill scoring never issue a
        # base .sample() concurrently. It never touches the trainer or on-policy generation.
        prefetch_pool = ThreadPoolExecutor(max_workers=1) if args.prefetch_baseline else None
        pending: Optional[Any] = None

        def _prefetch(peeked: List[Dict[str, Any]]) -> None:
            if peeked:
                baseline_rollout(base_sampler, peeked, base_dp, args, base_cache)

        # Baseline (round 0) eval BEFORE any training: measures the untrained skill model on
        # the fixed holdout so every later eval has a step-0 reference point on the same axis.
        if eval_records:
            eval_recs, eval_summary, eval_metrics = run_greedy_eval(
                base_sampler, skill_sampler, eval_records, -1, rounds, base_dp, skill_dp,
                args, eval_base_cache)
            for rec in eval_recs:
                _write(eval_f, rec)
            _write(eval_f, eval_summary)
            eval_f.flush()
            if use_swan:
                swanlab.log({f'eval/{k}': v for k, v in eval_metrics.items()}, step=0)
            sys.stderr.write(
                f'[eval] g-1 (init): n={eval_summary["n"]} viewB mean@1 '
                f'acc={eval_summary["baseline_acc_mean1"]:.3f}->{eval_summary["acc_mean1"]:.3f} '
                f'lift={eval_summary["lift_mean1"]:+.3f} '
                f'fmt={eval_summary["format_mean1"]:.2f} rounds={rounds}\n')

        # Each chunk is drawn fresh + RE-GENERATED with the current policy (on-policy);
        # with --balance, draw_chunk keeps drawing until the base fail:success mix hits target.
        while rounds < args.max_train_rounds:
            if pending is not None:
                pending.result()  # finish last round's prefetch before drawing (cache-warm)
                pending = None
            chunk, balance = draw_chunk(pool, base_sampler, base_dp, args, base_cache)
            if prefetch_pool is not None:
                peeked = pool.peek(int(args.chunk_size * (args.balance_max_draws_mult if args.balance else 1)))
                pending = prefetch_pool.submit(_prefetch, peeked)
            full, summary, groups = process_chunk(
                base_sampler, skill_sampler, chunk, gstep, base_dp, skill_dp,
                args, checker, rubric_cache, base_cache, neighbor_map, pool_answers)
            summary['balance'] = balance

            log = None
            if groups:
                log = _train_chunk(skill_model, ref_model, ckpt, groups, args)
                rounds += 1
                log.update({'record_type': 'train_round', 'round': rounds, 'chunk': gstep,
                            'epoch': pool.epoch, 'ts': int(time.time())})
                _write(tlog, log)
                tlog.flush()
                if rounds % args.save_rounds == 0:
                    skill_model.save(f'skill-rft-{rounds}', output_dir=args.output_dir)

            summary['rounds_done'], summary['epoch'] = rounds, pool.epoch
            for rec in full:
                _write(gen_f, rec)
            _write(gen_f, summary)
            gen_f.flush()
            for v in groups:
                _write(data_f, v)
            data_f.flush()

            sa, sb, sig = summary['view_A'], summary['view_B'], summary['signal']
            hist.append({'aA': sa['adoption_rate'], 'aB': sb['adoption_rate'],
                         'lift': summary['avg_lift'], 'pos': summary['n_reward_pos'],
                         'zero_grad': sig['zero_grad_frac']})
            bal_str = (f'bal {balance["selected_fail"]}f/{balance["selected_success"]}s '
                       f'(drew {balance["n_drawn"]}/fresh {balance["n_baseline_fresh"]}'
                       + ('!' if balance.get('budget_hit') else '') + ') ') if balance.get('enabled') else ''
            xp = summary.get('xproblem')
            xp_str = f'pair={xp["pair_rate"]:.2f}@{xp["neighbor_sim_mean"]:.2f} ' if xp else ''
            sys.stderr.write(
                f'[gen] e{pool.epoch} g{gstep}: {bal_str}n={summary["n"]} '
                f'clean={summary["n_clean"]} leak={summary["leak_rate"]:.2f} train={summary["n_train_samples"]} '
                f'0grad={sig["zero_grad_frac"]:.2f} R={sig["reward_mean"]:.2f}±{sig["reward_std"]:.2f} '
                f'acc={summary["avg_baseline_pass"]:.2f}->{summary["avg_withskill_pass"]:.2f} '
                f'lift={summary["avg_lift"]:+.3f} {xp_str}'
                f'A[{sa["n"]} {sa["adoption_rate"]:.2f}] B[{sb["n"]} {sb["adoption_rate"]:.2f}] '
                f'rounds={rounds}\n')
            if use_swan:
                swan_metrics = _swan_metrics(summary, log)
                swan_metrics.update(_view_a_rubric_leak_metrics(chunk, pool_answers))
                swanlab.log(swan_metrics, step=gstep)

            if eval_records and (gstep + 1) % args.eval_every == 0:
                eval_recs, eval_summary, eval_metrics = run_greedy_eval(
                    base_sampler, skill_sampler, eval_records, gstep, rounds, base_dp, skill_dp,
                    args, eval_base_cache)
                for rec in eval_recs:
                    _write(eval_f, rec)
                _write(eval_f, eval_summary)
                eval_f.flush()
                if use_swan:
                    swanlab.log({f'eval/{k}': v for k, v in eval_metrics.items()}, step=gstep)
                sys.stderr.write(
                    f'[eval] g{gstep}: n={eval_summary["n"]} viewB mean@1 '
                    f'acc={eval_summary["baseline_acc_mean1"]:.3f}->{eval_summary["acc_mean1"]:.3f} '
                    f'lift={eval_summary["lift_mean1"]:+.3f} '
                    f'fmt={eval_summary["format_mean1"]:.2f} rounds={rounds}\n')

            if (gstep + 1) % args.trend_every == 0:
                tl = _trend_line(hist, args.trend_every, rounds)
                if tl:
                    sys.stderr.write(tl + '\n')
            gstep += 1

    if prefetch_pool is not None:
        if pending is not None:
            pending.result()
        prefetch_pool.shutdown(wait=True)
    base_cache.close()
    eval_base_cache.close()
    rubric_cache.close()
    skill_model.save('skill-rft-final', output_dir=args.output_dir)
    sys.stderr.write(f'[rft] done: {rounds} rounds over {gstep} chunks / {pool.epoch} epochs -> {data_path}\n')


if __name__ == '__main__':
    main()
