"""Simplified GRPO + buffer-distill training for the reflexion skill generator (v2).

Key differences from train_reflexion_skill.py:
- No view A/B split: all skill-gen is query-only (deployment form).
- No baseline rollout in training, no balance selection.
- thinking OFF; skill model outputs optional analysis then <skills> block.
- Reward = parseable × (correct AND terminated) × min(1, len_budget/skill_len).
- Buffer A: adv=0 (all-fail) problems accumulate failure trajectories.
- Buffer B: batch rubric → regenerate skill → pass@k validate → SFT injection.
- SFT is event-driven: buffer B reaches threshold → one SFT pass → eval.

Launch:
    LLM_BACKUP_API_KEY=... python cookbook/exp/embedding/train_skill_v2.py \
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
from typing import Any, Dict, List, Optional, Set, Tuple

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

TRAIN_GPUS = int(os.environ.get('TRAIN_GPUS', 2))
REF_GPUS = int(os.environ.get('REF_GPUS', 2))
SKILL_SAMPLER_GPUS = int(os.environ.get('SKILL_SAMPLER_GPUS', 2))
BASE_SAMPLER_GPUS = int(os.environ.get('BASE_SAMPLER_GPUS', 2))
NUM_GPUS = TRAIN_GPUS + REF_GPUS + SKILL_SAMPLER_GPUS + BASE_SAMPLER_GPUS
TRAIN_FSDP = int(os.environ.get('TRAIN_FSDP', min(1, TRAIN_GPUS)))
REF_FSDP = int(os.environ.get('REF_FSDP', min(1, REF_GPUS)))
TRAIN_DP = TRAIN_GPUS // TRAIN_FSDP
REF_DP = REF_GPUS // REF_FSDP


# ===========================================================================
# Section A — boxed extraction + answer grading (verbatim from v1)
# ===========================================================================
_BOXED_RE = re.compile(r'\\boxed\s*\{')


def extract_boxed(text: str) -> Optional[str]:
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
    s = s.strip('$').strip().replace('\u2212', '-')
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
    s = re.sub(r'\^\\circ|\^\{\\circ\}|\u00b0|\\circ', 'deg', s)
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
            try: return float(m.group(1)) / float(m.group(2))
            except (ValueError, ZeroDivisionError): pass
        return None
    va, vb = _eval_frac(a), _eval_frac(b)
    return va is not None and vb is not None and abs(va - vb) < 1e-9 * max(1, abs(va), abs(vb))


_MCQ_REF_RE = re.compile(
    r'^\\(?:textbf|text|mathrm|mathbf)\{\(?([A-E])\)?\s*\}\s*(.+)$'
    r'|^\(?([A-E])\)\s+(.+)$')
_VAR_PREFIX_RE = re.compile(r'^(?:[A-Za-z](?:\([^)]*\))?|\([^)]*\))\s*=\s*(.+)$', re.DOTALL)


def _split_mcq(ans):
    s = ans.strip()
    m = _MCQ_REF_RE.match(s)
    if m:
        return (m.group(1) or m.group(3)), ((m.group(2) or m.group(4) or '').strip() or None)
    bl = re.match(r'^\\?(?:textbf|text|mathrm|mathbf|mathbb)?\{?\(?([A-E])\)?\}?$', s)
    return (bl.group(1), None) if bl else (None, s or None)


def _strip_var_prefix(ans):
    m = _VAR_PREFIX_RE.match((ans or '').strip())
    return m.group(1).strip() if m else (ans or '')


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
        tl, tr = re.sub(r'[\s()\[\]{}\\]', '', left or ''), re.sub(r'[\s()\[\]{}\\]', '', right or '')
        if ',' in tl and tl == tr:
            return True
    if '=' in norm_r and '=' not in norm_p:
        if any(part and (part == norm_p or _try_numeric_equal(part, norm_p)) for part in norm_r.split('=')):
            return True
    if '=' in norm_p and '=' not in norm_r:
        if any(part and (part == norm_r or _try_numeric_equal(part, norm_r)) for part in norm_p.split('=')):
            return True
    return _math_verify_equal(stripped_p or norm_p, stripped_r or norm_r)


_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _numeric_value(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().strip('$').strip()
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    s = re.sub(r'\\!|\\,|\\;|\\ |\\left|\\right|\s', '', s)
    for pat in (r'\\frac\{(-?\d+)\}\{(-?\d+)\}', r'(-?\d+)/(-?\d+)'):
        m = re.fullmatch(pat, s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return str(int(a / b)) if (b and a / b == int(a / b)) else (str(a / b) if b else None)
    return (str(int(float(s))) if float(s) == int(float(s)) else str(float(s))) if _NUM_RE.fullmatch(s) else None


def _answer_leaked(skill: str, reference: str) -> bool:
    if not skill:
        return False
    # Suffix guard: reject only a following DIGIT or a following '.<digit>' (decimal point),
    # NOT a sentence-ending '.'. Old '(?![\d.])' let leaks like "...= 675." slip through
    # because the trailing period satisfied the [\d.] class. 中文注释：尾断言只排除"后接数字"
    # 或"后接小数点+数字"，不排除句末句号，堵住 "答案." 这类泄漏漏检。
    for cand in {_numeric_value(reference), (str(reference).strip() or None)}:
        if cand and re.search(r'(?<![\d.])' + re.escape(cand) + r'(?!\d)(?!\.\d)', skill):
            return True
    return False


# ===========================================================================
# Section B — sampling / parsing utilities
# ===========================================================================
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


def _clean_text(decoded: Optional[str]) -> str:
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').rstrip()


def _extract_skill(text: str) -> Optional[str]:
    """Parse <skills>...</skills> block from skill-gen output."""
    low = text.lower()
    end_think = low.rfind('</think>')
    answer = text[end_think + len('</think>'):] if end_think >= 0 else text
    open_tag, close_tag = '<skills>', '</skills>'
    s = answer.lower().rfind(open_tag)
    if s < 0:
        return None
    inner = s + len(open_tag)
    e = answer.lower().find(close_tag, inner)
    if e < 0:
        return None
    block = answer[inner:e].strip()
    block = re.sub(r'</?(?:skills|skill|diagnose|pitfall|strategy|think)>', '', block, flags=re.IGNORECASE).strip()
    return block or None


def _parse_seq(seq, gold: str) -> Dict[str, Any]:
    text = _clean_text(getattr(seq, 'decoded', '') or '')
    pred = extract_boxed(text)
    correct = bool(pred) and answers_match(pred, gold)
    terminated = getattr(seq, 'stop_reason', None) != 'length'
    return {'pred': pred, 'correct': correct, 'terminated': terminated,
            'stop_reason': getattr(seq, 'stop_reason', None),
            'gen_tokens': len(getattr(seq, 'tokens', None) or []), 'text': text}


def _empty_roll():
    return {'pred': '', 'correct': False, 'terminated': False,
            'stop_reason': 'empty', 'gen_tokens': 0, 'text': ''}


def _run_samples(sampler, prompts, num_samples, max_tokens, gen_dp,
                 temperature=None, top_p=None, top_k=None):
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
# Section C — data loading (simplified: no balance, no xproblem, no views)
# ===========================================================================
def _boxed_batch(rows, dataset):
    sols = rows['solution']
    metas = rows.get('metadata', [None] * len(sols))
    refs = [extract_boxed(s or '') for s in sols]
    keep = [bool(ref) and (dataset != 'aops' or bool((meta or {}).get('boxed')))
            for ref, meta in zip(refs, metas)]
    return {**rows, 'reference_answer': refs, '_keep': keep}


def load_problems(dataset: str, n: int, seed: int) -> List[Dict[str, Any]]:
    ds_id = AOPS_DATASET_ID if dataset == 'aops' else os.environ.get('MATH_DATASET_ID', 'modelscope/competition_math')
    ds = Dataset(DatasetMeta(dataset_id=f'ms://{ds_id}', split='train'))
    nproc = min(32, os.cpu_count() or 1)
    ds.map(lambda rows: _boxed_batch(rows, dataset), num_proc=nproc)
    ds.filter(lambda row: row['_keep'], num_proc=nproc)
    out = [{'data_id': f'{dataset}:{i}', 'problem': row['problem'],
            'reference_answer': row['reference_answer']}
           for i, row in enumerate(ds.dataset)]
    rng = np.random.RandomState(seed)
    rng.shuffle(out)
    return out[:n] if (n and n < len(out)) else out


def _load_records(args):
    records = load_problems(args.dataset, 0, args.seed)
    raw_n = len(records)
    if args.numeric_only:
        records = [{**r, 'reference_answer': v}
                   for r, v in ((r, _numeric_value(r.get('reference_answer'))) for r in records)
                   if v is not None]
    np.random.RandomState(args.seed).shuffle(records)
    # exclude
    excl_ids, excl_probs = set(), set()
    for path in (args.exclude_data_ids or '').split(','):
        path = path.strip()
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row.get('record_type') in {'config', 'summary'}: continue
                did = str(row.get('data_id', '')).strip()
                if did: excl_ids.add(did)
                else:
                    p = str(row.get('problem', '')).strip()
                    if p: excl_probs.add(p)
    if excl_ids or excl_probs:
        records = [r for r in records
                   if str(r.get('data_id', '')) not in excl_ids
                   and str(r.get('problem', '')).strip() not in excl_probs]
    eval_n = min(args.eval_size, len(records)) if args.eval_size > 0 else 0
    eval_records = records[:eval_n]
    # Dedup by problem TEXT: index slices are disjoint, but duplicate problem statements
    # across the boundary would still leak eval into train. Drop any train record whose
    # problem appears in eval, then guard with an explicit overlap assertion.
    # 中文注释：train/eval 去重——按题面文本剔除，防止数据集内重复题目跨界泄漏；末尾硬断言无交集。
    eval_probs = {r['problem'] for r in eval_records}
    train_records = [r for r in records[eval_n:] if r['problem'] not in eval_probs]
    if args.n > 0:
        train_records = train_records[:args.n]
    if {r['problem'] for r in train_records} & eval_probs:
        raise ValueError('eval/train overlap detected after dedup')
    logger.info(f'[data] raw={raw_n} train={len(train_records)} eval={len(eval_records)}')
    return train_records, eval_records


# ===========================================================================
# Section D — DiskCache, ProblemPool, LockedSampler
# ===========================================================================
class DiskCache:
    def __init__(self, path: str, enabled: bool = True):
        self._mem: Dict[str, Any] = {}
        self._fh = None
        self._lock = threading.Lock()
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
    def key_for(*parts):
        return hashlib.md5('\x1f'.join(parts).encode('utf-8')).hexdigest()

    def get(self, key): return self._mem.get(key)
    def __contains__(self, key): return key in self._mem

    def put(self, key, value):
        with self._lock:
            self._mem[key] = value
            if self._fh:
                self._fh.write(json.dumps({'key': key, 'value': value}, ensure_ascii=False) + '\n')
                self._fh.flush()

    def close(self):
        if self._fh: self._fh.close()


class ProblemPool:
    def __init__(self, records, seed):
        self._records = list(records)
        self._seed, self._cursor, self.epoch = seed, 0, 0

    def draw(self, k):
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


class _LockedSampler:
    def __init__(self, sampler):
        self._sampler = sampler
        self._lock = threading.Lock()

    def sample(self, *a, **kw):
        with self._lock:
            return self._sampler.sample(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._sampler, name)


# ===========================================================================
# Section E — Rubric (teacher diagnosis, batched at distill time)
# ===========================================================================
_RFT_DIAG_SYSTEM = """\
You are a strategy-level process checker for a math solution attempt. You are given a
math problem, a rubric, and one attempted solution segment. Decide PASS or FAIL for each
criterion, and write the diagnosis so it can become useful reusable guidance for solving
similar problems without seeing this segment.

Output STRICT JSON (no prose outside it) with this shape:
{"items": [{"index": 1, "verdict": "PASS"|"FAIL", "reason": "...", "fix": ""}], "overall": "OK"|"ISSUES", "summary": "..."}

Rules:
- Judge every criterion independently.
- The diagnosis must stay answer-free.
- For FAIL items: describe the process problem at strategy level.
- A fix suggests the LOCAL correction direction without solving.
- Never reveal the final answer or a corrected expression.
- If segment was cut off (no \\boxed{}), mark length-budget as FAIL.
- Keep "reason" and "fix" concise: one short sentence each.
- Output only the JSON object."""

_RFT_DIAG_USER = """\
## Task / query
{query}

## Rubric
{rubric}

## Segment
{segment}

Now output the diagnostic JSON object."""

_MATH_RUBRIC = [
    ('The attempt chooses a method suitable for the problem structure', False),
    ('The attempt identifies the key constraint, invariant, or quantity before computing', False),
    ('Algebraic and logical transformations preserve validity at each step', True),
    ('The attempt checks required constraints, domains, boundary cases, or validity conditions', False),
    ('The attempt avoids redundant casework, looping, or re-deriving known facts', False),
    ('The attempt reaches a final boxed answer within the length budget', False),
    ('The approach stays focused on the actual question asked', False),
]
_RUBRIC_VERSION = 'rubric_v5_viewb_strategy'


class _RftRubricVerifier(RubricVerifier):
    def _diagnose_trajectory(self, query, rubric_block, segment_text):
        return {'messages': [
            {'role': 'system', 'content': _RFT_DIAG_SYSTEM},
            {'role': 'user', 'content': _RFT_DIAG_USER.format(
                query=query, rubric=rubric_block, segment=segment_text)}]}


def build_rubric_checker():
    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('LLM_BACKUP_BASE_URL')
            or os.environ.get('OPENAI_API_KEY')):
        return None
    return _RftRubricVerifier(
        fixed_rubric=[RubricItem(t, is_hard=h) for t, h in _MATH_RUBRIC], gate=True)


def _format_diagnosis(detail) -> str:
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


def _diagnose_entry(checker, entry: Dict[str, Any]) -> Optional[str]:
    """Run the teacher rubric on ONE buffer-A failure trajectory → formatted diagnosis text
    (or None on API error). Pure network/CPU (no GPU), so it can run on a background thread
    while GRPO trains. Shared by the background pre-diagnosis pool and distill_buffer's
    fallback for any entry the pool did not reach in time.
    中文注释：单条失败轨迹的 rubric 诊断（纯 API，不吃 GPU）。后台预诊断与 distill 补诊断共用。"""
    seg_text = entry['fail_segment']
    if entry.get('fail_stop_reason') == 'length':
        seg_text += ('\n\n[Process note: this attempt was cut off at the token budget '
                     'and never produced a final \\boxed{} answer.]')
    seg = {'messages': [{'role': 'user', 'content': entry['problem']},
                        {'role': 'assistant', 'content': seg_text}]}
    try:
        return _format_diagnosis(checker.diagnose(seg, query=entry['problem']))
    except Exception as exc:
        logger.warning(f'[rubric] diagnose error: {exc}')
        return None


# ===========================================================================
# Section F — NEW: prompts, reward, buffer logic
# ===========================================================================

# ---- Skill-gen system prompt (query-only, thinking OFF) ----
# 中文注释：skillmodel 系统提示词。thinking 关闭；允许在 <skills> 之前输出简短分析；
# 要求 skill ≤600 字符、切题、不给答案、不啰嗦；方向弱列举（pitfall/技术点/step/overview/确信输出）。
SKILL_GEN_SYSTEM = """\
You are a math guidance writer. Write short, reusable guidance for the problem below.

Rules:
- You may briefly analyze the problem BEFORE the <skills> tag.
- Output your guidance inside <skills>...</skills>.
- Keep the guidance within 600 characters, concise and problem-specific.
- Focus on: pitfalls may be happened to avoid, key techniques, brief step outlines, or how to help to converge to the answer quickly.
- Do NOT calculate the final answer.
- Do NOT be verbose or generic.

Output format:
[optional brief analysis]
<skills>
Your reusable solving guidance here.
</skills>"""

# ---- Executor system prompt (with skill injection) ----
# 中文注释：executor 系统提示词，将 skill 注入 solver 的 system prompt 前缀。
DIRECT_SYSTEM = (
    'You are an expert competition mathematician. Solve the following problem '
    'step by step. Provide your final answer inside \\boxed{}.')
_SKILL_SOLVE_PREFIX = (
    DIRECT_SYSTEM + '\n\n'
    'Before you start, keep these reminders in mind to avoid common mistakes on this '
    'type of problem:\n')
_SKILL_SOLVE_SUFFIX = '\nApply them where relevant, but rely on your own reasoning to reach the answer.'


def build_direct_prompt(problem):
    return {'messages': [{'role': 'system', 'content': DIRECT_SYSTEM},
                         {'role': 'user', 'content': problem}]}


def build_skill_solve_prompt(problem, skill):
    skill = (skill or '').strip()
    if not skill:
        return build_direct_prompt(problem)
    return {'messages': [
        {'role': 'system', 'content': _SKILL_SOLVE_PREFIX + skill + _SKILL_SOLVE_SUFFIX},
        {'role': 'user', 'content': problem}]}


# ---- Rubric-guided regeneration prompt (buffer B distillation) ----
# 中文注释：蒸馏重生成提示词。给旧 skill + rubric 诊断，要求产出改进后的 skill。
# 只输出 <skills> 块，≤600字符，不含答案。
REGEN_SYSTEM = """\
You are a math guidance writer. You previously wrote guidance for a problem, but the \
solver still failed. A process-check diagnosed the failure. Revise your guidance to \
address the diagnosed issues.

Rules:
- Output ONLY a <skills>...</skills> block (no analysis).
- Keep within 600 characters, concise, problem-specific.
- Address the diagnosed failure points, also keep the good parts of the old skills.
- Do NOT include the final answer."""

REGEN_USER = """\
Problem:
{problem}

Previous guidance (did not help):
{orig_skill}

Process-check diagnosis:
{rubric_diag}

Write improved <skills> guidance:"""


def _skillgen_prompt(problem: str) -> Dict[str, Any]:
    """Skill-gen prompt: query-only, no view split."""
    return {'messages': [
        {'role': 'system', 'content': SKILL_GEN_SYSTEM},
        {'role': 'user', 'content': f'Problem:\n{problem}'}]}


def _regen_prompt(problem: str, orig_skill: str, rubric_diag: str) -> Dict[str, Any]:
    """Regeneration prompt for buffer B distillation."""
    return {'messages': [
        {'role': 'system', 'content': REGEN_SYSTEM},
        {'role': 'user', 'content': REGEN_USER.format(
            problem=problem, orig_skill=orig_skill, rubric_diag=rubric_diag)}]}


# ---- Reward ----
# 中文注释：reward = parseable × (correct AND terminated) × min(1, budget/len)
# parseable=0 的候选 reward=0 仍参与 group（格式压力）；截断=失败；超长乘法衰减。
def _skill_reward(parseable: bool, correct: bool, terminated: bool,
                  skill_len: int, len_budget: int) -> float:
    if not parseable:
        return 0.0
    base = 1.0 if (correct and terminated) else 0.0
    len_factor = min(1.0, len_budget / max(skill_len, 1))
    return base * len_factor


# ---- Buffer A: collect adv=0 all-fail problems ----
def _collect_buffer_a(chunk, args) -> List[Dict[str, Any]]:
    """Collect problems where all candidates got reward 0 (adv=0, GRPO blind spot).
    Store one representative failure trajectory for later rubric diagnosis."""
    entries = []
    for r in chunk:
        cs = [c for c in r['_cands'] if c.get('reward') is not None]
        if len(cs) < 2:
            continue
        rewards = [c['reward'] for c in cs]
        if max(rewards) > 0:
            continue  # has signal, not all-fail
        # Representative trajectory for rubric + regen seed: prefer a terminated-wrong
        # parseable candidate (complete reasoning to diagnose). Its skill becomes the regen
        # seed, so pick the MOST SUBSTANTIAL one WITHIN budget (longest ≤ len_budget) — a
        # rich-but-not-bloated starting point — rather than an arbitrary [0] or a near-empty
        # skill. If all seeds exceed budget, take the one closest to budget (shortest-over).
        # 中文注释：代表轨迹既做 rubric 诊断又做 regen 种子——优先"跑完但答错"的候选（完整推理），
        # 其 skill 取预算内最长(最有实质)的作种子；若全超预算则取最接近预算的，避免随机/近空种子。
        budget = args.len_budget

        def _seed_key(c):
            L = len(c.get('skills') or '')
            return (L <= budget, L if L <= budget else -L)

        parseable = [c for c in cs if c.get('skills')]
        term_wrong = [c for c in parseable if c['rolls'] and c['rolls'][0].get('terminated')]
        pool_c = term_wrong or parseable or cs
        rep = max(pool_c, key=_seed_key)
        stop_dist = {}
        for c in cs:
            sr = c['rolls'][0]['stop_reason'] if c['rolls'] else 'none'
            stop_dist[sr] = stop_dist.get(sr, 0) + 1
        entries.append({
            'problem': r['problem'], 'reference_answer': r['reference_answer'],
            'data_id': r.get('data_id', ''),
            'orig_skill': rep.get('skills', ''),
            'orig_len': len(rep.get('skills', '')),
            'fail_segment': rep['rolls'][0]['text'] if rep['rolls'] else '',
            'fail_stop_reason': rep['rolls'][0]['stop_reason'] if rep['rolls'] else 'none',
            'stop_reason_dist': stop_dist,
        })
    return entries


# ---- Buffer B distillation ----
def distill_buffer(entries: List[Dict[str, Any]], skill_sampler, base_sampler,
                   checker, skill_dp: int, base_dp: int,
                   args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Batch rubric → regenerate K distinct skills → greedy-validate → return SFT records.
    中文注释：蒸馏流程（方案 B，多样性在 skill 侧、executor 用贪心）：
    1. 批量 rubric 诊断失败轨迹；2. 仅 [FAIL] 项用高温重生成 K 个不同候选 skill；
    3. 每个候选过 ≤budget+无leak+去重 过滤；4. 每个存活候选用 executor 贪心(T=0)解 1 次；
    5. gate：≥m 个不同候选达成 terminated-correct → select 长度最接近 budget 的一个入 buffer B。
    返回 (sft_records, distill_records)：后者逐 entry 记录 rubric_diag/候选 skill/贪心解结果/漏斗
    stage，落盘到 distill_records.jsonl 供复盘（否则 rubric 诊断与候选明细只存在于内存）。"""
    if not checker or not entries:
        return [], []

    # Step 1: ensure every entry has a rubric diagnosis. Entries pre-diagnosed in the
    # background (see _prediagnose in main) already carry '_rubric_diag'; only the misses
    # are diagnosed here (in parallel), so the GPU-idle API wait is normally hidden.
    # 中文注释：优先用后台预诊断结果；只对没预诊断到的条目并行补跑，隐藏 API 等待。
    pending = [e for e in entries if not e.get('_rubric_diag')]
    if pending:
        workers = min(args.rubric_workers, len(pending))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            diags = list(ex.map(lambda e: _diagnose_entry(checker, e), pending))
        for entry, diag in zip(pending, diags):
            entry['_rubric_diag'] = diag or ''
    for entry in entries:
        entry['rubric_diag'] = entry.get('_rubric_diag') or ''

    # Builder for the structured distill audit records (one per buffer-A entry). Closure over
    # `entries`/`args`; takes the per-entry regen skills + greedy solve results (may be empty
    # for the early-exit funnel stages). 中文注释：构造逐 entry 的蒸馏审计记录（含漏斗 stage）。
    def _mk_distill(results_by_entry, per_entry_skills, has_fail):
        hf_index = {id(e): ei for ei, e in enumerate(has_fail)}
        recs = []
        for e in entries:
            ei = hf_index.get(id(e))
            cand_results = results_by_entry.get(ei, []) if ei is not None else []
            n_pass = sum(1 for c in cand_results if c['correct'] and c['terminated'])
            n_cand = len(per_entry_skills[ei]) if (ei is not None and ei < len(per_entry_skills)) else 0
            if ei is None:
                stage = 'no_fail'          # rubric 未给出任何 [FAIL]
            elif n_cand == 0:
                stage = 'no_valid_regen'   # 有 [FAIL] 但重生成无一条过 ≤budget/无leak/去重
            elif n_pass >= args.passatk_m:
                stage = 'accepted'         # ≥m 个候选贪心解对 → 入 buffer B
            else:
                stage = 'rejected'         # 有候选但 <m 个解对
            recs.append({
                'record_type': 'distill', 'stage': stage,
                'data_id': e.get('data_id', ''), 'problem': e['problem'],
                'reference_answer': e['reference_answer'],
                'orig_skill': e.get('orig_skill', ''), 'orig_len': e.get('orig_len', 0),
                'fail_stop_reason': e.get('fail_stop_reason', ''),
                'rubric_diag': e.get('rubric_diag', ''),
                'n_cand_skills': n_cand, 'n_pass_skills': n_pass,
                'candidates': cand_results,  # each: {skill, len, correct, terminated}
            })
        return recs

    # Step 2: filter to entries with [FAIL] diagnosis
    has_fail = [e for e in entries if '[FAIL]' in e.get('rubric_diag', '')]
    if not has_fail:
        logger.info('[distill] no [FAIL] diagnoses, skipping regen')
        return [], _mk_distill({}, [], [])

    # Step 3: regenerate K DISTINCT candidate skills per [FAIL] entry (skill-model, high T).
    # 中文注释：方案 B——多样性放在 skill 侧。对每条 [FAIL] 条目用高温采 K 个不同候选 skill。
    k = args.passatk_k
    regen_prompts = [_regen_prompt(e['problem'], e['orig_skill'], e['rubric_diag']) for e in has_fail]
    regen_out = _run_samples(skill_sampler, regen_prompts, k, args.skill_max_tokens, skill_dp,
                             temperature=args.passatk_skill_temp, top_p=args.passatk_skill_top_p)

    # Per entry: keep the parseable, ≤budget, non-leaking candidate skills (deduped).
    per_entry_skills: List[List[str]] = []
    n_entries_with_cands = 0
    for entry, seqs in zip(has_fail, regen_out):
        seen, skills = set(), []
        for s in (seqs or []):
            resp = _clean_text(getattr(s, 'decoded', '') or '')
            skill = _extract_skill(resp)
            if not skill or len(skill) > args.len_budget:
                continue
            if _answer_leaked(skill, entry['reference_answer']):
                continue
            if skill in seen:
                continue  # 去重：同一 skill 只算一个"不同候选"
            seen.add(skill)
            skills.append(skill)
        per_entry_skills.append(skills)
        if skills:
            n_entries_with_cands += 1

    # Flatten to (entry_idx, skill) for ONE batched greedy executor solve per candidate skill.
    # 中文注释：每个候选 skill 只用 executor 贪心(T=0)解 1 次——与部署/eval 口径一致；
    # k 次 rollout 摊到 k 个不同 skill 上，多样性来自 skill 侧而非 executor 侧。
    flat_idx, flat_prompts = [], []
    for ei, skills in enumerate(per_entry_skills):
        for sk in skills:
            flat_idx.append((ei, sk))
            flat_prompts.append(build_skill_solve_prompt(has_fail[ei]['problem'], sk))
    if not flat_prompts:
        logger.info(f'[distill] {len(has_fail)} [FAIL] entries, 0 valid regen skills')
        return [], _mk_distill({}, per_entry_skills, has_fail)
    solve_out = _run_samples(base_sampler, flat_prompts, 1, args.max_tokens, base_dp,
                             temperature=0.0)

    # Gather greedy results back per entry: record EVERY candidate skill's solve outcome
    # (not just passers) so distill_records can show why an entry was rejected.
    # 中文注释：回收每个候选 skill 的贪心解结果（含未通过的），供审计记录还原被拒原因。
    results_by_entry: Dict[int, List[Dict[str, Any]]] = {}
    for (ei, sk), seqs in zip(flat_idx, solve_out):
        roll = _parse_seq(seqs[0], has_fail[ei]['reference_answer']) if seqs else _empty_roll()
        results_by_entry.setdefault(ei, []).append(
            {'skill': sk, 'len': len(sk), 'correct': roll['correct'], 'terminated': roll['terminated']})

    # Step 4: gate ≥ m distinct greedy-effective skills; select the survivor CLOSEST to the
    # length budget (short is the floor, but not so short it degrades to answer-dumping).
    # 中文注释：gate——≥m 个不同 skill 在贪心下 terminated-correct；select——在通过的候选里
    # 选长度最接近 budget 的一个入 buffer B（短是地板，但别短到退化成吐答案）。
    sft_records = []
    for ei, cand_results in results_by_entry.items():
        passers = [c['skill'] for c in cand_results if c['correct'] and c['terminated']]
        if len(passers) < args.passatk_m:
            continue
        entry = has_fail[ei]
        best = min(passers, key=lambda sk: abs(len(sk) - args.len_budget))
        sft_records.append({
            'problem': entry['problem'], 'reference_answer': entry['reference_answer'],
            'data_id': entry.get('data_id', ''),
            'response': f'<skills>\n{best}\n</skills>',
            'skills': best, 'sft': True,
            'n_pass_skills': len(passers), 'n_cand_skills': len(per_entry_skills[ei]),
        })

    logger.info(f'[distill] {len(entries)} A → {len(has_fail)} [FAIL] → '
                f'{n_entries_with_cands} w/cands → {len(sft_records)} validated B '
                f'(gate m={args.passatk_m}/k={k})')
    return sft_records, _mk_distill(results_by_entry, per_entry_skills, has_fail)



# ===========================================================================
# Section G — GRPO advantages + training
# ===========================================================================
def _assign_advantages(chunk, args):
    """Group-relative advantage: A = (R - mean) / (std + eps). std==0 → adv=0 (skipped)."""
    eps = 1e-6
    adv_clip = abs(float(getattr(args, 'adv_clip', 0.0) or 0.0))
    for r in chunk:
        for c in r['_cands']:
            c['advantage'], c['kept'] = 0.0, False
        cs = [c for c in r['_cands'] if c.get('reward') is not None]
        if len(cs) < 2:
            continue
        rewards = [c['reward'] for c in cs]
        mean_r = sum(rewards) / len(rewards)
        std = (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
        if std < 1e-9:
            continue
        for c in cs:
            raw = (c['reward'] - mean_r) / (std + eps)
            c['advantage'] = max(-adv_clip, min(adv_clip, raw)) if adv_clip > 0 else raw
            c['kept'] = c['reward'] > mean_r


def _train_trajectory(rec):
    """Rebuild the query-only skill-gen prompt (train/inference match) + response.
    GRPO records carry the full generated response; SFT records carry only the
    cleaned <skills> block. key_rounds selects the final assistant turn."""
    msgs = _skillgen_prompt(rec['problem'])['messages']
    return {'messages': msgs + [{'role': 'assistant', 'content': rec['response']}],
            'user_data': {'key_rounds': [len(msgs)]}}


def _train_step(skill_model, ref_model, ckpt, samples, args):
    """On-policy GRPO update over one batch, then sync weights. SFT samples ride the
    same GRPOLoss with a positive constant advantage (--sft-weight)."""
    trajs = [_train_trajectory(rec) for rec in samples]
    advs = [float(rec['advantage']) for rec in samples]
    rem = (-len(trajs)) % args.sft_batch_size
    if rem:
        trajs += [trajs[-1]] * rem
        advs += [0.0] * rem
    n, sft = len(trajs), args.sft_batch_size
    mini = args.ppo_mini_batch_size if args.ppo_mini_batch_size > 0 else n
    mini = max(sft, (mini // sft) * sft)
    multi_step = mini < n
    micro_ref, micro_old = [], []
    for i in range(0, n, sft):
        mb = trajs[i:i + sft]
        micro_ref.append(ref_model.forward_only(inputs=mb).get('logps'))
        micro_old.append(skill_model.forward_only(inputs=mb).get('logps') if multi_step else None)
    micro, n_steps = 0, 0
    for ms in range(0, n, mini):
        for i in range(ms, min(ms + mini, n), sft):
            k = i // sft
            skill_model.forward_backward(inputs=trajs[i:i + sft], advantages=advs[i:i + sft],
                                         old_logps=micro_old[k], ref_logps=micro_ref[k])
            micro += 1
        skill_model.clip_grad_and_step()
        n_steps += 1
    ckpt.sync_weights(merge_and_sync=True)
    metric = skill_model.calculate_metric(is_training=True)
    n_sft = sum(1 for s in samples if s.get('sft'))
    return {'n_samples': len(samples), 'n_sft': n_sft, 'n_grpo': len(samples) - n_sft,
            'n_steps': n_steps, 'n_micro_batches': micro,
            'metric': {k: (float(v) if _is_num(v) else v) for k, v in (metric or {}).items()}}


def _is_num(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


# ===========================================================================
# Section H — chunk processing, records, eval (+ hard-slice rescue)
# ===========================================================================
def process_chunk(base_sampler, skill_sampler, chunk, ci, base_dp, skill_dp, args):
    """skill-gen (query-only) → leak audit → with-skill greedy pass → reward → advantages.
    Returns (full_records, summary, grpo_train_records, buffer_a_entries)."""
    for r in chunk:
        r['_cands'] = []
    # skill-gen (thinking OFF), re-sample problems with no clean candidate
    flat = []
    pending = list(chunk)
    for _ in range(args.skill_retries + 1):
        if not pending:
            break
        sg_out = _run_samples(skill_sampler, [_skillgen_prompt(r['problem']) for r in pending],
                              args.n_skills, args.skill_max_tokens, skill_dp,
                              temperature=args.skill_gen_temperature, top_p=args.skill_gen_top_p,
                              top_k=args.skill_gen_top_k)
        still = []
        for r, seqs in zip(pending, sg_out):
            got = False
            for s in seqs:
                resp = _clean_text(getattr(s, 'decoded', '') or '')
                block = _extract_skill(resp) or ''
                cand = {'skills': block, 'response': resp, 'parseable': bool(block),
                        'leaked': None, 'with_pass': None, 'reward': None, 'rolls': [],
                        'advantage': 0.0, 'kept': False,
                        'skillgen_stop': getattr(s, 'stop_reason', None),
                        'skillgen_tokens': len(getattr(s, 'tokens', None) or [])}
                r['_cands'].append(cand)
                if block:
                    flat.append((r, cand))
                    got = True
            if not got:
                still.append(r)
        pending = still

    # leak audit (deterministic, observability only)
    for r, c in flat:
        c['leaked'] = _answer_leaked(c['skills'], r['reference_answer'])

    # with-skill greedy pass (T=0)
    if flat:
        ws_out = _run_samples(base_sampler,
                              [build_skill_solve_prompt(r['problem'], c['skills']) for r, c in flat],
                              1, args.max_tokens, base_dp, temperature=0.0)
        for (r, c), seqs in zip(flat, ws_out):
            roll = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
            c['rolls'] = [roll]
            c['with_pass'] = 1.0 if roll['correct'] else 0.0
            c['reward'] = _skill_reward(c['parseable'], roll['correct'], roll['terminated'],
                                        len(c['skills']), args.len_budget)
    # unparseable candidates score 0 and still join the group (format pressure)
    for r in chunk:
        for c in r['_cands']:
            if c['reward'] is None:
                c['reward'] = 0.0

    _assign_advantages(chunk, args)

    grpo = []
    for r in chunk:
        for c in r['_cands']:
            if abs(c.get('advantage') or 0.0) > 1e-9:
                grpo.append({'problem': r['problem'], 'reference_answer': r['reference_answer'],
                             'data_id': r.get('data_id', ''), 'response': c['response'],
                             'skills': c['skills'], 'advantage': c['advantage'],
                             'kept': c['kept'], 'reward': c['reward'], 'sft': False})
    buffer_a = _collect_buffer_a(chunk, args)
    return _full_records(chunk, ci), _chunk_summary(chunk, ci), grpo, buffer_a


def _roll(x):
    return {k: x[k] for k in ('pred', 'correct', 'terminated', 'stop_reason', 'gen_tokens', 'text')}


def _full_records(chunk, ci):
    out = []
    for r in chunk:
        out.append({
            'record_type': 'problem', 'chunk': ci, 'problem': r['problem'],
            'reference_answer': r['reference_answer'], 'data_id': r.get('data_id', ''),
            'candidates': [{
                'skills': c['skills'], 'response': c['response'], 'parseable': c['parseable'],
                'leaked': c['leaked'], 'with_pass': c['with_pass'], 'reward': c.get('reward'),
                'advantage': c.get('advantage'), 'kept': c.get('kept'),
                'skillgen_stop': c.get('skillgen_stop'), 'skillgen_tokens': c.get('skillgen_tokens'),
                'rolls': [_roll(x) for x in c['rolls']],
            } for c in r['_cands']],
        })
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _chunk_summary(chunk, ci):
    all_cands = [c for r in chunk for c in r['_cands']]
    cands = [c for c in all_cands if c['parseable']]
    scored = [c for c in cands if c['with_pass'] is not None]
    ws_rolls = [x for c in scored for x in c['rolls']]
    # signal: fraction of groups with zero reward variance (no gradient)
    group_vars, all_rewards, zero_grad, groups = [], [], 0, 0
    for r in chunk:
        rewards = [c['reward'] for c in r['_cands'] if c.get('reward') is not None]
        if len(rewards) < 2:
            continue
        groups += 1
        all_rewards.extend(rewards)
        v = _std(rewards)
        group_vars.append(v)
        if v < 1e-9:
            zero_grad += 1
    n_train = sum(1 for c in all_cands if abs(c.get('advantage') or 0.0) > 1e-9)
    trunc = sum(1 for x in ws_rolls if x['stop_reason'] == 'length')
    ws_acc = _mean([1.0 if any(c.get('reward') for c in r['_cands']) else 0.0
                    for r in chunk if r['_cands']])
    return {
        'record_type': 'summary', 'chunk': ci, 'n': len(chunk),
        'n_generated': len(all_cands), 'n_candidates_parseable': len(cands),
        'parse_rate': (len(cands) / len(all_cands)) if all_cands else 0.0,
        'n_leaked': sum(1 for c in cands if c['leaked']),
        'leak_rate': (sum(1 for c in cands if c['leaked']) / len(cands)) if cands else 0.0,
        'n_train_samples': n_train, 'n_groups': groups,
        'zero_grad_frac': (zero_grad / groups) if groups else 0.0,
        'reward_mean': _mean(all_rewards), 'reward_std': _std(all_rewards),
        'group_reward_std_mean': _mean(group_vars),
        'skill_tokens_mean': _mean([c.get('skillgen_tokens') or 0 for c in cands]),
        'skill_chars_mean': _mean([len(c['skills']) for c in cands]),
        'avg_withskill_pass': ws_acc,
        'candidate_withskill_pass': _mean([c['with_pass'] for c in scored]),
        'withskill_trunc_frac': (trunc / len(ws_rolls)) if ws_rolls else 0.0,
        'termination_rate_withskill': _mean([1.0 if x['terminated'] else 0.0 for x in ws_rolls]),
    }


def run_greedy_eval(base_sampler, skill_sampler, eval_records, ci, rounds,
                    base_dp, skill_dp, args, base_cache):
    """SEAM mean@1 on the fixed holdout: greedy skill (T=0) → greedy base solve (T=0).
    Adds hard-slice (baseline_pass==0) rescue rate as a zero-cost secondary readout."""
    # baseline (frozen, cached)
    todo = [r for r in eval_records if DiskCache.key_for(r['problem']) not in base_cache]
    if todo:
        out = _run_samples(base_sampler, [build_direct_prompt(r['problem']) for r in todo],
                           1, args.max_tokens, base_dp, temperature=0.0)
        for r, seqs in zip(todo, out):
            roll = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
            base_cache.put(DiskCache.key_for(r['problem']), roll)
    for r in eval_records:
        br = base_cache.get(DiskCache.key_for(r['problem']))
        r['_baseline_pass'] = 1.0 if br['correct'] else 0.0
    # skill-gen (greedy) → with-skill (greedy)
    sg_out = _run_samples(skill_sampler, [_skillgen_prompt(r['problem']) for r in eval_records],
                          1, args.skill_max_tokens, skill_dp, temperature=0.0)
    skills = []
    for seqs in sg_out:
        if not seqs:
            skills.append(('', ''))
            continue
        sresp = _clean_text(getattr(seqs[0], 'decoded', '') or '')
        skills.append((_extract_skill(sresp) or '', sresp))
    ws_out = _run_samples(base_sampler,
                          [build_skill_solve_prompt(r['problem'], sk) for r, (sk, _) in zip(eval_records, skills)],
                          1, args.max_tokens, base_dp, temperature=0.0)
    recs = []
    for r, (sk, sresp), seqs in zip(eval_records, skills, ws_out):
        roll = _parse_seq(seqs[0], r['reference_answer']) if seqs else _empty_roll()
        recs.append({
            'record_type': 'eval_problem', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
            'data_id': r.get('data_id', ''), 'problem': r['problem'],
            'reference_answer': r['reference_answer'], 'baseline_pass': r['_baseline_pass'],
            'skill': sk, 'skill_parseable': bool(sk), 'skill_chars': len(sk),
            'withskill_pred': roll['pred'], 'withskill_correct': roll['correct'],
            'withskill_terminated': roll['terminated'], 'withskill_stop_reason': roll['stop_reason'],
            'withskill_text': roll['text'],
        })
    n = len(recs)
    ws = (sum(1 for x in recs if x['withskill_correct']) / n) if n else 0.0
    base = (sum(x['baseline_pass'] for x in recs) / n) if n else 0.0
    fmt = (sum(1 for x in recs if x['skill_parseable']) / n) if n else 0.0
    term = (sum(1 for x in recs if x['withskill_terminated']) / n) if n else 0.0
    # 中文注释：难题子片救活率——baseline_pass==0 的子集里 with-skill 做对的比例。
    # 零成本（复用已算字段），是 buffer B 回路的目标量（见 skill_quality_analysis.md 第 14 节）。
    hard = [x for x in recs if not x['baseline_pass']]
    hard_rescued = sum(1 for x in hard if x['withskill_correct'])
    hard_rescue_rate = (hard_rescued / len(hard)) if hard else 0.0
    summary = {'record_type': 'eval_summary', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
               'n': n, 'acc_mean1': ws, 'baseline_acc_mean1': base, 'lift_mean1': ws - base,
               'format_mean1': fmt, 'term_mean1': term,
               'hard_n': len(hard), 'hard_rescued': hard_rescued, 'hard_rescue_rate': hard_rescue_rate}
    metrics = {'core/math/acc/mean@1': ws, 'core/math/baseline_acc/mean@1': base,
               'core/math/lift/mean@1': ws - base, 'core/math/format/mean@1': fmt,
               'core/math/term/mean@1': term, 'core/math/hard_rescue/mean@1': hard_rescue_rate}
    return recs, summary, metrics


# ===========================================================================
# Section I — components, args, main
# ===========================================================================
def init_components(args):
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
    skill_model.set_template(Template, model_id=MODEL_ID, enable_thinking=False,
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
    ref_model.set_template(Template, model_id=MODEL_ID, enable_thinking=False,
                           max_length=args.max_model_len, truncation_strategy='delete')
    ref_model.set_processor(InputProcessor, padding_free=False)
    ref_model.set_loss('GRPOLoss', epsilon=args.grpo_epsilon)

    def _sampler(group, world, enable_thinking):
        s = vLLMSampler(model_id=MODEL_ID,
                        engine_args={'gpu_memory_utilization': GPU_MEM,
                                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1},
                        device_mesh=DeviceMesh.from_sizes(world_size=world, dp_size=world),
                        remote_group=group)
        s.set_template(Template, model_id=MODEL_ID, enable_thinking=enable_thinking, max_length=args.max_model_len)
        return s

    skill_sampler = _sampler('skill_sampler', SKILL_SAMPLER_GPUS, enable_thinking=False)
    base_sampler = _LockedSampler(_sampler('base_sampler', BASE_SAMPLER_GPUS, enable_thinking=True))
    ckpt = CheckpointEngineManager(model=skill_model, sampler=skill_sampler)
    return skill_model, ref_model, skill_sampler, base_sampler, ckpt, SKILL_SAMPLER_GPUS, BASE_SAMPLER_GPUS


def _build_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    p.add_argument('--n', type=int, default=2000, help='Problems loaded into the draw pool (0=all).')
    p.add_argument('--exclude-data-ids', default='',
                   help='Comma-separated jsonl files whose data_id/problem keys are excluded.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval-size', type=int, default=128, help='Fixed holdout size (0 disables).')
    p.add_argument('--eval-every', type=int, default=5, help='Run holdout eval every N chunks.')
    p.add_argument('--chunk-size', type=int, default=16)
    p.add_argument('--n-skills', type=int, default=8)
    p.add_argument('--skill-retries', type=int, default=2)
    p.add_argument('--skill-gen-temperature', type=float, default=1.0)
    p.add_argument('--skill-gen-top-p', type=float, default=1.0)
    p.add_argument('--skill-gen-top-k', type=int, default=-1)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--skill-max-tokens', type=int, default=4096)
    p.add_argument('--len-budget', type=int, default=600,
                   help='Skill length budget (chars). Reward multiplied by min(1, budget/len).')
    # --- buffer / distillation ---
    p.add_argument('--distill-trigger', type=int, default=300,
                   help='Start draining buffer A into distillation once it reaches this many entries.')
    p.add_argument('--distill-batch', type=int, default=64,
                   help='Entries distilled per iteration while buffer A is over --distill-trigger '
                        '(incremental drain: bounds per-step latency instead of one big stall).')
    p.add_argument('--sft-trigger', type=int, default=100,
                   help='Run one SFT pass + eval when buffer B reaches this many validated entries. '
                        'Kept low: the distill funnel (has-FAIL × valid-regen × pass@k) yields only '
                        '~10-15%% of buffer A, so a high threshold would rarely fire the SFT loop.')
    # Plan B validation: diversity lives in the SKILL side, the executor stays at the
    # deployment (greedy) decoding口径. For each buffer-A problem we regenerate K distinct
    # candidate skills (high temperature), run each through ONE greedy (T=0) executor solve,
    # and accept the problem iff >= M distinct skills reach a terminated-correct solve. This
    # validates "the problem admits several skills that work under greedy decoding" (matches
    # eval口径) rather than "one skill passes m/k times under a high-temperature executor".
    p.add_argument('--passatk-k', type=int, default=8,
                   help='Plan B: number of DISTINCT candidate skills regenerated per problem '
                        '(skill-side diversity; executor stays greedy).')
    p.add_argument('--passatk-skill-temp', type=float, default=1.0,
                   help='Skill-model temperature when regenerating the K candidate skills '
                        '(needs >0 for diversity across candidates).')
    p.add_argument('--passatk-skill-top-p', type=float, default=1.0,
                   help='Skill-model top-p when regenerating the K candidate skills.')
    p.add_argument('--passatk-m', type=int, default=2,
                   help='Plan B: min number of DISTINCT candidate skills that must reach a '
                        'terminated-correct GREEDY solve to accept the problem into buffer B. '
                        'Lower than pass@k-over-one-skill (default 2): requiring m distinct '
                        'greedy-effective skills is already a strong, low-noise bar.')
    p.add_argument('--sft-weight', type=float, default=0.5,
                   help='Advantage magnitude for SFT distillation samples (-w*logp + beta*KL).')
    p.add_argument('--rubric-workers', type=int, default=16)
    # --- GRPO ---
    p.add_argument('--sft-batch-size', type=int, default=8)
    p.add_argument('--ppo-mini-batch-size', type=int, default=0)
    p.add_argument('--grpo-epsilon', type=float, default=0.2)
    p.add_argument('--adv-clip', type=float, default=3.0)
    p.add_argument('--kl-beta', type=float, default=0.001)
    p.add_argument('--lr', type=float, default=6e-6)
    p.add_argument('--max-train-rounds', type=int, default=1500)
    p.add_argument('--save-rounds', type=int, default=200)
    p.add_argument('--output-dir', default='./output/skill_v2')
    p.add_argument('--cache-dir', default='')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--swanlab-project', default='twinkle')
    p.add_argument('--swanlab-exp', default='')
    args = p.parse_args()
    if args.sft_batch_size % TRAIN_DP != 0:
        raise ValueError(f'--sft-batch-size ({args.sft_batch_size}) must be a multiple of train dp ({TRAIN_DP})')
    if args.chunk_size < 1:
        raise ValueError('--chunk-size must be >= 1')
    return args


def _write(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _swan_metrics(summary, log):
    # Lean metric set: each carries independent information. Dropped as redundant —
    # 中文注释：删除冗余项（换算重复）：n_groups(≈chunk_size)、reward_std(池化,组内方差已够)、
    # skill_tokens_mean(与chars重复)、leak/n(=rate×n)、candidate_withskill(与问题级重复)、
    # term/withskill(=1-trunc)、train/n_steps(恒为1)。
    d = {
        'signal/zero_grad_frac': summary['zero_grad_frac'],
        'signal/reward_mean': summary['reward_mean'],
        'signal/group_reward_std_mean': summary['group_reward_std_mean'],
        'signal/n_train_samples': summary['n_train_samples'],
        'skill/parse_rate': summary['parse_rate'], 'skill/chars_mean': summary['skill_chars_mean'],
        'leak/rate': summary['leak_rate'],
    }
    if summary['n_groups'] > 0:
        d.update({'acc/withskill_pass': summary['avg_withskill_pass'],
                  'term/withskill_trunc_frac': summary['withskill_trunc_frac']})
    if log:
        d['train/n_grpo'] = log['n_grpo']
        d['train/n_sft'] = log['n_sft']
        for k, v in (log.get('metric') or {}).items():
            if not _is_num(v):
                continue
            if k.startswith('learning rate'):
                if 'group 1' in k:
                    d['train/lr'] = float(v)
            else:
                d[f'train/{k.replace(" ", "_")}'] = float(v)
    return d


def main():
    args = _build_args()
    records, eval_records = _load_records(args)
    if len(records) < args.chunk_size:
        raise ValueError(f'--chunk-size ({args.chunk_size}) exceeds loaded ({len(records)}); raise --n')

    os.makedirs(args.output_dir, exist_ok=True)
    gen_path = os.path.join(args.output_dir, 'gen_records.jsonl')
    eval_path = os.path.join(args.output_dir, 'eval_records.jsonl')
    sft_path = os.path.join(args.output_dir, 'skill_dataset.jsonl')
    train_log_path = os.path.join(args.output_dir, 'train_log.jsonl')
    buffer_a_path = os.path.join(args.output_dir, 'buffer_a.jsonl')
    distill_path = os.path.join(args.output_dir, 'distill_records.jsonl')

    use_swan = swanlab is not None and os.environ.get('SWANLAB_MODE') != 'disabled'
    if use_swan:
        swanlab.init(project=args.swanlab_project, experiment_name=(args.swanlab_exp or None),
                     config={'model': MODEL_ID, 'dataset': args.dataset, 'n': len(records),
                             'eval_n': len(eval_records), 'n_skills': args.n_skills,
                             'len_budget': args.len_budget, 'distill_trigger': args.distill_trigger,
                             'sft_trigger': args.sft_trigger, 'passatk_k': args.passatk_k,
                             'passatk_m': args.passatk_m, 'passatk_skill_temp': args.passatk_skill_temp,
                             'sft_weight': args.sft_weight, 'lr': args.lr})

    skill_model, ref_model, skill_sampler, base_sampler, ckpt, skill_dp, base_dp = init_components(args)
    checker = build_rubric_checker()
    if checker is None:
        sys.stderr.write('[v2] no LLM backup env -> buffer B distillation DISABLED (GRPO only)\n')

    cache_dir = args.cache_dir or os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    eval_base_cache = DiskCache(os.path.join(cache_dir, 'eval_baseline.jsonl'), not args.no_cache)

    cfg = {'record_type': 'config', 'model': MODEL_ID, 'dataset': args.dataset,
           'n': len(records), 'eval_n': len(eval_records), 'seed': args.seed,
           'n_skills': args.n_skills, 'len_budget': args.len_budget,
           'distill_trigger': args.distill_trigger, 'sft_trigger': args.sft_trigger,
           'passatk_k': args.passatk_k, 'passatk_m': args.passatk_m,
           'passatk_skill_temp': args.passatk_skill_temp, 'passatk_skill_top_p': args.passatk_skill_top_p,
           'sft_weight': args.sft_weight,
           'grpo_epsilon': args.grpo_epsilon, 'kl_beta': args.kl_beta, 'lr': args.lr,
           'rubric_check': bool(checker), 'max_train_rounds': args.max_train_rounds,
           'started': int(time.time())}

    hist_a: List[Dict[str, Any]] = []   # buffer A accumulator (in-memory + jsonl)
    sft_queue: List[Dict[str, Any]] = []  # buffer B: validated SFT records awaiting an SFT pass
    rounds = 0        # GRPO rounds only (gates --max-train-rounds + save cadence)
    sft_rounds = 0    # SFT passes (separate: must NOT eat the GRPO round budget)
    pool = ProblemPool(records, args.seed)

    # Background rubric pre-diagnosis (做法 B): the moment a failure trajectory lands in
    # buffer A, fire its teacher-rubric call on a daemon thread pool. The API round-trip
    # then overlaps with GRPO GPU work, so by the time --distill-trigger fires the
    # diagnoses are usually already cached on each entry ('_rubric_diag'); distill_buffer
    # only pays for the stragglers. Entries are dicts held by reference, so the worker
    # writes the result straight onto the entry.
    # 中文注释：失败轨迹一进 buffer A 就后台异步跑 rubric，API 等待藏进 GPU 训练时间；
    # 到蒸馏时诊断多已缓存在条目上，distill_buffer 只补漏。
    prediag_pool = (ThreadPoolExecutor(max_workers=max(1, args.rubric_workers),
                                       thread_name_prefix='rubric-prediag')
                    if checker else None)

    def _prediagnose(entry: Dict[str, Any]):
        entry['_rubric_diag'] = _diagnose_entry(checker, entry) or ''

    with open(gen_path, 'w', encoding='utf-8') as gen_f, \
            open(eval_path, 'w', encoding='utf-8') as eval_f, \
            open(sft_path, 'w', encoding='utf-8') as sft_f, \
            open(train_log_path, 'w', encoding='utf-8') as tlog, \
            open(distill_path, 'w', encoding='utf-8') as distill_f, \
            open(buffer_a_path, 'w', encoding='utf-8') as buf_f:
        for f in (gen_f, eval_f, sft_f, tlog, distill_f):
            _write(f, cfg)

        def _do_eval(gstep):
            recs, summary, metrics = run_greedy_eval(
                base_sampler, skill_sampler, eval_records, gstep, rounds, base_dp, skill_dp,
                args, eval_base_cache)
            for rec in recs:
                _write(eval_f, rec)
            _write(eval_f, summary)
            eval_f.flush()
            if use_swan:
                swanlab.log({f'eval/{k}': v for k, v in metrics.items()}, step=max(gstep, 0))
            sys.stderr.write(
                f'[eval] g{gstep}: n={summary["n"]} acc={summary["baseline_acc_mean1"]:.3f}'
                f'->{summary["acc_mean1"]:.3f} lift={summary["lift_mean1"]:+.3f} '
                f'hard_rescue={summary["hard_rescue_rate"]:.3f}({summary["hard_rescued"]}/{summary["hard_n"]}) '
                f'fmt={summary["format_mean1"]:.2f} rounds={rounds}\n')

        if eval_records:
            _do_eval(-1)

        gstep = 0
        while rounds < args.max_train_rounds:
            chunk = pool.draw(args.chunk_size)
            full, summary, grpo, buffer_a = process_chunk(
                base_sampler, skill_sampler, chunk, gstep, base_dp, skill_dp, args)

            # accumulate buffer A (only when a rubric checker exists to consume it;
            # 中文注释：无 checker 时蒸馏永不触发，不累积以免内存无限增长)
            if checker:
                for e in buffer_a:
                    _write(buf_f, e)
                    prediag_pool.submit(_prediagnose, e)  # 后台异步预诊断，不阻塞主循环
                buf_f.flush()
                hist_a.extend(buffer_a)

            # GRPO train step (only when there is signal)
            log = None
            if grpo:
                log = _train_step(skill_model, ref_model, ckpt, grpo, args)
                rounds += 1
                log.update({'record_type': 'train_round', 'round': rounds, 'chunk': gstep,
                            'epoch': pool.epoch, 'kind': 'grpo', 'ts': int(time.time())})
                _write(tlog, log)
                tlog.flush()
                if rounds % args.save_rounds == 0:
                    skill_model.save(f'skill-v2-{rounds}', output_dir=args.output_dir)

            summary['rounds_done'], summary['epoch'] = rounds, pool.epoch
            summary['buffer_a_size'], summary['sft_queue_size'] = len(hist_a), len(sft_queue)
            for rec in full:
                _write(gen_f, rec)
            _write(gen_f, summary)
            gen_f.flush()

            sys.stderr.write(
                f'[gen] e{pool.epoch} g{gstep}: n={summary["n"]} '
                f'clean={summary["n_candidates_parseable"]} 0grad={summary["zero_grad_frac"]:.2f} '
                f'R={summary["reward_mean"]:.2f}+-{summary["reward_std"]:.2f} '
                f'ws_acc={summary["avg_withskill_pass"]:.2f} chars={summary["skill_chars_mean"]:.0f} '
                f'bufA={len(hist_a)} bufB={len(sft_queue)} rounds={rounds}\n')
            if use_swan:
                m = _swan_metrics(summary, log)
                m['buffer/a_size'] = float(len(hist_a))
                m['buffer/b_size'] = float(len(sft_queue))
                swanlab.log(m, step=gstep)

            # --- distillation: once buffer A fills, drain it INCREMENTALLY in bounded
            # batches (--distill-batch) so a large buffer never stalls the loop for tens
            # of minutes; each iteration processes one batch, interleaved with GRPO.
            # 中文注释：增量分批蒸馏——buffer A 满后每轮只处理 --distill-batch 条，把一次性
            # 几十分钟阻塞摊成每轮几分钟小停顿；两段验证(见 distill_buffer)再砍验证算力。
            if checker and len(hist_a) >= args.distill_trigger:
                batch = hist_a[:args.distill_batch]
                hist_a = hist_a[args.distill_batch:]
                new_sft, distill_recs = distill_buffer(batch, skill_sampler, base_sampler, checker,
                                                       skill_dp, base_dp, args)
                for rec in distill_recs:  # 逐 entry 审计记录：rubric_diag + 候选 skill + 贪心解 + stage
                    rec['chunk'] = gstep
                    _write(distill_f, rec)
                distill_f.flush()
                for rec in new_sft:
                    _write(sft_f, rec)
                sft_f.flush()
                sft_queue.extend(new_sft)

            # --- SFT trigger: buffer B full → one SFT pass + eval ---
            did_eval = False
            if len(sft_queue) >= args.sft_trigger:
                sys.stderr.write(f'[sft] triggered at bufB={len(sft_queue)}\n')
                sft_samples = [{**s, 'advantage': float(args.sft_weight)} for s in sft_queue]
                sft_log = _train_step(skill_model, ref_model, ckpt, sft_samples, args)
                sft_rounds += 1  # 中文注释：SFT 用独立计数，不占用 GRPO 的 rounds 配额/save 节奏
                sft_log.update({'record_type': 'train_round', 'round': rounds, 'sft_round': sft_rounds,
                                'chunk': gstep, 'epoch': pool.epoch, 'kind': 'sft', 'ts': int(time.time())})
                _write(tlog, sft_log)
                tlog.flush()
                sft_queue = []
                skill_model.save(f'skill-v2-sft{sft_rounds}', output_dir=args.output_dir)  # 大改动后落盘
                if eval_records:  # 中文注释：SFT 后立即 eval，测灾难性遗忘/真提升（第 11.3/13.4 节）
                    _do_eval(gstep)
                    did_eval = True

            if eval_records and not did_eval and (gstep + 1) % args.eval_every == 0:
                _do_eval(gstep)
            gstep += 1

    if prediag_pool is not None:
        prediag_pool.shutdown(wait=False, cancel_futures=True)  # 丢弃未完成的后台预诊断
    eval_base_cache.close()
    skill_model.save('skill-v2-final', output_dir=args.output_dir)
    sys.stderr.write(f'[v2] done: {rounds} rounds over {gstep} chunks / {pool.epoch} epochs\n')


if __name__ == '__main__':
    main()
