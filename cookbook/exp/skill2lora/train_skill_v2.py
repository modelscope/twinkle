"""Simplified GRPO + buffer-distill training for the reflexion skill generator (v2).

Key differences from train_reflexion_skill.py:
- No view A/B split: all skill-gen is query-only (deployment form).
- No baseline rollout in training, no balance selection.
- thinking ON for the skill model: actor reasons in <think> then emits a distilled
  <skills> block; the <think> is stripped by _extract_skill and NEVER reaches the executor
  (executor only consumes <skills>), so it is not SEAM-style think leakage.
- Reward = parseable × correct (aligned with SEAM lpem: no terminated, no length penalty).
- Buffer A: adv=0 (all-fail) problems accumulate failure trajectories.
- Buffer B: batch rubric → regenerate skill → pass@k validate → SFT injection.
- SFT is event-driven: buffer B reaches threshold → one SFT pass → eval.

Launch:
    LLM_BACKUP_API_KEY=... python cookbook/exp/skill2lora/train_skill_v2.py \
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
from twinkle.data_format import SamplingParams, pack_user_data
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.patch.no_split_modules import NoSplitModulesPatch
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle_agentic.verifier import RubricVerifier
from twinkle_agentic.verifier.rubric_verifier import RubricItem

# 任务适配器（BigCodeBench）。code_task 刻意不 import 本模块，所以这里可以顶层 import。
import code_task

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
# Section A0 — task switch (math = DeepMath \boxed{}, code = BigCodeBench unittest)
# ===========================================================================
# 与 _ALIGN_MODE / _SKILL_STYLE 同型的模块级开关（由 set_task 设置，trainer 在任何 prompt
# 构造之前调用）。**math 分支逐字不变**，所以 E1-E16 的行为、判分与可复现性不受影响。
# 分派点一共 7 处：build_direct_prompt / build_skill_solve_prompt / _skillgen_prompt /
# _parse_seq(+_parse_many) / _answer_leaked / build_rubric_checker / _diagnose_entry。
# 换成 code 时 reference_answer 不再是数值，而是 code_task.payload_of() 的判分载荷
# （task_id / entry_point / test / code_prompt / doc_struct / canonical_solution）。
_TASK = 'math'                # 'math' | 'code'
_CODE_TEST_WORKERS = 24       # 单测线程池（子进程并行度）
_CODE_TEST_TIMEOUT = 60       # 单题单测墙钟上限（秒）


def set_task(task: str, test_workers: int = 24, test_timeout: int = 60) -> None:
    """Set the task family. MUST run before any prompt is built or any roll is judged."""
    global _TASK, _CODE_TEST_WORKERS, _CODE_TEST_TIMEOUT, _RUBRIC_VERSION
    if task not in ('math', 'code'):
        raise ValueError(f"task must be 'math' or 'code', got {task!r}")
    _TASK = task
    _CODE_TEST_WORKERS = max(1, int(test_workers))
    _CODE_TEST_TIMEOUT = max(5, int(test_timeout))
    # 判据表换了，rubric 缓存键必须跟着换（rubric_cache 动态读 v2._RUBRIC_VERSION）。
    _RUBRIC_VERSION = code_task.RUBRIC_VERSION if task == 'code' else _RUBRIC_VERSION_MATH


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


# SEAM-style answer format: executor emits <think>...</think><answer>[numeric only]</answer>.
# 中文注释：executor 答案格式对齐 SEAM——优先解析 <answer>…</answer>，回退到 \boxed{}
# （兼容旧轨迹/rubric 提示）。取最后一个 <answer>，容忍缺失闭合标签（截断时取到 EOS）。
_ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
_ANSWER_OPEN_RE = re.compile(r'<answer>(.*)', re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer, preferring SEAM's <answer>…</answer>, falling back to \\boxed{}."""
    if not text:
        return None
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip() or None
    # tolerate a truncated / unclosed <answer> tag (e.g. cut at token budget)
    m = _ANSWER_OPEN_RE.search(text)
    if m:
        return m.group(1).strip() or None
    return extract_boxed(text)


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


def _answer_leaked(skill: str, reference) -> bool:
    if not skill:
        return False
    if _TASK == 'code':
        # 代码域：leak = skill 里出现参考解答的实质代码行（与数学域一致，只做监控）
        return code_task.leaked(skill, reference)
    # Suffix guard: reject only a following DIGIT or a following '.<digit>' (decimal point),
    # NOT a sentence-ending '.'. Old '(?![\d.])' let leaks like "...= 675." slip through
    # because the trailing period satisfied the [\d.] class. 中文注释：尾断言只排除"后接数字"
    # 或"后接小数点+数字"，不排除句末句号，堵住 "答案." 这类泄漏漏检。
    for cand in {_numeric_value(reference), (str(reference).strip() or None)}:
        if cand and re.search(r'(?<![\d.])' + re.escape(cand) + r'(?!\d)(?!\.\d)', skill):
            return True
    return False


# ---- SEAM lpem-style numeric correctness (seam=整段 sanitize；v2=\boxed{} 锚定) ----
# 中文注释：复刻 SEAM lpem 判分——抽 <answer>/boxed/$...$/分数/首个数字，float 归一后纯数值精确匹配。
_SEAM_TAG_RE = re.compile(r'<\s*answer\s*>(.*?)<\s*/\s*answer\s*>', re.I | re.S)
_SEAM_BOX_RE = re.compile(r'(?:\\{1,2}\(|)\\{1,2}boxed\s*\{\s*([^}]*)\s*}(?:\)|)', re.S)
_SEAM_INLINE_RE = re.compile(r'\$([^$]+)\$|\\\(([^)]+)\\\)', re.S)
_SEAM_FRAC_RE = re.compile(r'(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)')
_SEAM_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _seam_norm(num: str) -> str:
    try:
        f = float(num)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return num.strip()


def _seam_sanitize(txt: str, dfrac_fix: bool = True) -> str:
    """Port of SEAM lpem.sanitize_math_answer + normalize_number_format.

    ``dfrac_fix=False`` gives BIT parity with the upstream function (which only rewrites the
    literal ``\\frac``); it is used by the ``align='seam'`` judge so that E13's acc is
    reproducible against SEAM's step_summary. See _parse_seq."""
    txt = (txt or '').strip()
    if (m := _SEAM_TAG_RE.search(txt)):
        txt = m.group(1).strip()
    elif (m := _SEAM_BOX_RE.search(txt)):
        txt = m.group(1).strip()
    elif (m := _SEAM_INLINE_RE.search(txt)):
        txt = (m.group(1) or m.group(2)).strip()
    # bugfix 2026-07-29：\dfrac/\tfrac/\cfrac 不被下行字面 \frac 正则匹配，曾致
    # \boxed{\dfrac{1}{2}} 落到 _SEAM_NUM_RE 抓首个数字 → pred='1'（分数答案题全判错；
    # 实测被标"错"的 boxed rolls 中 70-85% 实为正确，见 skill_quality_analysis.md 末章）。
    # 注：SEAM 上游没有这一步，seam 对齐口径下必须关掉（dfrac_fix=False）。
    if dfrac_fix:
        txt = txt.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac').replace(r'\cfrac', r'\frac')
    txt = re.sub(r'\\frac\s*\{\s*([^}]+?)\s*}\s*\{\s*([^}]+?)\s*}', r'\1/\2', txt)
    if (m := _SEAM_FRAC_RE.search(txt)):
        p, q = map(float, m.groups())
        if q:
            return _seam_norm(str(p / q))
    if (m := _SEAM_NUM_RE.search(txt)):
        return _seam_norm(m.group())
    return txt


# ===========================================================================
# Section B — sampling / parsing utilities
# ===========================================================================
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>')


def _clean_text(decoded: Optional[str]) -> str:
    return _SPECIAL_TOKEN_RE.sub('', decoded or '').rstrip()


def _extract_skill(text: str) -> Optional[str]:
    """Parse the skill block: <memory_item> in seam mode (SEAM format_pass parity), else <skills>."""
    if _ALIGN_MODE == 'seam':
        # ⭐ 整段搜、取首个匹配，不先剔掉 <think> —— 与 SEAM 两处实现逐字一致：
        #   lpem.format_pass：MEMORY_RE.search(resp)；fsdp_workers.py:836：_re.search(..., response_text)。
        #   旧实现只搜 </think> 之后，会把"只在 think 里写了 memory_item"的候选当成格式失败
        #   （reward=0），而 SEAM 那边算格式通过 —— 直接影响 format 率与组内 reward 方差。
        m = re.search(r'<memory_item>(.*?)</memory_item>', text or '', re.DOTALL | re.IGNORECASE)
        return (m.group(1).strip() or None) if m else None
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


def _parse_seq(seq, gold) -> Dict[str, Any]:
    if _TASK == 'code':
        return _parse_many([(seq, gold)])[0]
    text = _clean_text(getattr(seq, 'decoded', '') or '')
    if _ALIGN_MODE == 'seam':
        # ⭐ seam 对齐口径（2026-08-02 修正）= SEAM_JUDGE=answer，即 train_deepmath_paper.sh 的默认值：
        #   整段文本走 sanitize 级联 <answer> → \boxed → $..$/\(..\) → 分数 → **首个数字**。
        #   之前这里走的是 boxed-only（等价 SEAM_JUDGE=boxed），与 executor prompt 要求
        #   "<think>...</think><answer>...</answer>" 直接冲突 —— 换成原版 prompt 后 boxed-only
        #   会把几乎所有 rollout 判错。prompt 与判分必须成对切换，见 _SEAM_SOLVE_ADVISORY 注释。
        #   dfrac_fix=False 是为了与 lpem.sanitize_math_answer 逐字节一致。
        pred = _seam_sanitize(text, dfrac_fix=False) or None
        correct = bool(pred) and (pred == _seam_sanitize(str(gold), dfrac_fix=False))
        terminated = getattr(seq, 'stop_reason', None) != 'length'
        return {'pred': pred, 'correct': correct, 'terminated': terminated,
                'stop_reason': getattr(seq, 'stop_reason', None),
                'gen_tokens': len(getattr(seq, 'tokens', None) or []), 'text': text}
    # v2（E1-E12/E14-E21）：只从 \boxed{} 抽取，再走同一套数值归一（frac/inline/number）后精确匹配；
    # 不做 lpem 式"整段抓数字"贪婪回退，保证这些臂之间的 acc/lift 横向可比。extract_boxed 取最后一个
    # 配平的 \boxed{}、截断时不误取；没有则判错。
    raw = extract_boxed(text)
    pred = _seam_sanitize(raw) if raw else None
    correct = bool(pred) and (pred == _seam_sanitize(str(gold)))
    terminated = getattr(seq, 'stop_reason', None) != 'length'
    return {'pred': pred, 'correct': correct, 'terminated': terminated,
            'stop_reason': getattr(seq, 'stop_reason', None),
            'gen_tokens': len(getattr(seq, 'tokens', None) or []), 'text': text}


def _parse_many(pairs) -> List[Dict[str, Any]]:
    """批量判分入口。``pairs`` = [(seq_or_None, gold)]，返回同序 roll 列表。

    math 分支逐条 _parse_seq（与逐条调用 bit 一致）；code 分支必须批量 —— 判分要起子进程跑
    unittest（典型 1-3s），一个 chunk 有几百次判分，串行会比同 chunk 的 GPU 时间还长一个量级。
    所有 rollout 汇合点（process_chunk / run_greedy_eval / methods / eval_reflexion）都走这里。
    """
    if _TASK != 'code':
        return [(_parse_seq(s, g) if s is not None else _empty_roll()) for s, g in pairs]
    items = []
    for s, g in pairs:
        if s is None:
            items.append(None)
            continue
        items.append((_clean_text(getattr(s, 'decoded', '') or ''),
                      getattr(s, 'stop_reason', None),
                      len(getattr(s, 'tokens', None) or []), g))
    return code_task.judge_many(items, _CODE_TEST_WORKERS, _CODE_TEST_TIMEOUT)


def _first_seq(seqs):
    """rollout 列表 -> 首个 sequence 或 None（判分批量化后统一用它取 seq）。"""
    return seqs[0] if seqs else None


def _empty_roll():
    if _TASK == 'code':
        return code_task.empty_roll()
    return {'pred': '', 'correct': False, 'terminated': False,
            'stop_reason': 'empty', 'gen_tokens': 0, 'text': ''}


def _run_samples(sampler, prompts, num_samples, max_tokens, gen_dp,
                 temperature=None, top_p=None, top_k=None, logprobs=None):
    if not prompts:
        return []
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=GEN_TEMPERATURE if temperature is None else temperature,
        top_p=GEN_TOP_P if top_p is None else top_p,
        num_samples=num_samples, **({} if top_k is None else {'top_k': top_k}),
        **({} if logprobs is None else {'logprobs': logprobs}))
    padded = prompts
    if gen_dp > 1 and 0 < len(prompts) < gen_dp:
        padded = prompts + [copy.deepcopy(prompts[-1]) for _ in range(gen_dp - len(prompts))]
    responses = sampler.sample(padded, params)[:len(prompts)]
    return [list(r.sequences) if (r and r.sequences) else [] for r in responses]


def sampler_logprobs(seq):
    """从 ``SampledSequence.logprobs`` 抽出采样 token 自己的 logprob（逐 token 一个 float）。

    采样器返回的形状是 ``[[(token_id, logprob), ...], ...]``（logprobs=1 时每位只有一项，
    就是被采中的那个 token）。抽出来喂给 GRPOMetric 的 ``sampler_logps``，它会把这串值与
    训练 forward 算出的 logp 逐 token 对账 —— 序列一致时两边只差引擎精度。
    采样时 T=1/top_p=1/top_k=-1（skill-gen 的固定参数），logits 没经过任何处理，所以
    vLLM 的 processed_logprobs 就等于原始 logprob，与 trainer 的取值口径一致。
    """
    out = []
    for item in (getattr(seq, 'logprobs', None) or []):
        if not item:
            return []
        out.append(float(item[0][1]))
    return out


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


def _load_seam_parquet(path: str) -> List[Dict[str, Any]]:
    """Read a SEAM ``build_aops_dataset.py`` parquet (VERL RLHF schema) into twinkle records,
    PRESERVING file row order. ``problem <- extra_info.problem`` and
    ``reference_answer <- reward_model.ground_truth``. No shuffle/filter: the parquet is already
    SEAM's numeric-filtered, seed-42-shuffled, truncated split.
    中文注释：直读 SEAM parquet、保持文件顺序，用于让 twinkle 与 SEAM 输入同一批数据。"""
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        ei = r.get('extra_info') or {}
        rm = r.get('reward_model') or {}
        problem = (ei.get('problem') or '').strip()
        ref = rm.get('ground_truth')
        if not problem or ref is None:
            continue
        out.append({'data_id': f"seam:{ei.get('split', '')}:{ei.get('index', i)}",
                    'problem': problem, 'reference_answer': str(ref)})
    return out


def load_train_order_file(path: str) -> List[Dict[str, Any]]:
    """Read a fixed training ORDER file (jsonl) -> records in file order, duplicates kept.

    Why this exists: verl's dataloader shuffles (data.shuffle defaults True), so SEAM's step-k
    batch is NOT train.parquet[k*128:(k+1)*128]. Measured 2026-08-02: twinkle chunk 0 and SEAM
    step 1 drew from the SAME 5000-problem pool but shared only 1 of 128 problems, which alone
    put ~6-7 accuracy points between the two curves. This file is SEAM's REALIZED batch
    sequence, reverse-engineered from its rollout dump (40 steps x 128 problems, in order), so
    feeding it with ProblemPool(fixed_order=True) makes chunk k == SEAM step k+1 problem by
    problem. Generated by .tmp_analysis/mk_seam_train_order.py.
    Each line: {'data_id','problem','reference_answer'[,'level','seam_step']}.
    """
    out: List[Dict[str, Any]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            problem, ref = (d.get('problem') or '').strip(), d.get('reference_answer')
            if not problem or ref is None:
                continue
            rec = {'data_id': str(d.get('data_id', '')), 'problem': problem,
                   'reference_answer': str(ref)}
            if d.get('level') is not None:
                rec['level'] = d['level']
            out.append(rec)
    if not out:
        raise ValueError(f'--train-order-file {path} produced 0 records')
    return out


def _load_records(args):
    seam_dir = (getattr(args, 'seam_parquet_dir', '') or '').strip()
    if seam_dir:  # 直读 SEAM parquet：按文件顺序取 train，val 整份当 eval，跳过 load/numeric/shuffle/split
        tp, vp = os.path.join(seam_dir, 'train.parquet'), os.path.join(seam_dir, 'val.parquet')
        if not (os.path.exists(tp) and os.path.exists(vp)):
            raise FileNotFoundError(
                f'--seam-parquet-dir needs both train.parquet and val.parquet in {seam_dir}')
        pool = _load_seam_parquet(tp)              # already SEAM-shuffled + truncated, file order
        eval_records = _load_seam_parquet(vp)      # SEAM's exact val holdout
        eval_probs = {r['problem'] for r in eval_records}
        train_records = [r for r in pool if r['problem'] not in eval_probs]
        if args.n > 0:
            train_records = train_records[:args.n]
        if {r['problem'] for r in train_records} & eval_probs:
            raise ValueError('eval/train overlap detected in SEAM parquet')
        logger.info(f'[data] SEAM parquet: train={len(train_records)} eval={len(eval_records)} dir={seam_dir}')
        return train_records, eval_records
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
    """Dataloader-like batch sampler: epoch-wise seeded RandomSampler + drop_last.

    SEAM's verl dataloader uses a sampler and drop_last=True; this mirrors that behavior more
    closely than the old cursor loop that carried a short epoch tail into the next batch.

    ``fixed_order=True`` disables the permutation and walks ``records`` in file order. That is
    what --train-order-file needs: the order file already IS verl's realized batch sequence
    (reverse-engineered from SEAM's rollout dump), so any reshuffle here would destroy it.
    """
    def __init__(self, records, seed, fixed_order=False):
        self._records = list(records)
        self._seed, self._cursor, self.epoch = seed, 0, 0
        self._fixed_order = bool(fixed_order)
        self._order: List[int] = []
        self._reset_epoch()

    def _reset_epoch(self):
        if self._fixed_order:
            self._order = list(range(len(self._records)))
        else:
            rng = np.random.RandomState(self._seed + self.epoch)
            self._order = list(rng.permutation(len(self._records)))
        self._cursor = 0

    def draw(self, k):
        if k > len(self._records):
            raise ValueError(f'batch size {k} exceeds dataset size {len(self._records)}')
        if self._cursor + k > len(self._order):
            self.epoch += 1
            self._reset_epoch()
        idx = self._order[self._cursor:self._cursor + k]
        self._cursor += k
        return [self._records[i] for i in idx]


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
- If segment was cut off (no final <answer> reached), mark the output-format criterion as FAIL.
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

# 判据按「错误类型」组织，但文本一律写成正向陈述 —— 全流程的语义是 PASS=没问题 /
# FAIL=该类错误存在（见 _format_diagnosis 与 gate），写成否定句会把 PASS/FAIL 反过来。
_MATH_RUBRIC = [
    # 1. 代数计算错误
    ('Arithmetic and algebraic manipulations are carried out correctly', True),
    # 2. 公式定理使用错误
    ('Formulas and theorems are invoked correctly and their preconditions hold', False),
    # 3. 起始方法论错误
    ('The initial approach is viable for this problem rather than a dead end', False),
    # 4. 题目目标分析错误
    ('The attempt correctly identifies what the problem actually asks for', False),
    # 5. 输出格式错误
    ('The attempt reaches a final answer in the required output format', False),
    # 6. 对计算过程反复犹豫
    ('The attempt commits to its computation instead of repeatedly second-guessing it', False),
    # 7. 构成自相矛盾
    ('The attempt stays internally consistent and never contradicts its own results', False),
]
# 版本号进 rubric 缓存键（GlobalRubricCache._key）：判据一改，旧诊断必须失效，
# 否则 rubric_cache_global.jsonl 里按 data_id 存的旧taxonomy诊断会被当成新判据的结果返回。
_RUBRIC_VERSION_MATH = 'rubric_v6_error_taxonomy'
_RUBRIC_VERSION = _RUBRIC_VERSION_MATH   # set_task('code') 会换成 code_task.RUBRIC_VERSION


class _RftRubricVerifier(RubricVerifier):
    def _diagnose_trajectory(self, query, rubric_block, segment_text):
        return {'messages': [
            {'role': 'system', 'content': _RFT_DIAG_SYSTEM},
            {'role': 'user', 'content': _RFT_DIAG_USER.format(
                query=query, rubric=rubric_block, segment=segment_text)}]}


class _CodeRubricVerifier(RubricVerifier):
    """代码域 judge：判据 = code_task.CODE_RUBRIC，且 segment 里带**单测真实报错**。"""

    def _diagnose_trajectory(self, query, rubric_block, segment_text):
        return {'messages': [
            {'role': 'system', 'content': code_task.DIAG_SYSTEM},
            {'role': 'user', 'content': code_task.DIAG_USER.format(
                query=query, rubric=rubric_block, segment=segment_text)}]}


def build_rubric_checker():
    if not (os.environ.get('LLM_BACKUP_API_KEY') or os.environ.get('LLM_BACKUP_BASE_URL')
            or os.environ.get('OPENAI_API_KEY')):
        return None
    if _TASK == 'code':
        return _CodeRubricVerifier(
            fixed_rubric=[RubricItem(t, is_hard=h) for t, h in code_task.CODE_RUBRIC], gate=True)
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
    if _TASK == 'code':
        # 代码域：query 里补上题面声明的硬约定（签名/返回/异常/示例，不含参考解答），
        # segment 由调用方（_rubric_entry）拼成"提交的代码 + 单测真实报错"。
        query = code_task.diag_query(entry['problem'], entry['reference_answer'])
        seg = {'messages': [{'role': 'user', 'content': query},
                            {'role': 'assistant', 'content': entry['fail_segment']}]}
        try:
            return _format_diagnosis(checker.diagnose(seg, query=query))
        except Exception as exc:
            logger.warning(f'[rubric] diagnose error: {exc}')
            return None
    seg_text = entry['fail_segment']
    if entry.get('fail_stop_reason') == 'length':
        seg_text += ('\n\n[Process note: this attempt was cut off at the token budget '
                     'and never produced a final <answer>.]')
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

# ---- Skill-gen system prompt (query-only; used in v2 mode) ----
# 中文注释：skillmodel 系统提示词（仅 v2 模式；seam 模式改用 _SEAM_EXPERIENCE_PROMPT）。
# 方案1：thinking 开启。让 skill 模型在 <think> 里“先把本题实际解一遍、想清楚”，再在 <skills> 里
# 只写抽象出来的“通用方法论”（不含本题任何具体数字/中间结果/答案）。<think> 会被 _extract_skill
# 用 rfind('</think>') 砍掉、绝不流给 executor（避免 SEAM 那种 think 泄漏），executor 只吃 <skills>。
# 之所以要开 thinking：nothinking 下模型无处安放解题过程，只能把“完整解答+答案”直接写进 <skills>
# （实测 <skills> 前置分析长度=0、且常把答案算出来写进块内），等于换标签的泄漏且质量差。开 thinking 后
# “先解题、再提炼”两步显式分离，<skills> 才可能是真正可迁移、不代入本题数值的方法论。
# skill_model/ref_model/skill_sampler 三者 enable_thinking 必须一致，否则训练轨迹 token 布局与采样对不上。
SKILL_GEN_SYSTEM = """\
You are a skill-generation model. Your <skills> block will be fed to a SEPARATE downstream executor model that must solve the problem on its own. The executor will NOT see your private reasoning — it only sees what is inside <skills>...</skills>.

First, think privately: actually work the problem out in your head to make sure you understand it, then step back and abstract WHAT MAKES THIS TYPE OF PROBLEM SOLVABLE into transferable methodology.

Then write the <skills> block following these rules:
- Give general, transferable solving techniques for this TYPE of problem: the key concepts/theorems it relies on, the recommended strategy and steps, and the common pitfalls to avoid — plus a brief reason for each piece of advice so the executor understands why.
- Write it as one coherent analysis narrative (not a bullet list): first name what the problem is essentially asking, then walk through how to approach it, blending concepts, steps, pitfalls and reasons into a single connected story.
- CRITICAL: Do NOT solve the problem for the executor. Do NOT reveal or compute the final answer, and do NOT substitute the problem's specific given numbers into the steps or state any intermediate numeric results. Leave ALL concrete numbers for the executor to compute on its own. If you catch yourself writing a specific number from the problem, replace it with a description of the quantity instead.
- Keep it concise: aim for roughly one focused paragraph.
- End the block with this exact sentence: "Avoid re-checking loops; box a bare number as soon as it is computed."

Put ONLY the methodology inside <skills></skills>.

Example:
<skills>
This problem is essentially asking for the units (last) digit of an integer raised to a high power; first get clear on what the problem is asking before deciding where to start. Since only the last digit matters, you should first look only at the units digit of the base, because the units digit of an integer power is determined solely by the units digit of the base and the higher digits do not affect the result — so at this step be careful not to expand or compute the whole large number, which is both unnecessary and error-prone. Next, repeatedly multiply this units digit by itself and record the units digit each time, until it starts to repeat, thereby obtaining its cycle period. The part about "determining the period length" is important here: be careful not to count one term too many or too few, otherwise all the later positioning will be off. Finally, take the given exponent modulo the period length and land on the corresponding term within the period; here pay special attention that when the remainder is 0 it corresponds to the last term of the period rather than the first. Overall, I summarize the approach for this kind of problem as "first recognize that it asks for the units digit of a power, then fix on the units digit to find the cycle period, and finally use the exponent modulo to locate the term", while leaving the concrete numbers for the downstream solver to substitute and compute on its own. Avoid re-checking loops; box a bare number as soon as it is computed.
</skills>
"""

# ---- Skill 文体消融（--skill-style）----
# 中文注释：探针实验（CONCLUSIONS_config/reflexion.md）验证的两种高性价比文体。
# 关键约束：同一文体在主链路（query-only 预判）与 buffer B regen（rubric 诊断条件）下
# 输出格式必须一致（toy=迷你题示范+迁移句；pitfall=WARNING/INSTEAD/纪律句），
# 否则 GRPO 与 SFT 样本分布不一致无法联合训练。
# toy 主链路：探针 P3_toy 原文（异数字玩具题，天然 answer-free）。
SKILL_GEN_TOY = """\
You are a skill-generation model. A separate executor model will solve the problem; it only sees your <skills> block.

First think privately and identify the core technique this problem needs. Then, inside <skills></skills>, do exactly one thing: invent a MINIATURE problem of the same type with DIFFERENT and much smaller numbers, and solve that miniature completely in at most 5 short lines, making the key trick explicit. Finish with one transfer sentence: "Your problem has the same shape - repeat these steps with its own numbers, then box a bare number."

Hard rules: never mention or use any number that appears in the original problem; never state the original problem's answer; keep the whole block under 100 words.
"""

# pitfall 主链路：探针 P5_pitfall 原文（预判最可能错误走向并拦截）。
SKILL_GEN_PITFALL = """\
You are a skill-generation model. A separate executor model will solve the problem; it only sees your <skills> block.

First think privately: solve the problem in your head AND identify the single most likely way a solver goes wrong on this type (a tempting but wrong turn, an off-by-one, a wasteful brute-force, a wrong branch). Then, inside <skills></skills>, write under 90 words:
- WARNING: name that most likely mistake concretely and say why it is wrong.
- INSTEAD: one or two sentences pointing to the correct turn (technique name + where to apply it), without solving the problem or revealing any numeric result.
- End with: "Avoid re-checking loops; box a bare number as soon as it is computed."
"""

# freeform 主链路（2026-08-01 用户拍板）：不锁死 narrative/pitfall/toy 固定文体，而是给模型一份
# "招式菜单"，让它按题自选最有用的形态（可组合、可极简）。设计目标是让 T=1.0×8 的候选自然铺开
# 到不同形态（分析 / 概念 / 预判纠错 / 迷你示范 / 直白执行指令，甚至 "let's think step by step"），
# 由 GRPO/BNPO 组内择优。依据：固定 hint 消融（good_skill_hard_fail/fixed_hint_probe.py）显示 hint
# 的"内容语义"贡献≈0、增益几乎全来自"存在一个 skill 块 + 催答案收尾"（A7_budget 最高 +0.16、
# exec_answered 3.95σ），所以放开文体、把"催收尾"作为可选招式之一，看模型能否自选出更优组合。
# 硬约束仍与 narrative 一致：只输出 <skills> 块、不解题、不代入本题数值、不给最终答案。
SKILL_GEN_FREEFORM = """\
You are a skill-generation model. Your <skills> block will be fed to a SEPARATE downstream executor model that must solve the problem on its own. The executor will NOT see your private reasoning — it only sees what is inside <skills>...</skills>.

First, think privately: actually work the problem out in your head until you understand what really makes it solvable, then decide what ONE kind of help would most raise a fresh solver's chance on THIS specific problem.

There is NO fixed format and no required style. Different problems are helped by different things — pick whatever you judge most useful here. Any of the following is allowed (the list is not exhaustive, and you may blend a couple if they genuinely help):
- a short transferable analysis of what this TYPE of problem is really asking and the recommended approach;
- naming the key concept / theorem / trick to reach for;
- a WARNING about the single most likely wrong turn on this type, and the correct move instead;
- a tiny worked example of the SAME type using DIFFERENT, smaller numbers (never the problem's own);
- a blunt execution directive that keeps the solver on track (e.g. "commit to one method and don't keep second-guessing", "let's think step by step", or "box a bare number as soon as it is computed");
- or plain, nothing-fancy encouragement if that is honestly all this problem needs.
- Any other freeform skill you can imagine to try on this query

Choose the form that fits THIS problem; do not pad. If one sharp sentence is the best help, give only that sentence; if a short focused paragraph is warranted, keep it tight. Being genuinely useful matters far more than being long or elaborate.

Hard rules (always apply, whatever form you pick):
- Do NOT solve the problem for the executor. Do NOT reveal or compute the final answer, and do NOT substitute the problem's specific given numbers or state any intermediate numeric result — leave ALL concrete numbers for the executor to compute.
- Put ONLY your chosen help inside <skills></skills>, and nothing else.

Whatever form you choose, it MUST be wrapped in a single <skills></skills> block with a proper closing tag. For example, a rich form:
<skills>
This is a modular-arithmetic problem: reduce each factor modulo the given modulus before multiplying, and never expand the full product — that is the whole trick. Commit to that reduction and don't second-guess it, then box a bare number as soon as it is computed.
</skills>
or, when the problem only needs a nudge, a minimal form is equally valid:
<skills>
Let's think step by step, and box a bare number as soon as it is computed.
</skills>
"""

_SKILL_STYLE = 'narrative'  # 'narrative' | 'toy' | 'pitfall' | 'freeform'；由 main() 依据 --skill-style 设置

# ---- Executor prompt (with skill injection) ----
# 中文注释：executor 提示词。答案格式已统一为 \boxed{}（人工拍板，2026-07-27）：seam/v2 两模式的
# 格式说明与判分口径完全一致，保证 E13(seam) 与 E1-E12 横向可比；prompt 结构差异（seam 嵌套/
# system+user vs v2 单 user）作为方案级差异保留。_ANSWER_FORMAT(<answer> 版) 已弃用。
_ANSWER_FORMAT = ('Present your reasoning and answer in the following format:\n'
                  '<think> Content of Thinking</think><answer>[Final numeric result only]</answer>')
# 统一执行器答案格式：把最终数值放进 \boxed{}；判分对应走 extract_boxed。
_ANSWER_FORMAT_V2 = ('Present your reasoning, then put ONLY the final numeric result inside '
                     '\\boxed{}. For example: \\boxed{42}.')
DIRECT_SYSTEM = (
    'You are an expert competition mathematician. Be concise and accurate. '
    + _ANSWER_FORMAT_V2)
_SKILL_SOLVE_PREFIX = (
    'You are an expert competition mathematician. Be concise and accurate.\n\n'
    'Before you start, keep these reminders in mind to avoid common mistakes on this '
    'type of problem:\n')
_SKILL_SOLVE_SUFFIX = ('\nApply them where relevant, but rely on your own reasoning to reach the answer.\n\n'
                       + _ANSWER_FORMAT)


# ---- Unified SEAM-alignment mode (toggle: prompt + skill-gen format only) ----
# 中文注释：SEAM 对齐开关（由 --align-mode 控制）。
#   1) executor 输入：seam=复刻 SEAM reward worker 的非空 experience 路径：
#      prompt_text + actor 原始 response_text（保留 <think> 与 <memory_item>）+ 解题 advisory。
#      v2=干净单 user turn（题目 + “Skill hint”advisory + 答案格式；空 skill 回退 direct）。
#   2) skill-gen prompt：seam=SEAM EXPERIENCE_PROMPT（单 user turn，输出 <memory_item>）；v2=SKILL_GEN_SYSTEM（<skills>）。
# 注：reward 判分（_parse_seq）与 loss 聚合（set_loss）已统一为 SEAM 口径（lpem 纯数值匹配 + token-mean）。
# 关于强制 <think>：SEAM 原版在 executor 末尾裸拼 "<think>\n"。v2 走 messages+模板路径，base_sampler
#   enable_thinking=True 时 Qwen3 生成起点默认进入 thinking；这里不改共享 sampler 的 assistant 前缀注入。
_ALIGN_MODE = 'v2'  # 'v2' | 'seam'；由 main() 依据 --align-mode 设置

# ===========================================================================================
# SEAM 对齐定案（2026-08-02，E13 = --align-mode seam）
# 目的：把追查过程固化下来，避免以后重复审查同样的地方。
#
# 【已逐条/逐字节校验对齐的 12 项】
#   1. 训练喂题序列 —— verl dataloader data.shuffle 默认 True，SEAM 的 step k 不是
#      train.parquet[k*128:(k+1)*128]。实测 twinkle chunk0 与 SEAM step1 的 128 题交集只有
#      1 题，单这一项就拉开 6-7 个 acc 点。已从 SEAM rollout dump 反推真实序列，用
#      --train-order-file + ProblemPool(fixed_order=True) 钉死（5120 行，missing=0）。
#   2. actor prompt / executor prompt（with-skill 与 baseline 两路）—— 含 chat template 与
#      Qwen3 的空 <think>\n\n</think>\n\n，逐字节 True。
#   3. templates/slove_qwen.txt advisory —— 逐字节 True（含第一行末尾那个空格、EOF 无换行）。
#   4. 判分口径 —— SEAM_JUDGE=answer 的整段级联（<answer> -> \boxed -> $..$ -> 分数 -> 首个
#      数字）。用 SEAM dump replay 20480/20480 与其 acc 逐条一致；旧的 boxed-only 判分对
#      SEAM 自己的输出只给 acc≈0.05。
#   5. format 抽取 —— 整段搜首个 <memory_item>，20480/20480 与 SEAM format 一致。
#   6. loss 聚合 —— BNPO token_mean_scope='micro'。verl 侧证据：dp_actor.py 在每个 micro 内部
#      masked_mean，再乘 loss_scale_factor = n_micro / ppo_mini_batch_size(=40/卡)，满 micro 时
#      恰好等权；即 verl 也是「每 5 条组等权」而不是全局 token-mean。
#   7. 聚合粒度 —— verl 的等权单元是 ppo_micro_batch_size_per_gpu=5 条。TRAIN_FSDP=2 腾出显存后
#      train_micro_batch=5、TRAIN_DP=1，每 optimizer step 32 个「5 条组」等权，与 verl 逐组一致。
#   8. 每 step 的 optimizer step 数 —— verl 的 ppo_mini_batch_size 经 fsdp_workers.py:198-199
#      归一化为 per-GPU 40，每卡 256 条 => 7 个 optimizer step；twinkle mini=160、n=1024 也是 7。
#   9. PPO 语义 —— clip_ratio 0.2 / clip_ratio_c 3.0 / ppo_epochs 1 / entropy_coeff 0 /
#      use_kl_loss+low_var_kl+coef 0.001 / clip_grad 1.0 / AdamW wd 0.01 全部一致；old_logps 在
#      任何更新前统一预计算（multi_step=True），所以 7 个 mini-step 的 PPO clip 真实有约束力。
#  10. advantage —— A=(r-mean)/std（unbiased std，norm_adv_by_std_in_grpo 语义）；drop_zero_adv=False，
#      零 adv 样本照样占分母。实测 |adv| 均值与 SEAM adv_absmax_mean 逐 step 吻合（0.20-0.28）。
#  11. 数值精度 —— fp32 master + bf16 计算（torch_dtype='float32' + mixed_precision='bf16'），
#      与 verl FSDP 的 fp32 master + param_dtype=bf16 等价；lr 恒定 1e-6 无 warmup/decay。
#  12. 采样超参 —— actor temperature 1.0 / top_p 1.0 / top_k 禁用；executor greedy + nothink；
#      8192/8192 预算，executor prompt 实测 ≤8522 < 10752 不触发左截断。
#
# 【残余偏离已定案并修复：截断 rollout 的思考段被 chat template 补了一个空思考块】
#   现象：两边起点几乎同一点（actor 输出 3851 vs 3842 tokens、format 0.874 vs 0.883），但
#   twinkle 6 个 chunk 就走完 SEAM 40 步的长度降幅（3851->3177 vs 3842->3239，均 -16~17%），
#   于是 format 0.874->0.957 而 SEAM 40 步才 0.883->0.943。acc/reward/lift/zero_grad/grp_std/
#   n_train 均已在 SEAM 自身 step 间噪声(sd≈0.03)内逐 step 对应。
#   定位方法（.tmp_analysis/probe_grad.py + probe_twinkle.py）：第 1 个 optimizer step 时 ratio≡1、
#   KL≡0，梯度完全由 (advantage, logits) 决定，所以拿 SEAM step1 dump 的同一批样本做一次
#   forward+backward 就能逐层对。结果：
#     * 环境差异排除 —— 同一份参考实现在 torch 2.11/tf 5.12 与 SEAM 的 torch 2.7/tf 4.53 下
#       tokenization md5 相同、loss 差 0.24%、grad_norm 差 0.9%。
#     * 逐组定位 —— 只有「非零 advantage 的样本恰好是截断样本」的 micro 组对不上（g4
#       0.5927 -> 2.0552，3.47 倍），不含截断样本的组差 <1%。
#     * 真因 —— 撞 8192 上限的 rollout 思考段没有 </think>（160 条里 18 条截断，其中 16 条
#       无 </think>，且无 </think> 的样本 100% 是截断样本），Template 的 pre-pipeline
#       _to_standard_reasoning_content 拆不出思考段，只能置 reasoning_content=''，Qwen3 模板于是
#       渲染成空思考块 + 原文，原文自带的 <think> 变成紧跟在闭合标签之后的 token —— 那个
#       位置模型输出 <think> 的概率≈0，logp 极低、梯度极大。
#     * 因果验证 —— 在参考实现里复刻这一编码（probe_grad.py --emulate-think-bug）后，逐组
#       grad_norm 从 [0.9999, 0.5733, 0.5927, 0.5307] 变成 [1.2520, 0.5672, 2.0539, 0.5310]，
#       与 twinkle 实测 [1.2573, 0.5787, 2.0552, 0.5338] 逐组重合。
#     * 修复验证 —— Template._fix_unfinished_last_round 上线后 g4 2.0552 -> 0.5922
#       （ref 0.5927），整批 160 条 32 组 0.3505 -> 0.1151（ref 0.1149，差 0.2%）。
#   为何只影响长度/format、不影响 acc/reward：截断样本 reward=0（没闭合 <memory_item>）、
#   advantage 为负，那个巨大梯度全压在“长输出”这一模式上；而截断率会随训练自我消解
#   （chunk0 11.6% -> chunk6 1.2%），所以偏差在前几步最猛、之后自行消失 —— 正好解释了
#   「twinkle 6 步冲完然后平稳 vs SEAM 40 步缓慢上升」。
#   另注：grad_norm 不可直接比（twinkle 只记 7 个 optimizer step 中的最后一个，verl 记均值）。
# ===========================================================================================

_SEAM_EXPERIENCE_PROMPT = (
    'You are a problem-solving guidance model. Read the math problem below and '
    'distill a concise, reusable piece of solving experience that will help a '
    'SEPARATE solver model reach the correct answer.\n'
    'Rules:\n'
    '- Do NOT solve the problem and do NOT reveal or compute the final answer.\n'
    '- State the key concepts/theorems, the recommended strategy/steps, and the '
    'common pitfalls to avoid.\n'
    '- Output ONLY the experience, wrapped EXACTLY as '
    '<memory_item> ... </memory_item>.\n\n'
    'Problem:\n{problem}')
# 逐字节复刻 SEAM templates/slove_qwen.txt（注意第一行末尾那个空格，EOF 无换行）。
_SEAM_SOLVE_ADVISORY = (
    'The above is a Q&A dialogue between a user and a problem-solving guidance model. \n'
    'Treat the output of the guidance model as advisory context to solve the math problem: '
    'prefer using its techniques when they fit, but you may use alternative correct methods '
    'if they are more efficient or clearer. If you diverge from the advisory context, briefly '
    'explain why. Be concise and accurate.\n'
    + _ANSWER_FORMAT)
# seam 模式的 baseline/空-skill 回退 system。必须用 _ANSWER_FORMAT（<think>/<answer>）而不是
# _ANSWER_FORMAT_V2（\boxed{}）：SEAM 的 fsdp_workers.py:850-856 在 SEAM_JUDGE=answer（默认）下
# 就是这个串，且 executor 关 thinking 时 Qwen3 会自动补一个空 <think></think>，与"请输出 <think>"
# 的要求撞在一起，抽答案更容易失败 —— 这正是 SEAM baseline 只有 0.57 的原因。用 boxed 会把
# baseline 抬到 0.72、给定可解析 skill 的 acc 抬到 0.923（SEAM 0.865），曲线水平就对不上。
# 判分侧无需改动：_seam_sanitize 优先匹配 <answer>、其次 boxed，两种格式都吃。
DIRECT_SYSTEM_SEAM = (
    'You are an expert competition mathematician. Be concise and accurate. '
    + _ANSWER_FORMAT)


def build_skill_solve_prompt_seam(problem, skill, raw_response=None, resp_terminated=True):
    """SEAM executor prompt. Non-empty skills use the actor's raw response_text, preserving
    actor <think> exactly as SEAM's reward worker does: prompt_text + response_text + grm.
    If raw_response is missing, fall back to reconstructing a minimal <memory_item> response."""
    skill = (skill or '').strip()
    response_text = (raw_response or '').strip()
    prompt_text = ('<|im_start|>user\n'
                   + _SEAM_EXPERIENCE_PROMPT.format(problem=problem)
                   + '<|im_end|>\n<|im_start|>assistant\n')
    if not response_text:
        response_text = f'<memory_item>{skill}</memory_item>'
    elif resp_terminated:
        # SEAM 的 response_text 是 skip_special_tokens=False 解码的（fsdp_workers.py:828），正常终止的
        # rollout 末尾带着 EOS，即 "</memory_item><|im_end|>"；twinkle 的 _clean_text 把 <|...|> 全剔了。
        # 差这一个 token 也会改变 executor 的贪心解码，补回来。截断的 rollout（stop=length）
        # SEAM 那边也没有 EOS，所以不补。
        response_text = response_text + '<|im_end|>'
    content = prompt_text + response_text + '\n' + _SEAM_SOLVE_ADVISORY
    return {'messages': [{'role': 'user', 'content': content}]}


def build_direct_prompt(problem):
    if _TASK == 'code':
        return code_task.direct_prompt(problem)
    if _ALIGN_MODE == 'seam':
        # seam 基线逐字复刻 SEAM fsdp_workers.py:850-858（system + user，<think>/<answer> 格式）
        return {'messages': [{'role': 'system', 'content': DIRECT_SYSTEM_SEAM},
                             {'role': 'user', 'content': problem}]}
    # v2：英文 executor 基线——与带 skill 版同格式，仅去掉“技巧提示”部分
    content = f'The problem you need to solve:\n{problem}\n\n' + _ANSWER_FORMAT_V2
    return {'messages': [{'role': 'user', 'content': content}]}


def build_skill_solve_prompt(problem, skill, raw_response=None, resp_terminated=True):
    skill = (skill or '').strip()
    if _TASK == 'code':
        # 空 skill -> 干净 direct（与数学分支同规则，见下方注释）
        return code_task.skill_solve_prompt(problem, skill)
    if not skill:
        # 空 skill → 干净 direct。训练侧根本不会用空 skill 走 executor（process_chunk 只对非空 flat 跑，
        # 空候选直接 reward=0），故此分支仅影响 eval 口径——让空 skill 题 withskill==baseline、对 lift 贡献 0，
        # 去掉空壳嵌套的框架水分，指标更干净。
        # （seam 模式下这正好等于 SEAM fsdp_workers.py:846-858 的 else 分支。）
        return build_direct_prompt(problem)
    if _ALIGN_MODE == 'seam':
        # 非空 skill 走 SEAM 原始 reward worker 路径：executor 可见 actor 完整 response_text（含 <think>）。
        return build_skill_solve_prompt_seam(problem, skill, raw_response=raw_response,
                                            resp_terminated=resp_terminated)
    # v2：英文 executor——题目 + 技巧提示(skill 作为 advisory) + 答案格式，单 user turn
    content = (f'The problem you need to solve:\n{problem}\n\n'
               'Skill hint:\nFor this problem, a skill-generation model has analyzed it and '
               'provided some advisory skills:\n'
               f'{skill}\n'
               'Prefer using its techniques when they fit, but if you have a more efficient or '
               'clearer correct method, you may use it. If you diverge from this advice, briefly '
               'explain why. Be concise and accurate.\n'
               + _ANSWER_FORMAT_V2)
    return {'messages': [{'role': 'user', 'content': content}]}


# ---- Rubric-guided regeneration prompt (buffer B distillation) ----
# 中文注释：蒸馏重生成提示词。给旧 skill + rubric 诊断，要求产出改进后的 skill（<skills>）。
# 输出要求第一人称自持句式、不指向外部上下文（防幻觉），含一个连贯叙述式示例。
REGEN_SYSTEM = """\
You are a skill-generation model. Your skill will be fed to a downstream executor model to help it solve the problem better.
You may give general, transferable solving techniques, together with why you give this advice, so the downstream model can follow it. Do NOT reveal or compute the final answer.

You previously generated a skill, but that skill did not help the model. The executor's actual solving process has now been analyzed. You need to regenerate the skill based on your previous skill and the mistakes the model actually made, so as to help the model solve this problem.

Your steps:
1. Re-read and understand the original problem.
2. Tell a coherent analysis story for this problem as one flowing narrative: first identify what it is essentially asking, then walk through how to approach it, naturally weaving together the solving points that were already correct last time, the pitfalls that actually tripped up the solving process and how to avoid them, and your reasoning for why you give this advice, blended into a single connected story, and leave the concrete numbers for the downstream solver to compute.
3. End the block with this exact sentence: "Avoid re-checking loops; box a bare number as soon as it is computed."
4. Put the above inside <skills></skills>.

Output requirement: Write your judgments and pitfall reminders about this problem directly in the first person (e.g. "I think this step tends to ...", "A common mistake is ..., so you need to ..."), and phrase the issues you find as self-contained, general techniques. Do NOT use phrasings that point to external context such as "according to the given analysis/hints" or "the previous skill" — the downstream executor cannot see that context, and such phrasings will cause hallucination.

Example:
<skills>
This problem is essentially asking "how many arrangements satisfy the given constraints", which is a counting problem; first get clear on "what exactly is being counted" before deciding whether to use permutations or combinations. Since it is counting, you should first clearly define the objects being counted and the constraints, and judge whether the elements are distinguishable and whether order matters, because this directly determines whether you will need to divide out duplicates later. Next, first compute a total as if things were "ordered/distinguishable", then find which seemingly different arrangements actually correspond to the same configuration. The part about "recognizing symmetry and determining the duplication factor" is important here: I think the step most likely to go wrong in this problem is ignoring symmetry and treating essentially identical configurations as different, which makes the result too large; I think it is also easy to directly miss the "divide by the duplication factor" step — as long as the choices can be interchanged, you must divide out duplicates, otherwise you overcount. Finally, divide the total by the duplication factor to get the truly non-duplicated count; here pay special attention not to jump straight to permutation/combination formulas, but first think clearly about whether the elements are distinguishable and then decide whether to divide out duplicates. Overall, I summarize the approach for this kind of problem as "first recognize that it is a counting problem and judge whether the elements are distinguishable, then compute the total, recognize symmetry and remove duplicates", because I judge that the loss points for such problems almost all concentrate on overcounting; while leaving the concrete numbers for the downstream solver to substitute and compute on its own. Avoid re-checking loops; box a bare number as soon as it is computed.
</skills>"""

REGEN_USER = """\
Original problem:
{problem}

Previously generated skill (did not help the executor):
{orig_skill}

Analysis of the executor's actual solving process:
{rubric_diag}

Now rewrite the improved <skills> guidance:"""

# 中文注释：buffer B regen 的 toy/pitfall 文体版（与主链路同文体，保证训练分布一致）。
# 源自 reflexion 探针 D3_toyfix / D1_needle（diag-only 口径），另加防指涉硬规则
# （不许写 "according to the diagnosis / the previous skill"，executor 看不到这些上下文）。
REGEN_TOY_SYSTEM = """\
You are a skill-generation model. A separate executor model previously FAILED this problem even with your earlier skill. You will see that earlier skill and an expert rubric diagnosis of the failure. The executor will retry seeing ONLY your new <skills> block.

First think privately: from the diagnosis, identify the ONE technique the executor got wrong. Then, inside <skills></skills>, do exactly this (under 110 words):
1. Invent a MINIATURE problem exercising that same technique with DIFFERENT, much smaller numbers, and solve the miniature completely in at most 5 short lines, making the correct move (the one the failed attempt missed) explicit.
2. One transfer sentence: "Your problem has the same shape - repeat these steps with its own numbers, then box a bare number."
Hard rules: never use any number from the original problem; never state its answer; the block must be self-contained - never reference "the diagnosis", "the previous skill" or any context the executor cannot see.
"""

REGEN_PITFALL_SYSTEM = """\
You are a skill-generation model. A separate executor model previously FAILED this problem even with your earlier skill. You will see that earlier skill and an expert rubric diagnosis of the failure. The executor will retry seeing ONLY your new <skills> block.

First think privately: from the diagnosis, pinpoint the decisive error. Then, inside <skills></skills>, write under 90 words:
- WARNING: the decisive mistake, stated concretely for THIS problem in self-contained first person (e.g. "I think the step most likely to go wrong is ...").
- INSTEAD: one or two sentences pointing to the correct turn (technique name + where to apply it), without solving the problem or revealing any numeric result.
- End with: "Avoid re-checking loops; box a bare number as soon as it is computed."
Hard rules: the block must be self-contained - never reference "the diagnosis" or "the previous skill"; the executor cannot see them.
"""

# freeform 的 regen 版（buffer B 蒸馏用）。bnpo/view-B 臂不会走 regen，此处仅为分派完整性与
# 未来 view-A + freeform 组合预留；同样放开形态、保留"自持、不指涉外部上下文、不泄漏"硬规则。
REGEN_FREEFORM_SYSTEM = """\
You are a skill-generation model. A separate executor model previously FAILED this problem even with your earlier skill. You will see that earlier skill and an expert rubric diagnosis of the failure. The executor will retry seeing ONLY your new <skills> block.

First think privately: from the diagnosis, pinpoint the ONE thing that actually went wrong. Then choose whatever form of help would best fix it for THIS problem — there is no fixed format. It may be a short transferable analysis, the key concept to reach for, a WARNING naming the decisive mistake plus the correct move instead, a tiny worked example with DIFFERENT smaller numbers, or a blunt execution directive (e.g. "box a bare number as soon as it is computed"). Blend a couple only if it genuinely helps, and do not pad.

Hard rules:
- Do NOT solve the problem or reveal/compute the final answer, and do NOT substitute the problem's own numbers.
- The block must be self-contained — never reference "the diagnosis", "the previous skill", or any context the executor cannot see, or it will hallucinate.
- Put ONLY your chosen help inside <skills></skills>.
"""


def _skillgen_prompt(problem: str) -> Dict[str, Any]:
    """Skill-gen prompt: query-only. seam mode uses SEAM EXPERIENCE_PROMPT (single user turn,
    <memory_item> output); v2 uses SKILL_GEN_SYSTEM (<skills>)."""
    if _TASK == 'code':
        # 代码域只做 narrative 一种文体（E4/E17 都是 narrative；toy/pitfall 未移植）
        return code_task.skillgen_prompt(problem)
    if _ALIGN_MODE == 'seam':
        return {'messages': [{'role': 'user', 'content': _SEAM_EXPERIENCE_PROMPT.format(problem=problem)}]}
    # 中文注释：按 --skill-style 选主链路文体（narrative=现版叙述式 / toy / pitfall）。
    sys_p = {'toy': SKILL_GEN_TOY, 'pitfall': SKILL_GEN_PITFALL,
             'freeform': SKILL_GEN_FREEFORM}.get(_SKILL_STYLE, SKILL_GEN_SYSTEM)
    return {'messages': [
        {'role': 'system', 'content': sys_p},
        {'role': 'user', 'content': f'Problem:\n{problem}'}]}


def _regen_prompt(problem: str, orig_skill: str, rubric_diag: str) -> Dict[str, Any]:
    """Regeneration prompt for buffer B distillation."""
    # 中文注释：regen 与主链路同文体（--skill-style），user 模板复用 REGEN_USER 三字段。
    sys_p = {'toy': REGEN_TOY_SYSTEM, 'pitfall': REGEN_PITFALL_SYSTEM,
             'freeform': REGEN_FREEFORM_SYSTEM}.get(_SKILL_STYLE, REGEN_SYSTEM)
    return {'messages': [
        {'role': 'system', 'content': sys_p},
        {'role': 'user', 'content': REGEN_USER.format(
            problem=problem, orig_skill=orig_skill, rubric_diag=rubric_diag)}]}


# ---- Reward ----
# 中文注释：reward = parseable × 通过率（对齐 SEAM lpem：去 terminated、去长度惩罚）。
# parseable=0 的候选 reward=0 仍参与 group（格式压力）。correct 兼容 bool（greedy 0/1，
# E1-E13）与 float 通过率（E14 多 rollout 判分，见 process_chunk reward_rollouts）。
def _skill_reward(parseable: bool, correct) -> float:
    return float(correct) if parseable else 0.0


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

    # Step 3: regenerate K DISTINCT candidate skills per [FAIL] entry (skill-model, high T) and
    # greedy-validate each. Entries that do NOT yet have >= m distinct greedy-passing skills are
    # RETRIED for up to --distill-retries extra rounds: regenerate more skills, dedup against those
    # already seen for that entry, greedy-solve, and accumulate — rescuing more problems into buffer B.
    # 中文注释：对每条 [FAIL] 高温采 K 个不同候选 skill 并贪心验证；还没凑够 m 个“贪心解对”的条目，
    # 再重生成 --distill-retries 轮（新候选去重、贪心解、累计），把更多题救进 buffer B。
    k = args.passatk_k
    per_entry_skills: List[List[str]] = [[] for _ in has_fail]        # 累计去重候选(跨轮)
    results_by_entry: Dict[int, List[Dict[str, Any]]] = {}            # 累计每候选贪心结果(跨轮)
    passers_by_entry: Dict[int, Set[str]] = {ei: set() for ei in range(len(has_fail))}
    pending_ei = list(range(len(has_fail)))                           # 还没凑够 m 个 passer 的条目
    for _ in range(args.distill_retries + 1):
        if not pending_ei:
            break
        regen_prompts = [_regen_prompt(has_fail[ei]['problem'], has_fail[ei]['orig_skill'],
                                       has_fail[ei]['rubric_diag']) for ei in pending_ei]
        regen_out = _run_samples(skill_sampler, regen_prompts, k, args.skill_max_tokens, skill_dp,
                                 temperature=args.passatk_skill_temp, top_p=args.passatk_skill_top_p)
        # 过滤(可解析/≤budget/无leak/对本条目去重) → 收集本轮新候选
        new_flat_idx, new_flat_prompts = [], []
        for ei, seqs in zip(pending_ei, regen_out):
            seen = set(per_entry_skills[ei])
            for s in (seqs or []):
                resp = _clean_text(getattr(s, 'decoded', '') or '')
                skill = _extract_skill(resp)
                if not skill or len(skill) > args.len_budget:
                    continue
                if _answer_leaked(skill, has_fail[ei]['reference_answer']):
                    continue
                if skill in seen:
                    continue
                seen.add(skill)
                per_entry_skills[ei].append(skill)
                new_flat_idx.append((ei, skill))
                new_flat_prompts.append(build_skill_solve_prompt(has_fail[ei]['problem'], skill))
        # 本轮新候选各用 executor 贪心(T=0)解 1 次，累计结果与 distinct passer
        if new_flat_prompts:
            solve_out = _run_samples(base_sampler, new_flat_prompts, 1, args.max_tokens, base_dp,
                                     temperature=0.0)
            for (ei, sk), seqs in zip(new_flat_idx, solve_out):
                roll = _parse_seq(seqs[0], has_fail[ei]['reference_answer']) if seqs else _empty_roll()
                results_by_entry.setdefault(ei, []).append(
                    {'skill': sk, 'len': len(sk), 'correct': roll['correct'], 'terminated': roll['terminated']})
                if roll['correct'] and roll['terminated']:
                    passers_by_entry[ei].add(sk)
        # 仍不足 m 个 distinct passer 的条目进入下一轮重试
        pending_ei = [ei for ei in pending_ei if len(passers_by_entry[ei]) < args.passatk_m]

    n_entries_with_cands = sum(1 for sk in per_entry_skills if sk)
    if not results_by_entry:
        logger.info(f'[distill] {len(has_fail)} [FAIL] entries, 0 valid regen skills')
        return [], _mk_distill({}, per_entry_skills, has_fail)

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
        # SEAM/verl uses torch.std's default unbiased=True for GRPO group std.
        import torch
        std = float(torch.std(torch.tensor(rewards, dtype=torch.float32)).item())
        if std < 1e-9:
            continue
        for c in cs:
            raw = (c['reward'] - mean_r) / (std + eps)
            c['advantage'] = max(-adv_clip, min(adv_clip, raw)) if adv_clip > 0 else raw
            c['kept'] = c['reward'] > mean_r


# ===========================================================================================
# 训练样本构造：response 段一律直接用采样返回的 token，严禁 decode 后重新过模板
# ===========================================================================================
# 为什么（2026-08-02 定案，.tmp_analysis/probe_token_direct.py 在 160 条真实样本上实测）：
# 把采样产出 decode 成文本、再塞回 messages 让 chat template 重新渲染一遍，可训练区就不再
# 等于模型真实生成的 token —— 模板会给 assistant 角色行带一个换行、给每条 message 补结尾
# EOS，思考段没闭合时还会先渲染一个空思考块。实测 160/160 条的可训练区都比采样 token 多；
# 截断样本更被塞进一个紧跟 </think> 之后的 <think>（p≈0、logp≈-20），单个 micro 的
# grad_norm 从 0.59 抬到 2.05，整批放大 3 倍，直接把 format_rate 曲线提前顶到 0.99。
#
# 正确形状：prompt 段照常编码（它本来就是模板产物，重编码无风险），response 段原样拼采样
# token。同一条实测里这条路径 160/160 逐 token 相等。采样器已经把 token 备好了
# （SampledSequence.tokens），所以候选记录只需把它带下来。
_ENCODE_TEMPLATE = None


def set_encode_template(template) -> None:
    """注入 skill 模型的客户端 Template 副本，供 build_train_feature 编码 prompt 段。"""
    global _ENCODE_TEMPLATE
    _ENCODE_TEMPLATE = template


def build_train_feature(prompt_messages, tokens, template=None):
    """prompt 段编码 + 原样拼采样 token，返回可直接喂模型的 InputFeature。

    labels 只盖住 ``tokens`` 那一段（prompt 段全 -100），与采样端真实生成的 token 逐一对应。
    ``template`` 默认用 skill 模型那份；executor 侧打分（E14 的 logP reward）要传自己的。
    """
    tmpl = template or _ENCODE_TEMPLATE
    assert tmpl is not None, 'call set_encode_template() before building train features'
    feat = tmpl.encode({'messages': [dict(m) for m in prompt_messages]}, add_generation_prompt=True)
    return tmpl.concat_input_feature(feat, [int(t) for t in tokens])


def _train_trajectory(rec):
    """query-only skill-gen prompt + 采样产出。

    GRPO 记录带 ``tokens``（采样端 vLLM 真实吐出的 token id），直接拼成 InputFeature；
    SFT 记录的 response 是程序合成的 ``<skills>`` 文本（本来就不是采样产出、没有对应 token），
    只能走 messages 编码，``key_rounds`` 标出最后一轮为唯一可训练区。
    """
    msgs = _skillgen_prompt(rec['problem'])['messages']
    if rec.get('tokens'):
        return build_train_feature(msgs, rec['tokens'])
    return {'messages': msgs + [{'role': 'assistant', 'content': rec['response']}],
            'user_data': pack_user_data({'key_rounds': [len(msgs)]})}


def _train_step(skill_model, ref_model, ckpt, samples, args):
    """On-policy GRPO update over one batch, then sync weights. SFT samples ride the
    same BNPOLoss with a positive constant advantage (--sft-weight)."""
    # 过滤空/纯空白 response：其可训练 token 为 0，会让持有它的 DP rank 跳过 backward，
    # 与对端 all-reduce 失步 → NCCL 死锁（find_unused_parameters=False）。高熵采样偶发首 token 即 EOS。
    n_in = len(samples)
    samples = [rec for rec in samples
               if (rec.get('tokens') if rec.get('tokens') is not None else (rec.get('response') or '').strip())]
    n_empty = n_in - len(samples)
    trajs = [_train_trajectory(rec) for rec in samples]
    advs = [float(rec['advantage']) for rec in samples]
    smp_all = [list(rec.get('logprobs') or []) for rec in samples]
    # drop_last 到 TRAIN_DP 整数倍：每个 micro（末尾那个可短于 sft）仍能被 dp 均分，零 padding 假样本。
    # 只丢尾部 ≤ dp-1 条真样本；若整批不足 dp（n<dp）则直接跳过该步。
    n_keep = (len(trajs) // TRAIN_DP) * TRAIN_DP
    if n_keep == 0:
        return {'n_samples': n_in, 'n_sft': 0, 'n_grpo': 0, 'n_empty': n_empty,
                'n_steps': 0, 'n_micro_batches': 0, 'metric': {}}
    trajs, advs = trajs[:n_keep], advs[:n_keep]
    # micro 尺寸可由 train_micro_batch 覆盖（ablate think/8192 实验防 backward OOM）；
    # 默认 0 = 跟随 sft_batch_size，主训练行为不变。梯度按 micro 数归一，切细数学等价。
    n = len(trajs)
    sft = getattr(args, 'train_micro_batch', 0) or args.sft_batch_size
    mini = args.ppo_mini_batch_size if args.ppo_mini_batch_size > 0 else n
    mini = max(sft, (mini // sft) * sft)
    multi_step = mini < n
    micro_ref, micro_old, micro_smp = [], [], []
    for i in range(0, n, sft):
        mb = trajs[i:i + sft]
        micro_ref.append(ref_model.forward_only(inputs=mb).get('logps'))
        micro_old.append(skill_model.forward_only(inputs=mb).get('logps') if multi_step else None)
        # 采样端 logprob，只给 GRPOMetric 做对账（sampler_logp_mae / sampler_token_delta）。
        # 整个 micro 都带齐了才传：SFT 样本的 response 是合成文本、根本没有采样 logprob，
        # 混进去只会让 token_delta 无法解释（它的语义是「应恒为 0」）。
        smp = [smp_all[j] for j in range(i, min(i + sft, n))]
        micro_smp.append(smp if all(smp) else None)
    micro, n_steps = 0, 0
    for ms in range(0, n, mini):
        for i in range(ms, min(ms + mini, n), sft):
            k = i // sft
            skill_model.forward_backward(inputs=trajs[i:i + sft], advantages=advs[i:i + sft],
                                         old_logps=micro_old[k], ref_logps=micro_ref[k],
                                         sampler_logps=micro_smp[k])
            micro += 1
        skill_model.clip_grad_and_step()
        n_steps += 1
    ckpt.sync_weights(merge_and_sync=True)
    metric = skill_model.calculate_metric(is_training=True)
    n_sft = sum(1 for s in samples if s.get('sft'))
    return {'n_samples': n_in, 'n_sft': n_sft, 'n_grpo': len(samples) - n_sft, 'n_empty': n_empty,
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
    # skill-gen (single pass, no retry)：每题恒 n_skills 个候选、组大小固定，对齐 SEAM rollout.n。
    # （全 0 的难题不在这里重采，而是进 buffer A → rubric 重生成，见 distill_buffer。）
    flat = []
    sg_out = _run_samples(skill_sampler, [_skillgen_prompt(r['problem']) for r in chunk],
                          args.n_skills, args.skill_max_tokens, skill_dp,
                          temperature=args.skill_gen_temperature, top_p=args.skill_gen_top_p,
                          top_k=args.skill_gen_top_k, logprobs=1)
    for r, seqs in zip(chunk, sg_out):
        for s in seqs:
            resp = _clean_text(getattr(s, 'decoded', '') or '')
            block = _extract_skill(resp) or ''
            # tokens = 采样端真实吐出的 token id，训练样本只能用它拼（见 build_train_feature）。
            # logprobs 同长，只给 GRPOMetric 做采样/训练对账（见 sampler_logprobs），不进 loss。
            _toks = [int(t) for t in (getattr(s, 'tokens', None) or [])]
            cand = {'skills': block, 'response': resp, 'parseable': bool(block),
                    'leaked': None, 'with_pass': None, 'reward': None, 'rolls': [],
                    'advantage': 0.0, 'kept': False, 'tokens': _toks,
                    'logprobs': sampler_logprobs(s),
                    'skillgen_stop': getattr(s, 'stop_reason', None),
                    'skillgen_tokens': len(_toks)}
            r['_cands'].append(cand)
            if block:
                flat.append((r, cand))

    # leak audit (deterministic, observability only)
    for r, c in flat:
        c['leaked'] = _answer_leaked(c['skills'], r['reference_answer'])

    # ⭐ executor 覆盖面（seam 对齐，2026-08-02）：SEAM 对**每一条**候选都跑 executor ——
    # fsdp_workers.py:842-858，抽不到 <memory_item> 时走 else 分支（direct_system + 裸题目），
    # 它的 acc 照样计入 reward_extra_info["acc"]，也就是 train/with_skill_accuracy 的分子。
    # 旧实现只跑 parseable 的，于是 twinkle 算不出同口径的 withskill 准确率，只能拿
    # P(correct|parseable) 去对 SEAM 的 P(correct)，两条曲线分母不同、根本无法对齐。
    # reward 口径不变：_skill_reward 仍然乘 parseable，空 skill 永远 reward=0。仅 seam 模式开，
    # 其余臂（E1-E12/E14-E21）行为与开销不动。
    exec_list = ([(r, c) for r in chunk for c in r['_cands']] if _ALIGN_MODE == 'seam' else flat)

    # with-skill executor pass：默认 greedy×1（E1-E13，reward 0/1 与旧口径 bit 一致）；
    # E14: reward_rollouts>1 时 T=reward_temperature × K 采样，reward = parseable × 通过率，
    # 把内容信号从 greedy 0/1 量化里释放出来（提升组内 std>0 比例）。
    if exec_list:
        K = max(1, int(getattr(args, 'reward_rollouts', 1) or 1))
        rT = float(getattr(args, 'reward_temperature', 0.0) or 0.0)
        ws_out = _run_samples(base_sampler,
                              [build_skill_solve_prompt(r['problem'], c['skills'], c.get('response'),
                                                        resp_terminated=(c.get('skillgen_stop') != 'length'))
                               for r, c in exec_list],
                              K, args.max_tokens, base_dp, temperature=rT)
        # 判分一次性批量化（code 任务要起子进程跑单测，逐条会比 GPU 还慢一个量级）
        pairs, spans = [], []
        for (r, c), seqs in zip(exec_list, ws_out):
            start = len(pairs)
            pairs.extend((s, r['reference_answer']) for s in (seqs or []))
            spans.append((start, len(pairs)))
        judged = _parse_many(pairs)
        for (r, c), (a, b) in zip(exec_list, spans):
            rolls = judged[a:b] or [_empty_roll()]
            for x in rolls[1:]:
                x['text'] = ''  # 磁盘保护：K>1 时只留首 rollout 全文（gen_records 体积控制）
            c['rolls'] = rolls
            c['with_pass'] = sum(1.0 for x in rolls if x['correct']) / len(rolls)
            c['reward'] = _skill_reward(c['parseable'], c['with_pass'])
    # unparseable candidates score 0 and still join the group (format pressure)
    for r in chunk:
        for c in r['_cands']:
            if c['reward'] is None:
                c['reward'] = 0.0

    # ⭐ 训练侧 no-skill baseline（seam 对齐，2026-08-02）。SEAM 只在第 1 个训练步跑一次
    # （ray_trainer.py:1461 `if self.global_steps == 1`，注释写明是提速、reward/梯度不受影响，
    # 后续 step 的 step_summary.lift 为 None），所以 SEAM 的 train lift 只有 step1 一个点
    # （0.7891-0.6338=+0.1553）。这里逐条照抄：仅 ci==0、direct prompt、T=0
    # （= SEAM use_experience=False 分支）。
    #
    # 每题必须跑 n_skills 次，不能跑 1 次再广播：SEAM 把**整个 1024 行 batch**送进
    # generate_sequences_as_grm_baseline，同一道题的 8 行是 8 个独立请求。executor 虽然是
    # greedy，vLLM 的批内非确定性（左 padding 长度随 batch 变、chunked prefill 的规约顺序）
    # 让同一 prompt 的 8 次结果并不总相同 —— step1 dump 实测 128 题里有 10 题的 8 次
    # baseline_acc 不全同。所以"同题 8 条结果相同、取 1 次即可"这个旧假设是错的，按题
    # 展开取均值才与 SEAM 的 np.mean(baseline_acc) 同口径。
    if _ALIGN_MODE == 'seam' and ci == 0 and chunk:
        K_b = max(1, int(args.n_skills))
        b_pairs = [r for r in chunk for _ in range(K_b)]
        b_out = _run_samples(base_sampler, [build_direct_prompt(r['problem']) for r in b_pairs],
                             1, args.max_tokens, base_dp, temperature=0.0)
        b_rolls = _parse_many([(_first_seq(seqs), r['reference_answer'])
                               for r, seqs in zip(b_pairs, b_out)])
        for i, r in enumerate(chunk):
            hits = [1.0 if x['correct'] else 0.0 for x in b_rolls[i * K_b:(i + 1) * K_b]]
            r['_train_baseline_pass'] = _mean(hits) if hits else None

    _assign_advantages(chunk, args)

    # SEAM/verl 对齐：每个 dataloader batch 都进入 actor update。零 adv 候选 PG 贡献 0，
    # 但仍计入 token-mean 分母，并在 beta>0 时贡献 KL 锚定；不再因整 chunk 无信号而跳过 step。
    grpo = []
    for r in chunk:
        for c in r['_cands']:
            if c.get('reward') is None:
                continue
            grpo.append({'problem': r['problem'], 'reference_answer': r['reference_answer'],
                         'data_id': r.get('data_id', ''), 'response': c['response'],
                         'skills': c['skills'], 'advantage': c['advantage'],
                         # tokens 是训练样本的唯一来源；response 只留给 reward/审计（见 _train_trajectory）
                         'tokens': c.get('tokens') or [],
                         'logprobs': c.get('logprobs') or [],
                         'kept': c['kept'], 'reward': c['reward'], 'sft': False})
    buffer_a = _collect_buffer_a(chunk, args)
    return _full_records(chunk, ci), _chunk_summary(chunk, ci), grpo, buffer_a


def _roll(x):
    out = {k: x[k] for k in ('pred', 'correct', 'terminated', 'stop_reason', 'gen_tokens', 'text')}
    # 代码域审计字段：判分结论 / 单测报错 / 用例数（离线分析错误类型分布靠它，_trim_err 已限长）
    for k in ('kind', 'error', 'n_tests'):
        if k in x:
            out[k] = x[k]
    return out


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
                'logp_base': c.get('logp_base'), 'logp_skill': c.get('logp_skill'),
                'logp_delta': c.get('logp_delta'),
            } for c in r['_cands']],
        })
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    import torch
    return float(torch.std(torch.tensor(xs, dtype=torch.float32)).item())


def _pstd(xs):
    """总体标准差（ddof=0）—— 报表用。

    SEAM 的 step_summary 用 ``np.std``（ddof=0）算 reward_std / group_reward_std_mean
    （ray_trainer.py:654,671），而 :func:`_std` 是 ``torch.std``（ddof=1）。组内只有 8 条时
    两者差 sqrt(8/7)=1.069，足以把 group_reward_std_mean 抬高 ~0.01（c0 实测
    0.14993 vs SEAM 0.14074；改 ddof=0 后为 0.14025）。注意分开：**advantage 的组内
    std 必须继续用 ddof=1**，verl 的 compute_grpo_outcome_advantage 用的也是
    ``torch.std`` 默认 unbiased，两边本来就一致，改了反而会把梯度弄不一致。
    """
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
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
        # 报表口径对齐 SEAM 的 np.std（ddof=0），见 _pstd。advantage 那边仍用 ddof=1。
        v = _pstd(rewards)
        group_vars.append(v)
        if v < 1e-9:
            zero_grad += 1
    n_train = sum(1 for c in all_cands if abs(c.get('advantage') or 0.0) > 1e-9)
    trunc = sum(1 for x in ws_rolls if x['stop_reason'] == 'length')
    # bugfix（ablate #6）：旧版 any(c.get('reward')) 用 truthiness，负 reward（E16 hinge/leak_gate、
    # E14 地板 -1.0）也被当“通过”；改用 with_pass>0（真正的 executor 通过率），greedy 0/1 臂语义不变。
    # 同时限 parseable：seam 模式下 unparseable 候选也有 with_pass（走 direct 回退），不限的话
    # 这条 pass@K 会被 baseline 成绩推高、与历史臂不可比。
    ws_acc = _mean([1.0 if any((c.get('with_pass') or 0) > 0 and c.get('parseable') for c in r['_cands'])
                    else 0.0
                    for r in chunk if r['_cands']])
    # ⭐ SEAM 同口径的 withskill 准确率：**全部**候选上的 mean(correct)，不看格式、不条件化。
    # = ray_trainer.py:1529 float(np.mean(reward_extra_infos_dict["acc"]))
    # = step_summary 的 withskill_pass = swanlab 的 train/with_skill_accuracy。
    # 只有 seam 模式会给 unparseable 候选跑 executor，其余臂这个值等于 candidate_withskill_pass。
    pass_all = [c['with_pass'] for c in all_cands if c['with_pass'] is not None]
    # 训练侧 baseline（仅 seam 模式的 chunk 0 有），按候选展开与 SEAM 的 np.mean(baseline_acc) 同口径。
    b_all = [r['_train_baseline_pass'] for r in chunk if r.get('_train_baseline_pass') is not None
             for _c in r['_cands']]
    base_pass = _mean(b_all) if b_all else None
    return {
        'record_type': 'summary', 'chunk': ci, 'n': len(chunk),
        'n_generated': len(all_cands), 'n_candidates_parseable': len(cands),
        'parse_rate': (len(cands) / len(all_cands)) if all_cands else 0.0,
        'n_leaked': sum(1 for c in cands if c['leaked']),
        'leak_rate': (sum(1 for c in cands if c['leaked']) / len(cands)) if cands else 0.0,
        'n_train_samples': n_train, 'n_groups': groups,
        'zero_grad_frac': (zero_grad / groups) if groups else 0.0,
        'reward_mean': _mean(all_rewards), 'reward_std': _pstd(all_rewards),
        'group_reward_std_mean': _mean(group_vars),
        'skill_tokens_mean': _mean([c.get('skillgen_tokens') or 0 for c in cands]),
        'skill_chars_mean': _mean([len(c['skills']) for c in cands]),
        'avg_withskill_pass': ws_acc,
        'candidate_withskill_pass': _mean([c['with_pass'] for c in scored]),
        'withskill_pass_all_cands': _mean(pass_all),
        'n_exec_cands': len(pass_all),
        'baseline_pass_train': base_pass,
        'lift_train': (_mean(pass_all) - base_pass) if base_pass is not None else None,
        'withskill_trunc_frac': (trunc / len(ws_rolls)) if ws_rolls else 0.0,
        'termination_rate_withskill': _mean([1.0 if x['terminated'] else 0.0 for x in ws_rolls]),
    }


def run_greedy_eval(base_sampler, skill_sampler, eval_records, ci, rounds,
                    base_dp, skill_dp, args, base_cache):
    """Holdout readout: skill-gen (T=args.eval_skill_temperature, args.eval_rollouts rollouts)
    -> greedy base solve (T=0). acc = per-problem mean correctness over the rollouts, averaged over
    problems (falls back to single greedy when eval_rollouts=1 & temp=0). Adds hard-slice
    (baseline_pass==0) rescue rate as a zero-cost secondary readout."""
    # baseline (frozen, cached)
    todo = [r for r in eval_records if DiskCache.key_for(r['problem']) not in base_cache]
    if todo:
        out = _run_samples(base_sampler, [build_direct_prompt(r['problem']) for r in todo],
                           1, args.max_tokens, base_dp, temperature=0.0)
        rolls = _parse_many([(_first_seq(seqs), r['reference_answer'])
                             for r, seqs in zip(todo, out)])
        for r, roll in zip(todo, rolls):
            base_cache.put(DiskCache.key_for(r['problem']), roll)
    for r in eval_records:
        br = base_cache.get(DiskCache.key_for(r['problem']))
        r['_baseline_pass'] = 1.0 if br['correct'] else 0.0
    # skill-gen (T=eval_skill_temperature, R rollouts) → with-skill greedy → mean acc over rollouts
    R = max(1, args.eval_rollouts)
    sg_out = _run_samples(skill_sampler, [_skillgen_prompt(r['problem']) for r in eval_records],
                          R, args.skill_max_tokens, skill_dp, temperature=args.eval_skill_temperature)
    # per problem -> list of R (skill, sresp)
    per_skills = []
    for seqs in sg_out:
        seqs = list(seqs or [])
        row = []
        for j in range(R):
            s = seqs[j] if j < len(seqs) else None
            if s is None:
                row.append(('', '', 'stop'))
            else:
                sresp = _clean_text(getattr(s, 'decoded', '') or '')
                row.append((_extract_skill(sresp) or '', sresp, getattr(s, 'stop_reason', None)))
        per_skills.append(row)
    # flatten R×N for a single batched greedy executor pass
    flat_prompts, flat_idx = [], []
    for pi, (r, row) in enumerate(zip(eval_records, per_skills)):
        for j, (sk, sresp, sstop) in enumerate(row):
            flat_prompts.append(build_skill_solve_prompt(r['problem'], sk, sresp,
                                                        resp_terminated=(sstop != 'length')))
            flat_idx.append((pi, j))
    ws_out = _run_samples(base_sampler, flat_prompts, 1, args.max_tokens, base_dp, temperature=0.0)
    judged = _parse_many([(_first_seq(seqs), eval_records[pi]['reference_answer'])
                          for (pi, _j), seqs in zip(flat_idx, ws_out)])
    roll_by = {idx: roll for idx, roll in zip(flat_idx, judged)}
    recs = []
    for pi, (r, row) in enumerate(zip(eval_records, per_skills)):
        rolls = [roll_by[(pi, j)] for j in range(len(row))]
        corr = [1.0 if x['correct'] else 0.0 for x in rolls]
        parses = [1.0 if sk else 0.0 for sk, _sresp, _sstop in row]
        terms = [1.0 if x['terminated'] else 0.0 for x in rolls]
        acc_mean = sum(corr) / len(corr) if corr else 0.0
        # bugfix（ablate #8）：unparseable skill 的 rollout 实际走了 direct 回退（≈baseline），
        # 主指标里格式崩塌会被 baseline 成绩掩护。strict 通道：unparseable 计 0，格式失败
        # 直接计入代价；主指标口径不变（与历史臂可比），两条曲线分叉即格式崩塌告警。
        acc_strict = (sum(c * p for c, p in zip(corr, parses)) / len(corr)) if corr else 0.0
        recs.append({
            'record_type': 'eval_problem', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
            'data_id': r.get('data_id', ''), 'problem': r['problem'],
            'reference_answer': r['reference_answer'], 'baseline_pass': r['_baseline_pass'],
            'n_rollouts': len(row), 'eval_skill_temperature': args.eval_skill_temperature,
            'withskill_acc_mean': acc_mean,                              # per-problem mean over R rollouts
            'withskill_acc_strict_mean': acc_strict,                     # unparseable counted wrong
            'withskill_pass_any': 1.0 if any(corr) else 0.0,            # pass@R (bonus readout)
            'skill_parseable_mean': sum(parses) / len(parses) if parses else 0.0,
            'withskill_terminated_mean': sum(terms) / len(terms) if terms else 0.0,
            # 首个 rollout 的明细留作肉眼抽查
            'skill': row[0][0], 'skill_parseable': bool(row[0][0]), 'skill_chars': len(row[0][0]),
            'withskill_pred': rolls[0]['pred'], 'withskill_correct': rolls[0]['correct'],
            'withskill_terminated': rolls[0]['terminated'], 'withskill_stop_reason': rolls[0]['stop_reason'],
            'withskill_text': rolls[0]['text'],
        })
    n = len(recs)
    # acc = 跨题平均的"每题 R 次平均正确率"（mean-over-rollouts）
    ws = (sum(x['withskill_acc_mean'] for x in recs) / n) if n else 0.0
    ws_strict = (sum(x['withskill_acc_strict_mean'] for x in recs) / n) if n else 0.0
    pass_any = (sum(x['withskill_pass_any'] for x in recs) / n) if n else 0.0
    base = (sum(x['baseline_pass'] for x in recs) / n) if n else 0.0
    fmt = (sum(x['skill_parseable_mean'] for x in recs) / n) if n else 0.0
    term = (sum(x['withskill_terminated_mean'] for x in recs) / n) if n else 0.0
    # 中文注释：难题子片救活率——baseline_pass==0 子集里 with-skill 的平均正确率（同 mean-over-rollouts 口径）。
    hard = [x for x in recs if not x['baseline_pass']]
    hard_rescue_rate = (sum(x['withskill_acc_mean'] for x in hard) / len(hard)) if hard else 0.0
    hard_rescued = sum(x['withskill_acc_mean'] for x in hard)  # 期望救活数(分数)
    summary = {'record_type': 'eval_summary', 'split': 'eval', 'chunk': ci, 'rounds_done': rounds,
               'n': n, 'acc_mean1': ws, 'baseline_acc_mean1': base, 'lift_mean1': ws - base,
               'acc_strict_mean1': ws_strict, 'lift_strict_mean1': ws_strict - base,
               'acc_pass_any': pass_any, 'n_rollouts': R, 'eval_skill_temperature': args.eval_skill_temperature,
               'format_mean1': fmt, 'term_mean1': term,
               'hard_n': len(hard), 'hard_rescued': hard_rescued, 'hard_rescue_rate': hard_rescue_rate}
    metrics = {'core/math/acc/mean@1': ws, 'core/math/baseline_acc/mean@1': base,
               'core/math/lift/mean@1': ws - base, 'core/math/format/mean@1': fmt,
               'core/math/acc_strict/mean@1': ws_strict, 'core/math/lift_strict/mean@1': ws_strict - base,
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
    # 主权重必须 fp32（对齐 verl actor：fp32 master + bf16 autocast）。twinkle 默认不传 dtype 时
    # transformers 会按 config 加载 bf16 主权重，lr=1e-6 的更新量(~1e-6)远小于 bf16 ulp(~4e-5)，
    # optimizer.step 的更新几乎全被舍入吞掉——这是 v2 学不动/与 SEAM 对不上的根因（A/B 实测差 10-20 倍）。
    # resume 旁路（skill_ablate）：skill_init_model_id 指向已保存的 checkpoint 目录时，仅 skill_model
    # 从该目录初始化（TransformersModel.load 无全量模型路径）；ref/samplers/template 仍用 MODEL_ID。
    _skill_init_id = getattr(args, 'skill_init_model_id', '') or MODEL_ID
    skill_model = TransformersModel(model_id=_skill_init_id, device_mesh=train_mesh, remote_group='train',
                                    torch_dtype='float32',
                                    ddp_config={'find_unused_parameters': False})
    skill_model.apply_patch(NoSplitModulesPatch({'Qwen3DecoderLayer'}))
    # 方案1：skill 模型开 thinking——让 actor 先在 <think> 里解题+提炼，<skills> 只放通用方法论。
    # <think> 由 _extract_skill 的 rfind('</think>') 砍掉，绝不流给 executor（executor 只吃 <skills>），
    # 因此不构成 SEAM 那种“把 think 喂给 executor”的泄漏。skill_model/ref_model/skill_sampler 三者
    # enable_thinking 必须一致，否则训练轨迹 token 布局与采样对不上。
    # 中文注释：skill_model/ref_model/skill_sampler 三者 enable_thinking 由 --skill-thinking 统一控制
    # （必须一致，否则训练轨迹 token 布局与采样对不上）；base_sampler（executor）走独立开关
    # --executor-thinking（默认 on，E1-E18 全部 on；E19/E20 为 off，见下方 base_sampler 处注释）。
    _think = args.skill_thinking == 'on'
    skill_model.set_template(Template, model_id=MODEL_ID, enable_thinking=_think,
                             max_length=args.max_model_len, truncation_strategy='delete')
    skill_model.set_processor(InputProcessor, padding_free=False)
    # 客户端同配置副本：训练样本的 prompt 段在这里编码，response 段直接拼采样返回的
    # token（见 build_train_feature）。skill_ablate/trainer.py 走自己那一句，两边不互干扰。
    set_encode_template(Template(model_id=MODEL_ID, enable_thinking=_think,
                                 max_length=args.max_model_len, truncation_strategy='delete'))
    # loss 统一用 SEAM 对齐的 SEAMBNPOLoss（verl PPO clip + low_var_kl + token-mean）；
    # v2/seam 两模式一致，不再随 align-mode 变。
    _loss_cls = 'SEAMBNPOLoss'
    skill_model.set_loss(_loss_cls, epsilon=args.grpo_epsilon, beta=args.kl_beta)
    # RL 异常 token 监控：除了 ratio/kl/entropy/clip 那一套，GRPOMetric 还会吐
    # train/logp_min、train/logp_frac_lt_10、train/sampler_logp_mae、train/sampler_token_delta
    # —— 训练序列一旦混进模型没生成过的 token（编码错位、模板凭空补的 EOS/空思考块），
    # 前两条会直接炸、后两条直接违反断言；而行为层指标（format/acc/reward）完全看不出来。
    # temperature 不传：skill-gen 就是 T=1，与采样端 logprob 取值口径天然一致。
    skill_model.add_metric('GRPOMetric', is_training=True, epsilon=args.grpo_epsilon)
    skill_model.set_optimizer('AdamW', lr=args.lr)
    # 对齐 SEAM：恒定 lr（无 warmup、无 decay）。SEAM 用 get_constant_schedule_with_warmup(
    # num_warmup_steps=0)+warmup_style=constant，全程恒定 1e-6。这里直接不设 scheduler，
    # skill_model.step() 对 lr_scheduler is None 有保护（transformers.py:822-824），lr 恒为 args.lr。
    # 之前的 CosineWarmupScheduler(warmup=10, cosine decay→0) 会让训练中后期有效 lr 持续衰减、
    # 更新幅度变小，与 SEAM 不一致，故移除。

    ref_mesh = DeviceMesh.from_sizes(world_size=REF_GPUS, dp_size=REF_DP, fsdp_size=REF_FSDP)
    ref_model = TransformersModel(model_id=MODEL_ID, device_mesh=ref_mesh, remote_group='ref',
                                  ddp_config={'find_unused_parameters': False})
    ref_model.apply_patch(NoSplitModulesPatch({'Qwen3DecoderLayer'}))
    # 方案1：与 skill_model 保持一致开 thinking（三者 enable_thinking 必须一致）。
    ref_model.set_template(Template, model_id=MODEL_ID, enable_thinking=_think,
                           max_length=args.max_model_len, truncation_strategy='delete')
    ref_model.set_processor(InputProcessor, padding_free=False)
    ref_model.set_loss('GRPOLoss', epsilon=args.grpo_epsilon)

    def _sampler(group, world, enable_thinking):
        # enable_prefix_caching 必须开：verl 的 vllm_rollout_spmd 把它硬编码成 True
        # （vllm_rollout_spmd.py:182），actor rollout 与 grm/executor rollout 共用这个引擎，
        # 所以 SEAM 两条采样通路全程带前缀缓存。twinkle 默认是 False，实测差别不只是速度：
        # 同一批里 8 个完全相同的 baseline prompt，缓存关掉时逐条重算、结果 128/128 题全同
        # （608=76x8 精确整除），缓存打开后首条算新 KV、后 7 条复用缓存 KV，数值路径不同，
        # SEAM 那边就有 10/128 题的 8 次贪心结果不全同。要和 SEAM 同口径就得同样开着。
        s = vLLMSampler(model_id=MODEL_ID,
                        engine_args={'gpu_memory_utilization': GPU_MEM,
                                     'max_model_len': args.max_model_len, 'tensor_parallel_size': 1,
                                     'enable_prefix_caching': True},
                        device_mesh=DeviceMesh.from_sizes(world_size=world, dp_size=world),
                        remote_group=group)
        s.set_template(Template, model_id=MODEL_ID, enable_thinking=enable_thinking, max_length=args.max_model_len)
        return s

    # 方案1：skill 采样器开 thinking，与 skill_model/ref_model 一致（actor 先想再写 <skills>）。
    skill_sampler = _sampler('skill_sampler', SKILL_SAMPLER_GPUS, enable_thinking=_think)
    # executor（base_sampler）的 thinking 单独一个开关，默认 on（E1-E18 全部如此）。
    # ⭐ 为什么要能关（2026-07-31 bcb 探针实测，n=275 同题配对）：BigCodeBench 上 think 的
    #   executor 有 34-50% 的 rollout 撞满预算、连代码块都没写出来（截断样本 8-gram 重复率
    #   p50=0.835、同一长句重复 92 次 = 字面死循环），把预算从 4096 加到 20000 也只把截断
    #   从 0.496 压到 0.338 且 pass 不涨。关掉 thinking 后截断归零、裸解反而更高
    #   （0.378 vs 0.324），rubric 增量从 +0.080 抬到 +0.135（p=1e-4）。
    #   见 bcb/bcb_eval0_{nothink,think12k,think20k}.jsonl。
    _exec_think = getattr(args, 'executor_thinking', 'on') == 'on'
    base_sampler = _LockedSampler(_sampler('base_sampler', BASE_SAMPLER_GPUS,
                                           enable_thinking=_exec_think))
    ckpt = CheckpointEngineManager(model=skill_model, sampler=skill_sampler)
    return skill_model, ref_model, skill_sampler, base_sampler, ckpt, SKILL_SAMPLER_GPUS, BASE_SAMPLER_GPUS


def _build_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('aops', 'math'), default='aops')
    p.add_argument('--n', type=int, default=0,
                   help='Problems loaded into the draw pool (0=all; keep 0 to match a SEAM run).')
    p.add_argument('--exclude-data-ids', default='',
                   help='Comma-separated jsonl files whose data_id/problem keys are excluded.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--numeric-only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval-size', type=int, default=128, help='Fixed holdout size (0 disables).')
    p.add_argument('--seam-parquet-dir', type=str, default='',
                   help='Read SEAM build_aops_dataset.py train.parquet/val.parquet directly, in '
                        'file order (problem<-extra_info.problem, answer<-reward_model.ground_truth). '
                        'val.parquet becomes the eval holdout. Bypasses load/--numeric-only/'
                        '--eval-size/internal shuffle so the input data matches a SEAM run exactly.')
    p.add_argument('--eval-every', type=int, default=5, help='Run holdout eval every N chunks.')
    p.add_argument('--eval-rollouts', type=int, default=1,
                   help='Eval: skill rollouts per holdout problem. SEAM validation uses one greedy rollout.')
    p.add_argument('--eval-skill-temperature', type=float, default=0.0,
                   help='Eval: skill-model sampling temperature. SEAM validation uses greedy T=0.')
    p.add_argument('--chunk-size', type=int, default=16)
    p.add_argument('--n-skills', type=int, default=8)
    p.add_argument('--skill-gen-temperature', type=float, default=1.0)
    p.add_argument('--skill-gen-top-p', type=float, default=1.0)
    p.add_argument('--skill-gen-top-k', type=int, default=-1)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192)
    # 方案1：开 thinking 后 skill 模型要先写 <think> 分析再写 <skills>，4096 装不下 think+完整 skills
    # 会截断成空块（_extract_skill 找不到 </skills> 返回 None）。提到 8192 给两段都留足空间。
    p.add_argument('--skill-max-tokens', type=int, default=8192)
    # 中文注释：文体消融开关——主链路与 buffer B regen 同文体（分布一致才可联合训练）。
    p.add_argument('--skill-style', choices=('narrative', 'toy', 'pitfall', 'freeform'), default='narrative',
                   help='skill文体: narrative=现版叙述式; toy=异数字玩具题示范; pitfall=预判纠错; '
                        'freeform=招式菜单/模型按题自选形态。主链路与 regen 同文体。')
    p.add_argument('--skill-thinking', choices=('on', 'off'), default='on',
                   help='skill_model/ref_model/skill_sampler 三者的 enable_thinking（必须一致）')
    p.add_argument('--executor-thinking', choices=('on', 'off'), default='on',
                   help='executor(base_sampler) 的 enable_thinking。off 用于 BigCodeBench 这类'
                        '"解答短、难点在选 API 而非多步推理"的任务：think 下 34-50% 的 rollout '
                        '陷入字面死循环撞满预算，关掉后截断归零且裸解更高（见 build 处注释）。')
    p.add_argument('--align-mode', choices=('v2', 'seam'), default='v2',
                   help="SEAM-alignment toggle for PROMPT/SKILL FORMAT only. "
                        "'v2'=clean single-user executor prompt + <skills> skill-gen. "
                        "'seam'=nested single-user executor prompt + EXPERIENCE_PROMPT/<memory_item> skill-gen. "
                        'Reward (lpem numeric-only) and loss (BNPOLoss token-mean) are SEAM-style in BOTH modes.')
    p.add_argument('--len-budget', type=int, default=1200,
                   help='Skill length budget (chars). ONLY used in distillation: drop regen skills '
                        'longer than this, pick the buffer-A seed / buffer-B survivor closest to it. '
                        'Does NOT affect GRPO reward or eval (reward = parseable AND correct).')
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
    p.add_argument('--distill-retries', type=int, default=1,
                   help='Extra regeneration rounds in distillation for [FAIL] entries that have '
                        'not yet reached m distinct greedy-passing skills (0=single pass). Each '
                        'extra round regenerates more skills (deduped) to rescue more into buffer B.')
    p.add_argument('--sft-weight', type=float, default=0.5,
                   help='Advantage magnitude for SFT distillation samples (-w*logp + beta*KL).')
    p.add_argument('--rubric-workers', type=int, default=16)
    # --- GRPO ---
    p.add_argument('--sft-batch-size', type=int, default=8)
    p.add_argument('--ppo-mini-batch-size', type=int, default=0)
    p.add_argument('--grpo-epsilon', type=float, default=0.2)
    p.add_argument('--adv-clip', type=float, default=0.0,
                   help='clip group-relative advantage to [-adv_clip, adv_clip]; '
                        '0 = no clipping (matches SEAM/verl GRPO which does not clip advantages)')
    # 与 skill_ablate/main.py 的默认值必须一致（两个入口默认值分叉 = 静默不可比）。
    # 2026-07-29 拍板 0.001 -> 0.01：对抗侵蚀 executor 收束能力的自发漂移。
    p.add_argument('--kl-beta', type=float, default=0.01)
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
        # ⭐ 与 SEAM 同名同口径的三条（ray_trainer.py:1569-1583），专为 swanlab 叠图对齐而发：
        #   train/with_skill_accuracy = 全部候选的 mean(correct)；acc/reward_mean = mean(correct∧format)；
        #   skill/format_rate = SEAM 的 format_mean。
        #   注：旧的 acc/withskill_pass 是**题级 pass@K**，与 SEAM 同名指标不同口径（它接近 1
        #   且会随训练小幅下行）—— 之前把这两条叠在一张图上看"趋势相反"就是这个原因。
        d['train/with_skill_accuracy'] = summary['withskill_pass_all_cands']
        d['acc/reward_mean'] = summary['reward_mean']
        d['skill/format_rate'] = summary['parse_rate']
        if summary.get('baseline_pass_train') is not None:
            # SEAM 只在 step1 跑训练侧 baseline，所以这两条也只在 chunk 0 有值（与 SEAM 同步）。
            d['acc/baseline_pass'] = summary['baseline_pass_train']
            d['acc/lift'] = summary['lift_train']
            d['train/baseline_accuracy'] = summary['baseline_pass_train']
            d['train/lift'] = summary['lift_train']
    if log:
        d['train/n_grpo'] = log['n_grpo']
        d['train/n_sft'] = log['n_sft']
        for k, v in (log.get('metric') or {}).items():
            if not _is_num(v):
                continue
            if k.startswith('learning rate'):
                if 'group 1' in k:
                    d['train/lr'] = float(v)
            elif k.startswith('train/'):
                # GRPOMetric 等已经自带 train/ 前缀，不能再套一层。
                d[k.replace(' ', '_')] = float(v)
            else:
                d[f'train/{k.replace(" ", "_")}'] = float(v)
    return d


def main():
    args = _build_args()
    global _ALIGN_MODE, _SKILL_STYLE
    _ALIGN_MODE = args.align_mode  # 'v2' | 'seam'
    _SKILL_STYLE = args.skill_style  # 'narrative' | 'toy' | 'pitfall' | 'freeform'
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
                             'sft_weight': args.sft_weight, 'lr': args.lr, 'align_mode': args.align_mode})

    skill_model, ref_model, skill_sampler, base_sampler, ckpt, skill_dp, base_dp = init_components(args)
    checker = build_rubric_checker()
    if checker is None:
        sys.stderr.write('[v2] no LLM backup env -> buffer B distillation DISABLED (GRPO only)\n')

    cache_dir = args.cache_dir or os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    # 每次启动强制重算 eval baseline：旧缓存可能来自不同环境/代码版本（torch/vllm/dtype 均影响 T=0 输出），
    # 跨 run 复用会造成 with-skill（现算）vs baseline（陈旧）不可比，lift 虚高/虚低（已实锤过一次）。
    _base_cache_path = os.path.join(cache_dir, 'eval_baseline.jsonl')
    if os.path.exists(_base_cache_path):
        os.remove(_base_cache_path)
        logger.info('stale eval_baseline cache removed (recomputed this run)')
    eval_base_cache = DiskCache(_base_cache_path, not args.no_cache)

    cfg = {'record_type': 'config', 'model': MODEL_ID, 'dataset': args.dataset,
           'n': len(records), 'eval_n': len(eval_records), 'seed': args.seed,
           'n_skills': args.n_skills, 'len_budget': args.len_budget,
           'distill_trigger': args.distill_trigger, 'sft_trigger': args.sft_trigger,
           'passatk_k': args.passatk_k, 'passatk_m': args.passatk_m,
           'passatk_skill_temp': args.passatk_skill_temp, 'passatk_skill_top_p': args.passatk_skill_top_p,
           'sft_weight': args.sft_weight,
           'grpo_epsilon': args.grpo_epsilon, 'kl_beta': args.kl_beta, 'lr': args.lr,
           'align_mode': args.align_mode,
           'rubric_check': bool(checker), 'max_train_rounds': args.max_train_rounds,
           'seam_parquet_dir': (getattr(args, 'seam_parquet_dir', '') or ''),
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
