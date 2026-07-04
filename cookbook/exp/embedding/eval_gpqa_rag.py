"""Math evaluation: direct vs RAG-augmented with Qwen3.5-4B.

Datasets (``--dataset``):
  - ``math`` (default): MATH (Hendrycks), stratified by difficulty (Level 1-5)
                        so RAG gain can be plotted against difficulty.
  - ``aops``:           AoPS competition problems (metadata.boxed only).

Modes (``--mode``):
  - ``direct``: The model solves problems directly (4 GPUs, TP=4).
  - ``rag`` (default): Retrieve top-k thinking traces from LanceDB, condense
                them (API qwen3.7-max), inject as 1-shot examples, then solve
                (8 GPUs: DP=4 embedding + TP=4 vLLM).

Defaults implement **raw RAG on MATH**: ``--dataset math --mode rag --condense``
with hint filtering OFF. The API condenser needs ``COMPRESS_API_KEY`` (or a
local condenser via ``EVAL_CONDENSER_GPUS``); otherwise pass ``--no-condense``.

Optional ``--hint`` flag (rag mode only):
  After retrieval + condensing, call an API model to filter and refine the
  traces — keeping only applicable methods — then inject the refined trace.

Reference answers are the ``\\boxed{...}`` content of each solution.

Launch examples:
    # Default: raw RAG on MATH, stratified 100/level (needs COMPRESS_API_KEY)
    COMPRESS_API_KEY=sk-... python cookbook/exp/embedding/eval_gpqa_rag.py

    # Paired direct baseline on the same MATH subset
    python cookbook/exp/embedding/eval_gpqa_rag.py --mode direct

    # Raw RAG without condenser (inject raw retrieved traces)
    python cookbook/exp/embedding/eval_gpqa_rag.py --no-condense

    # Add hint filtering back on top of condensing
    COMPRESS_API_KEY=sk-... python cookbook/exp/embedding/eval_gpqa_rag.py --hint

    # Fall back to the old AoPS dataset
    python cookbook/exp/embedding/eval_gpqa_rag.py --dataset aops
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams as TwinkleSamplingParams
from twinkle.loss import InfonceLoss
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

logger = get_logger()

# -- Condenser config ----------------------------------------------------------
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
CONDENSE_API_KEY = os.environ.get('COMPRESS_API_KEY', '')
CONDENSE_BASE_URL = os.environ.get('COMPRESS_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
CONDENSE_API_MODEL = os.environ.get('COMPRESS_MODEL', 'qwen3.7-max')
CONDENSE_API_CONCURRENCY = int(os.environ.get('API_CONCURRENCY', 32))
CONDENSE_API_MIN_INTERVAL = float(os.environ.get('API_MIN_INTERVAL', 0.1))
CONDENSE_TEMPERATURE = 0.2
CONDENSE_MAX_TOKENS = 8192

# -- Hint analysis config ------------------------------------------------------
HINT_ANALYSIS_MAX_TOKENS = int(os.environ.get('HINT_ANALYSIS_MAX_TOKENS', 2000))
HINT_ANALYSIS_TEMPERATURE = 0.2

HINT_ANALYSIS_SYSTEM = (
    'You are a mathematical reasoning trace filter. '
    'Given a target problem and reasoning traces retrieved from SIMILAR (but different) problems, '
    'your task is to FILTER and REFINE the traces into a clean reference.\n\n'
    'Rules:\n'
    '1. KEEP: solution steps, methods, formulas, techniques, and key insights '
    'that are directly applicable to solving the target problem.\n'
    '2. REMOVE: problem-specific numeric calculations that do not transfer, '
    'dead-end explorations, irrelevant approaches, verbose restatements, '
    'and any content that would mislead the solver on the target problem.\n'
    '3. Output the refined trace directly as actionable solution steps. '
    'Preserve the original mathematical expressions and step structure.\n'
    '4. Do NOT solve the target problem. Do NOT add your own solutions or commentary.\n'
    '5. Do NOT output the answer to either problem.\n'
    '6. If the traces are entirely irrelevant, output exactly: "No applicable methods."'
)

HINT_ANALYSIS_USER = (
    '## Target Problem\n{query}\n\n'
    '## Retrieved Reasoning Traces\n{thinking}'
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# -- Gen/Embed config ---------------------------------------------------------
GEN_MODEL_ID = os.environ.get('GEN_MODEL_ID', 'Qwen/Qwen3.5-4B')
EMBED_MODEL_ID = os.environ.get(
    'EMBED_MODEL_ID', 'output.oldemb/embedding_full_transformers/last-checkpoint')

GEN_GPUS = int(os.environ.get('GEN_GPUS', 4))
EMB_GPUS = int(os.environ.get('EMB_GPUS', 2))
EMBED_MAX_LENGTH = int(os.environ.get('EMBED_MAX_LENGTH', 20000))

GEN_GPU_MEM = float(os.environ.get('GEN_GPU_MEM', 0.85))
GEN_MAX_MODEL_LEN = int(os.environ.get('GEN_MAX_MODEL_LEN', 65536))
GEN_MAX_TOKENS = int(os.environ.get('GEN_MAX_TOKENS', 65536))
GEN_TEMPERATURE = float(os.environ.get('GEN_TEMPERATURE', 0.6))
GEN_TOP_P = float(os.environ.get('GEN_TOP_P', 0.95))

AOPS_DATASET_ID = os.environ.get('AOPS_DATASET_ID', 'AI-MO/aops')
MATH_DATA_DIR = os.environ.get('MATH_DATA_DIR', './output/math_data/MATH')


# ---------------------------------------------------------------------------
# Condenser prompts & validation
# ---------------------------------------------------------------------------

COMPRESS_SYSTEM = """\
You are a reasoning-trace condenser. Given a verbose reasoning trace, \
extract the TRANSFERABLE KNOWLEDGE as an EXECUTABLE SOLUTION SKELETON \
that would help a reader solve SIMILAR problems in the same domain.

Your output is the ENTIRE useful content — there is no expansion tool, no second pass. \
The reader will apply this knowledge to a DIFFERENT problem, so focus on what transfers.

Principles:
1. OUTPUT AN EXECUTABLE STEP CHAIN: numbered steps that a solver can directly follow. \
Each step should state WHAT to do and HOW (with the formula/technique), not just \
name the concept.
2. INCLUDE FULL FORMULAS: theorems, identities, inequalities — state each \
with its COMPLETE MATHEMATICAL EXPRESSION, not just the name.
3. STATE APPLICABILITY: what structural features of a problem signal that this \
approach works (e.g. "when the constraint is a sum of squares").
4. PRESERVE KEY INSIGHTS: the non-obvious ideas or tricks that make the approach \
work — the things a solver would NOT think of without guidance.
5. REMOVE: problem-specific numeric calculations, dead-end explorations, \
hesitations, verbose restatements, and trivial arithmetic.
6. FORMAT: Start with a one-line "Applicability" statement, then numbered steps, \
then key formulas. Keep it concise and actionable.
7. NO meta-commentary about the compression process. NO preamble.
"""

COMPRESS_USER = (
    '## Reader Problem (context only — do NOT solve it)\n{query}\n\n'
    '## Reasoning Trace to Condense\n{text}')


def _is_truncated_compression(text: str) -> bool:
    if not text or not text.strip():
        return True
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return True
    last_line = lines[-1]
    # Truncated if last line looks incomplete (no terminal punctuation/formula)
    if last_line and last_line[-1] not in '.。!！)）]】}\\$':
        # Allow lines ending with numbers, boxed answers, etc.
        if not re.search(r'\d$|\\boxed|\$|\)$', last_line):
            return True
    return False


# -- API rate limiter ----------------------------------------------------------
_api_semaphore = threading.Semaphore(CONDENSE_API_CONCURRENCY)
_api_bucket_lock = threading.Lock()
_api_tokens = [float(CONDENSE_API_CONCURRENCY)]
_api_last_refill = [time.monotonic()]


def _api_throttle():
    _api_semaphore.acquire()
    wait = 0.0
    try:
        with _api_bucket_lock:
            now = time.monotonic()
            elapsed = now - _api_last_refill[0]
            refill = elapsed / CONDENSE_API_MIN_INTERVAL
            _api_tokens[0] = min(float(CONDENSE_API_CONCURRENCY), _api_tokens[0] + refill)
            _api_last_refill[0] = now
            if _api_tokens[0] >= 1.0:
                _api_tokens[0] -= 1.0
            else:
                wait = (1.0 - _api_tokens[0]) * CONDENSE_API_MIN_INTERVAL
                _api_tokens[0] = 0.0
    finally:
        _api_semaphore.release()
    if wait > 0:
        time.sleep(wait)


def _api_condense_single(api_client: OpenAIClient, messages: List[Dict]) -> Optional[str]:
    _api_throttle()
    trajectory = {'messages': messages}
    sp = TwinkleSamplingParams(temperature=CONDENSE_TEMPERATURE, max_tokens=CONDENSE_MAX_TOKENS)
    try:
        reply = api_client(trajectory, sp, extra_body={'enable_thinking': False})
    except Exception as exc:
        logger.warning(f'[condense-api] error: {exc}')
        return None
    content = (reply.get('content') or '').strip()
    if not content:
        return None
    m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', content, re.DOTALL)
    if m:
        content = m.group(1).strip()
    return content


def _api_hint_analysis_batch(
    api_client: OpenAIClient,
    problems: List[str],
    condensed_examples: List[List[Dict[str, str]]],
) -> List[Optional[str]]:
    """Call API to pre-analyze RAG relevance for each problem."""
    _MAX_HINT_INPUT = 8000
    results: List[Optional[str]] = [None] * len(problems)
    tasks = []
    for i, prob in enumerate(problems):
        if not condensed_examples[i]:
            continue
        traces = [ex.get('thinking', '') for ex in condensed_examples[i]]
        merged_thinking = '\n---\n'.join(traces)
        if len(merged_thinking) > _MAX_HINT_INPUT:
            merged_thinking = merged_thinking[:_MAX_HINT_INPUT] + '\n[...truncated]'
        user_msg = HINT_ANALYSIS_USER.format(query=prob, thinking=merged_thinking)
        msgs = [
            {'role': 'system', 'content': HINT_ANALYSIS_SYSTEM},
            {'role': 'user', 'content': user_msg},
        ]
        tasks.append((i, msgs))

    if not tasks:
        return results

    def _call_one(idx, msgs):
        _api_throttle()
        try:
            trajectory = {'messages': msgs}
            sp = TwinkleSamplingParams(
                temperature=HINT_ANALYSIS_TEMPERATURE,
                max_tokens=HINT_ANALYSIS_MAX_TOKENS)
            reply = api_client(trajectory, sp, extra_body={'enable_thinking': False})
            content = (reply.get('content') or '').strip()
            # Treat "No applicable methods." as empty (will trigger fallback)
            if not content or content == 'No applicable methods.':
                return idx, None
            return idx, content
        except Exception as exc:
            logger.warning(f'[hint-analysis] error for idx={idx}: {exc}')
            return idx, None

    with ThreadPoolExecutor(max_workers=min(len(tasks), CONDENSE_API_CONCURRENCY)) as pool:
        futs = [pool.submit(_call_one, idx, msgs) for idx, msgs in tasks]
        for fut in as_completed(futs):
            idx, analysis = fut.result()
            results[idx] = analysis

    n_success = sum(1 for r in results if r)
    logger.info(f'[hint-analysis] completed {n_success}/{len(tasks)} analyses')
    return results


# ---------------------------------------------------------------------------
# LLM-based decontamination
# ---------------------------------------------------------------------------

_DECONTAM_JUDGE_PROMPT = (
    'We are building a RAG-augmented math training system. Problem A is the test '
    'question; Problem B was retrieved from a knowledge base.\n'
    'Answer YES only if A and B are essentially the SAME specific problem — '
    'i.e. solving B directly gives you A\'s answer (just different wording/notation/'
    'format/negation).\n'
    'Answer NO if they merely share the same method/topic but have different '
    'specific values, equations, or geometric configurations — learning B\'s '
    'approach still requires independent work to solve A.\n'
    'Problem A: {prob_a}\n'
    'Problem B: {prob_b}\n'
    'Answer only YES or NO.'
)


def _llm_judge_same_problem(
    api_client: OpenAIClient, pairs: List[tuple],
) -> List[bool]:
    """Batch LLM judge: are (problem_a, problem_b) the same problem?

    Returns list of bools (True = same problem = should filter).
    """
    if not pairs or not api_client:
        return [False] * len(pairs)

    results = [False] * len(pairs)

    def _judge_one(idx, pa, pb):
        prompt = _DECONTAM_JUDGE_PROMPT.format(prob_a=pa, prob_b=pb)
        msgs = [{'role': 'user', 'content': prompt}]
        _api_throttle()
        try:
            trajectory = {'messages': msgs}
            sp = TwinkleSamplingParams(temperature=0.1, max_tokens=8)
            reply = api_client(trajectory, sp, extra_body={'enable_thinking': False})
            answer = (reply.get('content') or '').strip().upper()
            return idx, 'YES' in answer
        except Exception:
            return idx, False

    with ThreadPoolExecutor(max_workers=min(len(pairs), CONDENSE_API_CONCURRENCY)) as pool:
        futs = [pool.submit(_judge_one, i, pa, pb) for i, (pa, pb) in enumerate(pairs)]
        for fut in as_completed(futs):
            idx, is_same = fut.result()
            results[idx] = is_same
    return results


def _llm_decontaminate(
    api_client: OpenAIClient,
    problems: List[str],
    all_examples: List[List[Dict[str, str]]],
) -> List[List[Dict[str, str]]]:
    """Apply LLM-based decontamination: remove retrievals judged as same problem."""
    judge_pairs = []  # (qi, ret_idx, prob_a, prob_b)
    for qi, exs in enumerate(all_examples):
        for ri, ex in enumerate(exs):
            judge_pairs.append((qi, ri, problems[qi], ex.get('query', '')))

    if not judge_pairs:
        return all_examples

    pairs_input = [(pa, pb) for _, _, pa, pb in judge_pairs]
    verdicts = _llm_judge_same_problem(api_client, pairs_input)
    to_remove = set()
    for vi, (qi, ri, _, _) in enumerate(judge_pairs):
        if verdicts[vi]:
            to_remove.add((qi, ri))

    if to_remove:
        logger.info(f'[decontam-llm] filtered {len(to_remove)} same-problem retrievals')
        for qi in range(len(all_examples)):
            all_examples[qi] = [
                ex for ri, ex in enumerate(all_examples[qi])
                if (qi, ri) not in to_remove
            ]
    return all_examples


def condense_traces(
    examples_batch: List[List[Dict[str, str]]],
    problems: List[str],
    api_client: OpenAIClient,
    condenser_sampler=None,
    compress_params=None,
    special_tokens: set = None,
    max_output_len: int = 2000,
    dp_size: int = 1,
) -> List[List[Dict[str, str]]]:
    """Compress retrieved thinking traces with query-aware condenser.

    Primary: local vLLM condenser (if provided).
    Fallback: API condenser.
    Final fallback: raw trace truncated to max_output_len.
    """
    result: List[List[Dict[str, str]]] = []
    # Flatten all (batch_idx, ex_idx, problem, example) for batch processing
    tasks = []
    for bi, (exs, prob) in enumerate(zip(examples_batch, problems)):
        for ei, ex in enumerate(exs):
            tasks.append((bi, ei, prob, ex))

    if not tasks:
        return [[] for _ in examples_batch]

    # Build condense prompts (aligned with make_embedding_dataset.py hard path)
    prompts = []
    for _, _, prob, ex in tasks:
        user_msg = COMPRESS_USER.format(query=prob, text=ex['thinking'])
        prompts.append([{'role': 'system', 'content': COMPRESS_SYSTEM},
                        {'role': 'user', 'content': user_msg}])

    # Phase 1: local vLLM condenser
    condensed = [None] * len(tasks)
    condense_sources = ['raw'] * len(tasks)
    fallback_indices = []

    if condenser_sampler is not None and compress_params is not None:
        sampler_inputs = [{'messages': p} for p in prompts]
        # The local vLLM sampler runs data-parallel across ``dp_size`` workers
        # and requires at least one item per worker (it errors with
        # "Batch too small for N workers" otherwise). Pad the batch up to a
        # multiple of dp_size by repeating the last item, run, then keep only
        # the first ``n_real`` responses and drop the padding.
        n_real = len(sampler_inputs)
        pad_size = 0
        if dp_size > 1 and n_real > 0 and n_real % dp_size != 0:
            pad_size = dp_size - (n_real % dp_size)
            sampler_inputs = sampler_inputs + [sampler_inputs[-1]] * pad_size
        try:
            responses = condenser_sampler.sample(sampler_inputs, compress_params)
        except Exception as exc:
            logger.warning(f'[condense] sampler error: {exc}')
            responses = [None] * len(sampler_inputs)
        if pad_size:
            responses = responses[:n_real]
        for ri, resp in enumerate(responses):
            seq = resp.sequences[0] if resp and resp.sequences else None
            text = ''
            if seq and seq.stop_reason != 'length' and seq.decoded:
                text = seq.decoded
                if special_tokens:
                    for tok in special_tokens:
                        text = text.replace(tok, '')
                text = text.rstrip()
            if text and not _is_truncated_compression(text):
                condensed[ri] = text
                condense_sources[ri] = 'local'
            else:
                fallback_indices.append(ri)
    else:
        fallback_indices = list(range(len(tasks)))

    # Phase 2: API fallback
    if fallback_indices and api_client:
        with ThreadPoolExecutor(max_workers=CONDENSE_API_CONCURRENCY) as pool:
            futures = {}
            for ri in fallback_indices:
                futures[pool.submit(_api_condense_single, api_client, prompts[ri])] = ri
            for fut in as_completed(futures):
                ri = futures[fut]
                api_result = fut.result()
                if api_result and not _is_truncated_compression(api_result):
                    condensed[ri] = api_result
                    condense_sources[ri] = 'api'

    # Phase 3: assemble results (fallback to raw truncation)
    result = [[] for _ in examples_batch]
    for ti, (bi, ei, prob, ex) in enumerate(tasks):
        compressed = condensed[ti]
        raw_len = len(ex['thinking'])
        sim_val = ex.get('_sim', 0.0)
        if compressed:
            result[bi].append({'query': ex['query'],
                               'thinking': _strip_condenser_markers(compressed),
                               '_condense_source': condense_sources[ti],
                               '_raw_trace_len': raw_len, '_sim': sim_val})
        else:
            result[bi].append({'query': ex['query'],
                               'thinking': ex['thinking'][:max_output_len],
                               '_condense_source': 'raw',
                               '_raw_trace_len': raw_len, '_sim': sim_val})

    n_ok = sum(1 for c in condensed if c)
    logger.info(f'[condense] {n_ok}/{len(tasks)} compressed ok, '
                f'{len(tasks) - n_ok} fell back to raw truncation')
    return result


def _strip_condenser_markers(text: str) -> str:
    """Light cleanup of condenser output.

    Removes any residual markdown headers or meta-lines that don't carry
    solution content. Keeps numbered steps and equations intact.
    """
    # Remove legacy ## headers if condenser still emits them
    if '## More' in text:
        text = text.split('## More', 1)[0]
    text = re.sub(r'^##\s*Summary\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Topic:\s*.*\n?', '', text, flags=re.MULTILINE)
    # Remove meta-commentary lines
    text = re.sub(r'^\s*\(Note:.*\)\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# Boxed answer extraction
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r'\\boxed\s*\{')


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content, handling nested braces."""
    if not text:
        return None
    last_match = None
    for m in _BOXED_RE.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            last_match = text[start:i - 1].strip()
    return last_match


def normalize_answer(ans: str) -> str:
    """Normalize a math answer string for comparison."""
    if not ans:
        return ''
    s = ans.strip()
    # MCQ: extract bare letter from \textbf{(D)}, \text{(A)}, \mathbb{A}, (B), etc.
    m = re.match(r'^\\?(?:textbf|text|mathrm|mathbf|mathbb)?\{?\(?([A-E])\)?\}?$', s)
    if m:
        return m.group(1)
    s = s.replace(' ', '')
    s = s.replace(r'\,', '')
    s = s.replace(r'\;', '')
    s = s.replace(r'\!', '')
    s = s.replace(r'\text', '')
    s = s.replace(r'\mathrm', '')
    s = s.replace(r'\displaystyle', '')
    s = re.sub(r'\\(?:left|right)[.()\[\]|]', '', s)
    s = s.replace(r'\dfrac', r'\frac')
    s = s.replace(r'\tfrac', r'\frac')
    s = s.strip('$').strip()
    # Strip unit-like brace suffixes: {cm}, {m}, {kg}, etc.
    s = re.sub(r'\{[a-zA-Z]+\}$', '', s)
    # Normalize degree: ^\circ, ^{\circ}, ° → deg
    s = re.sub(r'\^\\circ|\^\{\\circ\}|°', 'deg', s)
    # Canonicalize \frac{a}{b} → (a)/(b)
    def _frac_to_slash(m):
        # Handle nested braces in numerator/denominator
        text = m.group(0)
        pos = text.index('{') + 1
        depth, num_start = 1, pos
        while depth > 0:
            if text[pos] == '{': depth += 1
            elif text[pos] == '}': depth -= 1
            pos += 1
        numer = text[num_start:pos - 1]
        pos += 1  # skip '{'
        den_start = pos
        depth = 1
        while depth > 0:
            if text[pos] == '{': depth += 1
            elif text[pos] == '}': depth -= 1
            pos += 1
        denom = text[den_start:pos - 1]
        return f'({numer})/({denom})'
    s = re.sub(r'\\frac\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', _frac_to_slash, s)
    # Also handle bare a/b → (a)/(b) for consistent comparison
    # Only simple integer/variable fractions: 17/5 → (17)/(5)
    s = re.sub(r'(?<!\w)(\d+)/(\d+)(?!\w)', r'(\1)/(\2)', s)
    return s


def _try_numeric_equal(a: str, b: str) -> bool:
    """Try to evaluate both as floats; match if within 1e-9 relative tolerance."""
    try:
        va = float(a.replace('(', '').replace(')', ''))
        vb = float(b.replace('(', '').replace(')', ''))
        return abs(va - vb) < 1e-9 * max(1, abs(va), abs(vb))
    except (ValueError, ZeroDivisionError):
        pass
    # Try evaluating simple fraction expressions like (17)/(5)
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
    if va is not None and vb is not None:
        return abs(va - vb) < 1e-9 * max(1, abs(va), abs(vb))
    return False


# MCQ compound pattern: \text{(D) }49, \textbf{(C)}12, (B) 21, etc.
_MCQ_REF_RE = re.compile(
    r'^\\(?:textbf|text|mathrm|mathbf)\{\(?([A-E])\)?\s*\}\s*(.+)$'
    r'|^\(?([A-E])\)\s+(.+)$'
)


def _split_mcq(ans: str):
    """Split an MCQ answer into (letter, value) components.

    Handles compound forms (``\\text{(D) }49``, ``(B) 21``) as well as a
    bare letter (``D`` -> letter only) and a bare value (``21`` -> value only).
    Returns ``(letter_or_None, value_or_None)``.
    """
    s = ans.strip()
    m = _MCQ_REF_RE.match(s)
    if m:
        letter = m.group(1) or m.group(3)
        value = (m.group(2) or m.group(4) or '').strip()
        return letter, (value or None)
    # Bare single letter (with optional \text/\textbf/\mathbb wrapper or parens).
    # \mathbb{A} appears as a dirty reference label for option A in some rows.
    bl = re.match(r'^\\?(?:textbf|text|mathrm|mathbf|mathbb)?\{?\(?([A-E])\)?\}?$', s)
    if bl:
        return bl.group(1), None
    return None, s or None


def answers_match(predicted: str, reference: str) -> bool:
    """Check if two math answers are equivalent.

    Supports bidirectional MCQ matching: either side may be a bare option
    letter, a bare value, or a compound ``(letter) value`` form. The answer is
    considered correct if the letters match, or if the values match.
    """
    if not predicted or not reference:
        return False
    norm_p = normalize_answer(predicted)
    norm_r = normalize_answer(reference)
    if norm_p == norm_r:
        return True
    if _try_numeric_equal(norm_p, norm_r):
        return True

    # Symmetric MCQ matching: decompose both sides into (letter, value).
    p_letter, p_value = _split_mcq(predicted)
    r_letter, r_value = _split_mcq(reference)

    # Match on the option letter (only meaningful if both sides carry a letter).
    if p_letter and r_letter and p_letter == r_letter:
        return True

    # If both sides are letter-only (a bare option letter with no value), the
    # letters are the only signal; differing letters mean a mismatch. Do NOT
    # fall through to value comparison, which would spuriously match dirty
    # labels like \mathbb{A} vs \mathbb{B}.
    if (p_letter and p_value is None) and (r_letter and r_value is None):
        return False

    # Match on the value part (compare whichever value each side exposes; fall
    # back to the raw normalized string when a side has no separate value).
    p_val = normalize_answer(p_value) if p_value else norm_p
    r_val = normalize_answer(r_value) if r_value else norm_r
    if p_val and r_val:
        if p_val == r_val or _try_numeric_equal(p_val, r_val):
            return True
    return False


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_aops(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Load AoPS boxed problems, sample n, extract reference answers."""
    from modelscope import MsDataset
    ds = MsDataset.load(AOPS_DATASET_ID, split='train',
                        download_mode='reuse_dataset_if_exists')
    boxed = []
    for row in ds:
        if not row['metadata'].get('boxed'):
            continue
        ref = extract_boxed(row['solution'])
        if not ref:
            continue
        boxed.append({
            'problem': row['problem'],
            'solution': row['solution'],
            'reference_answer': ref,
            'tags': row.get('tags', []),
        })
    sys.stderr.write(f'[aops] {len(boxed)} boxed problems with extractable answers\n')
    rng = random.Random(seed)
    rng.shuffle(boxed)
    if n > 0 and n < len(boxed):
        boxed = boxed[:n]
        sys.stderr.write(f'[aops] sampled {n} problems\n')
    return boxed


def load_math(n: int, seed: int = 42, split: str = 'test',
              per_level: int = 0) -> List[Dict[str, Any]]:
    """Load the MATH (Hendrycks) dataset from local extracted JSON files.

    Each problem's reference answer is the ``\\boxed{}`` content of its
    ``solution`` (MATH solutions always end in a boxed answer).

    Sampling is *stratified by level* so every difficulty (Level 1-5) is
    represented equally — required to measure how RAG gain varies with
    difficulty. ``per_level`` (if >0) fixes the count per level; otherwise
    ``n`` is split evenly across the 5 levels. When both are 0, all problems
    are returned. The final list is shuffled with ``seed`` so index order is
    stable/comparable across direct vs rag runs.
    """
    import glob
    root = os.path.join(MATH_DATA_DIR, split)
    files = glob.glob(os.path.join(root, '*', '*.json'))
    if not files:
        raise FileNotFoundError(
            f'[math] no problems found under {root!r}; set MATH_DATA_DIR or '
            f'extract MATH.zip there')

    by_level: Dict[str, List[Dict[str, Any]]] = {}
    n_no_box = 0
    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8') as fin:
                row = json.load(fin)
        except Exception:
            continue
        ref = extract_boxed(row.get('solution', ''))
        if not ref:
            n_no_box += 1
            continue
        level = row.get('level', 'Unknown')
        by_level.setdefault(level, []).append({
            'problem': row['problem'],
            'solution': row['solution'],
            'reference_answer': ref,
            'level': level,
            'type': row.get('type', ''),
        })

    total = sum(len(v) for v in by_level.values())
    sys.stderr.write(
        f'[math] {total} problems with boxed answers across '
        f'{len(by_level)} levels (skipped {n_no_box} without boxed)\n')

    levels = sorted(by_level.keys())
    rng = random.Random(seed)

    # Decide how many per level.
    if per_level <= 0 and n > 0:
        per_level = max(1, n // max(1, len(levels)))

    sampled: List[Dict[str, Any]] = []
    for lv in levels:
        pool = by_level[lv]
        rng.shuffle(pool)
        take = pool if per_level <= 0 else pool[:per_level]
        sampled.extend(take)
        sys.stderr.write(f'[math]   {lv}: took {len(take)}/{len(pool)}\n')

    rng.shuffle(sampled)
    sys.stderr.write(f'[math] total sampled: {len(sampled)}\n')
    return sampled


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

DIRECT_SYSTEM = (
    'You are an expert competition mathematician. Solve the following problem '
    'step by step. Provide your final answer inside \\boxed{}.'
)

RAG_SYSTEM = (
    'You are an expert competition mathematician. Solve the following problem '
    'step by step. Provide your final answer inside \\boxed{}.\n\n'
    'You will first see example problem-solving traces or skills. '
    'Learn from the reasoning methodology demonstrated in these examples, '
    'then thinking to solve the actual problem.'
)

RAG_FOLLOWUP = (
    'The above is a reference solution to a similar problem. '
    'You may use any applicable techniques from it, or ignore it '
    'if you find a better approach. '
    'Solve the problem step by step and put your final answer in \\boxed{}.'
)

HINT_FOLLOWUP = (
    'The above are applicable solution approaches extracted from similar problems. '
    'You may use any applicable techniques from them, or ignore them '
    'if you find a better approach. '
    'Solve the problem step by step and put your final answer in \\boxed{}.'
)

# Reminder appended to the final user turn. Without this, the reasoning model
# can loop indefinitely on multiple-choice problems, oscillating between boxing
# the option letter and boxing the value (e.g. "I'll box B. I'll box 21. ...")
# and never terminating. Boxing BOTH the letter and value removes the ambiguity
# (the grader accepts either), so the model has no format decision to agonize over.
MCQ_INSTRUCTION = (
    '\n\nNote: If the problem is multiple-choice (it lists options such as '
    '(A), (B), (C), ...), put BOTH the option letter and its value in the box, '
    'e.g. \\boxed{(B) 21}. Otherwise, box the value directly. Decide the answer '
    'format once and do not deliberate over which form to box.'
)


def build_direct_prompt(problem: str) -> Dict[str, Any]:
    return {
        'messages': [
            {'role': 'system', 'content': DIRECT_SYSTEM},
            {'role': 'user', 'content': problem + MCQ_INSTRUCTION},
        ]
    }


def build_hint_prompt(problem: str, hint_analysis: str) -> Dict[str, Any]:
    """Build prompt with pre-analyzed hint in a multi-turn conversation.

    Mirrors ``build_rag_prompt``: the hint is presented as an assistant
    "extracted approaches" turn (instead of being buried in the system
    prompt), followed by a user instruction that provides a clear closing
    directive to solve the problem and box the answer. Keeping the final
    solve/box instruction in a dedicated user turn (rather than in the
    system prompt) helps the reasoning model terminate cleanly.
    """
    messages: List[Dict[str, str]] = [
        {'role': 'system', 'content': DIRECT_SYSTEM},
        {'role': 'user', 'content': problem},
        {'role': 'assistant',
         'content': ('Here are applicable solution approaches extracted from '
                     f'similar problems:\n\n{hint_analysis}')},
        {'role': 'user', 'content': HINT_FOLLOWUP + MCQ_INSTRUCTION},
    ]
    return {'messages': messages}


def build_rag_prompt(problem: str,
                     examples: List[Dict[str, str]]) -> Dict[str, Any]:
    """Approach B: multi-turn assistant format.

    The trace is presented as an assistant "retrieval" turn, followed by
    a user instruction that constrains the model to use methodology only.
    """
    messages: List[Dict[str, str]] = [{'role': 'system', 'content': DIRECT_SYSTEM}]
    messages.append({'role': 'user', 'content': problem})
    # Build trace content from retrieved examples
    trace_parts = []
    for i, ex in enumerate(examples, 1):
        trace_parts.append(f'[Retrieved Example {i}]\nProblem: {ex["query"]}\n'
                           f'Reasoning:\n{ex["thinking"]}')
    trace_text = '\n\n'.join(trace_parts)
    messages.append({'role': 'assistant',
                     'content': f'I found relevant reasoning traces from the knowledge base!\n\n{trace_text}'})
    messages.append({'role': 'user', 'content': RAG_FOLLOWUP + MCQ_INSTRUCTION})
    return {'messages': messages}


# ---------------------------------------------------------------------------
# 13-gram Jaccard decontamination
# ---------------------------------------------------------------------------

def _normalize_for_ngram(text: str) -> str:
    """Normalize text for n-gram comparison: strip LaTeX markup, lowercase."""
    text = text.lower()
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'\\[a-z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-z]+', ' ', text)
    text = re.sub(r'[{}\\^_$]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _ngram_jaccard(text_a: str, text_b: str, n: int = 13) -> float:
    """13-gram character-level Jaccard similarity."""
    a = _normalize_for_ngram(text_a)
    b = _normalize_for_ngram(text_b)
    if len(a) < n or len(b) < n:
        return 0.0
    grams_a = set(a[i:i + n] for i in range(len(a) - n + 1))
    grams_b = set(b[i:i + n] for i in range(len(b) - n + 1))
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


# ---------------------------------------------------------------------------
# Embedding / RAG helpers
# ---------------------------------------------------------------------------

def _wrap_anchor(text: str) -> List[Dict[str, str]]:
    return [
        {'role': 'user', 'content': text},
        {'role': 'assistant', 'content': 'Match the correct response here.'},
    ]


def get_embeddings(model: TransformersModel, template: Qwen3_5Template,
                   texts: List[str], dp_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    n = len(texts)
    pad_n = (-n) % dp_size
    padded = list(texts) + [' '] * pad_n if pad_n else list(texts)
    features = []
    for t in padded:
        feat = template.encode({'messages': _wrap_anchor(t or ' ')})
        feat['labels'] = [1]
        features.append(feat)
    out = model.forward_only(inputs=features, task='embedding', return_logits=True)
    emb = out['embeddings']
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().to(torch.float32).cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32)
    return emb[:n] if pad_n else emb


def retrieve_examples(tbl, query_vecs: np.ndarray, top_k: int,
                      use_thinking_raw: bool, sim_threshold: float = 0.0,
                      problems: List[str] = None,
                      decontam_threshold: float = 0.0,
                      ) -> List[List[Dict[str, str]]]:
    thinking_field = 'thinking_raw' if use_thinking_raw else 'cot_compressed'
    fetch_limit = top_k + 50 if decontam_threshold > 0 else top_k
    n_queries = len(query_vecs)
    all_examples: List[List[Dict[str, str]]] = [None] * n_queries
    decontam_skipped = 0
    _decontam_lock = threading.Lock()

    def _search_one(qi: int):
        nonlocal decontam_skipped
        vec = query_vecs[qi]
        results = (
            tbl.search(vec.astype(np.float32).tolist())
            .metric('dot')
            .limit(fetch_limit)
            .select(['query_raw', thinking_field, '_distance'])
            .to_list()
        )
        problem_text = problems[qi] if problems else ''
        examples = []
        local_skipped = 0
        for r in results:
            if len(examples) >= top_k:
                break
            sim = 1.0 - r.get('_distance', 0.0)
            if sim < sim_threshold:
                continue
            q = r.get('query_raw', '')
            t = r.get(thinking_field, '')
            if not t:
                continue
            if decontam_threshold > 0 and problem_text and q:
                ng_sim = _ngram_jaccard(problem_text, q)
                if ng_sim > decontam_threshold:
                    local_skipped += 1
                    continue
            examples.append({'query': q, 'thinking': t, '_sim': round(sim, 4),
                             '_raw_trace_len': len(t)})
        all_examples[qi] = examples
        if local_skipped:
            with _decontam_lock:
                decontam_skipped += local_skipped

    with ThreadPoolExecutor(max_workers=min(n_queries, 16)) as pool:
        list(pool.map(_search_one, range(n_queries)))

    if decontam_skipped > 0:
        logger.info(f'[decontam] skipped {decontam_skipped} leaked retrievals '
                    f'(13-gram Jaccard > {decontam_threshold})')
    return all_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['direct', 'rag'], default='rag')
    p.add_argument('--dataset', choices=['aops', 'math'], default='math',
                   help='Evaluation dataset. "math" = MATH (Hendrycks), '
                        'stratified by level for a difficulty-vs-gain curve.')
    p.add_argument('--math-split', default='test',
                   help='MATH split to load (test/train).')
    p.add_argument('--per-level', type=int, default=100,
                   help='MATH only: problems per difficulty level (default 100 '
                        '-> 500 total across Level 1-5). If 0, --n is split '
                        'evenly across the 5 levels.')
    p.add_argument('--n', type=int, default=0,
                   help='Pool size: sample this many problems (0 = all boxed). '
                        'In RAG mode with --target-eval, set this to 0 for max coverage.')
    p.add_argument('--target-eval', type=int, default=0,
                   help='Stop after this many problems are successfully evaluated '
                        '(0 = no limit, evaluate the entire sampled set — the '
                        'default, so all 500 stratified MATH problems are run). '
                        'RAG mode: counts problems with valid traces after '
                        'decontam; direct mode: ignored, evaluates all filtered.')
    p.add_argument('--db-path', default='./output.oldemb/thinking_rag/lance.db')
    p.add_argument('--table', default='thinking_traces')
    p.add_argument('--top-k', type=int, default=1)
    p.add_argument('--use-cot-compressed', action='store_true',
                   help='Use pre-compressed cot_compressed field instead of thinking_raw.')
    p.add_argument('--sim-threshold', type=float, default=0.75,
                   help='Minimum cosine similarity for retrieved traces. '
                        'Traces below this are discarded at retrieval time.')
    p.add_argument('--decontam-threshold', type=float, default=0.20,
                   help='13-gram Jaccard threshold for leak detection. '
                        'Retrieved traces above this are skipped (0=disabled).')
    p.add_argument('--llm-decontam', action='store_true', default=True,
                   help='LLM-based decontamination (default ON): API judges whether '
                        'retrieved problem is the same as the test problem. '
                        'Applied after 13-gram decontam, before condensing. '
                        'Use --no-llm-decontam to disable.')
    p.add_argument('--no-llm-decontam', dest='llm_decontam', action='store_false',
                   help='Disable LLM-based decontamination.')
    p.add_argument('--max-trace-len', type=int, default=12000)
    p.add_argument('--condense', action='store_true', default=True,
                   help='Enable condenser re-compression on retrieved traces '
                        '(default ON). Use --no-condense to inject raw traces.')
    p.add_argument('--no-condense', dest='condense', action='store_false',
                   help='Disable condenser; inject raw retrieved traces.')
    p.add_argument('--condense-max-len', type=int, default=2000,
                   help='Max chars of condensed trace (fallback truncation).')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hint', action='store_true', default=False,
                   help='Enable API hint filtering on retrieved traces (default OFF; '
                        'raw RAG injects the condensed trace directly). '
                        'In rag mode: retrieve → condense → API filters trace → refined system prompt. '
                        'In direct mode: ignored (no traces to filter).')
    p.add_argument('--no-hint', dest='hint', action='store_false',
                   help='Disable API hint filtering; inject condensed trace directly.')
    p.add_argument('--problem-ids-file', default=None,
                   help='File listing problem indices evaluated by RAG mode. '
                        'RAG mode writes this file; direct mode reads it to '
                        'evaluate the same subset (use --no-filter to disable). '
                        'Defaults to a dataset-specific path.')
    p.add_argument('--no-filter', action='store_true',
                   help='In direct mode, evaluate ALL sampled problems '
                        'instead of filtering to RAG subset.')
    p.add_argument('--output', default=None)
    args = p.parse_args()

    # Dataset-specific default paths (keeps aops and math runs from colliding).
    if args.problem_ids_file is None:
        args.problem_ids_file = (
            f'./output/thinking_rag/{args.dataset}_rag_problem_ids.json')

    if args.output is None:
        suffix = f'{args.mode}_hint' if (args.hint and args.mode == 'rag') else args.mode
        args.output = (
            f'./output/thinking_rag/{args.dataset}_{suffix}_results.jsonl')

    if args.condense and args.use_cot_compressed:
        logger.warning('--condense requires thinking_raw, ignoring --use-cot-compressed')
        args.use_cot_compressed = False

    if args.dataset == 'math':
        records = load_math(n=args.n, seed=args.seed, split=args.math_split,
                            per_level=args.per_level)
    else:
        records = load_aops(n=args.n, seed=args.seed)

    is_rag = (args.mode == 'rag')

    # Direct mode: filter to same problems RAG evaluated (controlled comparison)
    original_indices = list(range(len(records)))  # track original indices
    if not is_rag and not args.no_filter:
        if os.path.exists(args.problem_ids_file):
            with open(args.problem_ids_file) as f:
                content = f.read().strip()
            if content.startswith('['):
                valid_indices = set(json.loads(content))
            else:
                valid_indices = set(int(line) for line in content.splitlines() if line.strip())
            filtered = [(i, r) for i, r in enumerate(records) if i in valid_indices]
            original_indices = [i for i, _ in filtered]
            records = [r for _, r in filtered]
            sys.stderr.write(
                f'[direct] filtered to {len(records)} problems '
                f'from {args.problem_ids_file}\n')
        else:
            sys.stderr.write(
                f'[direct] WARNING: {args.problem_ids_file} not found, '
                f'running all {len(records)} problems\n')

    condenser_gpus = int(os.environ.get('EVAL_CONDENSER_GPUS', 0)) if args.condense else 0

    # Raw RAG relies on the API condenser (qwen3.7-max). Fail fast with a clear
    # message if it's enabled without an API key and without a local condenser.
    if is_rag and args.condense and not CONDENSE_API_KEY and condenser_gpus == 0:
        sys.stderr.write(
            '[condense] ERROR: --condense is ON but COMPRESS_API_KEY is unset '
            'and no local condenser (EVAL_CONDENSER_GPUS=0).\n'
            '  Fix one of:\n'
            '    - export COMPRESS_API_KEY=sk-...            (use API condenser)\n'
            '    - EVAL_CONDENSER_GPUS=2 python ...          (use local vLLM condenser)\n'
            '    - pass --no-condense                        (inject raw traces)\n')
        sys.exit(1)

    if is_rag:
        num_gpus = EMB_GPUS + GEN_GPUS + condenser_gpus
        device_groups = [
            DeviceGroup(name='emb_model', ranks=list(range(EMB_GPUS)),
                        device_type='GPU'),
            DeviceGroup(name='sampler',
                        ranks=list(range(EMB_GPUS, EMB_GPUS + GEN_GPUS)),
                        device_type='GPU', gpus_per_worker=GEN_GPUS),
        ]
        if condenser_gpus > 0:
            cond_start = EMB_GPUS + GEN_GPUS
            device_groups.append(
                DeviceGroup(name='condenser',
                            ranks=list(range(cond_start, cond_start + condenser_gpus)),
                            device_type='GPU'))
        emb_mesh = DeviceMesh.from_sizes(world_size=EMB_GPUS, dp_size=EMB_GPUS)
        gen_mesh = DeviceMesh.from_sizes(world_size=GEN_GPUS, tp_size=GEN_GPUS)
        twinkle.initialize(mode='ray', nproc_per_node=num_gpus,
                           groups=device_groups, lazy_collect=False)
    else:
        device_groups = [
            DeviceGroup(name='sampler', ranks=list(range(GEN_GPUS)),
                        device_type='GPU', gpus_per_worker=GEN_GPUS),
        ]
        gen_mesh = DeviceMesh.from_sizes(world_size=GEN_GPUS, tp_size=GEN_GPUS)
        twinkle.initialize(mode='ray', nproc_per_node=GEN_GPUS,
                           groups=device_groups, lazy_collect=False)

    sampler = vLLMSampler(
        model_id=GEN_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': GEN_GPU_MEM,
            'max_model_len': GEN_MAX_MODEL_LEN,
        },
        device_mesh=gen_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                         enable_thinking=True, max_length=GEN_MAX_MODEL_LEN)
    sys.stderr.write(f'[aops] vLLM sampler ready (model={GEN_MODEL_ID})\n')

    gen_params = TwinkleSamplingParams(
        max_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        num_samples=1,
    )

    emb_model = emb_template = tbl = None
    if is_rag:
        import lancedb
        db = lancedb.connect(args.db_path)
        if args.table not in db.table_names():
            raise SystemExit(f'Table "{args.table}" not found in {args.db_path}')
        tbl = db.open_table(args.table)
        sys.stderr.write(f'[aops] LanceDB rows={tbl.count_rows()}\n')

        emb_model = TransformersModel(
            model_id=EMBED_MODEL_ID, device_mesh=emb_mesh,
            remote_group='emb_model')
        emb_model.set_processor(InputProcessor)
        emb_model.set_loss(InfonceLoss, temperature=0.03, use_batch=True)
        emb_template = Qwen3_5Template(
            model_id=EMBED_MODEL_ID, max_length=EMBED_MAX_LENGTH,
            truncation_strategy='delete', enable_thinking=False)
        sys.stderr.write('[aops] embedding model ready\n')

    # -- Condenser setup (API primary + optional local vLLM) -------------------
    condenser_api_client = None
    condenser_sampler_obj = None
    condenser_params = None
    condenser_special_tokens = None

    if args.condense and is_rag:
        condenser_api_client = OpenAIClient(
            model=CONDENSE_API_MODEL, api_key=CONDENSE_API_KEY,
            base_url=CONDENSE_BASE_URL)
        sys.stderr.write(f'[condense] API client ready (model={CONDENSE_API_MODEL})\n')

        if condenser_gpus > 0:
            condenser_mesh = DeviceMesh.from_sizes(
                world_size=condenser_gpus, dp_size=condenser_gpus)
            condenser_sampler_obj = vLLMSampler(
                model_id=CONDENSE_MODEL_ID,
                engine_args={'gpu_memory_utilization': 0.8, 'max_model_len': 32768},
                device_mesh=condenser_mesh,
                remote_group='condenser',
            )
            condenser_sampler_obj.set_template(
                'Qwen3_5Template', model_id=CONDENSE_MODEL_ID,
                enable_thinking=False, truncation_strategy='delete',
                max_length=32768)
            condenser_template = Qwen3_5Template(
                model_id=CONDENSE_MODEL_ID, max_length=32768,
                enable_thinking=False, truncation_strategy='delete')
            condenser_special_tokens = set(condenser_template.tokenizer.all_special_tokens)
            condenser_params = TwinkleSamplingParams(
                max_tokens=CONDENSE_MAX_TOKENS,
                temperature=CONDENSE_TEMPERATURE,
                top_p=0.5, num_samples=1)
            sys.stderr.write(f'[condense] local vLLM ready (model={CONDENSE_MODEL_ID})\n')

    # -- Hint analysis API client (reuses condenser API config) -----------------
    hint_api_client = None
    if args.hint and is_rag:
        if condenser_api_client is not None:
            hint_api_client = condenser_api_client
        else:
            hint_api_client = OpenAIClient(
                model=CONDENSE_API_MODEL, api_key=CONDENSE_API_KEY,
                base_url=CONDENSE_BASE_URL)
        sys.stderr.write(f'[hint] API hint analysis enabled (model={CONDENSE_API_MODEL})\n')

    # -- LLM decontam API client ---------------------------------------------------
    decontam_api_client = None
    if args.llm_decontam and is_rag:
        if hint_api_client is not None:
            decontam_api_client = hint_api_client
        elif condenser_api_client is not None:
            decontam_api_client = condenser_api_client
        else:
            decontam_api_client = OpenAIClient(
                model=CONDENSE_API_MODEL, api_key=CONDENSE_API_KEY,
                base_url=CONDENSE_BASE_URL)
        sys.stderr.write(f'[decontam-llm] LLM decontamination enabled (model={CONDENSE_API_MODEL})\n')

    correct_count = 0
    total_count = 0
    skipped_indices: List[int] = []  # problems skipped by RAG (no valid trace)
    evaluated_indices: List[int] = []  # problems actually evaluated
    debug_records: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_f = open(args.output, 'w', encoding='utf-8')

    # Open problem-ids files for incremental writing (RAG mode only)
    ids_f = None
    skip_f = None
    if is_rag:
        os.makedirs(os.path.dirname(args.problem_ids_file) or '.', exist_ok=True)
        ids_f = open(args.problem_ids_file, 'w', encoding='utf-8')
        skip_path = args.problem_ids_file.replace('.json', '_skipped.json')
        skip_f = open(skip_path, 'w', encoding='utf-8')

    # -- RAG batch preparation (embed + retrieve + decontam + condense + hint) --
    def _prepare_rag_batch(batch_start: int):
        """Prepare a RAG batch: returns (prompts, batch, all_examples,
        hint_analyses, kept_global_indices, batch_skipped_indices) or None."""
        batch_end = min(batch_start + args.batch_size, len(records))
        batch = records[batch_start:batch_end]
        problems = [r['problem'] for r in batch]

        query_vecs = get_embeddings(emb_model, emb_template, problems, EMB_GPUS)
        use_raw = not args.use_cot_compressed
        all_examples = retrieve_examples(tbl, query_vecs, args.top_k,
                                         use_raw, args.sim_threshold,
                                         problems=problems,
                                         decontam_threshold=args.decontam_threshold)
        if args.use_cot_compressed:
            for exs in all_examples:
                for ex in exs:
                    ex['thinking'] = _strip_condenser_markers(ex['thinking'])

        if args.llm_decontam and decontam_api_client:
            all_examples = _llm_decontaminate(
                decontam_api_client, problems, all_examples)

        if args.condense and condenser_api_client:
            all_examples = condense_traces(
                all_examples, problems, condenser_api_client,
                condenser_sampler=condenser_sampler_obj,
                compress_params=condenser_params,
                special_tokens=condenser_special_tokens,
                max_output_len=args.condense_max_len,
                dp_size=condenser_gpus)

        hint_analyses = None
        if args.hint and hint_api_client:
            hint_analyses = _api_hint_analysis_batch(
                hint_api_client, problems, all_examples)

        keep_mask = []
        for pi, (r, examples) in enumerate(zip(batch, all_examples)):
            if not examples:
                keep_mask.append(False)
            elif hint_analyses and hint_analyses[pi]:
                keep_mask.append(True)
            else:
                usable = [ex for ex in examples
                          if len(ex['thinking']) <= args.max_trace_len]
                keep_mask.append(bool(usable))

        batch_skipped = []
        for pi, keep in enumerate(keep_mask):
            if not keep:
                batch_skipped.append(batch_start + pi)

        kept_batch = []
        kept_examples = []
        kept_hints = []
        kept_global_indices = []
        for pi, keep in enumerate(keep_mask):
            if keep:
                kept_batch.append(batch[pi])
                kept_examples.append(all_examples[pi])
                kept_hints.append(hint_analyses[pi] if hint_analyses else None)
                kept_global_indices.append(batch_start + pi)

        if not kept_batch:
            return None, None, None, None, None, batch_skipped

        prompts = []
        for pi, (r, examples) in enumerate(zip(kept_batch, kept_examples)):
            if kept_hints[pi]:
                prompts.append(build_hint_prompt(r['problem'], kept_hints[pi]))
            else:
                filtered = [{'query': ex['query'], 'thinking': ex['thinking']}
                            for ex in examples
                            if len(ex['thinking']) <= args.max_trace_len]
                prompts.append(build_rag_prompt(r['problem'], filtered))

        return prompts, kept_batch, kept_examples, kept_hints, kept_global_indices, batch_skipped

    target_reached = False
    batch_starts = list(range(0, len(records), args.batch_size))

    if is_rag:
        # Pipeline: prefetch next batch while current batch generates
        from concurrent.futures import Future
        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        # Prepare first batch synchronously
        cur_result = _prepare_rag_batch(batch_starts[0])

        for bi, batch_start in enumerate(batch_starts):
            if target_reached:
                break
            prompts, batch, all_examples, hint_analyses, kept_global_indices, batch_skipped = cur_result
            skipped_indices.extend(batch_skipped or [])
            if skip_f and batch_skipped:
                for sid in batch_skipped:
                    skip_f.write(f'{sid}\n')
                skip_f.flush()

            # Submit next batch preparation in background
            next_future: Optional[Future] = None
            if bi + 1 < len(batch_starts) and not target_reached:
                next_future = prefetch_pool.submit(_prepare_rag_batch, batch_starts[bi + 1])

            if prompts is None:
                # Entire batch skipped
                cur_result = next_future.result() if next_future else None
                continue

            # Generate (runs on gen GPU while next batch prepares on emb GPU + API)
            responses = sampler.sample(prompts, gen_params)

            for i, (rec, resp) in enumerate(zip(batch, responses)):
                seq = resp.sequences[0] if resp and resp.sequences else None
                raw_output = ''
                if seq is not None:
                    raw_output = seq.decoded or ''
                    raw_output = re.sub(r'<\|[^|]+\|>', '', raw_output).rstrip()

                predicted = extract_boxed(raw_output)
                is_correct = answers_match(predicted, rec['reference_answer'])
                if is_correct:
                    correct_count += 1
                total_count += 1

                global_idx = kept_global_indices[i]
                evaluated_indices.append(global_idx)
                if ids_f:
                    ids_f.write(f'{global_idx}\n')
                    ids_f.flush()

                debug_rec = {
                    'idx': global_idx,
                    'reference_answer': rec['reference_answer'],
                    'predicted': predicted,
                    'is_correct': is_correct,
                    'problem': rec['problem'],
                    'model_output': raw_output,
                }
                if rec.get('level'):
                    debug_rec['level'] = rec['level']
                if rec.get('type'):
                    debug_rec['type'] = rec['type']
                debug_rec['num_traces'] = len(all_examples[i])
                if all_examples[i]:
                    ex0 = all_examples[i][0]
                    debug_rec['similarity'] = ex0.get('_sim', 0.0)
                    debug_rec['retrieved_query'] = ex0.get('query', '')
                    debug_rec['raw_trace_len'] = ex0.get('_raw_trace_len', 0)
                    debug_rec['condensed_trace'] = ex0['thinking']
                    debug_rec['condensed_trace_len'] = len(ex0['thinking'])
                    debug_rec['condense_source'] = ex0.get('_condense_source', '')
                if hint_analyses and hint_analyses[i]:
                    debug_rec['hint_analysis'] = hint_analyses[i]
                debug_records.append(debug_rec)
                out_f.write(json.dumps(debug_rec, ensure_ascii=False) + '\n')
                out_f.flush()

            acc = correct_count / total_count if total_count else 0
            sys.stderr.write(
                f'  [{total_count}/{args.target_eval}] '
                f'acc={acc:.4f} ({correct_count}/{total_count})\n')

            if args.target_eval > 0 and total_count >= args.target_eval:
                target_reached = True

            # Collect prefetched result for next iteration (skip if done)
            if not target_reached and next_future:
                cur_result = next_future.result()
            else:
                cur_result = None

        prefetch_pool.shutdown(wait=True)
    else:
        # Direct mode: no pipeline needed, just batch generate
        for batch_start in batch_starts:
            batch_end = min(batch_start + args.batch_size, len(records))
            batch = records[batch_start:batch_end]
            prompts = [build_direct_prompt(r['problem']) for r in batch]

            responses = sampler.sample(prompts, gen_params)

            for i, (rec, resp) in enumerate(zip(batch, responses)):
                seq = resp.sequences[0] if resp and resp.sequences else None
                raw_output = ''
                if seq is not None:
                    raw_output = seq.decoded or ''
                    raw_output = re.sub(r'<\|[^|]+\|>', '', raw_output).rstrip()

                predicted = extract_boxed(raw_output)
                is_correct = answers_match(predicted, rec['reference_answer'])
                if is_correct:
                    correct_count += 1
                total_count += 1

                global_idx = original_indices[batch_start + i]
                evaluated_indices.append(global_idx)

                debug_rec = {
                    'idx': global_idx,
                    'reference_answer': rec['reference_answer'],
                    'predicted': predicted,
                    'is_correct': is_correct,
                    'problem': rec['problem'],
                    'model_output': raw_output,
                }
                if rec.get('level'):
                    debug_rec['level'] = rec['level']
                if rec.get('type'):
                    debug_rec['type'] = rec['type']
                debug_records.append(debug_rec)
                out_f.write(json.dumps(debug_rec, ensure_ascii=False) + '\n')
                out_f.flush()

            acc = correct_count / total_count if total_count else 0
            sys.stderr.write(
                f'  [{total_count}/{len(records)}] '
                f'acc={acc:.4f} ({correct_count}/{total_count})\n')

    overall_acc = correct_count / total_count if total_count else 0
    print(f'\n{"=" * 60}')
    print(f'{args.dataset.upper()} — mode={args.mode}, model={GEN_MODEL_ID}')
    print(f'  n={total_count}, seed={args.seed}')
    if is_rag:
        print(f'  evaluated={len(evaluated_indices)}, skipped={len(skipped_indices)}')
    print(f'{"=" * 60}')
    print(f'Overall accuracy: {overall_acc:.4f}  ({correct_count}/{total_count})')

    # Per-level breakdown (MATH: the difficulty-vs-gain curve we care about).
    if any(r.get('level') for r in debug_records):
        from collections import defaultdict
        per = defaultdict(lambda: [0, 0])  # level -> [correct, total]
        for r in debug_records:
            lv = r.get('level', 'Unknown')
            per[lv][1] += 1
            if r['is_correct']:
                per[lv][0] += 1
        print(f'\nPer-level accuracy:')
        for lv in sorted(per.keys()):
            c, t = per[lv]
            print(f'  {lv:>10}: {c/t:.4f}  ({c}/{t})')

    out_f.close()
    print(f'\n[output] {len(debug_records)} records saved to {args.output}')

    if ids_f:
        ids_f.close()
        print(f'[output] problem IDs ({len(evaluated_indices)}) saved to {args.problem_ids_file}')
    if skip_f:
        skip_f.close()
        if skipped_indices:
            print(f'[output] skipped IDs ({len(skipped_indices)}) saved to '
                  f'{args.problem_ids_file.replace(".json", "_skipped.json")}')


if __name__ == '__main__':
    main()
