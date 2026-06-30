"""RAG-hint GRPO training: retrieve thinking traces and condense as hints for RL.

Architecture (8 GPUs):
  - 1 GPU: condenser (vLLM, compress retrieved traces)
  - 1 GPU: embedding model (encode queries for retrieval)
  - 4 GPUs: sampler/rollout (vLLM TP=4)
  - 2 GPUs: training model (FSDP/DP)

Pipeline per step:
  1. DataLoader yields a batch of math problems
  2. [Async] Embedding model encodes problems → retrieve from LanceDB → condenser compresses
  3. Build RAG-hint prompts (one-shot in system, with analysis prefix)
  4. Sampler generates rollouts (response starts with forced analysis prefix)
  5. Reward (accuracy + format) → GRPO advantage → model update

Launch:
    python cookbook/exp/rl/rag_hint_grpo.py
"""
import json
import os
import re
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import InfonceLoss
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

logger = get_logger()

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')

# GPU layout: 1 condenser + 1 embedding + 4 rollout + 2 train = 8
CONDENSER_GPUS = int(os.environ.get('CONDENSER_GPUS', 1))
EMB_GPUS = int(os.environ.get('EMB_GPUS', 1))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 2))
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
NUM_GPUS = CONDENSER_GPUS + EMB_GPUS + SAMPLER_GPUS + MODEL_GPUS

# Training hyperparams
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 32768))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 5000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 100))
ADV_CLIP = float(os.environ.get('ADV_CLIP', 2.0))

# RAG config
DB_PATH = os.environ.get('DB_PATH', './output.oldemb/thinking_rag/lance.db')
DB_TABLE = os.environ.get('DB_TABLE', 'thinking_traces')
TOP_K = int(os.environ.get('TOP_K', 2))
SIM_THRESHOLD = float(os.environ.get('SIM_THRESHOLD', 0.65))
MAX_TRACE_LEN = int(os.environ.get('MAX_TRACE_LEN', 8192))
EMBED_MODEL_ID = os.environ.get(
    'EMBED_MODEL_ID', 'output.oldemb/embedding_full_transformers/last-checkpoint')
EMBED_MAX_LENGTH = int(os.environ.get('EMBED_MAX_LENGTH', 32000))

# Condenser config
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
CONDENSE_API_KEY = os.environ.get('COMPRESS_API_KEY', '')
CONDENSE_BASE_URL = os.environ.get('COMPRESS_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
CONDENSE_API_MODEL = os.environ.get('COMPRESS_MODEL', 'qwen3.7-max')
CONDENSE_API_CONCURRENCY = int(os.environ.get('API_CONCURRENCY', 16))
CONDENSE_TEMPERATURE = 0.2
CONDENSE_MAX_TOKENS = 8192

# Dataset
AOPS_DATASET_ID = os.environ.get('AOPS_DATASET_ID', 'AI-MO/aops')
AOPS_SEED = int(os.environ.get('AOPS_SEED', 100))

# Decontamination & RAG fallback
DECONTAM_THRESHOLD = float(os.environ.get('DECONTAM_THRESHOLD', 0.20))
RAG_FALLBACK_SIM = float(os.environ.get('RAG_FALLBACK_SIM', 0.60))

# Output / diagnostics
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './outputs/rag_hint_grpo')

# Forced analysis prefix appended at the start of assistant response
ANALYSIS_PREFIX = (
    "Let me think step by step. First, I will analyze the example above: "
    "identify which steps and concepts are CORRECT and APPLICABLE to this problem, "
    "and explicitly discard any reasoning that is WRONG, IRRELEVANT, or based on "
    "assumptions that do not hold here. Then I will solve the problem using only "
    "the validated useful parts:\n\n"
)

# ============================================================================
# Condenser prompt (strategy-level extraction)
# ============================================================================
COMPRESS_SYSTEM = """\
You are a reasoning-trace condenser. Given a verbose reasoning trace, \
extract the TRANSFERABLE KNOWLEDGE as an EXECUTABLE SOLUTION SKELETON \
that would help a reader solve SIMILAR problems in the same domain.

The reader will apply this knowledge to a DIFFERENT problem, so focus on what transfers. \
NEVER output the final answer or conclusion of the original problem. \
NEVER include problem-specific numeric results.

Principles:
1. OUTPUT AN EXECUTABLE STEP CHAIN: numbered steps that a solver can directly follow. \
Each step should state WHAT/WHY/HOW (with the formula/technique), not just name the concept.
2. INCLUDE FULL FORMULAS: theorems, identities — state each with COMPLETE MATHEMATICAL EXPRESSION.
3. STATE APPLICABILITY: what structural features signal that this approach works.
4. PRESERVE KEY INSIGHTS: non-obvious ideas that make the approach work.
5. REMOVE: problem-specific numeric calculations, final answers, dead-end explorations, hesitations.
6. FORMAT: Start with "Applicability:" one-line, then numbered steps. Keep concise.
7. NO meta-commentary. NO preamble. NO final answer.
"""

COMPRESS_USER = (
    '## Reader Problem (context only — do NOT solve it)\n{query}\n\n'
    '## Reasoning Trace to Condense\n{text}')

# ============================================================================
# RAG system prompt template (few-shot in system)
# ============================================================================
SYSTEM_WITH_RAG_HEADER = (
    'You are an expert competition mathematician. '
    'Below are condensed reasoning examples from similar problems. '
    'Analyze them critically — identify which steps/concepts are applicable '
    'and which may not apply. Then solve the actual problem step by step. '
    'Put your final answer inside \\boxed{}. '
    'For multiple-choice questions, put the option LETTER (A/B/C/D/E) inside \\boxed{}.\n\n'
)

EXAMPLE_TEMPLATE = (
    '--- Example {idx} ---\n'
    'Problem: {example_query}\n'
    'Methodology:\n{example_thinking}\n'
    '--- End Example {idx} ---\n'
)

SYSTEM_DIRECT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. Put your final answer inside \\boxed{}. '
    'For multiple-choice questions, put the option LETTER (A/B/C/D/E) inside \\boxed{}.'
)


# ============================================================================
# Condenser utilities
# ============================================================================
_api_semaphore = threading.Semaphore(CONDENSE_API_CONCURRENCY)


def _api_condense_single(api_client: OpenAIClient, messages: List[Dict]) -> Optional[str]:
    _api_semaphore.acquire()
    try:
        trajectory = {'messages': messages}
        sp = SamplingParams(temperature=CONDENSE_TEMPERATURE, max_tokens=CONDENSE_MAX_TOKENS)
        reply = api_client(trajectory, sp, extra_body={'enable_thinking': False})
        content = (reply.get('content') or '').strip()
        if not content:
            return None
        return content
    except Exception as exc:
        logger.warning(f'[condense-api] error: {exc}')
        return None
    finally:
        _api_semaphore.release()


# ============================================================================
# Embedding & Retrieval
# ============================================================================
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
    """13-gram character-level Jaccard similarity for decontamination."""
    a = _normalize_for_ngram(text_a)
    b = _normalize_for_ngram(text_b)
    if len(a) < n or len(b) < n:
        return 0.0
    grams_a = set(a[i:i + n] for i in range(len(a) - n + 1))
    grams_b = set(b[i:i + n] for i in range(len(b) - n + 1))
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


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


def retrieve_topk(tbl, query_vecs: np.ndarray, problems: List[str],
                  sim_threshold: float
                  ) -> List[List[Dict[str, Any]]]:
    """Retrieve top-K thinking_raw per query with decontamination and length filter.

    Returns per-query list of dicts with keys: query, thinking, sim.
    """
    results = []
    decontam_skipped = 0
    for qi, vec in enumerate(query_vecs):
        hits = (
            tbl.search(vec.astype(np.float32).tolist())
            .metric('dot')
            .limit(TOP_K + 50)
            .select(['query_raw', 'thinking_raw', '_distance'])
            .to_list()
        )
        matched = []
        problem_text = problems[qi] if problems else ''
        for h in hits:
            if len(matched) >= TOP_K:
                break
            sim = 1.0 - h.get('_distance', 0.0)
            if sim < sim_threshold:
                continue
            q = h.get('query_raw', '')
            t = h.get('thinking_raw', '')
            if not t:
                continue
            # Decontamination: skip if retrieved problem is too similar to current
            if DECONTAM_THRESHOLD > 0 and problem_text and q:
                if _ngram_jaccard(problem_text, q) > DECONTAM_THRESHOLD:
                    decontam_skipped += 1
                    continue
            # Drop traces exceeding max length (don't truncate — they'll be condensed poorly)
            if len(t) > MAX_TRACE_LEN * 4:
                continue
            matched.append({'query': q, 'thinking': t, 'sim': sim})
        results.append(matched)
    if decontam_skipped > 0:
        logger.info(f'[decontam] skipped {decontam_skipped} leaked retrievals')
    return results


# ============================================================================
# Reward
# ============================================================================
class AoPSAccuracyReward(Reward):
    """Accuracy reward via boxed answer extraction + robust equivalence matching."""

    @staticmethod
    def extract_boxed(text: str) -> str:
        idx = text.rfind('\\boxed{')
        if idx == -1:
            return ''
        start = idx + len('\\boxed{')
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            return text[start:j - 1].strip()
        return ''

    # --- MCQ letter regex (matches \textbf{(C) }value, (C) value, etc.) ---
    _MCQ_GT_RE = re.compile(
        r'^\\?(?:textbf|mathbf|text|mathrm)\{?\(?([A-E])[)}\s\\]*(.*)',
        re.DOTALL)
    _MCQ_PAREN_RE = re.compile(r'^\(?([A-E])\)?[\s\\]+(.*)', re.DOTALL)
    _MCQ_SINGLE_LETTER_RE = re.compile(r'^[A-E]$')
    # variable prefix: f(x)=..., m=..., N=..., P(n+1)=...
    _VAR_PREFIX_RE = re.compile(r'^[a-zA-Z](?:\([^)]*\))?\s*=\s*(.+)', re.DOTALL)
    # GT with derivation: "18×1+999×2=2016" → extract RHS
    _EQ_RHS_RE = re.compile(r'^.+=\s*(.+)$')

    @staticmethod
    def normalize_answer(ans: str) -> str:
        if not ans:
            return ''
        s = ans.strip()
        # Pure MCQ letter
        m = re.match(r'^\\?(?:textbf|text|mathrm|mathbf)?\{?\(?([A-E])\)?\}?$', s)
        if m:
            return m.group(1)
        s = s.replace(' ', '')
        s = s.replace(r'\,', '')
        s = s.replace(r'\;', '')
        s = s.replace(r'\!', '')
        # Remove text-mode wrappers but keep content (unwrap braces)
        s = re.sub(r'\\(?:text|mathrm|mathbf|textbf|operatorname)\{([^}]*)\}', r'\1', s)
        s = s.replace(r'\displaystyle', '')
        s = re.sub(r'\\(?:left|right)[.()\[\]|]', '', s)
        s = s.replace(r'\dfrac', r'\frac')
        s = s.replace(r'\tfrac', r'\frac')
        s = s.strip('$').strip()
        # Remove trailing unit braces: {cm}, {kg}, etc.
        s = re.sub(r'\{[a-zA-Z]+\}$', '', s)
        # Degree normalization — strip entirely (degrees are contextual)
        s = re.sub(r'\^\{\\circ\}|\^\\circ|°|\\circ', '', s)
        # Remove \quad, \qquad, \  etc spacing
        s = re.sub(r'\\(?:quad|qquad|\s)', '', s)
        # Normalize minus: \minus{} → -
        s = s.replace(r'\minus{}', '-').replace(r'\minus', '-')

        def _frac_to_slash(m):
            text = m.group(0)
            pos = text.index('{') + 1
            depth, num_start = 1, pos
            while depth > 0:
                if text[pos] == '{': depth += 1
                elif text[pos] == '}': depth -= 1
                pos += 1
            numer = text[num_start:pos - 1]
            pos += 1
            den_start = pos
            depth = 1
            while depth > 0:
                if text[pos] == '{': depth += 1
                elif text[pos] == '}': depth -= 1
                pos += 1
            denom = text[den_start:pos - 1]
            return f'({numer})/({denom})'

        s = re.sub(
            r'\\frac\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            _frac_to_slash, s)
        s = re.sub(r'(?<!\w)(\d+)/(\d+)(?!\w)', r'(\1)/(\2)', s)
        return s

    @classmethod
    def _strip_var_prefix(cls, s: str) -> str:
        """Strip variable assignment prefix: 'f(x)=x+1' → 'x+1', 'N=1006' → '1006'."""
        m = cls._VAR_PREFIX_RE.match(s)
        return m.group(1).strip() if m else s

    @classmethod
    def _extract_mcq_parts(cls, s: str):
        """Extract (letter, value) from MCQ-formatted string. Returns (None, None) if not MCQ."""
        m = cls._MCQ_GT_RE.match(s)
        if m:
            return m.group(1), m.group(2).strip()
        m = cls._MCQ_PAREN_RE.match(s)
        if m:
            return m.group(1), m.group(2).strip()
        # GT ends with " (A)" pattern: "2+2\sqrt{7} (A)"
        m2 = re.search(r'\(?([A-E])\)?\s*$', s)
        if m2 and len(s) > 3:
            return m2.group(1), s[:m2.start()].strip()
        return None, None

    @staticmethod
    def _try_numeric_equal(a: str, b: str) -> bool:
        """Try numeric equality after normalization. Handles fracs and simple expressions."""
        import math

        def _try_eval(s: str):
            # Direct float
            try:
                return float(s.replace('(', '').replace(')', ''))
            except (ValueError, ZeroDivisionError):
                pass
            # Strip trailing unit-like suffix and retry
            s_stripped = re.sub(r'[a-zA-Z]+$', '', s.replace('(', '').replace(')', '')).strip()
            if s_stripped and s_stripped != s:
                try:
                    return float(s_stripped)
                except (ValueError, ZeroDivisionError):
                    pass
            # Fraction pattern (a)/(b)
            frac_re = re.compile(r'^\(([^)]+)\)/\(([^)]+)\)$')
            m = frac_re.match(s)
            if m:
                try:
                    return float(m.group(1)) / float(m.group(2))
                except (ValueError, ZeroDivisionError):
                    pass
            # Try evaluating simple math expressions (pi, sqrt, etc.)
            expr = s
            expr = expr.replace('\\pi', str(math.pi))
            expr = expr.replace('\\e', str(math.e))
            expr = re.sub(r'\\sqrt\{([^}]+)\}', r'(\1)**0.5', expr)
            expr = re.sub(r'\\sqrt\[3\]\{([^}]+)\}', r'(\1)**(1/3)', expr)
            expr = re.sub(r'\\sqrt\[([^]]+)\]\{([^}]+)\}', r'(\2)**(1/\1)', expr)
            expr = expr.replace('{', '(').replace('}', ')')
            expr = expr.replace('\\cdot', '*').replace('\\times', '*')
            expr = re.sub(r'(\d)\(', r'\1*(', expr)
            try:
                val = eval(expr, {"__builtins__": {}, "math": math, "pi": math.pi, "e": math.e}, {})
                return float(val)
            except Exception:
                pass
            return None

        va, vb = _try_eval(a), _try_eval(b)
        if va is not None and vb is not None:
            return abs(va - vb) < 1e-6 * max(1, abs(va), abs(vb))
        return False

    @classmethod
    def _normalize_tuple(cls, s: str) -> str:
        """Normalize tuple formatting: (2, 5, 609) → 2,5,609."""
        return re.sub(r'[\s()\[\]]', '', s)

    @classmethod
    def _try_sympy_equal(cls, a: str, b: str) -> bool:
        """Optional sympy-based algebraic equivalence (graceful fallback if unavailable)."""
        try:
            from sympy.parsing.latex import parse_latex
            from sympy import simplify, nsimplify
            expr_a = parse_latex(a)
            expr_b = parse_latex(b)
            diff = simplify(nsimplify(expr_a - expr_b))
            return diff == 0
        except Exception:
            return False

    @classmethod
    def answers_match(cls, predicted: str, reference: str) -> bool:
        if not predicted or not reference:
            return False

        norm_p = cls.normalize_answer(predicted)
        norm_r = cls.normalize_answer(reference)

        # --- Strategy 1: direct string equality ---
        if norm_p == norm_r:
            return True

        # --- Strategy 2: case-insensitive ---
        if norm_p.lower() == norm_r.lower():
            return True

        # --- Strategy 3: numeric equality ---
        if cls._try_numeric_equal(norm_p, norm_r):
            return True

        # --- Strategy 4: variable-prefix stripping (both sides) ---
        stripped_p = cls.normalize_answer(cls._strip_var_prefix(predicted))
        stripped_r = cls.normalize_answer(cls._strip_var_prefix(reference))
        if stripped_p and stripped_r and stripped_p == stripped_r:
            return True
        if stripped_p and stripped_r and cls._try_numeric_equal(stripped_p, stripped_r):
            return True

        # --- Strategy 5: MCQ double matching ---
        # Extract letter+value from reference
        ref_letter, ref_value = cls._extract_mcq_parts(reference)
        if ref_letter:
            # pred matches the letter?
            if norm_p == ref_letter or predicted.strip().upper() == ref_letter:
                return True
            # pred matches the value?
            if ref_value:
                norm_ref_val = cls.normalize_answer(ref_value)
                if norm_p == norm_ref_val or cls._try_numeric_equal(norm_p, norm_ref_val):
                    return True
        # Extract from predicted side too (pred="B", ref has value)
        pred_letter, pred_value = cls._extract_mcq_parts(predicted)
        if pred_letter:
            if norm_r == pred_letter or reference.strip().upper() == pred_letter:
                return True
            if pred_value:
                norm_pred_val = cls.normalize_answer(pred_value)
                if norm_r == norm_pred_val or cls._try_numeric_equal(norm_r, norm_pred_val):
                    return True
        # MCQ: GT is single letter, pred is numeric/expression → match if pred chose option
        if cls._MCQ_SINGLE_LETTER_RE.match(reference.strip()):
            if cls._MCQ_SINGLE_LETTER_RE.match(predicted.strip().upper()):
                return predicted.strip().upper() == reference.strip().upper()
            # pred is a value, GT is just a letter: we accept pred=letter match only
            # (can't verify value without options text)

        # --- Strategy 6: tuple/set normalization ---
        tuple_p = cls._normalize_tuple(norm_p)
        tuple_r = cls._normalize_tuple(norm_r)
        if ',' in tuple_p and tuple_p == tuple_r:
            return True

        # --- Strategy 7: equation reorder (a+b=c vs c=a+b, or lhs=rhs swapped) ---
        if '=' in norm_r and '=' not in norm_p:
            # GT has derivation like "18*1+999*2=2016", pred is "2016"
            parts = norm_r.split('=')
            for part in parts:
                part = part.strip()
                if part == norm_p or cls._try_numeric_equal(part, norm_p):
                    return True
        if '=' in norm_p and '=' not in norm_r:
            parts = norm_p.split('=')
            for part in parts:
                part = part.strip()
                if part == norm_r or cls._try_numeric_equal(part, norm_r):
                    return True
        if '=' in norm_p and '=' in norm_r:
            # Both have =: try matching LHS=RHS in any order
            pp = [x.strip() for x in norm_p.split('=')]
            rp = [x.strip() for x in norm_r.split('=')]
            if set(pp) == set(rp):
                return True

        # --- Strategy 8: multiplicative reorder (27\pi\sqrt{6} vs 27\sqrt{6}\pi) ---
        def _sort_factors(s):
            tokens = re.findall(r'\\?[a-zA-Z]+\{[^}]*\}|\\?[a-zA-Z]+|\d+|[^a-zA-Z\d\\{}]', s)
            return ''.join(sorted(tokens))
        if _sort_factors(norm_p) == _sort_factors(norm_r):
            return True

        # --- Strategy 9: sympy algebraic equivalence (optional, slow) ---
        if cls._try_sympy_equal(predicted, reference):
            return True

        return False

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            user_data = traj.get('user_data') or []
            gt = ''
            for item in user_data:
                if item[0] == 'ground_truth':
                    gt = item[1]
                    break
            predicted = self.extract_boxed(completion)
            correct = self.answers_match(predicted, gt)
            rewards.append(1.0 if correct else 0.0)
        return rewards


class FormatReward(Reward):
    """Reward for having \\boxed{} in the output."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_boxed = '\\boxed{' in completion
            rewards.append(0.5 if has_boxed else 0.0)
        return rewards


class GibberishPenalty(Reward):
    """Negative reward for degenerate outputs (gibberish/random unicode tail)."""

    TAIL_CHARS = 400
    GIBBERISH_THRESHOLD = 0.20  # >20% non-math non-ascii in tail

    @classmethod
    def is_gibberish(cls, text: str) -> bool:
        if not text:
            return False
        tail = text[-cls.TAIL_CHARS:] if len(text) > cls.TAIL_CHARS else text
        non_math_non_ascii = 0
        for c in tail:
            code = ord(c)
            # Allow: ASCII, common CJK (for Chinese math), LaTeX symbols
            if code > 127 and not (0x4e00 <= code <= 0x9fff):
                non_math_non_ascii += 1
        return non_math_non_ascii > len(tail) * cls.GIBBERISH_THRESHOLD

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            rewards.append(-0.5 if self.is_gibberish(completion) else 0.0)
        return rewards


def compute_rewards(trajectories: List[Dict[str, Any]]
                    ) -> Tuple[List[float], List[float], List[float]]:
    acc_fn = AoPSAccuracyReward()
    fmt_fn = FormatReward()
    gib_fn = GibberishPenalty()
    acc = acc_fn(trajectories)
    fmt = fmt_fn(trajectories)
    gib = gib_fn(trajectories)
    total = [a + f + g for a, f, g in zip(acc, fmt, gib)]
    return total, fmt, acc


# ============================================================================
# Dataset: AoPS boxed problems
# ============================================================================
def create_aops_dataset():
    """Load AoPS and create GRPO-style dataset (prompt only, with ground_truth in user_data)."""
    from modelscope import MsDataset
    from twinkle.data_format import Message, Trajectory

    ds = MsDataset.load(AOPS_DATASET_ID, split='train',
                        download_mode='reuse_dataset_if_exists')
    rows = []
    for row in ds:
        if not row['metadata'].get('boxed'):
            continue
        ref = AoPSAccuracyReward.extract_boxed(row['solution'])
        if not ref:
            continue
        rows.append({'problem': row['problem'], 'ground_truth': ref})

    logger.info(f'[aops] loaded {len(rows)} boxed problems')
    rng = random.Random(AOPS_SEED)
    rng.shuffle(rows)

    # Build Trajectory list (prompt-only for GRPO)
    trajectories = []
    for r in rows:
        # Use direct system prompt as placeholder — will be replaced by RAG pipeline
        traj = Trajectory(
            messages=[
                Message(role='system', content=SYSTEM_DIRECT),
                Message(role='user', content=r['problem']),
            ],
            user_data=[('ground_truth', r['ground_truth'])],
        )
        trajectories.append(traj)

    data_meta = DatasetMeta(data=trajectories)
    dataset = Dataset(data_meta)
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID,
                         max_length=16384, truncation_strategy='delete',
                         enable_thinking=True)
    dataset.encode(add_generation_prompt=True)
    return dataset


# ============================================================================
# Main
# ============================================================================
def main():
    # GPU rank allocation
    cond_start = 0
    emb_start = cond_start + CONDENSER_GPUS
    sampler_start = emb_start + EMB_GPUS
    model_start = sampler_start + SAMPLER_GPUS

    device_groups = [
        DeviceGroup(name='condenser', ranks=list(range(cond_start, emb_start)),
                    device_type='GPU'),
        DeviceGroup(name='emb_model', ranks=list(range(emb_start, sampler_start)),
                    device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(sampler_start, model_start)),
                    device_type='GPU'),
        DeviceGroup(name='model', ranks=list(range(model_start, NUM_GPUS)),
                    device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, fsdp_size=MODEL_GPUS, ulysses_size=2)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    emb_mesh = DeviceMesh.from_sizes(world_size=EMB_GPUS, dp_size=EMB_GPUS)
    condenser_mesh = DeviceMesh.from_sizes(world_size=CONDENSER_GPUS, dp_size=CONDENSER_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS,
                       groups=device_groups, lazy_collect=False)

    # -- Training model (full-parameter) --
    model = TransformersModel(
        model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GSPOLoss', epsilon=0.2, epsilon_high=0.28, beta=0.0)
    model.set_processor(InputProcessor)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID,
                       enable_thinking=True, max_length=32768)

    # -- Rollout sampler --
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 32768,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID,
                         enable_thinking=True, max_length=32768)

    # -- Embedding model --
    emb_model = TransformersModel(
        model_id=EMBED_MODEL_ID, device_mesh=emb_mesh, remote_group='emb_model')
    emb_model.set_processor(InputProcessor)
    emb_model.set_loss(InfonceLoss, temperature=0.03, use_batch=True)
    emb_template = Qwen3_5Template(
        model_id=EMBED_MODEL_ID, max_length=EMBED_MAX_LENGTH,
        truncation_strategy='delete', enable_thinking=False)

    # -- Condenser sampler --
    condenser_sampler = vLLMSampler(
        model_id=CONDENSE_MODEL_ID,
        engine_args={'gpu_memory_utilization': 0.85, 'max_model_len': 32768},
        device_mesh=condenser_mesh,
        remote_group='condenser',
    )
    condenser_sampler.set_template(
        'Qwen3_5Template', model_id=CONDENSE_MODEL_ID,
        enable_thinking=False, truncation_strategy='delete', max_length=32768)
    condenser_template = Qwen3_5Template(
        model_id=CONDENSE_MODEL_ID, max_length=32768,
        enable_thinking=False, truncation_strategy='delete')
    condenser_special_tokens = set(condenser_template.tokenizer.all_special_tokens)
    compress_params = SamplingParams(
        max_tokens=CONDENSE_MAX_TOKENS, temperature=CONDENSE_TEMPERATURE,
        top_p=0.5, num_samples=1)

    # -- API client (condenser fallback) --
    api_client = None
    if CONDENSE_API_KEY:
        api_client = OpenAIClient(
            model=CONDENSE_API_MODEL, api_key=CONDENSE_API_KEY,
            base_url=CONDENSE_BASE_URL)

    # -- LanceDB --
    import lancedb
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(DB_TABLE)
    logger.info(f'[rag] LanceDB ready, rows={tbl.count_rows()}')

    # -- Checkpoint & DataLoader --
    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_aops_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95)

    optim_step = 0
    logger.info('Starting RAG-hint GRPO training')
    logger.info(get_device_placement())

    # -- Prefetch: overlap RAG data preparation with training --
    prefetch_pool = ThreadPoolExecutor(max_workers=1)

    def _extract_text(content) -> str:
        """Extract plain text from content (str or list-of-parts format)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return ''.join(
                p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
        return str(content) if content else ''

    def prepare_rag_batch(batch):
        """Embed → retrieve → condense → build prompts. Runs in background thread."""
        problems = []
        ground_truths = []
        for item in batch:
            msgs = item.get('messages', [])
            prob = ''
            for m in msgs:
                if m.get('role') == 'user':
                    prob = _extract_text(m.get('content', ''))
                    break
            problems.append(prob)
            ud = item.get('user_data', [])
            gt = ''
            for pair in ud:
                if pair[0] == 'ground_truth':
                    gt = pair[1]
                    break
            ground_truths.append(gt)

        # Embed & retrieve
        query_vecs = get_embeddings(emb_model, emb_template, problems, EMB_GPUS)
        retrieved = retrieve_topk(tbl, query_vecs, problems, SIM_THRESHOLD)

        # Condense (batch local vLLM + API fallback)
        condensed_examples: List[List[Dict[str, str]]] = [[] for _ in range(len(problems))]
        tasks_to_condense = []
        for i, rets in enumerate(retrieved):
            for j, ret in enumerate(rets):
                tasks_to_condense.append((i, j, problems[i], ret))

        if tasks_to_condense:
            condense_prompts = []
            for idx, _j, prob, ret in tasks_to_condense:
                user_msg = COMPRESS_USER.format(query=prob, text=ret['thinking'])
                condense_prompts.append({'messages': [
                    {'role': 'system', 'content': COMPRESS_SYSTEM},
                    {'role': 'user', 'content': user_msg}]})

            try:
                condense_responses = condenser_sampler.sample(condense_prompts, compress_params)
            except Exception as exc:
                logger.warning(f'[condense] local batch error: {exc}')
                condense_responses = [None] * len(condense_prompts)

            api_fallback_indices = []
            for ci, (idx, _j, prob, ret) in enumerate(tasks_to_condense):
                resp = condense_responses[ci] if condense_responses else None
                seq = resp.sequences[0] if resp and resp.sequences else None
                text = ''
                if seq and seq.stop_reason != 'length' and seq.decoded:
                    text = seq.decoded
                    for tok in condenser_special_tokens:
                        text = text.replace(tok, '')
                    text = text.strip()
                if text:
                    condensed_examples[idx].append({'query': ret['query'], 'thinking': text})
                else:
                    api_fallback_indices.append(ci)

            if api_fallback_indices and api_client:
                def _fallback(ci):
                    return ci, _api_condense_single(api_client, condense_prompts[ci]['messages'])
                with ThreadPoolExecutor(max_workers=CONDENSE_API_CONCURRENCY) as pool:
                    futs = [pool.submit(_fallback, ci) for ci in api_fallback_indices]
                    for fut in as_completed(futs):
                        ci, result = fut.result()
                        idx, _j, prob, ret = tasks_to_condense[ci]
                        text = result if result else ret['thinking'][:MAX_TRACE_LEN]
                        condensed_examples[idx].append({'query': ret['query'], 'thinking': text})
            elif api_fallback_indices:
                for ci in api_fallback_indices:
                    idx, _j, prob, ret = tasks_to_condense[ci]
                    condensed_examples[idx].append(
                        {'query': ret['query'], 'thinking': ret['thinking'][:MAX_TRACE_LEN]})

        # Build prompts with rag_fallback_sim check
        rag_prompts = []
        rag_debug_records = []
        for i, prob in enumerate(problems):
            examples = condensed_examples[i]
            rets = retrieved[i]
            best_sim = max((r['sim'] for r in rets), default=0.0)
            use_rag = bool(examples) and best_sim >= RAG_FALLBACK_SIM

            if use_rag:
                parts = [SYSTEM_WITH_RAG_HEADER]
                for eidx, ex in enumerate(examples, 1):
                    parts.append(EXAMPLE_TEMPLATE.format(
                        idx=eidx,
                        example_query=ex['query'],
                        example_thinking=ex['thinking']))
                sys_content = ''.join(parts)
            else:
                sys_content = SYSTEM_DIRECT

            prompt_feature = {
                'messages': [
                    {'role': 'system', 'content': sys_content},
                    {'role': 'user', 'content': prob},
                ],
                'user_data': [('ground_truth', ground_truths[i])],
                'assistant_prefix': ANALYSIS_PREFIX if use_rag else '',
            }
            rag_prompts.append(prompt_feature)

            # Diagnostic record
            debug_rec = {
                'problem': prob[:200],
                'ground_truth': ground_truths[i],
                'best_sim': round(best_sim, 4),
                'num_retrieved': len(rets),
                'num_condensed': len(examples),
                'use_rag': use_rag,
            }
            if rets:
                debug_rec['top_retrieved_query'] = rets[0]['query'][:200]
            if examples:
                debug_rec['condensed_len'] = len(examples[0].get('thinking', ''))
            rag_debug_records.append(debug_rec)

        return rag_prompts, rag_debug_records

    # Submit first batch prefetch
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rag_log_path = os.path.join(OUTPUT_DIR, 'rag_diagnostics.jsonl')
    rag_log_f = open(rag_log_path, 'a', encoding='utf-8')
    logger.info(f'[rag] diagnostics → {rag_log_path}')

    batch_iter = iter(dataloader)
    pending_future = None
    try:
        first_batch = next(batch_iter)
        pending_future = prefetch_pool.submit(prepare_rag_batch, first_batch)
    except StopIteration:
        pass

    try:
        while pending_future is not None:
            if optim_step >= MAX_STEPS:
                break

            metrics.reset()
            rag_prompts, rag_debug_records = pending_future.result()

            # Write RAG diagnostics
            for rec in rag_debug_records:
                rec['step'] = optim_step
                rag_log_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            rag_log_f.flush()

            # Submit next batch prefetch (overlaps with rollout + training)
            pending_future = None
            try:
                next_batch = next(batch_iter)
                pending_future = prefetch_pool.submit(prepare_rag_batch, next_batch)
            except StopIteration:
                pass

            # ---- Expand for NUM_GENERATIONS and sample ----
            expand_prompts = []
            for prompt in rag_prompts:
                expand_prompts.extend([prompt] * NUM_GENERATIONS)

            ckpt_manager.sync_weights(merge_and_sync=False)
            sampler.reset_prefix_cache()

            sample_responses = sampler.sample(expand_prompts, sampling_params)

            # ---- Collect rollouts ----
            all_input_data: List[Dict[str, Any]] = []
            all_old_logps: List[List[float]] = []
            all_completion_lengths: List[int] = []

            for sample_response in sample_responses:
                for sequence in sample_response.sequences:
                    all_input_data.append(sequence.new_input_feature)
                    all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                    all_completion_lengths.append(len(sequence.tokens))

            # ---- Rewards ----
            total_rewards, format_rewards, accuracy_rewards = compute_rewards(all_input_data)

            # Zero out rewards for rollouts that hit the max_tokens ceiling
            max_len_threshold = int(MAX_NEW_TOKENS * 0.95)
            for i in range(len(all_input_data)):
                if all_completion_lengths[i] >= max_len_threshold:
                    total_rewards[i] = 0.0
                    accuracy_rewards[i] = 0.0
                    format_rewards[i] = 0.0

            # Per-step reward summary to diagnostics
            n_correct = sum(1 for a in accuracy_rewards if a > 0)
            rag_log_f.write(json.dumps({
                'step': optim_step, 'type': 'reward_summary',
                'n_samples': len(accuracy_rewards),
                'accuracy': n_correct / len(accuracy_rewards) if accuracy_rewards else 0,
                'mean_reward': sum(total_rewards) / len(total_rewards) if total_rewards else 0,
            }, ensure_ascii=False) + '\n')

            metrics.accumulate(
                completion_lengths=all_completion_lengths,
                rewards={
                    'total': total_rewards,
                    'format': format_rewards,
                    'accuracy': accuracy_rewards,
                },
            )

            # ---- GRPO advantage ----
            advantages = advantage_fn(
                total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
            if ADV_CLIP > 0:
                advantages = [max(-ADV_CLIP, min(ADV_CLIP, a)) for a in advantages]

            # Log all rollout responses (after advantage computation)
            _extract_boxed = AoPSAccuracyReward.extract_boxed
            def _content_to_str(content):
                """Convert message content (str or list of blocks) to plain text."""
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return ''.join(
                        b.get('text', '') if isinstance(b, dict) else str(b)
                        for b in content)
                return str(content)

            for ridx, traj in enumerate(all_input_data):
                msgs = traj.get('messages', [])
                assistant_text = _content_to_str(next(
                    (m['content'] for m in reversed(msgs) if m.get('role') == 'assistant'), ''))
                user_text = _content_to_str(next(
                    (m['content'] for m in msgs if m.get('role') == 'user'), ''))
                sys_text = _content_to_str(next(
                    (m['content'] for m in msgs if m.get('role') == 'system'), ''))
                user_data = traj.get('user_data') or []
                gt = next((v for k, v in user_data if k == 'ground_truth'), '')
                problem_idx = ridx // NUM_GENERATIONS
                use_rag = 'condensed reasoning examples from similar problems' in sys_text
                # Per-problem group accuracy (all generations for same problem)
                grp_start = problem_idx * NUM_GENERATIONS
                grp_end = grp_start + NUM_GENERATIONS
                grp_acc = sum(accuracy_rewards[grp_start:grp_end]) / NUM_GENERATIONS

                rag_log_f.write(json.dumps({
                    'step': optim_step, 'type': 'rollout',
                    'idx': ridx,
                    'problem_idx': problem_idx,
                    'problem': user_text,
                    'system': sys_text,
                    'response': assistant_text,
                    'ground_truth': gt,
                    'predicted': _extract_boxed(assistant_text),
                    'use_rag': use_rag,
                    'best_sim': rag_debug_records[problem_idx].get('best_sim', 0.0) if problem_idx < len(rag_debug_records) else 0.0,
                    'reward': total_rewards[ridx],
                    'accuracy_reward': accuracy_rewards[ridx],
                    'format_reward': format_rewards[ridx],
                    'advantage': advantages[ridx],
                    'completion_length': all_completion_lengths[ridx],
                    'group_accuracy': grp_acc,
                }, ensure_ascii=False) + '\n')

            rag_log_f.flush()

            # ---- Filter out low-signal problem groups (DAPO-style dynamic sampling) ----
            # Skip groups where accuracy is too low (<0.1) or too high (>0.9)
            # to avoid gradient dominated by gibberish/format noise or no learning signal.
            filtered_inputs, filtered_old_logps, filtered_advantages = [], [], []
            for g in range(BATCH_SIZE):
                g_start = g * NUM_GENERATIONS
                g_end = g_start + NUM_GENERATIONS
                grp_adv = advantages[g_start:g_end]
                if all(abs(a) < 1e-8 for a in grp_adv):
                    continue
                grp_acc_rate = sum(accuracy_rewards[g_start:g_end]) / NUM_GENERATIONS
                if grp_acc_rate < 0.1 or grp_acc_rate > 0.9:
                    continue
                filtered_inputs.extend(all_input_data[g_start:g_end])
                filtered_old_logps.extend(all_old_logps[g_start:g_end])
                filtered_advantages.extend(grp_adv)

            # ---- Mini-batch training with gradient accumulation ----
            # Process MICRO_BATCH_SIZE samples per forward, accumulate grad_accum_steps
            # times before one optimizer step. clip_grad_norm normalizes by accumulated
            # num_tokens, ensuring mathematical equivalence with larger batch forward.
            total_completions = len(filtered_inputs)
            if total_completions == 0:
                logger.info(f'[Step {optim_step}] all groups filtered (uniform rewards), skip training')
                continue

            grad_accum_steps = MINI_BATCH_SIZE // MICRO_BATCH_SIZE
            accum_count = 0
            for mb_start in range(0, total_completions, MICRO_BATCH_SIZE):
                mb_end = min(mb_start + MICRO_BATCH_SIZE, total_completions)
                mb_inputs = filtered_inputs[mb_start:mb_end]
                mb_old_logps = filtered_old_logps[mb_start:mb_end]
                mb_advantages = filtered_advantages[mb_start:mb_end]

                model.forward_backward(
                    inputs=mb_inputs,
                    old_logps=mb_old_logps,
                    ref_logps=mb_old_logps,
                    advantages=mb_advantages,
                )
                accum_count += 1

                if accum_count % grad_accum_steps == 0:
                    model.clip_grad_and_step()
                    optim_step += 1

                    if optim_step >= MAX_STEPS:
                        break
                    if optim_step % SAVE_STEPS == 0:
                        model.save(f'rag-hint-grpo-checkpoint-{optim_step}')

            # Flush remaining accumulated gradients (incomplete window at tail)
            if accum_count % grad_accum_steps != 0:
                model.clip_grad_and_step()
                optim_step += 1

            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            metrics.reset()
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')
    finally:
        prefetch_pool.shutdown(wait=False)
        rag_log_f.close()

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('rag-hint-grpo-final')


if __name__ == '__main__':
    main()
