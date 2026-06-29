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
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 16))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 65536))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 5000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 4))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 1))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 100))

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
    'Put your final answer inside \\boxed{{}}.\n\n'
)

EXAMPLE_TEMPLATE = (
    '--- Example {idx} ---\n'
    'Problem: {example_query}\n'
    'Methodology:\n{example_thinking}\n'
    '--- End Example {idx} ---\n'
)

SYSTEM_DIRECT = (
    'You are an expert competition mathematician. '
    'Solve the problem step by step. Put your final answer inside \\boxed{}.'
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

    @staticmethod
    def normalize_answer(ans: str) -> str:
        if not ans:
            return ''
        s = ans.strip()
        m = re.match(r'^\\?(?:textbf|text|mathrm|mathbf)?\{?\(?([A-E])\)?\}?$', s)
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
        s = re.sub(r'\{[a-zA-Z]+\}$', '', s)
        s = re.sub(r'\^\\circ|\^\{\\circ\}|°', 'deg', s)

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

    @staticmethod
    def _try_numeric_equal(a: str, b: str) -> bool:
        try:
            va = float(a.replace('(', '').replace(')', ''))
            vb = float(b.replace('(', '').replace(')', ''))
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
        if va is not None and vb is not None:
            return abs(va - vb) < 1e-9 * max(1, abs(va), abs(vb))
        return False

    @classmethod
    def answers_match(cls, predicted: str, reference: str) -> bool:
        if not predicted or not reference:
            return False
        norm_p = cls.normalize_answer(predicted)
        norm_r = cls.normalize_answer(reference)
        if norm_p == norm_r:
            return True
        return cls._try_numeric_equal(norm_p, norm_r)

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


def compute_rewards(trajectories: List[Dict[str, Any]]
                    ) -> Tuple[List[float], List[float], List[float]]:
    acc_fn = AoPSAccuracyReward()
    fmt_fn = FormatReward()
    acc = acc_fn(trajectories)
    fmt = fmt_fn(trajectories)
    total = [a + f for a, f in zip(acc, fmt)]
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
                    device_type='GPU', gpus_per_worker=SAMPLER_GPUS),
        DeviceGroup(name='model', ranks=list(range(model_start, NUM_GPUS)),
                    device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, fsdp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    emb_mesh = DeviceMesh.from_sizes(world_size=EMB_GPUS, dp_size=EMB_GPUS)
    condenser_mesh = DeviceMesh.from_sizes(world_size=CONDENSER_GPUS, dp_size=CONDENSER_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS,
                       groups=device_groups, lazy_collect=False)

    # -- Training model (full-parameter) --
    model = TransformersModel(
        model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID,
                       enable_thinking=True, max_length=65536)

    # -- Rollout sampler --
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 65536,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID,
                         enable_thinking=True, max_length=65536)

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

            # Per-step reward summary to diagnostics
            n_correct = sum(1 for a in accuracy_rewards if a > 0)
            rag_log_f.write(json.dumps({
                'step': optim_step, 'type': 'reward_summary',
                'n_samples': len(accuracy_rewards),
                'accuracy': n_correct / len(accuracy_rewards) if accuracy_rewards else 0,
                'mean_reward': sum(total_rewards) / len(total_rewards) if total_rewards else 0,
            }, ensure_ascii=False) + '\n')
            rag_log_f.flush()

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

            # ---- Mini-batch training ----
            total_completions = len(all_input_data)
            for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
                mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
                mb_inputs = all_input_data[mb_start:mb_end]
                mb_old_logps = all_old_logps[mb_start:mb_end]
                mb_advantages = advantages[mb_start:mb_end]

                model.forward_backward(
                    inputs=mb_inputs,
                    old_logps=mb_old_logps,
                    advantages=mb_advantages,
                    micro_batch_size=MICRO_BATCH_SIZE,
                )
                model.clip_grad_and_step()
                optim_step += 1

                if optim_step >= MAX_STEPS:
                    break
                if optim_step % SAVE_STEPS == 0:
                    model.save(f'rag-hint-grpo-checkpoint-{optim_step}')

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
