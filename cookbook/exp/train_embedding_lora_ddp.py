"""LoRA embedding training with online condenser self-improvement.

Architecture (8 GPUs total):
  - Ranks 0-3 (``model``): Trainable embedding model with LoRA, InfoNCE loss.
  - Ranks 4-5 (``condenser_sampler``): Frozen vLLM condenser for online compression.
  - Ranks 6-7 (``condenser_model``): Trainable condenser with LoRA for self-improvement.

When the condenser sampler truncates (stop_reason='length'), an external OpenAI-
compatible API produces the correct compression. The failure is logged as SFT
training data. A background thread retrains the condenser on accumulated failures
mixed with condense_300K, then syncs weights back to the sampler.

Launch:
    python cookbook/exp/train_embedding_lora_ddp.py
"""
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import swanlab
import torch

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import InputFeature, SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import InfonceLoss
from twinkle.metric import EmbeddingMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle.utils.parallel import PosixFileLock
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_think import get_dataset  # noqa: E402

logger = get_logger()

# -- Backend selection --------------------------------------------------------
BACKEND: Literal['transformers', 'megatron'] = 'transformers'

MODEL_ID = os.environ.get('MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
TEMPLATE_NAME = 'Qwen3_5Template'

# -- GPU placement (8 total) --------------------------------------------------
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
CONDENSER_SAMPLER_GPUS = int(os.environ.get('CONDENSER_SAMPLER_GPUS', 2))
CONDENSER_MODEL_GPUS = int(os.environ.get('CONDENSER_MODEL_GPUS', 2))
NUM_GPUS = MODEL_GPUS + CONDENSER_SAMPLER_GPUS + CONDENSER_MODEL_GPUS

# -- Embedding training hyper-params ------------------------------------------
EMB_MAX_LENGTH = 4096
HARD_NEGATIVES = None
TEMPERATURE = 0.05

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
LEARNING_RATE = 5e-6
GRADIENT_ACCUMULATION_STEPS = 16
LOG_INTERVAL = 2
SAVE_INTERVAL = 1000
NUM_EPOCHS = 1

TOTAL_SAMPLES: Optional[int] = None

# -- Online-compression knobs -------------------------------------------------
MIN_COT_CHARS = 256
COMPRESS_MAX_TOKENS = 2048
COMPRESS_TEMPERATURE = 0.3
COMPRESS_TOP_P = 0.7
COMPRESS_MAX_MODEL_LEN = 32768

# -- OpenAI API fallback for truncated compressions ---------------------------
COMPRESS_API_KEY = os.environ.get('COMPRESS_API_KEY', '')
COMPRESS_BASE_URL = os.environ.get('COMPRESS_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
COMPRESS_MODEL = os.environ.get('COMPRESS_MODEL', 'qwen3.7-max')

# -- Condenser retraining knobs -----------------------------------------------
CONDENSER_DATASET_ID = 'ms://twinkle-kit/condense_300K'
CONDENSER_RETRAIN_SAMPLES = 128
CONDENSER_RETRAIN_EPOCHS = 3
CONDENSER_RETRAIN_LR = 1e-5

# -- Output paths -------------------------------------------------------------
OUTPUT_DIR = f'./output/embedding_lora_{BACKEND}'
RESPONSE_LOG = os.environ.get('RESPONSE_LOG', f'./output/embedding_lora_{BACKEND}/responses.jsonl')
FAILURE_LOG = os.environ.get('FAILURE_LOG', f'./output/embedding_lora_{BACKEND}/failures.jsonl')


# =============================================================================
# Prompts (from make_condenser_dataset.py — "## Read inline" format)
# =============================================================================

COMPRESS_SYSTEM = """\
You are a compression assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Read inline` directly, then calls `extract_compressed` \
on any topic-key listed under `## Call extract_compressed for` to recover its \
fuller content.

  `## Read inline`               — extreme-density text the reader reads directly.
  `## Call extract_compressed for` — a topic index whose keys are valid arguments \
to `extract_compressed` for recovering material not captured inline.

Together the two sections must form a COMPLETE, NON-DISTORTING inventory of the \
source for the query — nothing essential lost, nothing implied that the source \
does not support. NO preamble, NO meta-commentary, NO code fences wrapping the \
whole output.

Output skeleton:

## Read inline
Topic: <what the source is about + scope, one line>
<dense body answering the query>

## Call extract_compressed for
- <topic-key>: <one-line hint of what is revealed when expanded>
- ...

Format selection for the inline body (pick the MOST COMPACT form per query, mix \
when helpful):
- Interface / signature → code notation directly: `func(a:int)->str`
- Factual / entity → telegraphic prose; drop function words; ":" for "is", "," \
for "has"
- Skill / how-to / usage → lead with `Use when: <trigger>`; numbered telegraphic \
steps `1.do X 2.then Y`; close with `Output: <result>` when relevant
- Procedural → numbered short steps
- Analytical / design → hierarchical bullets with abbreviations

`## Read inline` rules:
1. TOPIC LINE — line 1 is ALWAYS `Topic: <subject — scope>`, even when the \
query is narrow. Anchors both the reader and the tool.
2. DENSITY — every token in the body carries query-relevant signal; cut filler.
3. PRIMARY-COMPLETE — never silently drop a fact essential to answering the \
query. Anything cut for length MUST appear as a key under \
`## Call extract_compressed for`.
4. NON-MISLEADING — phrasing must not let the reader infer anything the source \
does not support; partial truths that mislead are worse than honest omissions \
flagged in the index.
5. SELF-CONTAINED — the reader can act on the answer without re-opening the source.
6. FAITHFUL — only content the source supports; no fabrication, no extrapolation.
7. LANGUAGE — match the source language.
8. NO outer code fences around the whole answer; no meta-commentary.

`## Call extract_compressed for` rules (MANDATORY — this section is never omitted):
1. FORMAT — each bullet is `- <topic-key>: <one-line hint>`:
   • topic-key — short, unambiguous, grounded in source vocabulary so the \
`extract_compressed` tool can locate the aspect (e.g. `decorators`, \
`error handling`, `pitfalls`).
   • hint — tells WHAT the reader gains by expanding (concrete numbers, code \
listings, secondary cases, edge details, related context, …); do NOT restate \
the inline answer.
2. CRITERION — each bullet names an aspect that EXISTS in the source but is \
NOT fully captured inline. Material that genuinely fits inline without \
distortion MUST NOT be duplicated here.
3. FAITHFUL — hints must be grounded in the source; never speculate or invent.
4. ORDER — by relevance to the query, then by importance.
5. EMPTY CASE — if the source is so short / single-purpose that everything \
fits inline, write a single line `- (none)`.

Now begin.\
"""

COMPRESS_USER = "## Query\n{query}\n\n## Source\n{text}"

COMPRESS_SYSTEM_TRAIN = """\
You are a compression assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Read inline` directly, then calls `extract_compressed` \
on any topic-key listed under `## Call extract_compressed for` to recover its \
fuller content.

Output skeleton:

## Read inline
Topic: <subject — scope, one line>
<dense body answering the query>

## Call extract_compressed for
- <topic-key>: <one-line hint of what is revealed when expanded>
- ...

Rules:
1. Line 1 of `## Read inline` is ALWAYS `Topic: ...`.
2. Body is maximally dense; every token carries query-relevant signal.
3. Never silently drop a fact — anything cut for length MUST appear as a key \
under `## Call extract_compressed for` (do not duplicate inline material here).
4. No fabrication, no extrapolation, no misleading partial truths.
5. Match the source language. No outer code fences, no meta-commentary.\
"""


# =============================================================================
# Logging helpers
# =============================================================================

_response_lock: Optional[PosixFileLock] = None
_failure_lock: Optional[PosixFileLock] = None


def _log_responses(query_resp_text: str, cot_resp_text: str, idx: int,
                   query_raw: str = '', cot_raw: str = ''):
    global _response_lock
    if _response_lock is None:
        os.makedirs(os.path.dirname(RESPONSE_LOG) or '.', exist_ok=True)
        _response_lock = PosixFileLock(RESPONSE_LOG + '.lock')

    record = {
        'idx': idx,
        'query_raw': query_raw,
        'cot_raw': cot_raw,
        'query_compressed': query_resp_text,
        'cot_compressed': cot_resp_text,
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + '\n'
    with _response_lock:
        with open(RESPONSE_LOG, 'a', encoding='utf-8') as f:
            f.write(line)


def _log_failure(source_text: str, query: str, compressed: str, batch_idx: int):
    global _failure_lock
    if _failure_lock is None:
        os.makedirs(os.path.dirname(FAILURE_LOG) or '.', exist_ok=True)
        _failure_lock = PosixFileLock(FAILURE_LOG + '.lock')

    qhash = hashlib.md5(query.strip().encode('utf-8')).hexdigest()[:8]
    record = {
        'id': f'{batch_idx}__{qhash}',
        'source': 'online_failure',
        'query': query,
        'original_len': len(source_text),
        'compressed_len': len(compressed),
        'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM_TRAIN},
            {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=source_text)},
            {'role': 'assistant', 'content': compressed},
        ],
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + '\n'
    with _failure_lock:
        with open(FAILURE_LOG, 'a', encoding='utf-8') as f:
            f.write(line)


# =============================================================================
# Model builders
# =============================================================================

def build_model(device_mesh: DeviceMesh):
    if BACKEND == 'transformers':
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            remote_group='model',
            ddp_config={'find_unused_parameters': True},
        )
        from twinkle.patch.no_split_modules import NoSplitModulesPatch
        model.apply_patch(NoSplitModulesPatch({'Qwen3_5DecoderLayer'}))
        return model
    if BACKEND == 'megatron':
        from twinkle.model import MegatronModel
        return MegatronModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            remote_group='model',
            mixed_precision='bf16',
            variable_seq_lengths=True,
        )
    raise ValueError(f'Unknown BACKEND={BACKEND!r}')


def setup_optimizer(model, total_steps: int):
    if BACKEND == 'transformers':
        model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler(
            scheduler_cls='CosineWarmupScheduler',
            num_warmup_steps=50,
            num_training_steps=total_steps,
        )
        return
    if BACKEND == 'megatron':
        model.set_optimizer(optimizer_cls='default', lr=LEARNING_RATE)
        model.set_lr_scheduler(
            scheduler_cls='default',
            lr_warmup_steps=50,
            lr_decay_steps=total_steps,
        )
        return
    raise ValueError(f'Unknown BACKEND={BACKEND!r}')


def save_checkpoint(model, name: str):
    model.save(name, output_dir=OUTPUT_DIR)


# =============================================================================
# Compression prompt building
# =============================================================================

EMBED_QUERY_Q = (
    'What problem does this passage need to solve, and what kind of skill or '
    'method is required? Compress into a retrieval-friendly need description.')
EMBED_QUERY_COT = (
    'Extract the reusable skill: trigger conditions, key steps, and expected '
    'output. Compress into a standardized procedure for retrieval.')


def _extract_query_cot(row: Dict[str, Any]):
    messages = row.get('messages') or []
    query, cot = '', ''
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role') or ''
        if role == 'user' and not query:
            query = (m.get('content') or '').strip()
        elif role == 'assistant':
            cot = (m.get('reasoning_content') or '').strip()
            break
    return query, cot


def _build_compress_prompts(rows: List[Dict[str, Any]]) -> tuple:
    """Build prompts for compressing both query and cot per row.

    Returns (prompts, valid_indices, raw_pairs, prompt_queries) where:
    - prompts: flat-interleaved [query_0, cot_0, query_1, cot_1, ...]
    - valid_indices: which rows passed the min-length filter
    - raw_pairs: [(query, cot), ...]
    - prompt_queries: the query string used for each prompt (for failure logging)
    """
    prompts: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    raw_pairs: List[tuple] = []
    prompt_queries: List[str] = []
    for i, row in enumerate(rows):
        query, cot = _extract_query_cot(row)
        if not query or len(cot) < MIN_COT_CHARS:
            continue
        valid_indices.append(i)
        raw_pairs.append((query, cot))
        for text, qtpl in ((query, EMBED_QUERY_Q), (cot, EMBED_QUERY_COT)):
            user = COMPRESS_USER.format(query=qtpl, text=text)
            prompts.append({'messages': [
                {'role': 'system', 'content': COMPRESS_SYSTEM},
                {'role': 'user', 'content': user},
            ]})
            prompt_queries.append(qtpl)
    return prompts, valid_indices, raw_pairs, prompt_queries


def _get_first_feature(decoded_text: str, template: Template, role: str) -> Optional[Dict[str, Any]]:
    if not decoded_text:
        return None
    if role == 'anchor':
        feat = template.encode({'messages': [
            {'role': 'user', 'content': decoded_text},
            {'role': 'assistant', 'content': 'Match the correct response here.'},
        ]})
        feat['labels'] = [1]
    else:
        feat = template.encode({'messages': [
            {'role': 'user', 'content': 'Match the correct query here.'},
            {'role': 'assistant', 'content': decoded_text},
        ]})
        feat['labels'] = [0]
    return feat


# =============================================================================
# OpenAI API fallback
# =============================================================================

def _api_compress(api_client: OpenAIClient, prompt: Dict[str, Any]) -> Optional[str]:
    """Call external API to compress when vLLM truncates."""
    trajectory = {'messages': prompt['messages']}
    sp = SamplingParams(temperature=0.3, max_tokens=32000)
    try:
        reply = api_client(trajectory, sp, extra_body={'enable_thinking': False})
    except Exception as exc:
        logger.warning(f'[api_fallback] error: {exc}')
        return None
    content = (reply.get('content') or '').strip()
    if not content:
        return None
    # Strip outer code fence if present
    m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', content, re.DOTALL)
    if m:
        content = m.group(1).strip()
    return content


# =============================================================================
# Condenser Retrainer (background thread)
# =============================================================================

class CondenserRetrainer:
    """Async condenser self-improvement: retrains from failures, syncs to sampler."""

    def __init__(self, condenser_model, ckpt_manager: CheckpointEngineManager,
                 condenser_sampler):
        self._model = condenser_model
        self._ckpt_manager = ckpt_manager
        self._sampler = condenser_sampler
        self._signal = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._condense_300k_cache = None
        self._retrain_count = 0
        # Prevents sample() and sync_weights() from running concurrently
        self.sampler_lock = threading.Lock()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._signal.set()
        self._thread.join(timeout=10)

    def notify_failure(self):
        self._signal.set()

    def _loop(self):
        while not self._stop.is_set():
            self._signal.wait(timeout=60)
            if self._stop.is_set():
                break
            if not self._signal.is_set():
                continue
            self._signal.clear()
            try:
                self._retrain_and_sync()
            except Exception as exc:
                logger.error(f'[condenser_retrain] crashed: {exc}')

    def _load_condense_300k(self):
        if self._condense_300k_cache is None:
            dataset = Dataset(dataset_meta=DatasetMeta(CONDENSER_DATASET_ID, split='train'))
            dataset.set_template(TEMPLATE_NAME, model_id=MODEL_ID,
                                 max_length=40000, enable_thinking=False,
                                 truncation_strategy='delete')
            dataset.encode(load_from_cache_file=True, num_proc=4)
            self._condense_300k_cache = dataset
        return self._condense_300k_cache

    def _load_failures(self) -> List[Dict[str, Any]]:
        if not os.path.exists(FAILURE_LOG):
            return []
        rows = []
        with open(FAILURE_LOG, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _retrain_and_sync(self):
        failures = self._load_failures()
        if not failures:
            logger.info('[condenser_retrain] no failures to train on, skipping')
            return

        n_target = CONDENSER_RETRAIN_SAMPLES
        random.shuffle(failures)

        if len(failures) >= n_target:
            train_rows = failures[:n_target]
        else:
            condense_300k = self._load_condense_300k()
            n_fill = n_target - len(failures)
            indices = random.sample(range(len(condense_300k)), min(n_fill, len(condense_300k)))
            fill_rows = [condense_300k[i] for i in indices]
            train_rows = failures + fill_rows
            random.shuffle(train_rows)

        # Build dataset from failure rows (already have 'messages' field)
        dataset = Dataset()
        dataset.add_dataset(DatasetMeta(data=train_rows))
        dataset.set_template(TEMPLATE_NAME, model_id=MODEL_ID,
                             max_length=32768, enable_thinking=False,
                             truncation_strategy='delete')
        dataset.encode(load_from_cache_file=False)

        dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

        self._retrain_count += 1
        logger.info(f'[condenser_retrain] round {self._retrain_count}: '
                    f'{len(failures)} failures, {len(train_rows)} total samples, '
                    f'{CONDENSER_RETRAIN_EPOCHS} epochs')

        for epoch in range(CONDENSER_RETRAIN_EPOCHS):
            for batch in dataloader:
                self._model.forward_backward(inputs=batch)
                self._model.clip_grad_and_step()

        # Sync weights to sampler (exclusive with sampling)
        with self.sampler_lock:
            self._ckpt_manager.sync_weights()
            self._sampler.reset_prefix_cache()

        # Save checkpoint
        ckpt_name = f'condenser_retrain_{self._retrain_count}'
        self._model.save(ckpt_name, output_dir=OUTPUT_DIR)
        logger.info(f'[condenser_retrain] round {self._retrain_count} done, synced to sampler')


# =============================================================================
# Main training
# =============================================================================

def train():
    # -------- Device groups (3 groups) ----------------------------------------
    device_groups = [
        DeviceGroup(name='model',
                    ranks=list(range(MODEL_GPUS)),
                    device_type='GPU'),
        DeviceGroup(name='condenser_sampler',
                    ranks=list(range(MODEL_GPUS, MODEL_GPUS + CONDENSER_SAMPLER_GPUS)),
                    device_type='GPU'),
        DeviceGroup(name='condenser_model',
                    ranks=list(range(MODEL_GPUS + CONDENSER_SAMPLER_GPUS, NUM_GPUS)),
                    device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    condenser_sampler_mesh = DeviceMesh.from_sizes(
        world_size=CONDENSER_SAMPLER_GPUS, dp_size=CONDENSER_SAMPLER_GPUS)
    condenser_model_mesh = DeviceMesh.from_sizes(
        world_size=CONDENSER_MODEL_GPUS, dp_size=1, fsdp_size=CONDENSER_MODEL_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)

    # -------- Data -----------------------------------------------------------
    dataset = get_dataset(total=TOTAL_SAMPLES, load_from_cache_file=True)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    total_forward_steps = len(dataloader) * NUM_EPOCHS
    optimizer_steps = total_forward_steps // GRADIENT_ACCUMULATION_STEPS

    # -------- Embedding model (4 GPU) ----------------------------------------
    model = build_model(model_mesh)
    model.set_processor(InputProcessor)
    model.set_loss(InfonceLoss, temperature=TEMPERATURE, use_batch=True,
                   hard_negatives=HARD_NEGATIVES)
    setup_optimizer(model, optimizer_steps)
    model.add_metric(EmbeddingMetric, is_training=True)

    # -------- Condenser sampler (2 GPU, vLLM) --------------------------------
    emb_template = Template(model_id=MODEL_ID, max_length=EMB_MAX_LENGTH, enable_thinking=False)
    condenser_sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': COMPRESS_MAX_MODEL_LEN,
        },
        device_mesh=condenser_sampler_mesh,
        remote_group='condenser_sampler',
    )
    condenser_sampler.set_template(
        TEMPLATE_NAME, model_id=MODEL_ID, enable_thinking=False,
        truncation_strategy='delete', max_length=COMPRESS_MAX_TOKENS)
    compress_params = SamplingParams(
        max_tokens=COMPRESS_MAX_TOKENS,
        temperature=COMPRESS_TEMPERATURE,
        top_p=COMPRESS_TOP_P,
        num_samples=1,
    )

    # -------- Condenser model (2 GPU, trainable full-param) -------------------
    condenser_model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=condenser_model_mesh,
        remote_group='condenser_model',
    )
    condenser_model.set_optimizer(optimizer_cls='AdamW', lr=CONDENSER_RETRAIN_LR)

    # -------- CheckpointEngineManager: condenser_model → condenser_sampler ---
    condenser_ckpt_manager = CheckpointEngineManager(
        model=condenser_model, sampler=condenser_sampler)
    condenser_ckpt_manager.sync_weights()

    # -------- Background retrainer -------------------------------------------
    retrainer = CondenserRetrainer(condenser_model, condenser_ckpt_manager,
                                   condenser_sampler)
    retrainer.start()

    # -------- OpenAI API client for fallback ---------------------------------
    api_client = OpenAIClient(
        model=COMPRESS_MODEL,
        api_key=COMPRESS_API_KEY,
        base_url=COMPRESS_BASE_URL,
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total forward steps: {total_forward_steps}, optimizer steps: {optimizer_steps}')

    swanlab.init(project='twinkle', config={
        'backend': BACKEND,
        'model_id': MODEL_ID,
        'batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
        'temperature': TEMPERATURE,
        'emb_max_length': EMB_MAX_LENGTH,
        'compress_max_tokens': COMPRESS_MAX_TOKENS,
    })

    # -------- Train loop -----------------------------------------------------
    def _sample_batch(raw_batch):
        """Compress via vLLM sampler; fall back to API on truncation."""
        compress_prompts, valid_indices, raw_pairs, prompt_queries = \
            _build_compress_prompts(raw_batch)
        if not compress_prompts:
            return None

        with retrainer.sampler_lock:
            responses = condenser_sampler.sample(compress_prompts, compress_params)

        # Extract decoded texts; detect truncations and fall back to API
        decoded_texts: List[str] = []
        for ri, resp in enumerate(responses):
            seq = resp.sequences[0] if resp.sequences else None
            if seq and seq.stop_reason != 'length' and seq.decoded:
                decoded_texts.append(seq.decoded)
            else:
                # Truncated or empty — fall back to API
                api_result = _api_compress(api_client, compress_prompts[ri])
                if api_result:
                    decoded_texts.append(api_result)
                    # Determine source text for failure logging
                    pair_idx = ri // 2
                    q_raw, c_raw = raw_pairs[pair_idx]
                    source_text = q_raw if ri % 2 == 0 else c_raw
                    _log_failure(source_text, prompt_queries[ri], api_result,
                                 valid_indices[pair_idx])
                    retrainer.notify_failure()
                else:
                    decoded_texts.append('')

        # Build embedding features from decoded texts
        emb_features: List[Dict[str, Any]] = []
        for i in range(0, len(decoded_texts), 2):
            q_text = decoded_texts[i]
            c_text = decoded_texts[i + 1]
            q_raw, c_raw = raw_pairs[i // 2]
            _log_responses(q_text, c_text, valid_indices[i // 2],
                           query_raw=q_raw, cot_raw=c_raw)
            feat_q = _get_first_feature(q_text, emb_template, role='anchor')
            feat_c = _get_first_feature(c_text, emb_template, role='positive')
            if feat_q and feat_c:
                emb_features.append(feat_q)
                emb_features.append(feat_c)

        if len(emb_features) < 4:
            return None
        return emb_features

    cur_step = 0
    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    for epoch in range(NUM_EPOCHS):
        batch_iter = iter(dataloader)
        prefetch_future = None
        first_batch = next(batch_iter, None)
        if first_batch is not None:
            prefetch_future = prefetch_executor.submit(_sample_batch, first_batch)

        for raw_batch in batch_iter:
            emb_features = prefetch_future.result() if prefetch_future else None
            prefetch_future = prefetch_executor.submit(_sample_batch, raw_batch)

            if emb_features is None:
                continue

            model.forward_backward(inputs=emb_features, task='embedding')
            model.clip_grad_and_step(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
            cur_step += 1

            if cur_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(
                    f'Epoch {epoch} Step {cur_step}/{total_forward_steps}, metric: {metric}')
                log_dict = {}
                for k, v in metric.items():
                    if not v:
                        continue
                    try:
                        log_dict[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                log_dict['epoch'] = epoch
                swanlab.log(log_dict, step=cur_step)
            if cur_step % SAVE_INTERVAL == 0:
                save_checkpoint(model, f'step_{cur_step}')

        # Drain last prefetched batch
        if prefetch_future is not None:
            emb_features = prefetch_future.result()
            if emb_features is not None:
                model.forward_backward(inputs=emb_features, task='embedding')
                model.clip_grad_and_step()
                cur_step += 1

    prefetch_executor.shutdown(wait=False)
    retrainer.stop()
    save_checkpoint(model, 'last-checkpoint')


if __name__ == '__main__':
    train()
