"""LoRA embedding training with online compression via frozen vLLM condenser.

Architecture (8 GPUs total):
  - Ranks 0-3 (``model``): Trainable embedding model with LoRA, InfoNCE loss.
  - Ranks 4-7 (``condenser_sampler``): Frozen vLLM condenser for online compression.

When the condenser sampler truncates or regresses to the legacy schema, an
external OpenAI-compatible API produces the correct compression. The failure is
logged to failures.jsonl for offline SFT data regeneration.

Launch:
    python cookbook/exp/train_embedding_lora_ddp.py
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import swanlab

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.loss import InfonceLoss
from twinkle.metric import EmbeddingMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template, Template
from twinkle.utils.parallel import PosixFileLock
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_think import get_dataset as get_dataset_think  # noqa: E402
from dataset_index import get_dataset as get_dataset_index  # noqa: E402

logger = get_logger()

# -- Backend selection --------------------------------------------------------
BACKEND: Literal['transformers', 'megatron'] = 'transformers'

# Condenser (online compression + LoRA self-improvement); embedding model trains LoRA on top of MODEL_ID.
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
TEMPLATE_NAME = 'Qwen3_5Template'

# -- GPU placement (8 total) --------------------------------------------------
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
CONDENSER_SAMPLER_GPUS = int(os.environ.get('CONDENSER_SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + CONDENSER_SAMPLER_GPUS

# -- Embedding training hyper-params ------------------------------------------
EMB_MAX_LENGTH = 8192
HARD_NEGATIVES = None
# 0.07 keeps gradient on diag pairs until cosine clears ~0.75; 0.03 saturated near 0.40.
TEMPERATURE = 0.07

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 1
LOG_INTERVAL = 2
SAVE_INTERVAL = 2000
NUM_EPOCHS = 1

TOTAL_SAMPLES: Optional[int] = None
# Post-build caps on each loader (None = no cap). Applied via .select() before mix.
THINK_CAP: Optional[int] = 400_000
INDEX_CAP: Optional[int] = 400_000
MIX_SHUFFLE_SEED = 42

# -- Resume from checkpoint ---------------------------------------------------
# Empty by default — build_model falls back to MODEL_ID (the published emb model).
# Set both to point at a local in-progress run only when resuming the *same* schedule.
RESUME_CHECKPOINT = os.environ.get('RESUME_CHECKPOINT', '')
RESUME_STEP = int(os.environ.get('RESUME_STEP', 0))

# -- Online-compression knobs -------------------------------------------------
# Below this length, condenser fabricates content for open-ended short prompts;
# query passes through as qr verbatim and cot rows are dropped from training.
MIN_TEXT_CHARS = 256
DATASET_MAX_TOKENS = 32768
COMPRESS_TEMPERATURE = 0.2
COMPRESS_TOP_P = 0.5
COMPRESS_MAX_MODEL_LEN = 32768

# How many BATCH_SIZE chunks to fetch and compress in one vLLM call.
PREFETCH_BATCH_MULTIPLIER = int(os.environ.get('PREFETCH_BATCH_MULTIPLIER', 8))

# -- OpenAI API fallback for truncated compressions ---------------------------
COMPRESS_API_KEY = os.environ.get('COMPRESS_API_KEY', '')
COMPRESS_BASE_URL = os.environ.get('COMPRESS_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
COMPRESS_MODEL = os.environ.get('COMPRESS_MODEL', 'qwen3.7-max')
# Minimum gap between API calls (seconds); bounds dashscope qps under provider limits.
API_MIN_INTERVAL = float(os.environ.get('API_MIN_INTERVAL', 0.1))
API_CONCURRENCY = int(os.environ.get('API_CONCURRENCY', 8))
# vLLM sampler timeout (seconds); if a sample() call exceeds this, fall back to API.
SAMPLER_TIMEOUT = float(os.environ.get('SAMPLER_TIMEOUT', 300))

# -- Output paths -------------------------------------------------------------
OUTPUT_DIR = f'./output/embedding_lora_{BACKEND}'
RESPONSE_LOG = os.environ.get('RESPONSE_LOG', f'./output/embedding_lora_{BACKEND}/responses.jsonl')
FAILURE_LOG = os.environ.get('FAILURE_LOG', f'./output/embedding_lora_{BACKEND}/failures.jsonl')


# =============================================================================
# Prompts (from make_condenser_dataset.py — "## Summary" format)
# =============================================================================

COMPRESS_SYSTEM = """\
You are a compression and summary assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Summary` directly, then calls `extract_compressed` \
on any topic-key listed under `## More` to recover its \
fuller content.

  `## Summary`               — extreme-density text the reader reads directly.
  `## More` — a topic index whose keys are valid arguments \
to `extract_compressed` for recovering material not captured inline.

Together the two sections must form a COMPLETE, NON-DISTORTING inventory of the \
source for the query — nothing essential lost, nothing implied that the source \
does not support. NO preamble, NO meta-commentary, NO code fences wrapping the \
whole output.

Output skeleton:

## Summary
Topic: <what the source is about + scope, one line>
<dense body answering the query>

## More
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

`## Summary` rules:
1. TOPIC LINE — line 1 is ALWAYS `Topic: <subject — scope>`, even when the \
query is narrow. Anchors both the reader and the tool.
2. DENSITY — every token in the body carries query-relevant signal; cut filler.
3. PRIMARY-COMPLETE — never silently drop a fact essential to answering the \
query. Anything cut for length MUST appear as a key under \
`## More`.
4. NON-MISLEADING — phrasing must not let the reader infer anything the source \
does not support; partial truths that mislead are worse than honest omissions \
flagged in the index.
5. SELF-CONTAINED — the reader can act on the answer without re-opening the source.
6. FAITHFUL — only content the source supports; no fabrication, no extrapolation.
7. LANGUAGE — match the source language.
8. NO outer code fences around the whole answer; no meta-commentary.

`## More` rules (MANDATORY — this section is never omitted):
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

COMPRESS_USER = (
    'Downstream model will read your compressed block to decide whether to '
    'expand it. Compress faithfully: preserve the passage topic + core facts. '
    'Do NOT invent facts. Do NOT drop major facts. Do NOT write meta-commentary '
    'about the Query (never write "Query info: absent", "no X mention", etc.); '
    'if the passage does not address the Query, still summarize the passage. '
    'CRITICAL LANGUAGE RULE: detect the dominant language of the Passage '
    '(NOT the Query, NOT this instruction) and write the ENTIRE output in that '
    'same language; English passage → English output, Chinese passage → '
    'Chinese output, Japanese passage → Japanese output. NEVER translate, '
    'NEVER mix languages, NEVER copy these instructions into the output.\n\n'
    '## Query (ordering hint only — still summarize the whole passage)\n{query}\n\n'
    '## Passage\n{text}')


# =============================================================================
# Logging helpers
# =============================================================================

_response_lock: Optional[PosixFileLock] = None
_failure_lock: Optional[PosixFileLock] = None

# Monotonic global sample id; per-batch index would alias across batches.
_sample_counter = 0
_sample_counter_lock = threading.Lock()

_api_throttle_lock = threading.Lock()
_api_last_call = [0.0]


def _api_throttle():
    with _api_throttle_lock:
        gap = time.monotonic() - _api_last_call[0]
        if gap < API_MIN_INTERVAL:
            time.sleep(API_MIN_INTERVAL - gap)
        _api_last_call[0] = time.monotonic()


def _next_sample_id() -> int:
    global _sample_counter
    with _sample_counter_lock:
        sid = _sample_counter
        _sample_counter += 1
        return sid


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
            {'role': 'system', 'content': COMPRESS_SYSTEM},
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
    model_id = RESUME_CHECKPOINT if RESUME_CHECKPOINT else MODEL_ID
    if BACKEND == 'transformers':
        model = TransformersModel(
            model_id=model_id,
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
            num_warmup_steps=200,
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

# Hard-templated hints: the condenser SFT prior maps `Skill` to the legacy
# `Use when: / numbered steps / Output:` skeleton on long inputs; embedding the
# exact 4-line body template + explicit negative constraints is the only way to
# override it deterministically across query and cot sides.
EMBED_QUERY_Q = (
    'Summarize this query for retrieval. '
    'The body of ## Summary MUST follow this EXACT 4-line template — '
    'do NOT emit "Use when:", numbered procedure steps, or "Output:":\n'
    'Topic: <specific pattern name — scope>\n'
    'Problem: <what concrete problem is being asked>\n'
    'Skill: <which specific method/technique/pattern is required to solve it>\n'
    'Knowledge: <which domains/concepts/facts must be invoked>\n'
    'Then emit the mandatory ## More section as usual. '
    'Topic must name the specific pattern, never generic labels.')
EMBED_QUERY_COT = (
    'Summarize this reasoning trace for retrieval. '
    'The body of ## Summary MUST follow this EXACT 4-line template — '
    'do NOT emit "Use when:", numbered procedure steps, or "Output:":\n'
    'Topic: <specific pattern name — scope>\n'
    'Problem: <what concrete problem this trace tackled>\n'
    'Skill: <which specific method/technique/pattern was applied>\n'
    'Knowledge: <which domains/concepts/facts were used>\n'
    'Then emit the mandatory ## More section as usual. '
    'Topic must name the specific pattern, never generic labels.')

# Legacy schema (Use when: / numbered steps / Output:) — mixed in 50/50 with the
# new schema to expose the embedder to schema-invariant semantic alignment.
# Both query and cot of the SAME pair always use the SAME schema; cross-schema
# anchors and positives would re-introduce the schema asymmetry we just fixed.
EMBED_QUERY_Q_LEGACY = (
    'What problem does this passage address, and what skill or method is needed? '
    'Topic must name the specific pattern, never generic labels. '
    'Compress into a retrieval-friendly need description.')
EMBED_QUERY_COT_LEGACY = (
    'Extract the reusable skill: trigger conditions, key steps, and expected output. '
    'Topic names the method/pattern; format as "Use when: ...", numbered steps, '
    '"Output: ...". Compress into a standardized procedure for retrieval.')


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

    Returns (prompts, valid_indices, raw_pairs, prompt_queries, passthrough, schemas)
    where:
    - prompts: flat-interleaved [query_0, cot_0, query_1, cot_1, ...]; ``None`` means
      passthrough (use raw text directly, do not call sampler)
    - valid_indices: which rows passed the min-length filter
    - raw_pairs: [(query, cot), ...]
    - prompt_queries: the query string used for each prompt (for failure logging)
    - passthrough: parallel to prompts; non-None text means "use this verbatim as qc"
    - schemas: parallel to prompts; 'new' or 'legacy', drives validator branch
    """
    prompts: List[Optional[Dict[str, Any]]] = []
    valid_indices: List[int] = []
    raw_pairs: List[tuple] = []
    prompt_queries: List[str] = []
    passthrough: List[Optional[str]] = []
    schemas: List[str] = []
    # Conservative char budget: 32768 max_length - 8192 gen - ~2k prompt overhead = ~22k tokens.
    # 30k cap bounds vLLM batch latency (vLLM batches by max prompt length).
    _MAX_COT_CHARS = 30_000
    for i, row in enumerate(rows):
        query, cot = _extract_query_cot(row)
        if not query or len(cot) < MIN_TEXT_CHARS:
            continue
        if len(cot) > _MAX_COT_CHARS:
            continue
        valid_indices.append(i)
        raw_pairs.append((query, cot))
        # 50/50 schema mix; same schema for query+cot of one pair to keep alignment.
        schema = 'legacy' if (i % 2 == 0) else 'new'
        q_hint = EMBED_QUERY_Q_LEGACY if schema == 'legacy' else EMBED_QUERY_Q
        c_hint = EMBED_QUERY_COT_LEGACY if schema == 'legacy' else EMBED_QUERY_COT
        # Short query bypasses condenser to avoid skeleton-induced hallucination.
        if len(query) < MIN_TEXT_CHARS:
            prompts.append(None)
            passthrough.append(query)
        else:
            user = COMPRESS_USER.format(query=q_hint, text=query)
            prompts.append({'messages': [
                {'role': 'system', 'content': COMPRESS_SYSTEM},
                {'role': 'user', 'content': user},
            ]})
            passthrough.append(None)
        prompt_queries.append(q_hint)
        schemas.append(schema)
        user = COMPRESS_USER.format(query=c_hint, text=cot)
        prompts.append({'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': user},
        ]})
        prompt_queries.append(c_hint)
        passthrough.append(None)
        schemas.append(schema)
    return prompts, valid_indices, raw_pairs, prompt_queries, passthrough, schemas


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

_LEGACY_USE_WHEN_RE = re.compile(r'(?im)^\s*Use when\s*:')
_SCHEMA_MARKERS = ('Problem:', 'Skill:', 'Knowledge:')


def _is_truncated_compression(text: str, schema: str = 'new') -> bool:
    """Reject structurally incomplete OR schema-regressed condenser output.

    Triggers API fallback when the vLLM output:
      * lacks ``## Summary`` / ``## More``,
      * has an empty or unterminated ``## More`` bullet list, or
      * (schema='new' only) regresses to the legacy ``Use when: / numbered-steps /
        Output:`` skeleton instead of the mandated Problem/Skill/Knowledge 4-line
        body — the dominant cot-side failure mode that drives sim < 0.45 drops on
        the RAG index.

    For schema='legacy', body markers are intentionally NOT enforced: the legacy
    template legitimately emits ``Use when:`` and the SFT prior already produces
    that shape natively, so only structural completeness is checked.
    """
    if not text or not text.strip():
        return True
    if '## More' not in text or '## Summary' not in text:
        return True
    after_more = text.split('## More', 1)[1].strip()
    if not after_more:
        return True
    last_line = after_more.splitlines()[-1].strip()
    if not (last_line.startswith('-') or last_line.endswith(')')):
        return True
    if schema == 'new':
        summary_body = text.split('## Summary', 1)[1].split('## More', 1)[0]
        if _LEGACY_USE_WHEN_RE.search(summary_body):
            return True
        if not all(marker in summary_body for marker in _SCHEMA_MARKERS):
            return True
    return False


def _api_compress(api_client: OpenAIClient, prompt: Dict[str, Any]) -> Optional[str]:
    """Call external API to compress when vLLM truncates."""
    _api_throttle()
    trajectory = {'messages': prompt['messages']}
    # Cap max_tokens to leave ample prompt headroom inside the API model context.
    sp = SamplingParams(temperature=0.2, max_tokens=8192)
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
# Main training
# =============================================================================

def train():
    # -------- Device groups (2 groups) ----------------------------------------
    device_groups = [
        DeviceGroup(name='model',
                    ranks=list(range(MODEL_GPUS)),
                    device_type='GPU'),
        DeviceGroup(name='condenser_sampler',
                    ranks=list(range(MODEL_GPUS, MODEL_GPUS + CONDENSER_SAMPLER_GPUS)),
                    device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    condenser_sampler_mesh = DeviceMesh.from_sizes(
        world_size=CONDENSER_SAMPLER_GPUS, dp_size=CONDENSER_SAMPLER_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)

    # -------- Data -----------------------------------------------------------
    dataset = get_dataset_think(total=TOTAL_SAMPLES, load_from_cache_file=True)
    if THINK_CAP and len(dataset.dataset) > THINK_CAP:
        dataset.dataset = dataset.dataset.select(range(THINK_CAP))
    if INDEX_CAP != 0:
        from datasets import concatenate_datasets
        ds_index = get_dataset_index(total=None, load_from_cache_file=True)
        if INDEX_CAP and len(ds_index.dataset) > INDEX_CAP:
            ds_index.dataset = ds_index.dataset.select(range(INDEX_CAP))
        n_think = len(dataset.dataset)
        n_index = len(ds_index.dataset)
        # Both loaders emit identical {id, source, messages} schema post-QP.
        dataset.dataset = concatenate_datasets(
            [dataset.dataset, ds_index.dataset]).shuffle(seed=MIX_SHUFFLE_SEED)
        logger.info(f'[mix] think={n_think} + index={n_index} → total={len(dataset.dataset)}')
    _mega_batch_size = BATCH_SIZE * PREFETCH_BATCH_MULTIPLIER
    dataloader = DataLoader(dataset=dataset, batch_size=_mega_batch_size, shuffle=True)
    total_forward_steps = len(dataloader) * PREFETCH_BATCH_MULTIPLIER * NUM_EPOCHS
    optimizer_steps = total_forward_steps // GRADIENT_ACCUMULATION_STEPS

    # -------- Embedding model (4 GPU) ----------------------------------------
    model = build_model(model_mesh)
    model.set_processor(InputProcessor)
    model.set_loss(InfonceLoss, temperature=TEMPERATURE, use_batch=True,
                   hard_negatives=HARD_NEGATIVES)
    setup_optimizer(model, optimizer_steps)
    model.add_metric(EmbeddingMetric, is_training=True)

    # -------- Condenser sampler (4 GPU, vLLM) --------------------------------
    emb_template = Qwen3_5Template(model_id=MODEL_ID, max_length=EMB_MAX_LENGTH, enable_thinking=False)
    # Special tokens come from the condenser tokenizer because the leak we strip is in its decoded output.
    condenser_template = Qwen3_5Template(model_id=CONDENSE_MODEL_ID, max_length=DATASET_MAX_TOKENS,
                                  enable_thinking=False)
    _special_tokens = set(condenser_template.tokenizer.all_special_tokens)
    condenser_sampler = vLLMSampler(
        model_id=CONDENSE_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': COMPRESS_MAX_MODEL_LEN,
        },
        device_mesh=condenser_sampler_mesh,
        remote_group='condenser_sampler',
    )
    condenser_sampler.set_template(
        TEMPLATE_NAME, model_id=CONDENSE_MODEL_ID, enable_thinking=False,
        truncation_strategy='delete', max_length=DATASET_MAX_TOKENS)
    compress_params = SamplingParams(
        max_tokens=8192,
        temperature=COMPRESS_TEMPERATURE,
        top_p=COMPRESS_TOP_P,
        num_samples=1,
    )

    condenser_sampler._ray_get_timeout = SAMPLER_TIMEOUT
    _sampler_epoch = 0

    def _rebuild_sampler():
        """Kill stuck actors and recreate the vLLM sampler from scratch."""
        nonlocal condenser_sampler, _sampler_epoch
        import ray
        for actor in getattr(condenser_sampler, '_actors', []):
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        logger.warning('[sampler] killed stuck actors, recreating sampler \u2026')
        new = vLLMSampler(
            model_id=CONDENSE_MODEL_ID,
            engine_args={'gpu_memory_utilization': 0.8, 'max_model_len': COMPRESS_MAX_MODEL_LEN},
            device_mesh=condenser_sampler_mesh,
            remote_group='condenser_sampler',
        )
        new.set_template(
            TEMPLATE_NAME, model_id=CONDENSE_MODEL_ID, enable_thinking=False,
            truncation_strategy='delete', max_length=DATASET_MAX_TOKENS)
        new._ray_get_timeout = SAMPLER_TIMEOUT
        condenser_sampler = new
        _sampler_epoch += 1
        logger.warning('[sampler] sampler rebuilt successfully')

    # -------- OpenAI API client for fallback ---------------------------------
    api_client = OpenAIClient(
        model=COMPRESS_MODEL,
        api_key=COMPRESS_API_KEY,
        base_url=COMPRESS_BASE_URL,
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total forward steps: {total_forward_steps}, optimizer steps: {optimizer_steps}')
    if RESUME_STEP > 0:
        logger.info(f'Resuming from step {RESUME_STEP}, checkpoint: {RESUME_CHECKPOINT}')
        logger.info(f'Starting at epoch {RESUME_STEP // (total_forward_steps // NUM_EPOCHS)}, '
                    f'skipping {RESUME_STEP - (RESUME_STEP // (total_forward_steps // NUM_EPOCHS)) * (total_forward_steps // NUM_EPOCHS)} batches')

    swanlab.init(project='twinkle', config={
        'backend': BACKEND,
        'model_id': MODEL_ID,
        'condense_model_id': CONDENSE_MODEL_ID,
        'batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
        'temperature': TEMPERATURE,
        'emb_max_length': EMB_MAX_LENGTH,
        'DATASET_MAX_TOKENS': DATASET_MAX_TOKENS,
    })

    # -------- Train loop -----------------------------------------------------
    def _sample_batch(raw_batch):
        """Compress via vLLM sampler; fall back to API on truncation."""
        _t_enter = time.monotonic()
        compress_prompts, valid_indices, raw_pairs, prompt_queries, passthrough, schemas = \
            _build_compress_prompts(raw_batch)
        _t_build = time.monotonic()
        if len(compress_prompts) < 4:
            return None

        # Only submit non-passthrough prompts to the sampler.
        sampler_input = [p for p in compress_prompts if p is not None]
        sampler_pos = [ri for ri, p in enumerate(compress_prompts) if p is not None]
        if sampler_input:
            try:
                sampler_responses = condenser_sampler.sample(sampler_input, compress_params)
            except Exception as exc:
                logger.warning(f'[sampler] error \u2192 API fallback: {exc}')
                sampler_responses = [None] * len(sampler_input)
                if 'Timeout' in type(exc).__name__:
                    try:
                        _rebuild_sampler()
                    except Exception as re_exc:
                        logger.error(f'[sampler] rebuild failed: {re_exc}')
        else:
            sampler_responses = []
        _t_sample = time.monotonic()

        responses = [None] * len(compress_prompts)
        for resp, pos in zip(sampler_responses, sampler_pos):
            responses[pos] = resp

        # Extract decoded texts; detect truncations and fall back to API
        decoded_texts: List[Optional[str]] = [None] * len(compress_prompts)
        fallback_indices: List[int] = []
        for ri in range(len(compress_prompts)):
            if passthrough[ri] is not None:
                decoded_texts[ri] = passthrough[ri]
                continue
            resp = responses[ri]
            seq = resp.sequences[0] if resp and resp.sequences else None
            text = ''
            if seq and seq.stop_reason != 'length' and seq.decoded:
                text = seq.decoded
                for tok in _special_tokens:
                    text = text.replace(tok, '')
                text = text.rstrip()

            needs_fallback = (not seq or seq.stop_reason == 'length'
                              or _is_truncated_compression(text, schemas[ri]))
            if not needs_fallback:
                decoded_texts[ri] = text
            else:
                fallback_indices.append(ri)

        _api_calls = len(fallback_indices)
        if fallback_indices:
            from concurrent.futures import as_completed
            api_futures = {}
            with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as api_pool:
                for ri in fallback_indices:
                    api_futures[api_pool.submit(_api_compress, api_client, compress_prompts[ri])] = ri
                for fut in as_completed(api_futures):
                    ri = api_futures[fut]
                    api_result = fut.result()
                    if api_result and not _is_truncated_compression(api_result, schemas[ri]):
                        decoded_texts[ri] = api_result
                        pair_idx = ri // 2
                        q_raw, c_raw = raw_pairs[pair_idx]
                        source_text = q_raw if ri % 2 == 0 else c_raw
                        _log_failure(source_text, prompt_queries[ri], api_result,
                                     valid_indices[pair_idx])
                    else:
                        decoded_texts[ri] = ''
        _t_api = time.monotonic()

        # Build embedding features from decoded texts
        emb_features: List[Dict[str, Any]] = []
        for i in range(0, len(decoded_texts), 2):
            q_text = decoded_texts[i]
            c_text = decoded_texts[i + 1]
            q_raw, c_raw = raw_pairs[i // 2]
            _log_responses(q_text, c_text, _next_sample_id(),
                           query_raw=q_raw, cot_raw=c_raw)
            feat_q = _get_first_feature(q_text, emb_template, role='anchor')
            feat_c = _get_first_feature(c_text, emb_template, role='positive')
            if feat_q and feat_c:
                emb_features.append(feat_q)
                emb_features.append(feat_c)
        _t_feat = time.monotonic()

        logger.info(
            f'[prefetch] prompts={len(sampler_input)} api={_api_calls} feats={len(emb_features)} | '
            f'build={_t_build - _t_enter:.1f}s '
            f'vllm={_t_sample - _t_build:.1f}s '
            f'api={_t_api - _t_sample:.1f}s feat={_t_feat - _t_api:.1f}s '
            f'total={_t_feat - _t_enter:.1f}s')

        _target = BATCH_SIZE * 2
        minibatches = [emb_features[i:i + _target] for i in range(0, len(emb_features), _target)]
        minibatches = [mb for mb in minibatches if len(mb) >= 4]
        return minibatches if minibatches else None

    cur_step = RESUME_STEP
    _batches_per_epoch = len(dataloader)
    _steps_per_mega = PREFETCH_BATCH_MULTIPLIER
    _start_epoch = cur_step // (_batches_per_epoch * _steps_per_mega) if cur_step > 0 else 0
    _skip_batches_in_epoch = max(0, cur_step // _steps_per_mega - _start_epoch * _batches_per_epoch)

    _ema_prefetch = 0.0
    _ema_train = 0.0
    _ema_alpha = 0.1

    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    for epoch in range(_start_epoch, NUM_EPOCHS):
        if _skip_batches_in_epoch > 0:
            dataloader.skip_consumed_samples(_skip_batches_in_epoch * _mega_batch_size)
        batch_iter = iter(dataloader)
        _skip_batches_in_epoch = 0

        first = next(batch_iter, None)
        future = prefetch_executor.submit(_sample_batch, first) if first else None

        for raw_mega_batch in batch_iter:
            t0 = time.monotonic()
            minibatches = future.result() if future else None
            t_prefetch = time.monotonic() - t0
            future = prefetch_executor.submit(_sample_batch, raw_mega_batch)

            if not minibatches:
                continue

            for mb in minibatches:
                t1 = time.monotonic()
                model.forward_backward(inputs=mb, task='embedding')
                model.clip_grad_and_step(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
                t_train = time.monotonic() - t1
                cur_step += 1

                _ema_prefetch = _ema_alpha * t_prefetch + (1 - _ema_alpha) * _ema_prefetch if cur_step > RESUME_STEP + 1 else t_prefetch
                _ema_train = _ema_alpha * t_train + (1 - _ema_alpha) * _ema_train if cur_step > RESUME_STEP + 1 else t_train

                if cur_step % LOG_INTERVAL == 0:
                    metric = model.calculate_metric(is_training=True)
                    _bottleneck = 'PREFETCH' if _ema_prefetch > _ema_train else 'TRAIN'
                    logger.info(
                        f'Epoch {epoch} Step {cur_step}/{total_forward_steps}, metric: {metric} | '
                        f'prefetch={t_prefetch:.1f}s(ema {_ema_prefetch:.1f}) '
                        f'train={t_train:.1f}s(ema {_ema_train:.1f}) '
                        f'bottleneck={_bottleneck}')
                    log_dict = {}
                    for k, v in metric.items():
                        if not v:
                            continue
                        try:
                            log_dict[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                    log_dict['epoch'] = epoch
                    log_dict['prefetch_sec'] = round(t_prefetch, 2)
                    log_dict['train_sec'] = round(t_train, 2)
                    swanlab.log(log_dict, step=cur_step)
                if cur_step % SAVE_INTERVAL == 0:
                    save_checkpoint(model, f'step_{cur_step}')
                t_prefetch = 0.0

        # Drain final mega-batch
        if future:
            minibatches = future.result()
            future = None
            if minibatches:
                for mb in minibatches:
                    model.forward_backward(inputs=mb, task='embedding')
                    model.clip_grad_and_step(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
                    cur_step += 1
                    if cur_step % SAVE_INTERVAL == 0:
                        save_checkpoint(model, f'step_{cur_step}')

    prefetch_executor.shutdown(wait=False)
    save_checkpoint(model, 'last-checkpoint')


if __name__ == '__main__':
    train()
