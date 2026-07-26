"""Offline compression pipeline: raw datasets → condenser → pre-compressed embedding dataset.

Loads think/index/hard datasets, compresses query/cot/negatives via vLLM condenser
with API fallback, saves a single HF Dataset ready for embedding training.

Output schema: {anchor_text, positive_text, negative_texts, source}

Launch (8 GPUs — 4 for vLLM condenser):
    python cookbook/exp/embedding/make_embedding_dataset.py
"""
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle.utils.parallel import PosixFileLock
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_think import get_dataset as get_dataset_think  # noqa: E402
from dataset_index import get_dataset as get_dataset_index  # noqa: E402
from dataset_hard import get_dataset as get_dataset_hard  # noqa: E402

logger = get_logger()

# -- Model config -------------------------------------------------------------
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
TEMPLATE_NAME = 'Qwen3_5Template'

# -- GPU placement (condenser only) -------------------------------------------
CONDENSER_GPUS = int(os.environ.get('CONDENSER_GPUS', 8))

# -- Dataset caps -------------------------------------------------------------
TOTAL_SAMPLES: Optional[int] = None
THINK_CAP: Optional[int] = int(os.environ.get('THINK_CAP', 100_000))
INDEX_CAP: Optional[int] = int(os.environ.get('INDEX_CAP', 100_000))
HARD_CAP: Optional[int] = int(os.environ.get('HARD_CAP', 0)) or None
HARD_MAX_NEGATIVES = int(os.environ.get('HARD_MAX_NEGATIVES', 8))

# -- Compression params -------------------------------------------------------
MIN_TEXT_CHARS = 256
DATASET_MAX_TOKENS = 32768
COMPRESS_TEMPERATURE = 0.2
COMPRESS_TOP_P = 0.5
COMPRESS_MAX_MODEL_LEN = 32768
BATCH_SIZE = int(os.environ.get('COMPRESS_BATCH_SIZE', 128))

# -- API fallback -------------------------------------------------------------
COMPRESS_API_KEY = os.environ.get('COMPRESS_API_KEY', '')
COMPRESS_BASE_URL = os.environ.get('COMPRESS_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
COMPRESS_MODEL = os.environ.get('COMPRESS_MODEL', 'qwen3.7-max')
API_MIN_INTERVAL = float(os.environ.get('API_MIN_INTERVAL', 0.1))
API_CONCURRENCY = int(os.environ.get('API_CONCURRENCY', 24))
SAMPLER_TIMEOUT = float(os.environ.get('SAMPLER_TIMEOUT', 300))

# -- Output -------------------------------------------------------------------
OUTPUT_DIR = os.environ.get('EMB_DATASET_OUTPUT', './output/embedding_dataset')
RESULTS_JSONL = f'{OUTPUT_DIR}/results.jsonl'
PROGRESS_FILE = f'{OUTPUT_DIR}/progress.json'

# =============================================================================
# Prompts
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

EMBED_QUERY_Q_LEGACY = (
    'What problem does this passage address, and what skill or method is needed? '
    'Topic must name the specific pattern, never generic labels. '
    'Compress into a retrieval-friendly need description.')

EMBED_QUERY_COT_LEGACY = (
    'Extract the reusable skill: trigger conditions, key steps, and expected output. '
    'Topic names the method/pattern; format as "Use when: ...", numbered steps, '
    '"Output: ...". Compress into a standardized procedure for retrieval.')

EMBED_QUERY_REASONIR_Q = (
    'Extract the abstract PROBLEM TYPE from this query. '
    'IGNORE all specific numbers, values, variable names, and parameters — '
    'focus ONLY on the CLASS of problem and the METHODOLOGY required. '
    'The body of ## Summary MUST follow this EXACT 4-line template:\n'
    'Topic: <problem class — mathematical/logical domain>\n'
    'Problem: <what abstract TYPE of problem needs solving, no specific values>\n'
    'Skill: <which general method/technique is required>\n'
    'Knowledge: <which theoretical concepts/formulas must be invoked>\n'
    'Then emit the mandatory ## More section as usual. '
    'Topic must name the method class, never mention specific numbers.')

EMBED_QUERY_REASONIR_COT = (
    'Extract the abstract METHODOLOGY demonstrated in this solution. '
    'IGNORE all specific numbers, values, and computed results — '
    'focus ONLY on the general TECHNIQUE and key reasoning STEPS. '
    'The body of ## Summary MUST follow this EXACT 4-line template:\n'
    'Topic: <method/technique name — scope>\n'
    'Problem: <what abstract type of problem this method solves>\n'
    'Skill: <key steps of the methodology in abstract terms>\n'
    'Knowledge: <theoretical basis and prerequisites>\n'
    'Then emit the mandatory ## More section as usual. '
    'Topic must name the method class, never mention specific numbers.')


# =============================================================================
# Validation & API fallback
# =============================================================================

_LEGACY_USE_WHEN_RE = re.compile(r'(?im)^\s*Use when\s*:')
_SCHEMA_MARKERS = ('Problem:', 'Skill:', 'Knowledge:')


def _is_truncated_compression(text: str, schema: str = 'new') -> bool:
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


_api_semaphore = threading.Semaphore(API_CONCURRENCY)
_api_bucket_lock = threading.Lock()
_api_tokens = [float(API_CONCURRENCY)]
_api_last_refill = [time.monotonic()]


def _api_throttle():
    """Token-bucket rate limiter: API_CONCURRENCY requests per API_MIN_INTERVAL*API_CONCURRENCY window."""
    _api_semaphore.acquire()
    try:
        with _api_bucket_lock:
            now = time.monotonic()
            elapsed = now - _api_last_refill[0]
            refill = elapsed / API_MIN_INTERVAL
            _api_tokens[0] = min(float(API_CONCURRENCY), _api_tokens[0] + refill)
            _api_last_refill[0] = now
            if _api_tokens[0] >= 1.0:
                _api_tokens[0] -= 1.0
            else:
                wait = (1.0 - _api_tokens[0]) * API_MIN_INTERVAL
                _api_tokens[0] = 0.0
                time.sleep(wait)
    finally:
        _api_semaphore.release()


def _api_compress(api_client: OpenAIClient, prompt: Dict[str, Any]) -> Optional[str]:
    _api_throttle()
    trajectory = {'messages': prompt['messages']}
    sp = SamplingParams(temperature=0.2, max_tokens=8192)
    try:
        reply = api_client(trajectory, sp, extra_body={'enable_thinking': False})
    except Exception as exc:
        logger.warning(f'[api_fallback] error: {exc}')
        return None
    content = (reply.get('content') or '').strip()
    if not content:
        return None
    m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', content, re.DOTALL)
    if m:
        content = m.group(1).strip()
    return content


# =============================================================================
# Core compression logic
# =============================================================================

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


def _compress_batch_phase1(
    rows: List[Dict[str, Any]],
    condenser_sampler,
    compress_params: SamplingParams,
    special_tokens: set,
    source_type: str,
) -> Optional[Dict[str, Any]]:
    """Phase 1 (GPU): build prompts → vLLM sample → validate. Returns state for phase 2."""
    _MAX_COT_CHARS = 30_000

    if source_type == 'hard':
        return _compress_hard_phase1(rows, condenser_sampler, compress_params,
                                     special_tokens, source_type)

    prompts: List[Optional[Dict[str, Any]]] = []
    meta: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        query, cot = _extract_query_cot(row)
        if not query or len(cot) < MIN_TEXT_CHARS or len(cot) > _MAX_COT_CHARS:
            continue
        schema = 'legacy' if (i % 2 == 0) else 'new'
        q_hint = EMBED_QUERY_Q_LEGACY if schema == 'legacy' else EMBED_QUERY_Q
        c_hint = EMBED_QUERY_COT_LEGACY if schema == 'legacy' else EMBED_QUERY_COT

        if len(query) < MIN_TEXT_CHARS:
            prompts.append(None)
        else:
            user = COMPRESS_USER.format(query=q_hint, text=query)
            prompts.append({'messages': [
                {'role': 'system', 'content': COMPRESS_SYSTEM},
                {'role': 'user', 'content': user},
            ]})
        user_c = COMPRESS_USER.format(query=c_hint, text=cot)
        prompts.append({'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': user_c},
        ]})
        meta.append({'query_raw': query, 'cot_raw': cot, 'schema': schema,
                     'q_hint': q_hint, 'source': source_type,
                     'row_id': row.get('id', str(i))})

    if not prompts:
        return {'final': []}

    sampler_input = [p for p in prompts if p is not None]
    sampler_pos = [ri for ri, p in enumerate(prompts) if p is not None]
    try:
        sampler_responses = condenser_sampler.sample(sampler_input, compress_params)
    except Exception as exc:
        logger.warning(f'[compress] sampler error: {exc}')
        sampler_responses = [None] * len(sampler_input)

    responses = [None] * len(prompts)
    for resp, pos in zip(sampler_responses, sampler_pos):
        responses[pos] = resp

    decoded: List[str] = []
    fallback_indices: List[int] = []
    for ri in range(len(prompts)):
        pair_idx = ri // 2
        schema = meta[pair_idx]['schema']
        if prompts[ri] is None:
            decoded.append(meta[pair_idx]['query_raw'])
            continue
        resp = responses[ri]
        seq = resp.sequences[0] if resp and resp.sequences else None
        text = ''
        if seq and seq.stop_reason != 'length' and seq.decoded:
            text = seq.decoded
            for tok in special_tokens:
                text = text.replace(tok, '')
            text = text.rstrip()
        if not _is_truncated_compression(text, schema):
            decoded.append(text)
        else:
            decoded.append('')
            fallback_indices.append(ri)

    return {'prompts': prompts, 'meta': meta, 'decoded': decoded,
            'fallback_indices': fallback_indices}


def _compress_batch_phase2(
    state: Dict[str, Any],
    api_client: OpenAIClient,
) -> List[Dict[str, Any]]:
    """Phase 2 (no GPU): API fallback → build results."""
    if 'final' in state:
        return state['final']

    prompts = state['prompts']
    decoded = state['decoded']
    fallback_indices = state['fallback_indices']
    is_hard = state.get('hard', False)
    meta = state.get('meta')  # None for hard

    # Track which prompts used API fallback
    api_set: set = set()
    if fallback_indices:
        api_futures = {}
        with ThreadPoolExecutor(max_workers=API_CONCURRENCY) as pool:
            for ri in fallback_indices:
                api_futures[pool.submit(_api_compress, api_client, prompts[ri])] = ri
            for fut in as_completed(api_futures):
                ri = api_futures[fut]
                api_result = fut.result()
                schema = 'new' if is_hard else meta[ri // 2]['schema']
                if api_result and not _is_truncated_compression(api_result, schema):
                    decoded[ri] = api_result
                    api_set.add(ri)

    state['api_set'] = api_set
    if is_hard:
        return _build_hard_results(state)
    return _build_think_index_results(state)


def _build_think_index_results(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = state['meta']
    decoded = state['decoded']
    api_set = state.get('api_set', set())
    results = []
    for pair_idx in range(len(meta)):
        q_text = decoded[pair_idx * 2]
        c_text = decoded[pair_idx * 2 + 1]
        if not q_text or not c_text:
            continue
        q_method = 'api' if (pair_idx * 2) in api_set else 'vllm'
        c_method = 'api' if (pair_idx * 2 + 1) in api_set else 'vllm'
        results.append({
            'anchor_text': q_text,
            'positive_text': c_text,
            'negative_texts': [],
            'source': meta[pair_idx]['source'],
            'query_raw': meta[pair_idx]['query_raw'],
            'cot_raw': meta[pair_idx]['cot_raw'],
            'anchor_method': q_method,
            'positive_method': c_method,
        })
    return results


def _build_hard_results(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    group_sizes = state['group_sizes']
    decoded = state['decoded']
    source_type = state['source_type']
    raw_groups = state['raw_groups']
    api_set = state.get('api_set', set())
    results = []
    offset = 0
    for gi, gs in enumerate(group_sizes):
        q_text = decoded[offset]
        c_text = decoded[offset + 1]
        if not q_text or not c_text:
            offset += gs
            continue
        neg_texts = []
        neg_raws = []
        neg_methods = []
        for ni in range(2, gs):
            nt = decoded[offset + ni]
            if nt:
                neg_texts.append(nt)
                neg_raws.append(raw_groups[gi]['negs_raw'][ni - 2])
                neg_methods.append('api' if (offset + ni) in api_set else 'vllm')
        q_method = 'api' if offset in api_set else 'vllm'
        c_method = 'api' if (offset + 1) in api_set else 'vllm'
        results.append({
            'anchor_text': q_text,
            'positive_text': c_text,
            'negative_texts': neg_texts,
            'source': source_type,
            'query_raw': raw_groups[gi]['query_raw'],
            'cot_raw': raw_groups[gi]['cot_raw'],
            'negs_raw': neg_raws,
            'anchor_method': q_method,
            'positive_method': c_method,
            'neg_methods': neg_methods,
        })
        offset += gs
    return results


def _compress_hard_phase1(
    rows: List[Dict[str, Any]],
    condenser_sampler,
    compress_params: SamplingParams,
    special_tokens: set,
    source_type: str,
) -> Dict[str, Any]:
    """Phase 1 for hard rows: vLLM sample + validate. Returns state for phase 2."""
    _MAX_COT_CHARS = 30_000

    prompts: List[Dict[str, Any]] = []
    group_sizes: List[int] = []
    row_ids: List[str] = []
    raw_groups: List[Dict[str, Any]] = []

    for row in rows:
        query, cot = _extract_query_cot(row)
        if not query or not cot or len(cot) > _MAX_COT_CHARS:
            continue
        negatives = row.get('negatives') or []
        valid_negs = [n for n in negatives
                      if n and len(n) <= _MAX_COT_CHARS]

        user_q = COMPRESS_USER.format(query=EMBED_QUERY_REASONIR_Q, text=query)
        prompts.append({'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': user_q},
        ]})
        user_c = COMPRESS_USER.format(query=EMBED_QUERY_REASONIR_COT, text=cot)
        prompts.append({'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': user_c},
        ]})
        for neg in valid_negs:
            user_n = COMPRESS_USER.format(query=EMBED_QUERY_REASONIR_COT, text=neg)
            prompts.append({'messages': [
                {'role': 'system', 'content': COMPRESS_SYSTEM},
                {'role': 'user', 'content': user_n},
            ]})
        group_sizes.append(2 + len(valid_negs))
        row_ids.append(row.get('id', ''))
        raw_groups.append({'query_raw': query, 'cot_raw': cot, 'negs_raw': valid_negs})

    if not prompts:
        return {'hard': True, 'prompts': [], 'group_sizes': [], 'row_ids': [],
                'decoded': [], 'fallback_indices': [], 'source_type': source_type,
                'raw_groups': []}

    try:
        responses = condenser_sampler.sample(prompts, compress_params)
    except Exception as exc:
        logger.warning(f'[compress-hard] sampler error: {exc}')
        responses = [None] * len(prompts)

    decoded: List[str] = []
    fallback_indices: List[int] = []
    for ri, resp in enumerate(responses):
        seq = resp.sequences[0] if resp and resp.sequences else None
        text = ''
        if seq and seq.stop_reason != 'length' and seq.decoded:
            text = seq.decoded
            for tok in special_tokens:
                text = text.replace(tok, '')
            text = text.rstrip()
        if text and not _is_truncated_compression(text, 'new'):
            decoded.append(text)
        else:
            decoded.append('')
            fallback_indices.append(ri)

    return {'hard': True, 'prompts': prompts, 'group_sizes': group_sizes,
            'row_ids': row_ids, 'decoded': decoded,
            'fallback_indices': fallback_indices, 'source_type': source_type,
            'raw_groups': raw_groups}


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    device_groups = [
        DeviceGroup(name='condenser_sampler',
                    ranks=list(range(CONDENSER_GPUS)),
                    device_type='GPU'),
    ]
    condenser_mesh = DeviceMesh.from_sizes(
        world_size=CONDENSER_GPUS, dp_size=CONDENSER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=CONDENSER_GPUS, groups=device_groups)

    # -- Load raw datasets ----------------------------------------------------
    from datasets import Dataset as HFDataset

    dataset_think = get_dataset_think(total=TOTAL_SAMPLES, load_from_cache_file=True)
    if THINK_CAP and len(dataset_think.dataset) > THINK_CAP:
        dataset_think.dataset = dataset_think.dataset.select(range(THINK_CAP))
    ds_think = dataset_think.dataset
    logger.info(f'[load] think={len(ds_think)}')

    ds_index_obj = get_dataset_index(total=None, load_from_cache_file=True)
    ds_index = ds_index_obj.dataset
    if INDEX_CAP and len(ds_index) > INDEX_CAP:
        ds_index = ds_index.select(range(INDEX_CAP))
    logger.info(f'[load] index={len(ds_index)}')

    ds_hard_raw = get_dataset_hard(max_negatives=HARD_MAX_NEGATIVES, load_from_cache_file=True)
    if HARD_CAP and len(ds_hard_raw) > HARD_CAP:
        ds_hard_raw = ds_hard_raw.select(range(HARD_CAP))
    n_hard = len(ds_hard_raw)
    logger.info(f'[load] hard={n_hard}')

    # Convert hard to messages schema
    hard_rows_list = []
    if n_hard > 0:
        h_ids = ds_hard_raw['id']
        h_queries = ds_hard_raw['query']
        h_cots = ds_hard_raw['cot']
        h_responses = ds_hard_raw['response'] if 'response' in ds_hard_raw.column_names else [''] * n_hard
        h_negatives = ds_hard_raw['negatives']
        for i in range(n_hard):
            hard_rows_list.append({
                'id': h_ids[i],
                'messages': [
                    {'role': 'user', 'content': h_queries[i]},
                    {'role': 'assistant', 'reasoning_content': h_cots[i],
                     'content': h_responses[i] or ''},
                ],
                'negatives': h_negatives[i],
            })

    # Batch-convert HF Datasets to list-of-dicts
    def _ds_to_rows(ds):
        return [dict(zip(ds.column_names, vals)) for vals in zip(*(ds[c] for c in ds.column_names))]

    think_rows = _ds_to_rows(ds_think)
    index_rows = _ds_to_rows(ds_index)

    # -- Setup condenser ------------------------------------------------------
    condenser_template = Qwen3_5Template(
        model_id=CONDENSE_MODEL_ID, max_length=DATASET_MAX_TOKENS,
        enable_thinking=False, truncation_strategy='delete')
    special_tokens = set(condenser_template.tokenizer.all_special_tokens)

    condenser_sampler = vLLMSampler(
        model_id=CONDENSE_MODEL_ID,
        engine_args={'gpu_memory_utilization': 0.8, 'max_model_len': COMPRESS_MAX_MODEL_LEN},
        device_mesh=condenser_mesh,
        remote_group='condenser_sampler',
    )
    condenser_sampler.set_template(
        TEMPLATE_NAME, model_id=CONDENSE_MODEL_ID, enable_thinking=False,
        truncation_strategy='delete', max_length=DATASET_MAX_TOKENS)
    condenser_sampler._ray_get_timeout = SAMPLER_TIMEOUT
    compress_params = SamplingParams(
        max_tokens=8192, temperature=COMPRESS_TEMPERATURE,
        top_p=COMPRESS_TOP_P, num_samples=1)

    api_client = OpenAIClient(
        model=COMPRESS_MODEL, api_key=COMPRESS_API_KEY, base_url=COMPRESS_BASE_URL)

    # -- Resume support ----------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = {'think': 0, 'index': 0, 'hard': 0}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
        logger.info(f'[resume] loaded progress: {progress}')

    _results_lock = PosixFileLock(RESULTS_JSONL + '.lock')

    def _flush_results(records: List[Dict[str, Any]]):
        if not records:
            return
        lines = [json.dumps(r, ensure_ascii=False) + '\n' for r in records]
        with _results_lock:
            with open(RESULTS_JSONL, 'a', encoding='utf-8') as f:
                f.writelines(lines)

    def _save_progress():
        tmp = PROGRESS_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(progress, f)
        os.replace(tmp, PROGRESS_FILE)

    # -- Process in batches (pipelined: vLLM batch N+1 overlaps API fallback N) -
    total_flushed = 0
    if os.path.exists(RESULTS_JSONL):
        with open(RESULTS_JSONL, 'r', encoding='utf-8') as f:
            total_flushed = sum(1 for l in f if l.strip())
        if total_flushed:
            logger.info(f'[resume] {total_flushed} records already in results.jsonl')

    def _process_source(rows, source_type, label):
        nonlocal total_flushed
        n_total = len(rows)
        skip = progress.get(source_type, 0)
        if skip >= n_total:
            logger.info(f'[{label}] skipped (already done {skip}/{n_total})')
            return
        if skip > 0:
            logger.info(f'[{label}] resuming from row {skip}/{n_total}')

        bg_pool = ThreadPoolExecutor(max_workers=1)
        pending = None  # (future, batch_start, batch_len)

        def _drain_pending():
            nonlocal total_flushed, pending
            if pending is None:
                return
            fut, p_start, p_len = pending
            batch_results = fut.result()
            _flush_results(batch_results)
            total_flushed += len(batch_results)
            progress[source_type] = p_start + p_len
            _save_progress()
            pending = None

        for start in range(skip, n_total, BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            state = _compress_batch_phase1(
                batch, condenser_sampler, compress_params,
                special_tokens, source_type)
            _drain_pending()
            pending = (
                bg_pool.submit(_compress_batch_phase2, state, api_client),
                start, len(batch))
            n_done = start + len(batch)
            if n_done % (BATCH_SIZE * 10) == 0 or n_done >= n_total:
                logger.info(f'[{label}] {n_done}/{n_total} vLLM done, '
                            f'{total_flushed} records flushed (last batch pending)')

        _drain_pending()
        bg_pool.shutdown(wait=False)
        logger.info(f'[{label}] complete, {total_flushed} total records flushed')

    _process_source(hard_rows_list, 'hard', 'hard')
    _process_source(think_rows, 'think', 'think')
    _process_source(index_rows, 'index', 'index')

    # -- Convert JSONL → HF Dataset -------------------------------------------
    logger.info(f'[save] converting results.jsonl to HF Dataset...')
    all_results = []
    with open(RESULTS_JSONL, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f'[save] skipping malformed line {line_no} (truncated resume?)')
    logger.info(f'[save] total records: {len(all_results)}')
    out_ds = HFDataset.from_dict({
        'anchor_text': [r['anchor_text'] for r in all_results],
        'positive_text': [r['positive_text'] for r in all_results],
        'negative_texts': [r['negative_texts'] for r in all_results],
        'source': [r['source'] for r in all_results],
        'query_raw': [r.get('query_raw', '') for r in all_results],
        'cot_raw': [r.get('cot_raw', '') for r in all_results],
        'negs_raw': [r.get('negs_raw', []) for r in all_results],
    })
    out_ds.save_to_disk(OUTPUT_DIR + '/dataset')
    logger.info(f'[save] dataset saved to {OUTPUT_DIR}/dataset')
    logger.info(f'[stats] think={sum(1 for r in all_results if r["source"]=="think")} '
                f'index={sum(1 for r in all_results if r["source"]=="index")} '
                f'hard={sum(1 for r in all_results if r["source"]=="hard")}')


if __name__ == '__main__':
    main()
