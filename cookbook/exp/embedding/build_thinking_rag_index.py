"""Build a thinking-trace RAG index from condensed (query, cot) pairs.

Pipeline (per row, batched):
  1. Load (user_query, reasoning_content) pairs from ``dataset_think.get_dataset``.
  2. Compress query with ``FIXED_QUERY_NEED`` and cot with ``FIXED_QUERY_SKILL``
     using a Twinkle ``vLLMSampler`` (TP=4 across GPUs 0-3). Reuses the prompt
     suite from ``cookbook/exp/condenser/make_condenser_dataset.py``.
  3. On condenser truncation (``stop_reason='length'`` or skeleton-incomplete
     output), fall back to an external OpenAI-compatible API.
  4. Encode the condensed pair via the trained embedding model — Twinkle
     ``TransformersModel`` on the ``emb_model`` device group (DP=4 across GPUs
     4-7) using ``forward_only(task='embedding')``, the same code path as
     training.
  5. Compute cosine similarity for each (query, thinking) pair, drop pairs with
     ``sim < SIM_THRESHOLD``, and insert kept rows into LanceDB. The vector
     column carries the **positive (compressed-skill)** embedding so a search
     keyed by an anchor-encoded query retrieves the matching thinking trace.
  6. Each row stores the **raw thinking** alongside its embedding, so a hit
     in the index can directly surface the original CoT.

Eval mode (``--mode eval`` or ``--mode both``):
  * Self-recall test — encode a sample of dataset queries (whose corresponding
    rows are already in the index) as anchors and report recall@1/5/10 plus
    a per-source breakdown.

Architecture (8 GPUs):
  * GPU 0-3: vLLM condenser (tensor-parallel, ``DeviceGroup name='sampler'``)
  * GPU 4-7: TransformersModel embedding (data-parallel, ``DeviceGroup name='emb_model'``)
  * Single ``twinkle.initialize(mode='ray', ...)`` call wires both groups.

Launch examples:
  python build_thinking_rag_index.py --mode build --total 500000
  python build_thinking_rag_index.py --mode eval  --eval-size 1000
  python build_thinking_rag_index.py --mode both  --total 200000 --eval-size 500
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reuse condenser prompts (single source of truth)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'condenser'))
sys.path.insert(0, str(_HERE))
from make_condenser_dataset import (  # noqa: E402  (after sys.path tweak)
    COMPRESS_SYSTEM,
    COMPRESS_USER,
    FIXED_QUERY_NEED,
    FIXED_QUERY_SKILL,
)
# Default dataset loader is the index-time corpus (broader retrieval profile);
# pass --dataset-module dataset_think to fall back to the training mix.
from dataset_index import get_dataset as _default_get_dataset  # noqa: E402

_GET_DATASET = _default_get_dataset

import twinkle  # noqa: E402
from twinkle import DeviceGroup, DeviceMesh, get_logger  # noqa: E402
from twinkle.data_format import SamplingParams as TwinkleSamplingParams  # noqa: E402
from twinkle.loss import InfonceLoss  # noqa: E402
from twinkle.model import TransformersModel  # noqa: E402
from twinkle.processor import InputProcessor  # noqa: E402
from twinkle.sampler import vLLMSampler  # noqa: E402
from twinkle.template import Qwen3_5Template  # noqa: E402
from twinkle.utils.parallel import PosixFileLock  # noqa: E402
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient  # noqa: E402

logger = get_logger()


# ===========================================================================
# Config (most fields overridable via CLI / env)
# ===========================================================================

EMBED_MODEL_ID = os.environ.get(
    'EMBED_MODEL_ID',
    'ms://twinkle-kit/Qwen3.5-4B-QA-emb',
)
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')

# Twinkle device topology: TP=4 sampler on 0-3, DP=4 embedding on 4-7.
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
EMB_GPUS = int(os.environ.get('EMB_GPUS', 4))
NUM_GPUS = SAMPLER_GPUS + EMB_GPUS

# vLLM engine sizing.
CONDENSE_GPU_MEM = float(os.environ.get('CONDENSE_GPU_MEM', 0.85))
CONDENSE_MAX_MODEL_LEN = int(os.environ.get('CONDENSE_MAX_MODEL_LEN', 32768))
CONDENSE_MAX_TOKENS = int(os.environ.get('CONDENSE_MAX_TOKENS', 8192))
COMPRESS_TEMPERATURE = float(os.environ.get('COMPRESS_TEMPERATURE', 0.2))
COMPRESS_TOP_P = float(os.environ.get('COMPRESS_TOP_P', 0.5))

# Embedding sizing.
EMBED_MAX_LENGTH = int(os.environ.get('EMBED_MAX_LENGTH', 8192))

SIM_THRESHOLD = float(os.environ.get('SIM_THRESHOLD', 0.65))
MIN_TEXT_CHARS = int(os.environ.get('MIN_TEXT_CHARS', 256))

# OpenAI API fallback (used when vLLM truncates).
COMPRESS_API_KEY = os.environ.get('COMPRESS_API_KEY', '')
COMPRESS_BASE_URL = os.environ.get(
    'COMPRESS_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
COMPRESS_API_MODEL = os.environ.get('COMPRESS_API_MODEL', 'qwen3.7-max')

# Source → coarse domain (for filtered eval).
DOMAIN_MAP = {
    'CodeX-2M-Thinking': 'code',
    'OpenThoughts3-1.2M': 'reasoning',
    'LIMO-v2': 'math',
    'Chinese-DeepSeek-R1-Distill-data-110k': 'reasoning_zh',
    'Opus-4.6-Reasoning-3000x-filtered': 'reasoning',
    'claude-opus-4.6-10000x': 'mixed',
    'angrygiraffe-claude-opus-4.6-4.7-reasoning-8.7k': 'mixed',
}


# ===========================================================================
# Small helpers
# ===========================================================================

def _is_truncated_compression(text: str) -> bool:
    """Detect structurally incomplete condenser output even when stop='stop'.

    The skeleton mandates a ``## Summary`` plus a ``## More`` bullet list whose
    last line begins with ``-`` or ends with ``)`` (the ``(none)`` sentinel).
    Anything short of that signals truncation; the API fallback is invoked.
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
    return False


def _strip_outer_codefence(text: str) -> str:
    m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _wrap_anchor(text: str) -> List[Dict[str, str]]:
    """Anchor-side message wrapping (must match training)."""
    return [
        {'role': 'user', 'content': text},
        {'role': 'assistant', 'content': 'Match the correct response here.'},
    ]


def _wrap_positive(text: str) -> List[Dict[str, str]]:
    """Positive-side message wrapping (must match training)."""
    return [
        {'role': 'user', 'content': 'Match the correct query here.'},
        {'role': 'assistant', 'content': text},
    ]


def _short(text: str, n: int = 96) -> str:
    text = (text or '').replace('\n', ' ').strip()
    return text[:n] + ('…' if len(text) > n else '')


def _detect_lang(text: str) -> str:
    if not text:
        return 'unknown'
    cjk = sum(1 for ch in text[:512] if '\u4e00' <= ch <= '\u9fff')
    return 'zh' if cjk >= 8 else 'en'


def _build_compress_messages(text: str, query: str) -> List[Dict[str, str]]:
    return [
        {'role': 'system', 'content': COMPRESS_SYSTEM},
        {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
    ]


# ===========================================================================
# Twinkle component wrappers
# ===========================================================================

def initialize_twinkle() -> Tuple[DeviceMesh, DeviceMesh]:
    """Wire two device groups (sampler / emb_model) and return their meshes."""
    device_groups = [
        DeviceGroup(
            name='sampler',
            ranks=list(range(SAMPLER_GPUS)),
            device_type='GPU',
            gpus_per_worker=SAMPLER_GPUS,  # TP=4 → one worker spans all 4 GPUs
        ),
        DeviceGroup(
            name='emb_model',
            ranks=list(range(SAMPLER_GPUS, NUM_GPUS)),
            device_type='GPU',
        ),
    ]
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    emb_mesh = DeviceMesh.from_sizes(world_size=EMB_GPUS, dp_size=EMB_GPUS)
    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
        lazy_collect=False,
    )
    return sampler_mesh, emb_mesh


def build_sampler(sampler_mesh: DeviceMesh) -> vLLMSampler:
    sampler = vLLMSampler(
        model_id=CONDENSE_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': CONDENSE_GPU_MEM,
            'max_model_len': CONDENSE_MAX_MODEL_LEN,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(
        'Qwen3_5Template',
        model_id=CONDENSE_MODEL_ID,
        enable_thinking=False,
        max_length=CONDENSE_MAX_MODEL_LEN,
    )
    return sampler


def build_emb_model(emb_mesh: DeviceMesh) -> Tuple[TransformersModel, Qwen3_5Template]:
    model = TransformersModel(
        model_id=EMBED_MODEL_ID,
        device_mesh=emb_mesh,
        remote_group='emb_model',
    )
    model.set_processor(InputProcessor)
    # InfonceLoss is required by the framework even though forward_only does
    # not actually invoke it; matches the training-time configuration.
    model.set_loss(InfonceLoss, temperature=0.03, use_batch=True)
    # Qwen3.5-specific subclass applies orphan-</think> chat-template patches.
    template = Qwen3_5Template(
        model_id=EMBED_MODEL_ID,
        max_length=EMBED_MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    return model, template


# ===========================================================================
# Compression helpers (vLLMSampler) + API fallback
# ===========================================================================

def _vllm_compress(sampler: vLLMSampler, texts: List[str], query_hint: str
                   ) -> List[Tuple[str, str]]:
    """Compress ``texts`` via the sampler; return ``(decoded, stop_reason)``."""
    if not texts:
        return []
    prompts = [{'messages': _build_compress_messages(t, query_hint)} for t in texts]
    params = TwinkleSamplingParams(
        max_tokens=CONDENSE_MAX_TOKENS,
        temperature=COMPRESS_TEMPERATURE,
        top_p=COMPRESS_TOP_P,
        num_samples=1,
    )
    responses = sampler.sample(prompts, params)
    results: List[Tuple[str, str]] = []
    for resp in responses:
        seq = resp.sequences[0] if resp and resp.sequences else None
        if seq is None:
            results.append(('', 'error'))
            continue
        text = seq.decoded or ''
        # Strip any leaked chat-template special tokens like ``<|im_end|>``.
        text = re.sub(r'<\|[^|]+\|>', '', text).rstrip()
        text = _strip_outer_codefence(text)
        results.append((text, seq.stop_reason or 'stop'))
    return results


def _api_compress(api: OpenAIClient, messages: List[Dict[str, str]]) -> Optional[str]:
    sp = TwinkleSamplingParams(temperature=COMPRESS_TEMPERATURE, max_tokens=CONDENSE_MAX_TOKENS)
    try:
        reply = api({'messages': messages}, sp, extra_body={'enable_thinking': False})
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        sys.stderr.write(f'[api_fallback] error: {exc}\n')
        return None
    content = (reply.get('content') or '').strip()
    if not content:
        return None
    return _strip_outer_codefence(content)


def _resolve_compressed(sampler: vLLMSampler, api: Optional[OpenAIClient],
                        texts: List[str], query_hint: str) -> List[Optional[str]]:
    """Run vLLM batch; replace truncations / skeleton-incomplete with API output."""
    pairs = _vllm_compress(sampler, texts, query_hint)
    results: List[Optional[str]] = []
    for (text, stop), src_text in zip(pairs, texts):
        if stop != 'length' and not _is_truncated_compression(text):
            results.append(text)
            continue
        if api is None:
            results.append(None)
            continue
        api_text = _api_compress(api, _build_compress_messages(src_text, query_hint))
        if api_text is None or _is_truncated_compression(api_text):
            results.append(None)
        else:
            results.append(api_text)
    return results


# ===========================================================================
# Embedding helpers (TransformersModel.forward_only(task='embedding'))
# ===========================================================================

def _build_features(template: Qwen3_5Template, texts: List[str], role: str
                    ) -> List[Dict[str, Any]]:
    """Wrap each text into the role-specific anchor / positive feature dict."""
    features: List[Dict[str, Any]] = []
    for text in texts:
        if not text or not text.strip():
            # Pad with a single space so positional alignment holds against
            # the input list — the caller filters out empty-text rows upstream.
            text = ' '
        if role == 'anchor':
            feat = template.encode({'messages': _wrap_anchor(text)})
            feat['labels'] = [1]
        else:
            feat = template.encode({'messages': _wrap_positive(text)})
            feat['labels'] = [0]
        features.append(feat)
    return features


def get_embeddings(model: TransformersModel, template: Qwen3_5Template,
                   texts: List[str], role: str) -> np.ndarray:
    """Return ``[N, H]`` float32 L2-normalised embeddings for ``texts``.

    Inputs are padded up to a multiple of ``EMB_GPUS`` and sliced back to the
    original ``N``: the dispatch layer (``_dispatch_args``) starves any rank
    whose chunk lands beyond ``len(texts)``, so a single forward of fewer than
    ``EMB_GPUS`` items (e.g. the probe) would otherwise raise
    ``Batch too small for {EMB_GPUS} workers``.
    """
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    n = len(texts)
    pad_n = (-n) % EMB_GPUS
    padded = list(texts) + [' '] * pad_n if pad_n else list(texts)
    features = _build_features(template, padded, role)
    out = model.forward_only(inputs=features, task='embedding', return_logits=True)
    emb = out['embeddings']
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().to(torch.float32).cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32)
    return emb[:n] if pad_n else emb


def _probe_hidden_size(model: TransformersModel, template: Qwen3_5Template) -> int:
    """One-shot warmup forward to read out the embedding dimension."""
    emb = get_embeddings(model, template, ['probe'], role='anchor')
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise RuntimeError(f'unexpected embedding shape from probe: {emb.shape}')
    return int(emb.shape[1])


# ===========================================================================
# LanceDB I/O
# ===========================================================================

def _make_arrow_schema(hidden_size: int):
    import pyarrow as pa
    return pa.schema([
        pa.field('id', pa.string()),
        pa.field('vector', pa.list_(pa.float32(), hidden_size)),
        pa.field('thinking_raw', pa.string()),
        pa.field('query_raw', pa.string()),
        pa.field('cot_compressed', pa.string()),
        pa.field('query_compressed', pa.string()),
        pa.field('source', pa.string()),
        pa.field('domain', pa.string()),
        pa.field('language', pa.string()),
        pa.field('sim', pa.float32()),
    ])


def _open_or_create_table(db_path: str, table_name: str, hidden_size: int,
                          mode: str):
    """Open an existing table for append/eval, or create a fresh one."""
    import lancedb
    db = lancedb.connect(db_path)
    schema = _make_arrow_schema(hidden_size)
    if table_name in db.table_names():
        if mode == 'overwrite':
            db.drop_table(table_name)
            tbl = db.create_table(table_name, schema=schema, mode='overwrite')
        else:
            tbl = db.open_table(table_name)
    else:
        tbl = db.create_table(table_name, schema=schema, mode='create')
    return db, tbl


def _existing_ids(table) -> set:
    try:
        col = table.to_pandas(columns=['id'])
        return set(col['id'].astype(str).tolist())
    except Exception:  # noqa: BLE001
        return set()


# ===========================================================================
# Build pipeline
# ===========================================================================

def _stream_corpus(total: Optional[int], load_from_cache_file: bool,
                   max_rows: int = 0) -> Iterator[Dict[str, Any]]:
    ds = _GET_DATASET(total=total, load_from_cache_file=load_from_cache_file)
    n_full = len(ds)
    cap = max_rows if (max_rows and max_rows < n_full) else n_full
    sys.stderr.write(f'[corpus] get_dataset: {n_full} rows'
                     + (f' → yielding first {cap}\n' if cap < n_full else '\n'))
    for i, row in enumerate(ds):
        if i >= cap:
            break
        yield row


def _extract_query_cot(row: Dict[str, Any]) -> Tuple[str, str]:
    user_query, cot = '', ''
    for m in row.get('messages') or []:
        if not isinstance(m, dict):
            continue
        role = m.get('role') or ''
        if role == 'user' and not user_query:
            user_query = (m.get('content') or '').strip()
        elif role == 'assistant':
            cot = (m.get('reasoning_content') or '').strip()
            break
    return user_query, cot


def _log_miss(misses_path: str, lock: PosixFileLock, record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str) + '\n'
    with lock:
        with open(misses_path, 'a', encoding='utf-8') as fh:
            fh.write(line)


def build_index(args: argparse.Namespace,
                sampler: vLLMSampler,
                emb_model: TransformersModel,
                emb_template: Qwen3_5Template,
                api: Optional[OpenAIClient]) -> None:
    # ---- Probe embedding dimension -----------------------------------------
    sys.stderr.write('[build] probing embedding hidden size...\n')
    hidden_size = _probe_hidden_size(emb_model, emb_template)
    sys.stderr.write(f'[build] hidden_size={hidden_size}\n')

    # ---- LanceDB ------------------------------------------------------------
    db, tbl = _open_or_create_table(
        args.db_path, args.table, hidden_size,
        mode='overwrite' if args.overwrite else 'append',
    )
    indexed = _existing_ids(tbl) if not args.overwrite else set()
    sys.stderr.write(f'[build] table "{args.table}" — {len(indexed)} existing rows.\n')

    misses_path = args.misses_log or (str(Path(args.db_path) / f'{args.table}.misses.jsonl'))
    Path(misses_path).parent.mkdir(parents=True, exist_ok=True)
    misses_lock = PosixFileLock(misses_path + '.lock')

    # ---- Streaming loop -----------------------------------------------------
    n_seen = n_kept = n_dropped_short = n_dropped_compress = n_dropped_sim = 0
    n_dropped_dup = 0
    pbar = tqdm(desc='index', unit='row', dynamic_ncols=True)

    batch: List[Dict[str, Any]] = []

    def _flush(rows: List[Dict[str, Any]]) -> None:
        nonlocal n_kept, n_dropped_compress, n_dropped_sim
        if not rows:
            return
        # Phase 1 — compress query (FIXED_QUERY_NEED) and cot (FIXED_QUERY_SKILL).
        q_compressed = _resolve_compressed(
            sampler, api, [r['query_raw'] for r in rows], FIXED_QUERY_NEED)
        c_compressed = _resolve_compressed(
            sampler, api, [r['cot_raw'] for r in rows], FIXED_QUERY_SKILL)
        kept_rows: List[Dict[str, Any]] = []
        for r, q_cmp, c_cmp in zip(rows, q_compressed, c_compressed):
            if not q_cmp or not c_cmp:
                n_dropped_compress += 1
                _log_miss(misses_path, misses_lock, {
                    'id': r['id'], 'source': r['source'], 'reason': 'compress_fail',
                    'query_raw_head': _short(r['query_raw'], 200),
                    'cot_raw_head': _short(r['cot_raw'], 200),
                })
                continue
            r['query_compressed'] = q_cmp
            r['cot_compressed'] = c_cmp
            kept_rows.append(r)
        if not kept_rows:
            return
        # Phase 2 — encode anchor (compressed query) + positive (compressed cot).
        anchor_emb = get_embeddings(
            emb_model, emb_template, [r['query_compressed'] for r in kept_rows], role='anchor')
        positive_emb = get_embeddings(
            emb_model, emb_template, [r['cot_compressed'] for r in kept_rows], role='positive')
        sims = (anchor_emb * positive_emb).sum(axis=1).astype(np.float32)
        # Phase 3 — sim filter + LanceDB insert.
        to_insert: List[Dict[str, Any]] = []
        for idx, (r, sim_val) in enumerate(zip(kept_rows, sims)):
            tag = 'KEEP' if sim_val >= SIM_THRESHOLD else 'DROP'
            print(f'[{tag} sim={sim_val:.4f}] {r["source"][:24]} '
                  f'q={_short(r["query_raw"], 60)!r} '
                  f'cot={_short(r["cot_raw"], 60)!r}', flush=True)
            if sim_val < SIM_THRESHOLD:
                n_dropped_sim += 1
                _log_miss(misses_path, misses_lock, {
                    'id': r['id'], 'source': r['source'], 'reason': 'sim_low',
                    'sim': float(sim_val),
                    'query_raw': r['query_raw'],
                    'cot_raw': r['cot_raw'],
                    'query_compressed': r['query_compressed'],
                    'cot_compressed': r['cot_compressed'],
                })
                continue
            to_insert.append({
                'id': r['id'],
                'vector': positive_emb[idx].tolist(),
                'thinking_raw': r['cot_raw'],
                'query_raw': r['query_raw'],
                'cot_compressed': r['cot_compressed'],
                'query_compressed': r['query_compressed'],
                'source': r['source'],
                'domain': DOMAIN_MAP.get(r['source'], 'mixed'),
                'language': _detect_lang(r['cot_raw']),
                'sim': float(sim_val),
            })
        if to_insert:
            tbl.add(to_insert)
            n_kept += len(to_insert)
            indexed.update(r['id'] for r in to_insert)

    try:
        for row in _stream_corpus(total=args.total, load_from_cache_file=not args.no_cache,
                                  max_rows=args.max_rows):
            n_seen += 1
            if args.limit and n_kept >= args.limit:
                break
            rid = row.get('id') or ''
            if not rid:
                continue
            if rid in indexed:
                n_dropped_dup += 1
                continue
            user_query, cot = _extract_query_cot(row)
            if not user_query or len(cot) < MIN_TEXT_CHARS:
                n_dropped_short += 1
                continue
            batch.append({
                'id': rid,
                'source': row.get('source') or 'unknown',
                'query_raw': user_query,
                'cot_raw': cot,
            })
            if len(batch) >= args.batch_size:
                _flush(batch)
                batch.clear()
                pbar.set_postfix(kept=n_kept, sim_drop=n_dropped_sim,
                                 cmp_drop=n_dropped_compress, refresh=False)
            pbar.update(1)
        if batch:
            _flush(batch)
            batch.clear()
    finally:
        pbar.close()

    sys.stderr.write(
        f'[build] seen={n_seen} kept={n_kept} sim_drop={n_dropped_sim} '
        f'cmp_drop={n_dropped_compress} short_drop={n_dropped_short} '
        f'dup_skip={n_dropped_dup}\n')

    # ---- Build vector index for fast retrieval ------------------------------
    if n_kept >= 64 and not args.skip_index:
        sys.stderr.write('[build] creating IVF_PQ index (metric=dot)...\n')
        n_partitions = max(8, min(256, n_kept // 1000 + 1))
        try:
            tbl.create_index(
                metric='dot',
                vector_column_name='vector',
                num_partitions=n_partitions,
                num_sub_vectors=16,
                index_type='IVF_PQ',
                replace=True,
            )
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f'[build] index build failed: {exc} '
                             '(table is still queryable via brute-force scan)\n')
    sys.stderr.write(f'[build] done. table rows={tbl.count_rows()}\n')


# ===========================================================================
# Eval pipeline (self-recall on indexed rows)
# ===========================================================================

def eval_recall(args: argparse.Namespace,
                sampler: vLLMSampler,
                emb_model: TransformersModel,
                emb_template: Qwen3_5Template,
                api: Optional[OpenAIClient]) -> None:
    """Probe each gold query against the index; report recall@k.

    Self-recall semantics: only rows whose ``id`` is already present in the
    index are probed. The corresponding ``cot``-keyed vector must be retrieved
    by encoding the **raw user query** through the condenser → embedder
    pipeline (anchor side). The match is correct iff the retrieved row's
    ``id`` equals the probe row's ``id``.
    """
    import lancedb
    db = lancedb.connect(args.db_path)
    if args.table not in db.table_names():
        raise SystemExit(f'[eval] table "{args.table}" does not exist in {args.db_path}')
    tbl = db.open_table(args.table)
    indexed_ids = _existing_ids(tbl)
    sys.stderr.write(f'[eval] table rows={tbl.count_rows()} indexed_ids={len(indexed_ids)}\n')
    if not indexed_ids:
        sys.stderr.write('[eval] empty index — nothing to evaluate.\n')
        return

    ks = sorted({1, 5, 10, args.top_k})
    hits = {k: 0 for k in ks}
    per_source_hits: Dict[str, Dict[int, int]] = {}
    per_source_total: Dict[str, int] = {}
    probed = 0

    pbar = tqdm(desc='eval', unit='probe', dynamic_ncols=True)
    batch_rows: List[Dict[str, Any]] = []

    def _flush(rows: List[Dict[str, Any]]) -> None:
        nonlocal probed
        if not rows:
            return
        compressed = _resolve_compressed(
            sampler, api, [r['query_raw'] for r in rows], FIXED_QUERY_NEED)
        useful = [(r, c) for r, c in zip(rows, compressed) if c]
        if not useful:
            return
        anchor_emb = get_embeddings(
            emb_model, emb_template, [c for _, c in useful], role='anchor')
        for (r, _), vec in zip(useful, anchor_emb):
            res = (
                tbl.search(vec.astype(np.float32).tolist())
                .metric('dot')
                .limit(max(ks))
                .select(['id', 'source'])
                .to_list()
            )
            hit_ids = [item['id'] for item in res]
            try:
                rank = hit_ids.index(r['id'])
            except ValueError:
                rank = -1
            for k in ks:
                if 0 <= rank < k:
                    hits[k] += 1
                    per_source_hits.setdefault(r['source'], {kk: 0 for kk in ks})[k] += 1
            per_source_total[r['source']] = per_source_total.get(r['source'], 0) + 1
            per_source_hits.setdefault(r['source'], {kk: 0 for kk in ks})
            probed += 1
        pbar.update(len(useful))

    try:
        for row in _stream_corpus(total=args.total, load_from_cache_file=not args.no_cache,
                                  max_rows=args.max_rows):
            if probed + len(batch_rows) >= args.eval_size:
                break
            rid = row.get('id') or ''
            if not rid or rid not in indexed_ids:
                continue
            user_query, _ = _extract_query_cot(row)
            if not user_query or len(user_query) < MIN_TEXT_CHARS:
                continue
            batch_rows.append({
                'id': rid,
                'source': row.get('source') or 'unknown',
                'query_raw': user_query,
            })
            if len(batch_rows) >= args.batch_size:
                _flush(batch_rows)
                batch_rows.clear()
        if batch_rows:
            _flush(batch_rows)
    finally:
        pbar.close()

    if probed == 0:
        sys.stderr.write(
            '[eval] no probed rows — index empty, queries too short, or '
            'corpus exhausted before eval-size?\n')
        return

    print('\n=== Recall @ k (self-recall, gold present in index) ===')
    print(f'probed = {probed}')
    for k in ks:
        print(f'  recall@{k:<3} = {hits[k]/probed:.4f}  ({hits[k]}/{probed})')

    print('\n=== Per-source recall@10 ===')
    for src in sorted(per_source_total):
        tot = per_source_total[src]
        h10 = per_source_hits.get(src, {}).get(10, 0)
        print(f'  {src:<48s} {h10/tot:.4f}  ({h10}/{tot})')


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['build', 'eval', 'both'], default='build')
    p.add_argument('--db-path', default='./output/thinking_rag/lance.db',
                   help='LanceDB on-disk directory (persisted across runs).')
    p.add_argument('--table', default='thinking_traces',
                   help='LanceDB table name within --db-path.')
    p.add_argument('--total', type=int, default=0,
                   help='Total dataset rows to scale corpus to (0 = base sizes from the loader module).')
    p.add_argument('--dataset-module', default='dataset_index',
                   choices=['dataset_index', 'dataset_think'],
                   help='Which loader to use: dataset_index (RAG profile) or '
                        'dataset_think (training mix).')
    p.add_argument('--limit', type=int, default=0,
                   help='Stop building once this many rows are kept (0 = no cap).')
    p.add_argument('--max-rows', type=int, default=0,
                   help='Truncate corpus to this many rows AFTER get_dataset (0 = no cap). '
                        'Use this instead of --total to avoid invalidating the dataset cache.')
    p.add_argument('--batch-size', type=int, default=64,
                   help='Rows per condense+encode batch.')
    p.add_argument('--no-cache', action='store_true',
                   help='Disable load_from_cache_file in dataset_think.get_dataset.')
    p.add_argument('--overwrite', action='store_true',
                   help='Drop the table before build and start fresh.')
    p.add_argument('--skip-index', action='store_true',
                   help='Skip IVF_PQ index build at the end (debug).')
    p.add_argument('--misses-log', default='',
                   help='Path for filtered-row JSONL log (defaults to <db-path>/<table>.misses.jsonl).')

    # eval-only
    p.add_argument('--eval-size', type=int, default=500,
                   help='Number of probes for self-recall evaluation.')
    p.add_argument('--top-k', type=int, default=10,
                   help='Largest k to report. Smaller ks (1, 5) are always reported.')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.db_path).mkdir(parents=True, exist_ok=True)

    global _GET_DATASET
    if args.dataset_module == 'dataset_think':
        from dataset_think import get_dataset as _swap
        _GET_DATASET = _swap
    sys.stderr.write(f'[main] dataset loader: {args.dataset_module}\n')

    # Build/eval both depend on the same Twinkle stack — initialize once.
    sampler_mesh, emb_mesh = initialize_twinkle()
    sys.stderr.write(f'[main] twinkle initialized: '
                     f'sampler ranks 0-{SAMPLER_GPUS - 1} (TP={SAMPLER_GPUS}), '
                     f'emb_model ranks {SAMPLER_GPUS}-{NUM_GPUS - 1} (DP={EMB_GPUS}).\n')

    sys.stderr.write('[main] starting vLLM condenser sampler...\n')
    sampler = build_sampler(sampler_mesh)
    sys.stderr.write('[main] starting embedding TransformersModel...\n')
    emb_model, emb_template = build_emb_model(emb_mesh)

    api: Optional[OpenAIClient] = None
    if COMPRESS_API_KEY:
        api = OpenAIClient(
            model=COMPRESS_API_MODEL,
            api_key=COMPRESS_API_KEY,
            base_url=COMPRESS_BASE_URL,
        )
    else:
        sys.stderr.write(
            '[main] WARNING: COMPRESS_API_KEY unset — truncated rows will be dropped.\n')

    if args.mode in ('build', 'both'):
        build_index(args, sampler, emb_model, emb_template, api)
    if args.mode in ('eval', 'both'):
        eval_recall(args, sampler, emb_model, emb_template, api)


if __name__ == '__main__':
    main()
