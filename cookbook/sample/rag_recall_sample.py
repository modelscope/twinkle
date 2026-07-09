"""RAG recall test: compress a query via condenser → embed → search LanceDB.

End-to-end validation that the thinking-trace RAG index built by
``cookbook/exp/embedding/build_thinking_rag_index.py`` is retrievable.

Architecture (8 GPUs, same as build script):
  * GPU 0-3: vLLM condenser (TP=4)
  * GPU 4-7: TransformersModel embedding (DP=4)

Launch:
    python cookbook/sample/rag_recall_sample.py
    python cookbook/sample/rag_recall_sample.py --query "How to implement binary search?"
    python cookbook/sample/rag_recall_sample.py --db-path ./output/thinking_rag/lance.db --top-k 5
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.loss import InfonceLoss
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template

logger = get_logger()

# ---------------------------------------------------------------------------
# Config (mirrors build_thinking_rag_index.py)
# ---------------------------------------------------------------------------
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
EMBED_MODEL_ID = os.environ.get(
    'EMBED_MODEL_ID', 'output/embedding_lora_transformers/last-checkpoint')
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
EMB_GPUS = int(os.environ.get('EMB_GPUS', 4))
NUM_GPUS = SAMPLER_GPUS + EMB_GPUS

CONDENSE_GPU_MEM = float(os.environ.get('CONDENSE_GPU_MEM', 0.85))
CONDENSE_MAX_MODEL_LEN = int(os.environ.get('CONDENSE_MAX_MODEL_LEN', 32768))
CONDENSE_MAX_TOKENS = int(os.environ.get('CONDENSE_MAX_TOKENS', 8192))
COMPRESS_TEMPERATURE = float(os.environ.get('COMPRESS_TEMPERATURE', 0.2))
COMPRESS_TOP_P = float(os.environ.get('COMPRESS_TOP_P', 0.5))
EMBED_MAX_LENGTH = int(os.environ.get('EMBED_MAX_LENGTH', 8192))
MIN_TEXT_CHARS = int(os.environ.get('MIN_TEXT_CHARS', 256))

# ---------------------------------------------------------------------------
# Compress prompts — MUST match build_thinking_rag_index.py exactly.
# ---------------------------------------------------------------------------
COMPRESS_SYSTEM = """\
You are a compression and summary assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Summary` directly, then calls `extract_compressed` \
on any topic-key listed under `## More` to recover its \
fuller content.

  `## Summary`               \u2014 extreme-density text the reader reads directly.
  `## More` \u2014 a topic index whose keys are valid arguments \
to `extract_compressed` for recovering material not captured inline.

Together the two sections must form a COMPLETE, NON-DISTORTING inventory of the \
source for the query \u2014 nothing essential lost, nothing implied that the source \
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
- Interface / signature \u2192 code notation directly: `func(a:int)->str`
- Factual / entity \u2192 telegraphic prose; drop function words; \":\" for \"is\", \",\" \
for \"has\"
- Skill / how-to / usage \u2192 lead with `Use when: <trigger>`; numbered telegraphic \
steps `1.do X 2.then Y`; close with `Output: <result>` when relevant
- Procedural \u2192 numbered short steps
- Analytical / design \u2192 hierarchical bullets with abbreviations

`## Summary` rules:
1. TOPIC LINE \u2014 line 1 is ALWAYS `Topic: <subject \u2014 scope>`, even when the \
query is narrow. Anchors both the reader and the tool.
2. DENSITY \u2014 every token in the body carries query-relevant signal; cut filler.
3. PRIMARY-COMPLETE \u2014 never silently drop a fact essential to answering the \
query. Anything cut for length MUST appear as a key under \
`## More`.
4. NON-MISLEADING \u2014 phrasing must not let the reader infer anything the source \
does not support; partial truths that mislead are worse than honest omissions \
flagged in the index.
5. SELF-CONTAINED \u2014 the reader can act on the answer without re-opening the source.
6. FAITHFUL \u2014 only content the source supports; no fabrication, no extrapolation.
7. LANGUAGE \u2014 match the source language.
8. NO outer code fences around the whole answer; no meta-commentary.

`## More` rules (MANDATORY \u2014 this section is never omitted):
1. FORMAT \u2014 each bullet is `- <topic-key>: <one-line hint>`:
   \u2022 topic-key \u2014 short, unambiguous, grounded in source vocabulary so the \
`extract_compressed` tool can locate the aspect (e.g. `decorators`, \
`error handling`, `pitfalls`).
   \u2022 hint \u2014 tells WHAT the reader gains by expanding (concrete numbers, code \
listings, secondary cases, edge details, related context, \u2026); do NOT restate \
the inline answer.
2. CRITERION \u2014 each bullet names an aspect that EXISTS in the source but is \
NOT fully captured inline. Material that genuinely fits inline without \
distortion MUST NOT be duplicated here.
3. FAITHFUL \u2014 hints must be grounded in the source; never speculate or invent.
4. ORDER \u2014 by relevance to the query, then by importance.
5. EMPTY CASE \u2014 if the source is so short / single-purpose that everything \
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
    'same language; English passage \u2192 English output, Chinese passage \u2192 '
    'Chinese output, Japanese passage \u2192 Japanese output. NEVER translate, '
    'NEVER mix languages, NEVER copy these instructions into the output.\n\n'
    '## Query (ordering hint only \u2014 still summarize the whole passage)\n{query}\n\n'
    '## Passage\n{text}')

RAG_QUERY_HINT = (
    'Summarize this query for retrieval. '
    'The body of ## Summary MUST follow this EXACT 4-line template \u2014 '
    'do NOT emit "Use when:", numbered procedure steps, or "Output:":\n'
    'Topic: <specific pattern name \u2014 scope>\n'
    'Problem: <what concrete problem is being asked>\n'
    'Skill: <which specific method/technique/pattern is required to solve it>\n'
    'Knowledge: <which domains/concepts/facts must be invoked>\n'
    'Then emit the mandatory ## More section as usual. '
    'Topic must name the specific pattern, never generic labels.')

# ---------------------------------------------------------------------------
# Demo queries (diverse domains to exercise retrieval)
# ---------------------------------------------------------------------------
DEMO_QUERIES = [
    'How can I implement binary search in Python and what are the edge cases?',
    'Explain the Free-Energy Principle in neuroscience and how it relates to active inference.',
    '如何用动态规划解决最长公共子序列问题？',
    'What is the optimal turbulence model for simulating airflow around a building?',
    '请详细解释快速排序的分治策略及其时间复杂度分析',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_outer_codefence(text: str) -> str:
    m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _short(text: str, n: int = 120) -> str:
    text = (text or '').replace('\n', ' ').strip()
    return text[:n] + ('\u2026' if len(text) > n else '')


def _build_compress_messages(text: str, query: str) -> List[Dict[str, str]]:
    return [
        {'role': 'system', 'content': COMPRESS_SYSTEM},
        {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
    ]


def _wrap_anchor(text: str) -> List[Dict[str, str]]:
    return [
        {'role': 'user', 'content': text},
        {'role': 'assistant', 'content': 'Match the correct response here.'},
    ]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def compress_query(sampler: vLLMSampler, query: str) -> str:
    """Compress a query using the condenser; short queries pass through."""
    if len(query) < MIN_TEXT_CHARS:
        return query
    prompts = [{'messages': _build_compress_messages(query, RAG_QUERY_HINT)}]
    params = SamplingParams(
        max_tokens=CONDENSE_MAX_TOKENS,
        temperature=COMPRESS_TEMPERATURE,
        top_p=COMPRESS_TOP_P,
        num_samples=1,
    )
    responses = sampler.sample(prompts, params)
    seq = responses[0].sequences[0] if responses and responses[0].sequences else None
    if seq is None:
        return query
    text = seq.decoded or ''
    text = re.sub(r'<\|[^|]+\|>', '', text).rstrip()
    text = _strip_outer_codefence(text)
    return text if text.strip() else query


def embed_query(model: TransformersModel, template: Qwen3_5Template,
                text: str) -> np.ndarray:
    """Encode a single text as an anchor embedding, returns [H] float32."""
    feat = template.encode({'messages': _wrap_anchor(text)})
    feat['labels'] = [1]
    # Pad to EMB_GPUS to avoid dispatch starvation.
    pad_n = EMB_GPUS - 1
    pad_feat = template.encode({'messages': _wrap_anchor(' ')})
    pad_feat['labels'] = [1]
    features = [feat] + [pad_feat] * pad_n
    out = model.forward_only(inputs=features, task='embedding', return_logits=True)
    emb = out['embeddings']
    if hasattr(emb, 'detach'):
        emb = emb.detach().cpu().numpy()
    return np.asarray(emb[0], dtype=np.float32)


def search_lancedb(db_path: str, table_name: str, vector: np.ndarray,
                   top_k: int) -> List[Dict[str, Any]]:
    """Search LanceDB table and return top-k results."""
    import lancedb
    db = lancedb.connect(db_path)
    available = db.list_tables()
    table_list = available.tables if hasattr(available, 'tables') else list(available)
    if table_name not in table_list:
        raise SystemExit(f'Table "{table_name}" not found in {db_path}. '
                         f'Available: {table_list}')
    tbl = db.open_table(table_name)
    results = (
        tbl.search(vector.tolist())
        .metric('dot')
        .limit(top_k)
        .select(['id', 'source', 'query_raw', 'thinking_raw',
                 'query_compressed', 'cot_compressed', 'sim', '_distance'])
        .to_list()
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--query', type=str, nargs='*', default=None,
                   help='Custom queries to test (overrides built-in demos).')
    p.add_argument('--db-path', default='./output/thinking_rag/lance.db',
                   help='LanceDB directory (same as build script).')
    p.add_argument('--table', default='thinking_traces',
                   help='LanceDB table name.')
    p.add_argument('--top-k', type=int, default=3,
                   help='Number of results to retrieve per query.')
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.db_path).exists():
        raise SystemExit(f'DB path does not exist: {args.db_path}\n'
                         f'Run build_thinking_rag_index.py first.')

    queries = args.query if args.query else DEMO_QUERIES

    # ── 1. Initialize Twinkle ───────────────────────────────────────────
    device_groups = [
        DeviceGroup(
            name='sampler',
            ranks=list(range(SAMPLER_GPUS)),
            device_type='GPU',
            gpus_per_worker=SAMPLER_GPUS,
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
        mode='ray', nproc_per_node=NUM_GPUS,
        groups=device_groups, lazy_collect=False)

    # ── 2. vLLM condenser ───────────────────────────────────────────────
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
        'Qwen3_5Template', model_id=CONDENSE_MODEL_ID,
        enable_thinking=False, max_length=CONDENSE_MAX_MODEL_LEN)

    # ── 3. Embedding model ──────────────────────────────────────────────
    emb_model = TransformersModel(
        model_id=EMBED_MODEL_ID,
        device_mesh=emb_mesh,
        remote_group='emb_model',
    )
    emb_model.set_processor(InputProcessor)
    emb_model.set_loss(InfonceLoss, temperature=0.03, use_batch=True)
    emb_template = Qwen3_5Template(
        model_id=EMBED_MODEL_ID,
        max_length=EMBED_MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )

    logger.info(f'Initialized: sampler GPUs 0-{SAMPLER_GPUS-1}, '
                f'emb GPUs {SAMPLER_GPUS}-{NUM_GPUS-1}')
    logger.info(f'DB: {args.db_path} / table: {args.table}')
    logger.info(f'Queries to test: {len(queries)}')

    # ── 4. Per-query: compress → embed → search ─────────────────────────
    for i, raw_query in enumerate(queries):
        print(f'\n{"="*80}')
        print(f'[Query {i+1}/{len(queries)}]')
        print(f'  Raw: {_short(raw_query, 200)}')

        # Compress
        compressed = compress_query(sampler, raw_query)
        is_passthrough = len(raw_query) < MIN_TEXT_CHARS
        if is_passthrough:
            print(f'  Compressed: (passthrough, len={len(raw_query)} < {MIN_TEXT_CHARS})')
        else:
            print(f'  Compressed ({len(raw_query)}\u2192{len(compressed)} chars):')
            for line in compressed.split('\n')[:8]:
                print(f'    {line}')
            if compressed.count('\n') > 8:
                print(f'    ... ({compressed.count(chr(10))+1} lines total)')

        # Embed
        vec = embed_query(emb_model, emb_template, compressed)
        print(f'  Embedding: shape={vec.shape}, norm={np.linalg.norm(vec):.4f}')

        # Search
        results = search_lancedb(args.db_path, args.table, vec, args.top_k)
        print(f'\n  Top-{args.top_k} Results:')
        if not results:
            print('    (no results)')
            continue
        for rank, r in enumerate(results, 1):
            dist = r.get('_distance', None)
            sim = (1.0 - dist) if isinstance(dist, (int, float)) else None
            sim_str = f'{sim:.4f}' if sim is not None else '?'
            dist_str = f'{dist:.4f}' if isinstance(dist, (int, float)) else '?'
            print(f'  [{rank}] cos_sim={sim_str} (dist={dist_str})  source={r["source"]}')
            print(f'      query: {_short(r["query_raw"], 100)}')
            print(f'      thinking: {_short(r["thinking_raw"], 150)}')
            print()

    print(f'\n{"="*80}')
    print('RAG recall test complete.')


if __name__ == '__main__':
    main()
