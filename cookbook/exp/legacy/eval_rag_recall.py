"""Self-recall evaluation: sample rows from LanceDB, re-encode query, check retrieval.

Unlike the full build pipeline (which needs 8 GPUs for condenser + embedding),
this script only needs the embedding model (4 GPUs) since it uses the
already-compressed ``query_compressed`` stored in the index.

Launch:
    python cookbook/exp/embedding/eval_rag_recall.py
    python cookbook/exp/embedding/eval_rag_recall.py --n 200 --top-k 20
    python cookbook/exp/embedding/eval_rag_recall.py --db-path ./output/thinking_rag/lance.db
"""
import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.loss import InfonceLoss
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import Qwen3_5Template

logger = get_logger()

EMBED_MODEL_ID = os.environ.get(
    'EMBED_MODEL_ID', 'output/embedding_full_transformers/last-checkpoint')
EMB_GPUS = int(os.environ.get('EMB_GPUS', 4))
EMBED_MAX_LENGTH = int(os.environ.get('EMBED_MAX_LENGTH', 8192))


def _wrap_anchor(text: str) -> List[Dict[str, str]]:
    return [
        {'role': 'user', 'content': text},
        {'role': 'assistant', 'content': 'Match the correct response here.'},
    ]


def get_embeddings(model: TransformersModel, template: Qwen3_5Template,
                   texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    n = len(texts)
    pad_n = (-n) % EMB_GPUS
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db-path', default='./output/thinking_rag/lance.db')
    p.add_argument('--table', default='thinking_traces')
    p.add_argument('--n', type=int, default=100, help='Number of samples to probe.')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output', default='./output/thinking_rag/recall_debug.jsonl',
                   help='JSONL file to dump per-sample debug info.')
    args = p.parse_args()

    import lancedb
    db = lancedb.connect(args.db_path)
    if args.table not in db.table_names():
        raise SystemExit(f'Table "{args.table}" not found in {args.db_path}')
    tbl = db.open_table(args.table)
    total_rows = tbl.count_rows()
    sys.stderr.write(f'[eval] table={args.table} rows={total_rows}\n')

    df = tbl.to_pandas()
    n_sample = min(args.n, len(df))
    random.seed(args.seed)
    sample_indices = random.sample(range(len(df)), n_sample)
    samples = df.iloc[sample_indices].reset_index(drop=True)
    sys.stderr.write(f'[eval] sampled {n_sample} rows for self-recall test\n')

    # Init embedding model only (no condenser needed).
    device_groups = [
        DeviceGroup(name='emb_model', ranks=list(range(EMB_GPUS)), device_type='GPU'),
    ]
    emb_mesh = DeviceMesh.from_sizes(world_size=EMB_GPUS, dp_size=EMB_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=EMB_GPUS, groups=device_groups,
                       lazy_collect=False)

    model = TransformersModel(model_id=EMBED_MODEL_ID, device_mesh=emb_mesh,
                              remote_group='emb_model')
    model.set_processor(InputProcessor)
    model.set_loss(InfonceLoss, temperature=0.03, use_batch=True)
    template = Qwen3_5Template(model_id=EMBED_MODEL_ID, max_length=EMBED_MAX_LENGTH,
                               truncation_strategy='delete', enable_thinking=False)
    sys.stderr.write('[eval] embedding model ready\n')

    ks = sorted({1, 5, 10, args.top_k})
    hits = {k: 0 for k in ks}
    per_source_hits: Dict[str, Dict[int, int]] = {}
    per_source_total: Dict[str, int] = {}
    debug_records: List[Dict[str, Any]] = []

    # Batch encode and search.
    for batch_start in range(0, n_sample, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_sample)
        batch = samples.iloc[batch_start:batch_end]
        queries = batch['query_compressed'].tolist()
        ids = batch['id'].tolist()
        sources = batch['source'].tolist()
        thinkings = batch['thinking_raw'].tolist()
        query_raws = batch['query_raw'].tolist()
        cot_compresseds = batch['cot_compressed'].tolist()

        anchor_emb = get_embeddings(model, template, queries)

        for i, (rid, src, vec) in enumerate(zip(ids, sources, anchor_emb)):
            res = (
                tbl.search(vec.astype(np.float32).tolist())
                .metric('dot')
                .limit(max(ks))
                .select(['id', 'source', 'query_compressed', 'cot_compressed',
                         'thinking_raw', 'query_raw'])
                .to_list()
            )
            hit_ids = [item['id'] for item in res]
            try:
                rank = hit_ids.index(rid)
            except ValueError:
                rank = -1

            for k in ks:
                if 0 <= rank < k:
                    hits[k] += 1
                    per_source_hits.setdefault(src, {kk: 0 for kk in ks})[k] += 1
            per_source_total[src] = per_source_total.get(src, 0) + 1
            per_source_hits.setdefault(src, {kk: 0 for kk in ks})

            top1 = res[0] if res else {}
            debug_records.append({
                'id': rid,
                'source': src,
                'rank': rank,
                'query_raw': query_raws[i],
                'query_compressed': queries[i],
                'cot_compressed': cot_compresseds[i],
                'thinking_raw': thinkings[i][:2000],
                'top1_id': top1.get('id'),
                'top1_source': top1.get('source'),
                'top1_query_compressed': top1.get('query_compressed'),
                'top1_cot_compressed': top1.get('cot_compressed'),
                'top1_query_raw': top1.get('query_raw'),
                'top1_thinking_raw': (top1.get('thinking_raw') or '')[:2000],
                'top1_is_self': top1.get('id') == rid,
            })

        sys.stderr.write(f'  probed {batch_end}/{n_sample}\n')

    print(f'\n=== Self-Recall @ k (n={n_sample}, seed={args.seed}) ===')
    for k in ks:
        print(f'  recall@{k:<3} = {hits[k]/n_sample:.4f}  ({hits[k]}/{n_sample})')

    print(f'\n=== Per-source recall@{max(ks)} ===')
    for src in sorted(per_source_total, key=lambda s: -per_source_total[s]):
        tot = per_source_total[src]
        h = per_source_hits.get(src, {}).get(max(ks), 0)
        print(f'  {src:<48s} {h/tot:.4f}  ({h}/{tot})')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for rec in debug_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'\n[debug] {len(debug_records)} records saved to {args.output}')


if __name__ == '__main__':
    main()
