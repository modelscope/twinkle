"""Hard-negative dataset for embedding training.

Provides ReasonIR (AI-ModelScope/reasonir-data, hq subset):
  - query: reasoning-intensive question
  - positive: BRIGHT document (resolved via xlangai/BRIGHT documents corpus)
  - negatives: plausibly related but ultimately unhelpful documents

Output schema: ``{id, source, query, cot, response, negatives}``
where ``negatives`` is a list of strings (each a separate hard negative).
"""
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset
from modelscope import MsDataset

_CACHE_DIR = Path(__file__).resolve().parent / '.cache_hard'


def _hash_id(prefix: str, content: str) -> str:
    return f'{prefix}__{hashlib.md5(content.encode("utf-8")).hexdigest()[:16]}'


# ---------------------------------------------------------------------------
# BRIGHT document corpus (lazy singleton)
# ---------------------------------------------------------------------------

_BRIGHT_SPLITS = [
    'aops', 'biology', 'earth_science', 'economics', 'leetcode', 'pony',
    'psychology', 'robotics', 'stackoverflow', 'sustainable_living',
    'theoremqa_questions', 'theoremqa_theorems',
]

_bright_docs: Optional[Dict[str, str]] = None


def _load_bright_docs() -> Dict[str, str]:
    """Load all BRIGHT document splits into {id -> content} lookup dict."""
    global _bright_docs
    if _bright_docs is not None:
        return _bright_docs
    sys.stderr.write('[dataset_hard] Loading BRIGHT documents corpus...\n')
    _bright_docs = {}
    for split in _BRIGHT_SPLITS:
        try:
            ds = MsDataset.load(
                'xlangai/BRIGHT', subset_name='documents', split=split,
                download_mode='reuse_dataset_if_exists')
            for row in ds:
                doc_id = row.get('id', '')
                content = row.get('content', '')
                if doc_id and content:
                    _bright_docs[doc_id] = content
                    short = doc_id.rsplit('/', 1)[-1] if '/' in doc_id else doc_id
                    if short not in _bright_docs:
                        _bright_docs[short] = content
            sys.stderr.write(f'  [{split}] loaded {len(ds)} docs\n')
        except Exception as e:
            sys.stderr.write(f'  [{split}] FAILED: {e}\n')
    sys.stderr.write(f'[dataset_hard] BRIGHT total: {len(_bright_docs)} entries\n')
    return _bright_docs


# ---------------------------------------------------------------------------
# ReasonIR dataset
# ---------------------------------------------------------------------------

def get_dataset_reasonir(max_rows: Optional[int] = None,
                         max_negatives: int = 16,
                         load_from_cache_file: bool = True) -> HFDataset:
    """Load AI-ModelScope/reasonir-data (hq subset) with BRIGHT doc resolution.

    Schema: {id, source, query, cot, response, negatives}
    """
    cache_key = f'reasonir_neg{max_negatives}'
    cache_path = _CACHE_DIR / cache_key
    if load_from_cache_file and cache_path.exists():
        sys.stderr.write(f'[reasonir] loading from cache: {cache_path}\n')
        ds = HFDataset.load_from_disk(str(cache_path))
        if max_rows and len(ds) > max_rows:
            ds = ds.select(range(max_rows))
        sys.stderr.write(f'[reasonir] {len(ds)} rows (cached)\n')
        return ds

    ds = MsDataset.load(
        'AI-ModelScope/reasonir-data', subset_name='hq', split='train',
        download_mode='reuse_dataset_if_exists')
    if max_rows and len(ds) > max_rows:
        ds = ds.select(range(max_rows))

    bright = _load_bright_docs()
    rows = []
    n_miss = 0
    for row in ds:
        query_parts = row.get('query', [])
        if not isinstance(query_parts, list) or len(query_parts) < 2:
            continue
        query = query_parts[1].strip()
        if not query:
            continue

        pos_list = row.get('pos', [])
        if not pos_list:
            continue
        pos_id = pos_list[0][1] if isinstance(pos_list[0], list) and len(pos_list[0]) > 1 else ''
        cot = bright.get(pos_id, '')
        if not cot:
            n_miss += 1
            continue

        neg_list = row.get('neg', [])
        negatives = []
        for neg in neg_list:
            if isinstance(neg, list) and len(neg) > 1:
                neg_text = neg[1].strip()
                if neg_text:
                    negatives.append(neg_text)
                    if len(negatives) >= max_negatives:
                        break

        if not negatives:
            continue

        rows.append({
            'id': _hash_id('reasonir', f'{query}\n{pos_id}'),
            'source': 'reasonir-hq',
            'query': query,
            'cot': cot,
            'response': '',
            'negatives': negatives,
        })

    if n_miss:
        sys.stderr.write(f'[reasonir] {n_miss} rows skipped (BRIGHT doc not found)\n')
    sys.stderr.write(f'[reasonir] {len(rows)} rows with hard negatives\n')
    result = HFDataset.from_dict(_rows_to_cols(rows))
    # Persist full dataset; max_rows is applied post-cache for flexibility.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(str(cache_path))
    sys.stderr.write(f'[reasonir] cached to {cache_path}\n')
    if max_rows and len(result) > max_rows:
        result = result.select(range(max_rows))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows_to_cols(rows: List[Dict[str, Any]]) -> Dict[str, list]:
    if not rows:
        return {'id': [], 'source': [], 'query': [], 'cot': [],
                'response': [], 'negatives': []}
    keys = rows[0].keys()
    return {k: [r[k] for r in rows] for k in keys}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset(
    reasonir_max: Optional[int] = None,
    max_negatives: int = 16,
    load_from_cache_file: bool = True,
    **kwargs,
) -> HFDataset:
    """Load hard-negative dataset (reasonir only).

    Returns HF Dataset with schema: {id, source, query, cot, response, negatives}
    """
    ds = get_dataset_reasonir(max_rows=reasonir_max, max_negatives=max_negatives,
                              load_from_cache_file=load_from_cache_file)
    if len(ds) == 0:
        sys.stderr.write('[dataset_hard] WARNING: reasonir dataset empty\n')
    else:
        sys.stderr.write(f'[dataset_hard] reasonir={len(ds)}\n')
    return ds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reasonir-max', type=int, default=1000)
    args = parser.parse_args()

    ds = get_dataset(reasonir_max=args.reasonir_max)
    print(f'Total rows: {len(ds)}')
    print(f'Features: {ds.features}')
    if len(ds) > 0:
        row = ds[0]
        print(f'\nSample[0]:')
        print(f'  id: {row["id"]}')
        print(f'  source: {row["source"]}')
        print(f'  query: {row["query"][:100]}...')
        print(f'  cot: {row["cot"][:100]}...')
        print(f'  negatives: {len(row["negatives"])} items')
        if row['negatives']:
            print(f'    [0]: {row["negatives"][0][:80]}...')
