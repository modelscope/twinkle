# Copyright (c) ModelScope Contributors. All rights reserved.
"""Near-duplicate removal via MinHash-LSH (AUDIT D1) — OFFLINE ONLY.

``DedupFilter`` collapses only exact/prefix duplicates; a single edited character
slips through. This adds fuzzy near-dup detection over shingled trajectory text.

OFFLINE CONTRACT (same as :class:`DedupFilter`): this needs a *global* view of
the dataset — it must see all rows in one ``__call__`` and is NOT a per-batch
``QualityPreprocessor`` step. Running near-dup on a per-batch stream would judge
similarity against only the current batch, causing severe false negatives (and,
if used to drop, unstable results). Materialize the dataset, run this once, then
re-wrap the kept rows.

Uses ``datasketch`` when installed (fast LSH); otherwise falls back to a pure
O(n²) MinHash comparison — correct but slower, fine for modest offline batches.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger

from ..message_utils import msg_content_text

logger = get_logger()

_WORD_RE = re.compile(r'\w+', re.UNICODE)


def _row_text(row: Dict[str, Any]) -> str:
    messages = row.get('messages') or []
    return '\n'.join(msg_content_text(m) for m in messages if isinstance(m, dict))


def _shingles(text: str, k: int) -> Set[str]:
    tokens = _WORD_RE.findall(text.lower())
    if len(tokens) < k:
        return {' '.join(tokens)} if tokens else set()
    return {' '.join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def _minhash_signature(shingles: Set[str], num_perm: int) -> List[int]:
    """Pure-python MinHash: for each of ``num_perm`` salted hashes, take the min."""
    if not shingles:
        return [0] * num_perm
    sig: List[int] = []
    for p in range(num_perm):
        salt = str(p).encode()
        mn = min(int(hashlib.md5(salt + s.encode('utf-8')).hexdigest(), 16) for s in shingles)
        sig.append(mn)
    return sig


class NearDupFilter(Preprocessor):
    """Global near-duplicate removal over a fully materialized row collection.

    Args:
        threshold: Jaccard similarity at/above which two rows are near-duplicates.
        shingle_size: word n-gram size for shingling.
        num_perm: MinHash permutations (higher = more accurate, slower).
        keep: within a near-dup cluster, keep the ``'longest'`` (most messages)
            or ``'first'`` seen row.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.8,
        shingle_size: int = 5,
        num_perm: int = 128,
        keep: str = 'longest',
    ):
        if keep not in ('longest', 'first'):
            raise ValueError("keep must be 'longest' or 'first'")
        self.threshold = float(threshold)
        self.shingle_size = int(shingle_size)
        self.num_perm = int(num_perm)
        self.keep = keep

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        n = len(rows)
        if n <= 1:
            return rows, []
        shingle_sets = [_shingles(_row_text(r), self.shingle_size) for r in rows]

        try:
            from datasketch import MinHash, MinHashLSH
            clusters = self._cluster_lsh(shingle_sets, MinHash, MinHashLSH)
        except Exception as e:
            logger.info(f'[NearDupFilter] datasketch unavailable ({e}); pure-python O(n^2) fallback.')
            clusters = self._cluster_bruteforce(shingle_sets)

        keep_flag = [True] * n
        dropped: List[Dict[str, Any]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            winner = self._pick_winner(rows, cluster)
            for idx in cluster:
                if idx != winner:
                    keep_flag[idx] = False
                    dropped.append(dict(rows[idx], drop_reason='near_duplicate'))
        kept = [rows[i] for i in range(n) if keep_flag[i]]
        return kept, dropped

    def _pick_winner(self, rows: List[Dict[str, Any]], cluster: List[int]) -> int:
        if self.keep == 'first':
            return min(cluster)
        return max(cluster, key=lambda i: len(rows[i].get('messages') or []))

    def _cluster_lsh(self, shingle_sets, MinHash, MinHashLSH) -> List[List[int]]:
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        mh_list = []
        for i, sh in enumerate(shingle_sets):
            mh = MinHash(num_perm=self.num_perm)
            for s in sh:
                mh.update(s.encode('utf-8'))
            mh_list.append(mh)
            lsh.insert(str(i), mh)
        return self._union_find([(i, [int(x) for x in lsh.query(mh_list[i])]) for i in range(len(shingle_sets))],
                                len(shingle_sets))

    def _cluster_bruteforce(self, shingle_sets) -> List[List[int]]:
        n = len(shingle_sets)
        neighbors: List[Tuple[int, List[int]]] = []
        for i in range(n):
            adj = [i]
            for j in range(i + 1, n):
                a, b = shingle_sets[i], shingle_sets[j]
                if not a and not b:
                    continue
                inter = len(a & b)
                union = len(a | b) or 1
                if inter / union >= self.threshold:
                    adj.append(j)
            neighbors.append((i, adj))
        return self._union_find(neighbors, n)

    @staticmethod
    def _union_find(adjacency: List[Tuple[int, List[int]]], n: int) -> List[List[int]]:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, adj in adjacency:
            for j in adj:
                union(i, j)
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())
