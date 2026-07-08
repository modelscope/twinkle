# Copyright (c) ModelScope Contributors. All rights reserved.
"""Benchmark decontamination via n-gram overlap (AUDIT D2) — OFFLINE ONLY.

Removes (or tags) training rows that overlap with evaluation benchmarks, so
reported metrics aren't inflated by leakage. Follows the standard 13-gram
overlap recipe (GPT-3 / Llama / Dolma): build an n-gram set from the benchmark
texts once, then flag any row whose text shares an n-gram with it.

OFFLINE CONTRACT: the benchmark index is static and global; build it once and
reuse across the whole dataset. This is not a per-batch pipeline step — but
unlike near-dup it *is* embarrassingly parallel per row, so it can also run as a
standalone batch pass. Default mode ``'drop'`` removes contaminated rows;
``'tag'`` keeps them and only records a ``contaminated`` label (safer default for
real-time-ish contexts where false positives must never delete data).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set, Tuple

from twinkle.preprocessor import Preprocessor

from .. import label_schema as L
from ..message_utils import msg_content_text

KEY_CONTAMINATED = 'contaminated'

_WORD_RE = re.compile(r'\w+', re.UNICODE)


def _ngrams(text: str, n: int) -> Set[str]:
    tokens = _WORD_RE.findall(text.lower())
    if len(tokens) < n:
        return set()
    return {' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def build_benchmark_index(texts: Iterable[str], n: int = 13) -> Set[str]:
    """Build a static n-gram set from benchmark texts (build once, reuse)."""
    index: Set[str] = set()
    for t in texts:
        index |= _ngrams(t or '', n)
    return index


class Decontaminator(Preprocessor):
    """Flag/drop rows that share an n-gram with a static benchmark index.

    Args:
        benchmark_ngrams: prebuilt index from :func:`build_benchmark_index`.
        n: n-gram size (must match the index's n). Default 13.
        min_overlap: number of shared n-grams to count as contaminated.
        mode: ``'drop'`` removes contaminated rows; ``'tag'`` keeps them and only
            writes the ``contaminated`` label (fail-open).
        scan: which roles to scan — 'user' (default), 'assistant', or 'all'.
    """

    def __init__(
        self,
        benchmark_ngrams: Set[str],
        *,
        n: int = 13,
        min_overlap: int = 1,
        mode: str = 'drop',
        scan: str = 'user',
    ):
        if mode not in ('drop', 'tag'):
            raise ValueError("mode must be 'drop' or 'tag'")
        if scan not in ('user', 'assistant', 'all'):
            raise ValueError("scan must be 'user', 'assistant', or 'all'")
        self.index = benchmark_ngrams or set()
        self.n = int(n)
        self.min_overlap = int(min_overlap)
        self.mode = mode
        self.scan = scan

    def _row_text(self, row: Dict[str, Any]) -> str:
        messages = row.get('messages') or []
        parts = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get('role')
            if self.scan == 'all' or role == self.scan:
                parts.append(msg_content_text(m))
        return '\n'.join(p for p in parts if p)

    def _overlap(self, row: Dict[str, Any]) -> int:
        if not self.index:
            return 0
        grams = _ngrams(self._row_text(row), self.n)
        if not grams:
            return 0
        return len(grams & self.index)

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for row in rows:
            overlap = self._overlap(row)
            contaminated = overlap >= self.min_overlap
            if contaminated and self.mode == 'drop':
                dropped.append(dict(row, drop_reason='benchmark_contamination'))
                continue
            if self.mode == 'tag':
                kept.append(L.set_label(row, KEY_CONTAMINATED, contaminated))
            else:
                kept.append(row)
        return kept, dropped
