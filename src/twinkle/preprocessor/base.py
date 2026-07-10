# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, List, Tuple

from twinkle.data_format import Trajectory


class Preprocessor:
    """Base for pipeline steps.

    Concrete steps take a batch of rows (list-of-dict, or the columnar
    dict-of-lists produced by HF ``datasets``) and return a
    ``(kept, dropped)`` tuple of row lists. ``map_col_to_row`` normalizes the
    input; a step that never removes rows (a *mapper*) returns
    ``(rows, [])`` — see :class:`Mapper`. Steps that select rows (a *filter*)
    return ``(kept, dropped)`` — see :class:`Filter`.

    The pipeline runner (:class:`~twinkle_agentic.preprocessor.QualityPreprocessor`)
    consumes the tuple, logs the dropped rows, and re-columnarizes ``kept`` before
    handing it to the next step.
    """

    @staticmethod
    def map_col_to_row(rows) -> List[Dict[str, Any]]:
        if isinstance(rows, list):
            return rows
        if not rows:
            return []
        _new_rows = []
        total_count = len(rows[next(iter(list(rows.keys())))])
        for i in range(total_count):
            row = {}
            for key in rows:
                row[key] = rows[key][i]
            _new_rows.append(row)
        return _new_rows

    @staticmethod
    def map_row_to_col(rows, keys: List[str] = None) -> Dict[str, List[Any]]:
        if isinstance(rows, dict):
            return rows
        if not rows:
            return {k: [] for k in keys} if keys else {}

        columns: Dict[str, List[Any]] = {}
        row_keys = list(rows[0].keys())
        out_keys = row_keys if not keys else list(dict.fromkeys(row_keys + [k for k in keys if k not in row_keys]))

        for key in out_keys:
            columns[key] = [row.get(key) for row in rows]

        return columns

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return ``(kept, dropped)`` row lists. Subclasses must override."""
        raise NotImplementedError


class Mapper(Preprocessor):
    """A step that annotates/transforms rows and never drops any.

    Subclasses implement :meth:`map` (row-in, row-out); the ``(rows, [])``
    contract is provided so mappers compose with filters in the same pipeline.
    """

    def map(self, row: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        return [self.map(r) for r in rows], []


class Filter(Preprocessor):
    """A step that selects rows, returning ``(kept, dropped)``.

    Subclasses implement :meth:`keep` (row-in, bool-out). Dropped rows are
    returned so the runner can log them.
    """

    def keep(self, row: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def drop_reason(self, row: Dict[str, Any]) -> str:
        """Short reason tag for dropped-row logs (override in subclasses)."""
        return type(self).__name__

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for r in rows:
            if self.keep(r):
                kept.append(r)
            else:
                dropped.append(dict(r, drop_reason=self.drop_reason(r)))
        return kept, dropped


class DataFilter:

    def __call__(self, row) -> bool:
        ...
