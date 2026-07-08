# Copyright (c) ModelScope Contributors. All rights reserved.
"""Data-lineage / provenance stamping — tag only, never drop (AUDIT D10).

Industry data pipelines keep provenance so any training example is traceable
back to its source (which dataset, which teacher/student model produced it, when
it was ingested, which cleaning pipeline version touched it). In a self-evolving
distillation loop this is what lets us later attribute a regression to a bad
source or a specific teacher, and to reproduce a training mix.

This mapper writes a single ``provenance`` blob into ``user_data`` (JSON-packed,
PyArrow-stable via A5). It reads whatever lineage fields already exist on the row
(``model_id`` and any configured passthroughs) and adds an ingest timestamp so
the record is self-describing downstream.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Sequence

from twinkle.preprocessor import Mapper

from . import label_schema as L


class ProvenanceStamp(Mapper):
    """Stamp each row with a provenance blob in ``user_data`` (never drops).

    Args:
        source: a static source/dataset identifier for this ingest batch.
        pipeline_version: version string of the cleaning pipeline for audit.
        model_field: row field holding the producing model id (default 'model_id').
        extra_fields: additional row fields to copy verbatim into provenance
            (e.g. 'teacher_model', 'student_model', 'request_id').
        add_timestamp: include a unix ingest timestamp. Default True.
        overwrite: if False, rows that already carry a provenance blob are left
            untouched (idempotent re-runs / preserve upstream lineage). Default False.
    """

    def __init__(
        self,
        *,
        source: str = '',
        pipeline_version: str = '',
        model_field: str = 'model_id',
        extra_fields: Sequence[str] = (),
        add_timestamp: bool = True,
        overwrite: bool = False,
    ):
        self.source = source
        self.pipeline_version = pipeline_version
        self.model_field = model_field
        self.extra_fields = tuple(extra_fields)
        self.add_timestamp = bool(add_timestamp)
        self.overwrite = bool(overwrite)

    def map(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not self.overwrite and L.get_label(row, L.KEY_PROVENANCE, None) is not None:
            return row
        blob: Dict[str, Any] = {}
        if self.source:
            blob['source'] = self.source
        if self.pipeline_version:
            blob['pipeline_version'] = self.pipeline_version
        model = row.get(self.model_field)
        if model:
            blob['model'] = model
        for f in self.extra_fields:
            v = row.get(f)
            if v is not None:
                blob[f] = v
        if self.add_timestamp:
            blob['ingested_at'] = int(time.time())
        return L.set_label(row, L.KEY_PROVENANCE, blob)
