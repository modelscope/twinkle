# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline batch-only preprocessors (AUDIT D1 / D2).

These steps require a GLOBAL view of the dataset and must NOT be dropped into the
per-batch :class:`~twinkle_agentic.preprocessor.QualityPreprocessor` pipeline:

- :class:`NearDupFilter` (D1) — MinHash-LSH near-duplicate removal; per-batch use
  would only compare within a batch, causing severe false negatives.
- :class:`Decontaminator` (D2) — benchmark n-gram overlap removal against a static
  index; kept out of the real-time path to avoid false-positive deletions
  (defaults to a safe ``'tag'``-friendly design).

They are deliberately kept out of the main package namespace. Import explicitly::

    from twinkle_agentic.preprocessor.offline import NearDupFilter, Decontaminator
    from twinkle_agentic.preprocessor.offline import build_benchmark_index

Usage: materialize the dataset to ``List[Dict]``, run these once, then re-wrap
the kept rows before/after the streaming QualityPreprocessor pipeline.
"""
from .decontaminate import Decontaminator, build_benchmark_index  # noqa: F401
from .near_dedup import NearDupFilter  # noqa: F401

__all__ = ['NearDupFilter', 'Decontaminator', 'build_benchmark_index']
