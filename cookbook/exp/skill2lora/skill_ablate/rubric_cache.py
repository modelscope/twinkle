# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rubric double-cache (skill_quality_analysis.md #4, #17).

Two cache scopes, both wrapping v2's ``DiskCache`` (append-only jsonl, in-memory index):

- GlobalRubricCache (key = data_id): the RL / SFT lines diagnose the BARE-PROBLEM greedy
  trajectory. Because the executor is frozen at T=0, that trajectory is deterministic and
  identical across experiments, so its rubric diagnosis can be shared across ALL runs via one
  global file (``rubric_cache_global.jsonl``) — diagnose each problem once, reuse everywhere.

- LocalRubricCache (key = md5(data_id + skill)): the improve-skill+SFT / OPSD lines diagnose a
  WITH-SKILL trajectory whose skill evolves with the policy, so the diagnosis is experiment-
  and step-specific and lives only in that experiment's directory.

Both reuse v2's ``_diagnose_entry`` (pure teacher-API call, no GPU) for the actual diagnosis,
so there is a single source of truth for the rubric prompt / parsing.
"""
import os
from typing import Any, Dict, Optional

from train_skill_v2 import DiskCache, _diagnose_entry


class _BaseRubricCache:
    """Shared get-or-diagnose logic over a DiskCache; subclasses define the key."""

    def __init__(self, path: str, enabled: bool = True):
        self._cache = DiskCache(path, enabled)

    def _key(self, entry: Dict[str, Any], skill: Optional[str]) -> str:
        raise NotImplementedError

    def get(self, entry: Dict[str, Any], skill: Optional[str] = None) -> Optional[str]:
        return self._cache.get(self._key(entry, skill))

    def get_or_diagnose(self, entry: Dict[str, Any], checker,
                        skill: Optional[str] = None) -> str:
        """Return cached diagnosis, else run the teacher rubric once and cache it.

        ``entry`` must carry the fields ``_diagnose_entry`` needs: ``problem``,
        ``fail_segment``, ``fail_stop_reason`` (and ``reference_answer`` is unused by the
        diagnosis but kept for auditing). Returns '' when there is no checker or on API error
        (never raises), so the caller can treat "no diagnosis" uniformly.
        """
        if checker is None:
            return ''
        key = self._key(entry, skill)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        diag = _diagnose_entry(checker, entry) or ''
        self._cache.put(key, diag)
        return diag

    def put(self, entry: Dict[str, Any], diag: str, skill: Optional[str] = None) -> None:
        self._cache.put(self._key(entry, skill), diag)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def close(self) -> None:
        self._cache.close()


class GlobalRubricCache(_BaseRubricCache):
    """key = data_id: bare-problem trajectory diagnosis, shareable across experiments."""

    def _key(self, entry: Dict[str, Any], skill: Optional[str] = None) -> str:
        return DiskCache.key_for('rubric_global', str(entry.get('data_id', '')))


class LocalRubricCache(_BaseRubricCache):
    """key = md5(data_id + skill): with-skill trajectory diagnosis, per-experiment only."""

    def _key(self, entry: Dict[str, Any], skill: Optional[str] = None) -> str:
        return DiskCache.key_for('rubric_local', str(entry.get('data_id', '')), skill or '')


def build_rubric_cache(scope: str, output_dir: str,
                       global_dir: Optional[str] = None, enabled: bool = True):
    """Factory: scope='global' -> shared file under ``global_dir`` (default output_dir/..);
    scope='local' -> per-experiment file under ``output_dir/cache``."""
    if scope == 'global':
        base = global_dir or os.path.dirname(os.path.abspath(output_dir.rstrip('/')))
        os.makedirs(base, exist_ok=True)
        return GlobalRubricCache(os.path.join(base, 'rubric_cache_global.jsonl'), enabled)
    if scope == 'local':
        cache_dir = os.path.join(output_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        return LocalRubricCache(os.path.join(cache_dir, 'rubric_cache_local.jsonl'), enabled)
    raise ValueError(f"scope must be 'global' or 'local', got {scope!r}")
