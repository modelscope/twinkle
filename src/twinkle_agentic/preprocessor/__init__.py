# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger
from twinkle.utils.parallel import PosixFileLock
from .data_juicer import FixUnicodeFilter, RemoveRepeatSentencesFilter, SpecialCharsFilter, TokenNumFilter
from .dead_loop_filter import DeadLoopFilter
from .dedup_filter import DedupFilter
from .hard_filter import HardFilter
from .intent_classifier import IntentClassifier
from .language_filter import LanguageFilter  # noqa: F401
from .message_normalizer import MessageNormalizer  # noqa: F401
from .message_sanity import MessageSanityFilter
from .model_filter import ModelFilter
from .outcome_filter import TrajectoryOutcomeFilter  # noqa: F401
from .pii_presidio_filter import PIIPresidioFilter
from .provenance import ProvenanceStamp  # noqa: F401
from .refuse_filter import RefuseFilter
from .safety_scorer import SafetyScorer  # noqa: F401
from .structural_noise import StructuralNoiseTagger  # noqa: F401
from .token_soup import TokenSoupFilter
from .trajectory_scorer import TrajectoryScorer  # noqa: F401
from .value_selector import ValueSelector, select_top_for_rubric  # noqa: F401

logger = get_logger()


def truncate_dropped_logs(dropped_log_path: str) -> None:
    """Remove prior dropped log shards (call once from the main process before map)."""
    if not dropped_log_path:
        return
    import glob
    for p in [dropped_log_path] + glob.glob(f'{dropped_log_path}.*'):
        if p.endswith('.lock'):
            continue
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def merge_dropped_shards(dropped_log_path: str) -> None:
    """Merge per-worker ``dropped.jsonl.<pid>`` shards into ``dropped.jsonl``."""
    if not dropped_log_path:
        return
    import glob
    shards = sorted(
        p for p in glob.glob(f'{dropped_log_path}.*')
        if not p.endswith('.lock'))
    if not shards:
        return
    os.makedirs(os.path.dirname(os.path.abspath(dropped_log_path)) or '.', exist_ok=True)
    with open(dropped_log_path, 'w', encoding='utf-8') as out:
        for sp in shards:
            with open(sp, encoding='utf-8') as fin:
                for line in fin:
                    if line.strip():
                        out.write(line if line.endswith('\n') else line + '\n')
            try:
                os.remove(sp)
            except FileNotFoundError:
                pass


def run_quality_pipeline(dataset, pipeline: 'QualityPreprocessor', *,
                         num_proc: int = 1, **map_kwargs):
    """Run a ``drop_mode='mark'`` pipeline as map(equal-length) + filter(keep).

    This is the ghost-proof way to run a filtering pipeline: ``map`` never
    changes row count (every batch returns equal-length columns with a
    ``_keep`` flag), then a single ``Dataset.filter`` on that flag does the
    actual removal. Returns the dataset (mutated in place).
    """
    if getattr(pipeline, '_drop_mode', None) != 'mark':
        raise ValueError("run_quality_pipeline requires a pipeline built with drop_mode='mark'")
    flag = QualityPreprocessor.KEEP_FLAG
    map_kwargs.pop('remove_columns', None)  # mark mode keeps row count; not needed
    dataset.map(pipeline, num_proc=num_proc, **map_kwargs)
    dataset.filter(lambda row: bool(row.get(flag, True)))
    # Drop the transient keep-flag column so downstream schema stays clean.
    hf = dataset.dataset
    if flag in hf.column_names:
        dataset.dataset = hf.remove_columns([flag])
        datasets = getattr(dataset, 'datasets', None)
        if isinstance(datasets, dict) and len(datasets) == 1:
            for k in list(datasets.keys()):
                datasets[k] = dataset.dataset
    return dataset


class QualityPreprocessor(Preprocessor):
    """Thin pipeline runner: accepts a list of callables, runs them in order.

    Each step must accept and return List[Dict[str, Any]].
    Per-step logging (before/after count) and optional dropped-row JSONL are provided.
    """

    #: Column name for the keep flag emitted in ``drop_mode='mark'``.
    KEEP_FLAG = '_keep'

    def __init__(self, pipeline: List[Callable], dropped_log_path: str = '',
                 drop_mode: str = 'inline'):
        super().__init__()
        if drop_mode not in ('inline', 'mark'):
            raise ValueError("drop_mode must be 'inline' or 'mark'")
        # 'inline': the batch returns only surviving rows (shorter columns). HF
        #   then needs remove_columns to change row count cleanly, else ghost
        #   rows appear. Kept as the backward-compatible default.
        # 'mark': the batch ALWAYS returns equal-length columns; dropped rows are
        #   returned too, flagged KEEP_FLAG=False (survivors True). No row-count
        #   change happens inside map, so no ghosting is possible. The caller
        #   materializes the drop with a follow-up ``Dataset.filter`` on KEEP_FLAG
        #   (see ``run_quality_pipeline``).
        self._drop_mode = drop_mode
        self._pipelines = list(pipeline)
        self._dropped_log_path = dropped_log_path
        if dropped_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(dropped_log_path)) or '.', exist_ok=True)
        lock_path = (dropped_log_path + '.lock') if dropped_log_path else ''
        self._lock: Optional[PosixFileLock] = PosixFileLock(lock_path) if lock_path else None
        # Truncation is explicit (see truncate_dropped_logs) so HF num_proc workers
        # do not race to delete each other's shard files on unpickle/re-init.

    def __call__(self, rows):
        input_col_keys = list(rows.keys()) if isinstance(rows, dict) else None
        rows_list = self.map_col_to_row(rows)
        total_start = len(rows_list)
        # In 'mark' mode we must return every input row (equal-length columns), so
        # remember each row's identity to reconcile survivors vs. dropped at the
        # end. A per-batch position index is stable and needs no unique id, and a
        # snapshot preserves dropped rows' original columns for re-emission.
        original_rows = None
        if self._drop_mode == 'mark':
            original_rows = [dict(r) for r in rows_list]
            for i, r in enumerate(rows_list):
                r['_row_idx'] = i
        stats = []
        for step in self._pipelines:
            if not rows_list:
                break
            step_name = getattr(step, '__name__', None) or type(step).__name__
            before = len(rows_list)
            t0 = time.perf_counter()
            kept, dropped = step(rows_list)
            rows_list = self.map_col_to_row(kept)
            elapsed = time.perf_counter() - t0
            after = len(rows_list)
            stats.append(f'  {step_name}: {before}->{after} (dropped {before - after}, {elapsed:.3f}s)')
            self._log_dropped(step_name, dropped)
        summary = '\n'.join(stats)
        logger.info(f'[QualityPreprocessor] {total_start} -> {len(rows_list)}\n{summary}')

        if self._drop_mode == 'mark':
            return self._emit_marked(rows_list, total_start, input_col_keys, original_rows)
        # 'inline': HF ``datasets.map(batched=True)`` changes row count only when
        # the batch returns shorter columns AND the caller passes remove_columns
        # so the old columns are rebuilt (else survivors of a partially-filtered
        # batch leave the un-dropped originals behind as ghost rows). Emitting an
        # empty dict would also leave ghosts, so always emit explicit columns.
        return self.map_row_to_col(rows_list, keys=input_col_keys)

    def _emit_marked(self, survivors, total_start, input_col_keys, original_rows):
        """Return ALL input rows with equal-length columns, flagging survivors.

        Survivors carry ``KEEP_FLAG=True`` plus their tags; dropped rows are
        re-emitted from their original input state with ``KEEP_FLAG=False`` so no
        column ever changes length inside ``map`` (ghost-proof). The caller then
        does a single ``Dataset.filter`` on ``KEEP_FLAG``.
        """
        by_idx = {r.get('_row_idx'): r for r in survivors}
        merged = []
        for i in range(total_start):
            if i in by_idx:
                row = by_idx[i]
                row[self.KEEP_FLAG] = True
            else:
                # dropped: re-emit the original input row so its columns still
                # exist (values are irrelevant — the caller filters it out).
                row = dict(original_rows[i])
                row[self.KEEP_FLAG] = False
            row.pop('_row_idx', None)
            merged.append(row)
        # Emit the UNION of every row's keys so a tag added only to survivors
        # (e.g. `intent`) is present as a real column (None for dropped rows) —
        # rows[0] alone is not enough (that is the original ghosting bug).
        key_union: List[str] = list(input_col_keys or [])
        for row in merged:
            for k in row.keys():
                if k not in key_union:
                    key_union.append(k)
        columns = {k: [row.get(k) for row in merged] for k in key_union}
        return columns

    def _log_dropped(self, step_name: str, dropped: List[Dict[str, Any]]) -> None:
        if not self._lock or not dropped:
            return
        shard = f'{self._dropped_log_path}.{os.getpid()}'
        with self._lock:
            with open(shard, 'a', encoding='utf-8') as f:
                for r in dropped:
                    rec = self._compact_drop_record(step_name, r)
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

    @staticmethod
    def _compact_drop_record(step_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """Log metadata only — full messages are huge and break multiprocess merges."""
        return {
            'step': step_name,
            'reason': row.get('drop_reason') or step_name,
            'id': row.get('id'),
            'model_id': row.get('model_id'),
            'n_msgs': len(row.get('messages') or []),
        }
