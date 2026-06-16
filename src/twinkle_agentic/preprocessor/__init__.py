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
from .llm_backend import LLMBackend, OpenAIBackend, SamplerBackend  # noqa: F401
from .message_normalizer import MessageNormalizer  # noqa: F401
from .message_sanity import MessageSanityFilter
from .model_filter import ModelFilter
from .pii_presidio_filter import PIIPresidioFilter
from .refuse_filter import RefuseFilter
from .score_filter import ScoreFilter
from .token_soup import TokenSoupFilter

logger = get_logger()


class QualityPreprocessor(Preprocessor):
    """Thin pipeline runner: accepts a list of callables, runs them in order.

    Each step must accept and return List[Dict[str, Any]].
    Per-step logging (before/after count) and optional dropped-row JSONL are provided.
    """

    def __init__(self, pipeline: List[Callable], dropped_log_path: str = ''):
        super().__init__()
        self._pipelines = list(pipeline)
        self._dropped_log_path = dropped_log_path
        if dropped_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(dropped_log_path)), exist_ok=True)
        self._lock: Optional[PosixFileLock] = (PosixFileLock(dropped_log_path + '.lock') if dropped_log_path else None)
        if dropped_log_path and os.path.exists(dropped_log_path):
            os.remove(dropped_log_path)

    def __call__(self, rows):
        rows_list = self.map_col_to_row(rows)
        total_start = len(rows_list)
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
        return self.map_row_to_col(rows_list)

    def _log_dropped(self, step_name: str, dropped: List[Dict[str, Any]]) -> None:
        if not self._lock or not dropped:
            return
        with self._lock:
            with open(self._dropped_log_path, 'a', encoding='utf-8') as f:
                for r in dropped:
                    f.write(json.dumps({'step': step_name, 'row': r}, ensure_ascii=False, default=str) + '\n')
