# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from typing import Any, Callable, Dict, List, Optional

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger
from twinkle.utils.parallel import PosixFileLock

from .consistency_filter import ConsistencyFilter
from .data_juicer import (
    AlphanumericFilter,
    CharRepeatFilter,
    FlaggedWordsFilter,
    FixUnicodeFilter,
    KenLMFilter,
    LanguageFilter,
    LLMConditionFilter,
    LLMDifficultyFilter,
    LLMQualityFilter,
    LLMTaskRelevanceFilter,
    MinHashDedupFilter,
    RemoveRepeatSentencesFilter,
    SpecialCharsFilter,
    StopwordsFilter,
    TextActionFilter,
    TokenNumFilter,
    WordRepeatFilter,
)
from .dead_loop_filter import DeadLoopFilter
from .hard_filter import HardFilter
from .intent_classifier import IntentClassifier
from .llm_backend import LLMBackend, OpenAIBackend, SamplerBackend  # noqa: F401
from .majority_vote import MajorityVoteFilter
from .message_sanity import MessageSanityFilter
from .perplexity import PerplexityFilter
from .refuse_filter import RefuseFilter
from .response_refiner import ResponseRefiner
from .score_filter import ScoreFilter
from .token_soup import TokenSoupFilter

logger = get_logger(only_local_master=False)


class QualityPreprocessor(Preprocessor):
    """Thin pipeline runner: accepts a list of callables, runs them in order.

    Each step must accept and return List[Dict[str, Any]].
    Per-step logging (before/after count) and optional dropped-row JSONL are provided.
    """

    def __init__(self, pipeline: List[Callable], dropped_log_path: str = ''):
        import os
        super().__init__()
        self._pipelines = list(pipeline)
        self._dropped_log_path = dropped_log_path
        self._lock: Optional[PosixFileLock] = (
            PosixFileLock(dropped_log_path + '.lock') if dropped_log_path else None)
        if dropped_log_path and os.path.exists(dropped_log_path):
            os.remove(dropped_log_path)

    def __call__(self, rows):
        rows_list = self.map_col_to_row(rows)
        for step in self._pipelines:
            if not rows_list:
                break
            step_name = getattr(step, '__name__', None) or type(step).__name__
            before = len(rows_list)
            prev = rows_list
            rows_list = self.map_col_to_row(step(rows_list))
            after = len(rows_list)
            logger.info(f'[QualityPreprocessor] {step_name}: {before} -> {after} (dropped {before - after})')
            self._log_dropped(step_name, prev, rows_list)
        return self.map_row_to_col(rows_list)

    def _log_dropped(self, step_name: str, prev: List[Dict[str, Any]],
                     kept: List[Dict[str, Any]]) -> None:
        if not self._lock or len(kept) == len(prev):
            return
        kept_ids = {id(r) for r in kept}
        dropped = [r for r in prev if id(r) not in kept_ids]
        if not dropped:
            return
        with self._lock:
            with open(self._dropped_log_path, 'a', encoding='utf-8') as f:
                for r in dropped:
                    f.write(json.dumps({'step': step_name, 'row': r},
                                       ensure_ascii=False, default=str) + '\n')
