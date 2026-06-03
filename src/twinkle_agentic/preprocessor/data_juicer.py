# Copyright (c) ModelScope Contributors. All rights reserved.
# Data-Juicer integration for trajectory quality filtering.
#
# Each class below is a standalone Preprocessor with __call__ interface.
# They share a module-level op cache for model/tokenizer reuse.
from typing import Any, Dict, List, Optional, Union

from twinkle.preprocessor import Preprocessor


# ── Shared helpers ────────────────────────────────────────────────────────────

_OP_CACHE: Dict = {}


def _get_op(op_class, **kwargs):
    key = (op_class, repr(tuple(sorted(kwargs.items()))))
    if key not in _OP_CACHE:
        _OP_CACHE[key] = op_class(**kwargs)
    return _OP_CACHE[key]


def _get_tokenizer(hf_tokenizer: str):
    key = ('_tokenizer', hf_tokenizer)
    if key not in _OP_CACHE:
        from modelscope import AutoTokenizer
        _OP_CACHE[key] = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
    return _OP_CACHE[key]


def _get_text(row: Dict[str, Any], role: str = 'assistant') -> str:
    """Concatenate all turns for a given role from messages."""
    parts = []
    for msg in row.get('messages') or []:
        if msg.get('role') == role:
            content = msg.get('content') or ''
            if isinstance(content, list):
                content = ' '.join(b.get('text', '') for b in content if isinstance(b, dict))
            parts.append(str(content))
    return ' '.join(parts)


def _keep_mask(op, texts: List[str]) -> List[bool]:
    """Run a DJ Filter op directly; no dataset/multiprocessing overhead."""
    from data_juicer.utils.constant import Fields
    samples = {op.text_key: texts, Fields.stats: [{} for _ in texts], Fields.meta: [{} for _ in texts]}
    samples = op.compute_stats_batched(samples)
    return list(op.process_batched(samples))


# ── Wrapper classes ───────────────────────────────────────────────────────────


class FixUnicodeFilter(Preprocessor):
    def __init__(self, normalization: str = 'NFC', role: str = 'assistant'):
        self._normalization = normalization
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.mapper import FixUnicodeMapper
        op = _get_op(FixUnicodeMapper, normalization=self._normalization)
        indices, texts = [], []
        for ri, row in enumerate(rows):
            for mi, msg in enumerate(row.get('messages') or []):
                if msg.get('role') == self._role:
                    texts.append(msg.get('content') or '')
                    indices.append((ri, mi))
        if not texts:
            return rows
        result = op.process_batched({op.text_key: list(texts)})
        for (ri, mi), new_text in zip(indices, result[op.text_key]):
            rows[ri]['messages'][mi]['content'] = new_text
        return rows


class RemoveRepeatSentencesFilter(Preprocessor):
    def __init__(self, lowercase: bool = False, ignore_special_character: bool = True, role: str = 'assistant'):
        self._lowercase = lowercase
        self._ignore = ignore_special_character
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.mapper import RemoveRepeatSentencesMapper
        op = _get_op(RemoveRepeatSentencesMapper, lowercase=self._lowercase, ignore_special_character=self._ignore)
        indices, texts = [], []
        for ri, row in enumerate(rows):
            for mi, msg in enumerate(row.get('messages') or []):
                if msg.get('role') == self._role:
                    texts.append(msg.get('content') or '')
                    indices.append((ri, mi))
        if not texts:
            return rows
        result = op.process_batched({op.text_key: list(texts)})
        for (ri, mi), new_text in zip(indices, result[op.text_key]):
            rows[ri]['messages'][mi]['content'] = new_text
        return rows


class WordRepeatFilter(Preprocessor):
    def __init__(self, rep_len: int = 10, max_ratio: float = 0.4, role: str = 'assistant'):
        self._rep_len = rep_len
        self._max_ratio = max_ratio
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import WordRepetitionFilter
        op = _get_op(WordRepetitionFilter, rep_len=self._rep_len, min_ratio=0.0, max_ratio=self._max_ratio)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class CharRepeatFilter(Preprocessor):
    def __init__(self, rep_len: int = 10, max_ratio: float = 0.4, role: str = 'assistant'):
        self._rep_len = rep_len
        self._max_ratio = max_ratio
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import CharacterRepetitionFilter
        op = _get_op(CharacterRepetitionFilter, rep_len=self._rep_len, min_ratio=0.0, max_ratio=self._max_ratio)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class SpecialCharsFilter(Preprocessor):
    def __init__(self, max_ratio: float = 0.25, role: str = 'assistant'):
        self._max_ratio = max_ratio
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import SpecialCharactersFilter
        op = _get_op(SpecialCharactersFilter, min_ratio=0.0, max_ratio=self._max_ratio)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class AlphanumericFilter(Preprocessor):
    def __init__(self, min_ratio: float = 0.25, role: str = 'assistant'):
        self._min_ratio = min_ratio
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import AlphanumericFilter as DJAlphanumericFilter
        op = _get_op(DJAlphanumericFilter, tokenization=False, min_ratio=self._min_ratio)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class TokenNumFilter(Preprocessor):
    def __init__(self, hf_tokenizer: str = 'Qwen/Qwen2.5-0.5B', min_num: int = 10, max_num: int = 8192, role: str = 'assistant'):
        self._hf_tokenizer = hf_tokenizer
        self._min_num = min_num
        self._max_num = max_num
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        tokenizer = _get_tokenizer(self._hf_tokenizer)
        texts = [_get_text(r, self._role) for r in rows]
        encoded = tokenizer(texts, add_special_tokens=False)
        return [r for r, ids in zip(rows, encoded['input_ids']) if self._min_num <= len(ids) <= self._max_num]


class TextActionFilter(Preprocessor):
    def __init__(self, lang: str = 'en', min_action_num: int = 1, role: str = 'assistant'):
        self._lang = lang
        self._min_action_num = min_action_num
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import TextActionFilter as DJTextActionFilter
        op = _get_op(DJTextActionFilter, lang=self._lang, min_action_num=self._min_action_num)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class StopwordsFilter(Preprocessor):
    def __init__(self, lang: str = 'en', min_ratio: float = 0.1, max_ratio: float = 1.0, role: str = 'assistant'):
        self._lang = lang
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import StopWordsFilter
        op = _get_op(StopWordsFilter, lang=self._lang, min_ratio=self._min_ratio, max_ratio=self._max_ratio)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class FlaggedWordsFilter(Preprocessor):
    def __init__(self, lang: str = 'en', max_ratio: float = 0.045, role: str = 'assistant'):
        self._lang = lang
        self._max_ratio = max_ratio
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import FlaggedWordFilter
        op = _get_op(FlaggedWordFilter, lang=self._lang, min_ratio=0.0, max_ratio=self._max_ratio)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class LanguageFilter(Preprocessor):
    def __init__(self, lang: Union[str, List[str]] = '', min_score: float = 0.7, role: str = 'assistant'):
        self._lang = lang
        self._min_score = min_score
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import LanguageIDScoreFilter
        op = _get_op(LanguageIDScoreFilter, lang=self._lang, min_score=self._min_score)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class KenLMFilter(Preprocessor):
    def __init__(self, lang: str = 'en', min_ppl: float = 0, max_ppl: float = 1500, role: str = 'assistant'):
        self._lang = lang
        self._min_ppl = min_ppl
        self._max_ppl = max_ppl
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import PerplexityFilter as KenLMPPLFilter
        op = _get_op(KenLMPPLFilter, lang=self._lang, min_ppl=self._min_ppl, max_ppl=self._max_ppl)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class MinHashDedupFilter(Preprocessor):
    def __init__(self, tokenization: str = 'character', window_size: int = 5, num_permutations: int = 256, jaccard_threshold: float = 0.7, role: str = 'assistant'):
        self._tokenization = tokenization
        self._window_size = window_size
        self._num_permutations = num_permutations
        self._jaccard_threshold = jaccard_threshold
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.deduplicator import DocumentMinhashDeduplicator
        from data_juicer.core.data import NestedDataset
        from data_juicer.utils.constant import Fields
        import datasets

        texts = [_get_text(r, self._role) for r in rows]
        ds = datasets.Dataset.from_dict({'text': texts})
        ds = ds.map(lambda x: {Fields.stats: {}, Fields.meta: {}}, batched=False)
        nd = NestedDataset(ds)

        op = _get_op(DocumentMinhashDeduplicator,
            tokenization=self._tokenization,
            window_size=self._window_size,
            num_permutations=self._num_permutations,
            jaccard_threshold=self._jaccard_threshold,
        )
        nd = op.run(nd)
        keep_texts = set(nd['text'])
        seen, result = set(), []
        for r, t in zip(rows, texts):
            if t in keep_texts and t not in seen:
                seen.add(t)
                result.append(r)
        return result


class LLMQualityFilter(Preprocessor):
    def __init__(self, api_endpoint: str, model: str = 'default', min_score: float = 0.5, role: str = 'assistant'):
        self._api_endpoint = api_endpoint
        self._model = model
        self._min_score = min_score
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import LLMQualityScoreFilter
        op = _get_op(LLMQualityScoreFilter, api_or_hf_model=self._model, api_endpoint=self._api_endpoint, min_score=self._min_score)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class LLMDifficultyFilter(Preprocessor):
    def __init__(self, api_endpoint: str, model: str = 'default', min_score: float = 0.4, max_score: float = 1.0, role: str = 'user'):
        self._api_endpoint = api_endpoint
        self._model = model
        self._min_score = min_score
        self._max_score = max_score
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import LLMDifficultyScoreFilter
        op = _get_op(LLMDifficultyScoreFilter, api_or_hf_model=self._model, api_endpoint=self._api_endpoint, min_score=self._min_score, max_score=self._max_score)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class LLMConditionFilter(Preprocessor):
    def __init__(self, condition: str, api_endpoint: str, model: str = 'default', role: str = 'assistant'):
        self._condition = condition
        self._api_endpoint = api_endpoint
        self._model = model
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import LLMConditionFilter as DJLLMConditionFilter
        op = _get_op(DJLLMConditionFilter, condition=self._condition, api_or_hf_model=self._model, api_endpoint=self._api_endpoint)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]


class LLMTaskRelevanceFilter(Preprocessor):
    def __init__(self, api_endpoint: str, task_desc: str = '', model: str = 'default', min_score: float = 0.5, role: str = 'assistant'):
        self._api_endpoint = api_endpoint
        self._task_desc = task_desc
        self._model = model
        self._min_score = min_score
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import LLMTaskRelevanceFilter as DJLLMTaskRelevanceFilter
        op = _get_op(DJLLMTaskRelevanceFilter, api_or_hf_model=self._model, api_endpoint=self._api_endpoint, min_score=self._min_score, valid_dataset=None, task_desc=self._task_desc)
        texts = [_get_text(r, self._role) for r in rows]
        mask = _keep_mask(op, texts)
        return [r for r, keep in zip(rows, mask) if keep]
