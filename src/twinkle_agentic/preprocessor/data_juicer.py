# Copyright (c) ModelScope Contributors. All rights reserved.
# Data-Juicer integration for trajectory quality filtering.
#
# Each class below is a standalone Preprocessor with __call__ interface.
# They share a module-level op cache for model/tokenizer reuse.
from typing import Any, Dict, List

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


def _has_tool_calls(row: Dict[str, Any], role: str = 'assistant') -> bool:
    for msg in row.get('messages') or []:
        if msg.get('role') == role and msg.get('tool_calls'):
            return True
    return False


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


class SpecialCharsFilter(Preprocessor):
    def __init__(self, max_ratio: float = 0.25, role: str = 'assistant'):
        self._max_ratio = max_ratio
        self._role = role

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import SpecialCharactersFilter
        op = _get_op(SpecialCharactersFilter, min_ratio=0.0, max_ratio=self._max_ratio)
        texts = [_get_text(r, self._role) for r in rows]
        # Tool-only assistant turns have empty content; ratio is meaningless → exempt.
        exempt = [not t.strip() and _has_tool_calls(r, self._role) for r, t in zip(rows, texts)]
        mask = _keep_mask(op, texts)
        return [r for r, keep, ex in zip(rows, mask, exempt) if ex or keep]


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
