# Copyright (c) ModelScope Contributors. All rights reserved.
"""Data-Juicer integration for trajectory quality filtering.

Each class is a standalone Preprocessor with __call__ interface; they share a
module-level op cache for model/tokenizer reuse.
"""
from typing import Any, Dict, List, Tuple

from twinkle.preprocessor import Preprocessor
from .utils import msg_content_text

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
    """Concatenate text-projected content of all turns matching `role`."""
    return ' '.join(
        msg_content_text(msg) for msg in (row.get('messages') or [])
        if isinstance(msg, dict) and msg.get('role') == role)


def _keep_mask(op, texts: List[str]) -> List[bool]:
    """Run a DJ Filter op directly; no dataset/multiprocessing overhead."""
    from data_juicer.utils.constant import Fields
    samples = {op.text_key: texts, Fields.stats: [{} for _ in texts], Fields.meta: [{} for _ in texts]}
    samples = op.compute_stats_batched(samples)
    return list(op.process_batched(samples))


def _apply_mapper(op, rows: List[Dict[str, Any]], role: str) -> None:
    """Run a DJ Mapper on string-content messages of `role`. Non-string content is preserved verbatim."""
    indices: List[Tuple[int, int]] = []
    texts: List[str] = []
    for ri, row in enumerate(rows):
        for mi, msg in enumerate(row.get('messages') or []):
            if not isinstance(msg, dict) or msg.get('role') != role:
                continue
            content = msg.get('content')
            # Skip multimodal/None content — mapper only mutates plain string turns.
            if not isinstance(content, str):
                continue
            indices.append((ri, mi))
            texts.append(content)
    if not texts:
        return
    result = op.process_batched({op.text_key: texts})
    for (ri, mi), new_text in zip(indices, result[op.text_key]):
        rows[ri]['messages'][mi]['content'] = new_text


# ── Wrapper classes ───────────────────────────────────────────────────────────


class FixUnicodeFilter(Preprocessor):

    def __init__(self, normalization: str = 'NFC', role: str = 'assistant'):
        super().__init__()
        self._normalization = normalization
        self._role = role

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.mapper import FixUnicodeMapper
        _apply_mapper(_get_op(FixUnicodeMapper, normalization=self._normalization), rows, self._role)
        return rows, []


class RemoveRepeatSentencesFilter(Preprocessor):

    def __init__(self, lowercase: bool = False, ignore_special_character: bool = True, role: str = 'assistant'):
        super().__init__()
        self._lowercase = lowercase
        self._ignore = ignore_special_character
        self._role = role

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.mapper import RemoveRepeatSentencesMapper
        op = _get_op(RemoveRepeatSentencesMapper, lowercase=self._lowercase, ignore_special_character=self._ignore)
        _apply_mapper(op, rows, self._role)
        return rows, []


class SpecialCharsFilter(Preprocessor):

    def __init__(self, max_ratio: float = 0.25, role: str = 'assistant'):
        super().__init__()
        self._max_ratio = max_ratio
        self._role = role

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        from data_juicer.ops.filter import SpecialCharactersFilter
        op = _get_op(SpecialCharactersFilter, min_ratio=0.0, max_ratio=self._max_ratio)
        texts = [_get_text(r, self._role) for r in rows]
        # Filter only non-empty text; empty-text rows (e.g. tool-only assistants) are kept verbatim.
        non_empty = [i for i, t in enumerate(texts) if t.strip()]
        keep = [True] * len(rows)
        if non_empty:
            sub = _keep_mask(op, [texts[i] for i in non_empty])
            for i, m in zip(non_empty, sub):
                keep[i] = bool(m)
        out: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for r, k in zip(rows, keep):
            if k:
                out.append(r)
            else:
                dropped.append(dict(r, drop_reason='special_chars_ratio'))
        return out, dropped


class TokenNumFilter(Preprocessor):

    def __init__(self,
                 hf_tokenizer: str = 'Qwen/Qwen2.5-0.5B',
                 min_num: int = 10,
                 max_num: int = 8192,
                 role: str = 'assistant'):
        super().__init__()
        self._hf_tokenizer = hf_tokenizer
        self._min_num = min_num
        self._max_num = max_num
        self._role = role

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        tokenizer = _get_tokenizer(self._hf_tokenizer)
        texts = [_get_text(r, self._role) for r in rows]
        encoded = tokenizer(texts, add_special_tokens=False)
        out: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for r, ids in zip(rows, encoded['input_ids']):
            if self._min_num <= len(ids) <= self._max_num:
                out.append(r)
            else:
                dropped.append(dict(r, drop_reason='token_count_out_of_range'))
        return out, dropped
