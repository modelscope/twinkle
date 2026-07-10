# Copyright (c) ModelScope Contributors. All rights reserved.
"""Language-identification filter (AUDIT D4).

Keeps only rows whose user-facing language is in an allow-list. Uses ``langid``
when installed (proper LID over 97 languages); otherwise degrades gracefully to
a script-ratio heuristic (CJK vs Latin) so the filter is usable with zero extra
dependencies — just coarser. This complements the existing ``cjk_ratio`` checks
in :class:`HardFilter`, which only measure script mix, not language.

The language is judged from the concatenated user turns (the request defines the
expected response language; assistant text can legitimately quote other
languages, e.g. code or translations).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

from twinkle.preprocessor import Filter
from twinkle.utils import get_logger

from .message_utils import cjk_ratio, msg_content_text

logger = get_logger()

# Injected scaffolding that is NOT the user's own request and would skew language
# detection (usually English system boilerplate wrapping a non-English query, or
# vice versa). Stripped before LID so we judge the real user text.
_INJECTION_BLOCK_RE = re.compile(
    r'<(system-reminder|system_reminder|system|instructions?|context|'
    r'important_instructions|env|environment|tools?)\b[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)
# Self-closing / unmatched openers of the same tags (defensive).
_INJECTION_TAG_RE = re.compile(
    r'</?(system-reminder|system_reminder|system|instructions?|context|'
    r'important_instructions|env|environment|tools?)\b[^>]*/?>',
    re.IGNORECASE,
)


def _strip_injections(text: str) -> str:
    """Remove injected system-scaffolding blocks so LID sees the real user text."""
    text = _INJECTION_BLOCK_RE.sub(' ', text)
    text = _INJECTION_TAG_RE.sub(' ', text)
    return text.strip()


class LanguageFilter(Filter):
    """Keep rows whose detected user language is allowed.

    Args:
        allowed: allowed ISO 639-1 codes (e.g. ``('en', 'zh')``).
        min_chars: skip detection (keep) for user text shorter than this — LID is
            unreliable on very short strings.
        cjk_threshold: fallback heuristic boundary; user text with CJK ratio above
            this is treated as ``zh``, else ``en``. Only used when ``langid`` is absent.
        keep_undetected: keep rows where language can't be determined. Default True
            (fail-open) so the filter never silently deletes ambiguous data.
    """

    def __init__(
        self,
        allowed: Sequence[str] = ('en', 'zh'),
        *,
        min_chars: int = 20,
        cjk_threshold: float = 0.15,
        keep_undetected: bool = True,
    ):
        self.allowed = {a.lower() for a in allowed}
        self.min_chars = int(min_chars)
        self.cjk_threshold = float(cjk_threshold)
        self.keep_undetected = bool(keep_undetected)
        self._identifier = self._load_langid()
        if self._identifier is None:
            logger.info('[LanguageFilter] langid not installed; using CJK/Latin script heuristic.')

    @staticmethod
    def _load_langid():
        try:
            from langid.langid import LanguageIdentifier, model
            return LanguageIdentifier.from_modelstring(model, norm_probs=True)
        except Exception:
            return None

    def _user_text(self, row: Dict[str, Any]) -> str:
        messages = row.get('messages') or []
        parts = [_strip_injections(msg_content_text(m)) for m in messages
                 if isinstance(m, dict) and m.get('role') == 'user']
        return '\n'.join(p for p in parts if p).strip()

    def _detect(self, text: str) -> Optional[str]:
        if self._identifier is not None:
            try:
                lang, _prob = self._identifier.classify(text)
                return lang
            except Exception:
                return None
        # heuristic fallback: CJK ratio -> zh, else en
        return 'zh' if cjk_ratio(text) > self.cjk_threshold else 'en'

    def keep(self, row: Dict[str, Any]) -> bool:
        text = self._user_text(row)
        if len(text) < self.min_chars:
            return True  # too short to judge reliably
        lang = self._detect(text)
        if lang is None:
            return self.keep_undetected
        return lang.lower() in self.allowed

    def drop_reason(self, row: Dict[str, Any]) -> str:
        text = self._user_text(row)
        lang = self._detect(text) if len(text) >= self.min_chars else None
        return f'language_{lang or "undetected"}'
