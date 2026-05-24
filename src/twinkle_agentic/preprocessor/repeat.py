# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from twinkle.preprocessor import Preprocessor

# ── Thresholds ────────────────────────────────────────────────────────────────

# N-gram sizes: larger = more specific, less likely to false-positive
_N_LATIN = 5   # word-level 5-gram for Latin scripts
_N_CJK   = 4   # char-level 4-gram for CJK (~2 Chinese words per gram)

# Self-repetition: (total_ngrams - unique_ngrams) / total_ngrams
_SELF_REPEAT_THRESHOLD_LATIN = 0.35
_SELF_REPEAT_THRESHOLD_CJK   = 0.45  # CJK char n-grams have more natural overlap

# Instruction copy: |asst_ngrams ∩ user_ngrams| / |asst_ngrams|  (set-based)
_COPY_THRESHOLD = 0.60

# Skip copy check when user message is substantially longer than the response
# (e.g., user provides code and asks to fix it — some overlap is expected)
_COPY_SKIP_USER_RATIO = 1.5

# Minimum token count below which n-gram stats are unreliable
_MIN_TOKENS = 20

# ── CJK detection ─────────────────────────────────────────────────────────────

_CJK_RE = re.compile(
    r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]'
)


def _is_cjk_dominant(text: str) -> bool:
    return len(_CJK_RE.findall(text)) > len(text) * 0.25


# ── Tokenization ───────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Word-level for Latin; character-level (no spaces) for CJK."""
    if _is_cjk_dominant(text):
        return [c for c in text if not c.isspace()]
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def _ngrams(tokens: List[str], n: int) -> List[str]:
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _self_repeat_ratio(text: str) -> Tuple[float, bool]:
    """Return (ratio, is_cjk).

    ratio = (total_ngrams - unique_ngrams) / total_ngrams
    A high ratio means the model regenerated the same phrases multiple times.
    """
    is_cjk = _is_cjk_dominant(text)
    n = _N_CJK if is_cjk else _N_LATIN
    tokens = _tokenize(text)
    if len(tokens) < _MIN_TOKENS:
        return 0.0, is_cjk
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0, is_cjk
    unique = len(set(grams))
    return (len(grams) - unique) / len(grams), is_cjk


def _copy_ratio(user_text: str, asst_text: str) -> float:
    """Return fraction of unique assistant n-grams that also appear in the user message.

    High value means the assistant largely echoed/copy-pasted the user's input.
    Skip if the user message is much longer than the response (e.g. code-fix task).
    """
    if len(user_text) > len(asst_text) * _COPY_SKIP_USER_RATIO:
        return 0.0
    is_cjk = _is_cjk_dominant(asst_text)
    n = _N_CJK if is_cjk else _N_LATIN
    user_tokens = _tokenize(user_text)
    asst_tokens = _tokenize(asst_text)
    if len(asst_tokens) < _MIN_TOKENS:
        return 0.0
    user_gram_set = set(_ngrams(user_tokens, n))
    asst_gram_set = set(_ngrams(asst_tokens, n))
    if not asst_gram_set:
        return 0.0
    overlap = len(asst_gram_set & user_gram_set)
    return overlap / len(asst_gram_set)


def _is_repetitive(user_text: str, asst_text: str) -> bool:
    """Return True if the assistant reply is low-quality due to excessive repetition."""
    sr, is_cjk = _self_repeat_ratio(asst_text)
    threshold = _SELF_REPEAT_THRESHOLD_CJK if is_cjk else _SELF_REPEAT_THRESHOLD_LATIN
    if sr > threshold:
        return True
    if _copy_ratio(user_text, asst_text) > _COPY_THRESHOLD:
        return True
    return False


# ── Preprocessor ─────────────────────────────────────────────────────────────

class RepeatFilter(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.repeat_filter(rows)
        rows = self.map_row_to_col(rows)
        return rows

    def repeat_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop rows where the assistant reply is repetitive or copies the user message.

        Two independent signals:
          1. Self-repetition: (total - unique) n-grams / total > threshold
             — catches the model regenerating the same passage multiple times.
          2. Instruction copy: |asst ∩ user| / |asst| (set n-gram overlap) > threshold
             — catches the model echoing the user's question as its answer.
             Skipped when the user message is ≥1.5× longer than the response
             (legitimate code-correction / rewriting tasks).
        """
        out = []
        for row in rows:
            messages = row.get('messages') or []

            user_msgs = [m for m in messages if isinstance(m, dict) and m.get('role') == 'user']
            asst_msgs = [m for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']

            if not asst_msgs:
                out.append(row)
                continue

            # Concatenate all user turns as the "instruction" context
            user_text = ' '.join((m.get('content') or '') for m in user_msgs).strip()
            asst_text = ' '.join((m.get('content') or '') for m in asst_msgs).strip()

            if not _is_repetitive(user_text, asst_text):
                out.append(row)
        return out
