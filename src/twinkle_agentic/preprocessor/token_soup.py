# Copyright (c) ModelScope Contributors. All rights reserved.
"""Token-soup / garbled output detector for assistant messages."""
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from twinkle.preprocessor import Preprocessor
from .utils import msg_content_text

# ── Pre-compiled patterns ─────────────────────────────────────────────────────

_REPLACEMENT_CHAR_RE = re.compile(r'\ufffd')

# Non-printable control chars; \t \n \r kept as legitimate whitespace.
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

_PRIVATE_USE_RE = re.compile(r'[\ue000-\uf8ff\U000f0000-\U000fffff\U00100000-\U0010ffff]')

# Bracket BERT-style tokens stay case-sensitive via (?-i:...) — lowercase "[mask]"/"[pad]"
# collide with bitmask-DP variables like dp[mask].
_SPECIAL_TOKEN_RE = re.compile(
    r'(<\|[^|>\n]{1,40}\|>|</s>|(?-i:\[/?(?:PAD|UNK|SEP|CLS|MASK)\])|</?unk>|</?pad>|<0x[0-9A-Fa-f]{2}>)',
    re.IGNORECASE,
)

# Same printable char repeated 20+ times. Excludes ASCII rule chars, digits, box-drawing,
# block elements, geometric shapes, braille, and dashes.
_SINGLE_CHAR_REPEAT_RE = re.compile(
    r'([^\s\n\-=_.\*\+~#|><0-9\u2013-\u2015\u2500-\u25ff\u2800-\u28ff\u30fc\uff0d])\1{19,}')

# ── Unicode script classifier ─────────────────────────────────────────────────


def _script_of(cp: int) -> str:
    """Map a codepoint to a coarse script bucket."""
    if cp <= 0x024F:
        return 'latin'
    if 0x0370 <= cp <= 0x03FF:
        return 'greek'
    if 0x0400 <= cp <= 0x04FF:
        return 'cyrillic'
    if 0x0590 <= cp <= 0x05FF:
        return 'hebrew'
    if 0x0600 <= cp <= 0x06FF:
        return 'arabic'
    if 0x0900 <= cp <= 0x097F:
        return 'devanagari'
    if 0x0E00 <= cp <= 0x0E7F:
        return 'thai'
    if 0x3040 <= cp <= 0x309F:
        return 'hiragana'
    if 0x30A0 <= cp <= 0x30FF:
        return 'katakana'
    if 0x4E00 <= cp <= 0x9FFF:
        return 'cjk'
    if 0xAC00 <= cp <= 0xD7A3:
        return 'hangul'
    if 0xE000 <= cp <= 0xF8FF:
        return 'private'
    return 'other'


def _script_chaos(text: str, min_chars: int = 40) -> float:
    """Fraction of adjacent letter/number pairs that switch script."""
    chars = [c for c in text if unicodedata.category(c)[0] in ('L', 'N')]
    if len(chars) < min_chars:
        return 0.0
    scripts = [_script_of(ord(c)) for c in chars]
    switches = sum(a != b for a, b in zip(scripts, scripts[1:]))
    return switches / (len(scripts) - 1)


# ── Detector ──────────────────────────────────────────────────────────────────


def _ratio(pattern: re.Pattern, text: str) -> float:
    return len(pattern.findall(text)) / max(len(text), 1)


def _is_token_soup(
    text: str,
    replacement_char_ratio: float = 0.02,
    control_char_ratio: float = 0.01,
    private_use_ratio: float = 0.03,
    special_token_count: int = 20,
    script_chaos_threshold: float = 0.55,
    script_chaos_min_chars: int = 40,
    max_chars: int = 0,
) -> bool:
    """True if `text` exhibits any garbled-output signal."""
    if not text:
        return False
    # Token-soup signals are statistical/uniform; head sample is sufficient.
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    if _ratio(_REPLACEMENT_CHAR_RE, text) > replacement_char_ratio:
        return True
    if _ratio(_CONTROL_CHAR_RE, text) > control_char_ratio:
        return True
    if _ratio(_PRIVATE_USE_RE, text) > private_use_ratio:
        return True
    if len(_SPECIAL_TOKEN_RE.findall(text)) >= special_token_count:
        return True
    if _SINGLE_CHAR_REPEAT_RE.search(text):
        return True
    if _script_chaos(text, script_chaos_min_chars) > script_chaos_threshold:
        return True
    return False


# ── Preprocessor ──────────────────────────────────────────────────────────────


class TokenSoupFilter(Preprocessor):

    def __init__(
        self,
        replacement_char_ratio: float = 0.02,
        control_char_ratio: float = 0.01,
        private_use_ratio: float = 0.03,
        special_token_count: int = 20,
        script_chaos_threshold: float = 0.55,
        script_chaos_min_chars: int = 40,
        max_chars: int = 0,
    ) -> None:
        super().__init__()
        self._cfg = dict(
            replacement_char_ratio=replacement_char_ratio,
            control_char_ratio=control_char_ratio,
            private_use_ratio=private_use_ratio,
            special_token_count=special_token_count,
            script_chaos_threshold=script_chaos_threshold,
            script_chaos_min_chars=script_chaos_min_chars,
            max_chars=max_chars,
        )

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for row in rows:
            asst_texts = [
                msg_content_text(m).strip() for m in (row.get('messages') or [])
                if isinstance(m, dict) and m.get('role') == 'assistant'
            ]
            if any(_is_token_soup(t, **self._cfg) for t in asst_texts):
                dropped.append(dict(row, drop_reason='token_soup'))
            else:
                out.append(row)
        return out, dropped
