# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shape of a piece of text, and word-list matching against it.

These take plain strings, not messages: which script a string is written in, and
whether it hits a banned-word list. Both questions come up wherever text arrives
from a model or a dataset -- filtering a corpus, deciding a reply's language,
refusing to train on something -- so they do not belong to any one of those.

The CJK class covers Han, Hiragana, Katakana and Hangul, which is what callers
mean by "CJK" here even though Korean is not Chinese-Japanese.
"""
import os
import re
from typing import Optional, Set

__all__ = ['CJK_CHARS_RE', 'build_sensitive_regex', 'cjk_ratio', 'load_sensitive_words']

CJK_CHARS_RE = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]')


def cjk_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are CJK."""
    chars = text.replace(' ', '').replace('\n', '').replace('\t', '')
    if not chars:
        return 0.0
    return len(CJK_CHARS_RE.findall(chars)) / len(chars)


def load_sensitive_words(path: Optional[str]) -> Set[str]:
    """Load from external file (one word per line). Blank lines and #-comments ignored."""
    if not path or not os.path.isfile(path):
        return set()
    words: Set[str] = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                words.add(line)
    return words


def build_sensitive_regex(words: Set[str]) -> Optional['re.Pattern']:
    """Build a compiled regex from a set of words. Returns None if empty.

    Latin words get word boundaries, CJK ones cannot: there is no ``\\b`` between
    two Han characters, so a boundary there would never match.
    """
    if not words:
        return None
    cjk_words = []
    latin_words = []
    for w in sorted(words):
        if CJK_CHARS_RE.search(w):
            cjk_words.append(re.escape(w))
        else:
            latin_words.append(re.escape(w))
    parts = []
    if latin_words:
        parts.append(r'\b(' + '|'.join(latin_words) + r')\b')
    if cjk_words:
        parts.append('(' + '|'.join(cjk_words) + ')')
    return re.compile('|'.join(parts), re.IGNORECASE)
