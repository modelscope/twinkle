# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import List, Dict, Any

from twinkle.preprocessor import Preprocessor

# ── Thresholds ────────────────────────────────────────────────────────────────

# Hesitation markers per 1000 chars above which the reply is likely stuck
_HESITATION_DENSITY_THRESHOLD = 5.0

# Number of self-correction signals within a sliding window (chars) to flag a cascade
_CASCADE_WINDOW = 800
_CASCADE_THRESHOLD = 5

# Fraction of repeated n-grams above which the reply is considered looping
_REPETITION_THRESHOLD = 0.45
_NGRAM_SIZE = 8        # word n-gram size for repetition check
_NGRAM_MIN_WORDS = 30  # skip check for very short texts

# Relaxed thresholds for <think> sections where hesitation is expected
_THINK_HESITATION_DENSITY_THRESHOLD = 15.0
_THINK_CASCADE_THRESHOLD = 20
_THINK_REPETITION_THRESHOLD = 0.65

# ── Hesitation-marker regexes ─────────────────────────────────────────────────
#
# Matches thinking-aloud / self-interruption signals.
# Each pattern intentionally targets SURFACE FORM, not semantic meaning,
# to avoid false positives on normal explanatory language.

_EN_HESITATE = re.compile(
    r'\b('
    # Direct hesitation tokens
    r'wait[,\s]*\.{2,}|wait[,\s]+(wait|no|actually|hmm|let)|'
    r'no\s+wait|oh\s+wait|but\s+wait|'
    # Thinking aloud with self-doubt
    r'hmm+[,\s]*\.{0,3}|uh+m*[,\s]*\.{0,3}|'
    # Self-correction cascade starters
    r'actually[,\s]+no|actually[,\s]+wait|actually[,\s]+i\s+was|'
    r'no[,\s]+actually[,\s]+(that|this|i)|'
    # Explicit restart / reconsideration  
    r'let\s+me\s+(re-?think|try\s+again|start\s+over|reconsider)|'
    r'i\'?ll\s+(start\s+over|try\s+again|redo\s+this)|'
    # Confusion / disorientation
    r'i\'?m\s+(getting\s+confused|going\s+in\s+circles|lost\s+here|not\s+sure\s+where)|'
    r'this\s+is\s+(getting|becoming)\s+(messy|complicated\s+fast|circular)|'
    # Repeated-mistake acknowledgement
    r'i\s+keep\s+(making|getting)\s+(the\s+same\s+)?error|'
    r'i\s+(made|keep\s+making)\s+(the\s+same\s+)?(mistake|error)\s+again'
    r')\b',
    re.IGNORECASE,
)

_ZH_HESITATE = re.compile(
    r'('
    # Direct hesitation tokens
    r'等等[，,。\s]*\.{0,3}|等一下[，,。]?|哦等等|不不不+|'
    # Note: 哦 is excluded (95%+ sentence-final particle, e.g. "拍拍我哦"); 嗯 requires
    # repetition (single 嗯 is often affirmation, e.g. "嗯，好的").
    r'嗯{2,}[，,。\s]*\.{0,3}|呃+[，,。\s]*\.{0,3}|'
    # Self-correction
    r'不对[，,。]?[，,\s]?(等等|重新|让我)|错了[，,。]?\s*让我|'
    r'让我(重新|再次?)(想|试|来|考虑|计算)|'
    r'我(再|重新)(想想|试试|来一次|考虑)|'
    # Confusion / disorientation
    r'我(越来越|有点)?(搞不清楚?|不确定|迷糊了?|乱了?)|'
    r'这(变得|太|越来越)(复杂|乱|难以?理清)|'
    # Repeated-mistake
    r'我(好像|似乎|又)(搞|弄)错(了)?|我(又犯|再次犯)(了)?错|'
    r'一直(出错|犯错|搞错)'
    r')',
    re.UNICODE,
)

_JA_HESITATE = re.compile(
    r'('
    r'ちょっと待って|待って待って|いや待って|えっと+[、。\s]*\.{0,3}|'
    r'うーん+[、。\s]*\.{0,3}|あれ[、。]?[、。\s]*(また|もう一度)|'
    r'もう一度考え直|やり直し|混乱してきた|わからなくなって'
    r')',
    re.UNICODE,
)

_KO_HESITATE = re.compile(
    r'('
    r'잠깐[,\s]*\.{0,3}|아\s*잠깐|잠깐만요?|'
    r'음+[,\s]*\.{0,3}|어+[,\s]*\.{0,3}|'
    r'다시\s*(생각|시작|해보|해야)|'
    r'헷갈(리기|리네|려서)|'
    r'계속\s*(틀리|실수|잘못)'
    r')',
    re.UNICODE,
)

# Combined list for density scan
_HESITATE_PATTERNS = (_EN_HESITATE, _ZH_HESITATE, _JA_HESITATE, _KO_HESITATE)

# Lightweight per-char cascade pattern (fast scan for dense clusters)
_CASCADE_RE = re.compile(
    r'\b(wait|actually|hmm|no\s+wait|oh\s+wait|let\s+me|'
    r'i\s+was\s+wrong|i\s+made\s+an?\s+(error|mistake))\b|'
    r'(等等|不对|重新|错了|嗯{2,}|让我再)',
    re.IGNORECASE | re.UNICODE,
)


# ── Detection helpers ─────────────────────────────────────────────────────────

def _hesitation_density(text: str) -> float:
    """Count hesitation markers per 1000 chars across all language patterns."""
    count = sum(len(p.findall(text)) for p in _HESITATE_PATTERNS)
    return count / max(len(text), 1) * 1000


def _has_correction_cascade(text: str) -> bool:
    """True if CASCADE_THRESHOLD signals appear within any CASCADE_WINDOW-char span."""
    return _has_correction_cascade_with_threshold(text, _CASCADE_THRESHOLD)


def _has_correction_cascade_with_threshold(text: str, threshold: int) -> bool:
    matches = [m.start() for m in _CASCADE_RE.finditer(text)]
    if len(matches) < threshold:
        return False
    for i in range(len(matches) - threshold + 1):
        if matches[i + threshold - 1] - matches[i] <= _CASCADE_WINDOW:
            return True
    return False


def _high_repetition(text: str) -> bool:
    """True if repeated word n-grams dominate the text (content looping)."""
    return _high_repetition_with_threshold(text, _REPETITION_THRESHOLD)


def _high_repetition_with_threshold(text: str, threshold: float) -> bool:
    words = text.split()
    if len(words) < _NGRAM_MIN_WORDS:
        return False
    ngrams = [' '.join(words[i:i + _NGRAM_SIZE]) for i in range(len(words) - _NGRAM_SIZE + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return (1.0 - unique_ratio) > threshold


def _is_stuck(text: str) -> bool:
    """Return True if the text exhibits signs of a hesitation / dead-loop.

    Uses relaxed thresholds for <think> sections.
    """
    import re as _re
    think_match = _re.search(r'<think>(.*?)</think>', text, _re.DOTALL)
    if think_match:
        think_part = think_match.group(1)
        response_part = text[think_match.end():]
        # Check think part with relaxed thresholds
        think_stuck = (
            _hesitation_density(think_part) > _THINK_HESITATION_DENSITY_THRESHOLD
            or _has_correction_cascade_with_threshold(think_part, _THINK_CASCADE_THRESHOLD)
            or _high_repetition_with_threshold(think_part, _THINK_REPETITION_THRESHOLD)
        )
        # Check response part with normal thresholds
        response_stuck = response_part.strip() and (
            _hesitation_density(response_part) > _HESITATION_DENSITY_THRESHOLD
            or _has_correction_cascade(response_part)
            or _high_repetition(response_part)
        )
        return think_stuck or response_stuck
    return (
        _hesitation_density(text) > _HESITATION_DENSITY_THRESHOLD
        or _has_correction_cascade(text)
        or _high_repetition(text)
    )


# ── Preprocessor ─────────────────────────────────────────────────────────────

class DeadLoopFilter(Preprocessor):

    def __call__(self, rows) -> List[Dict[str, Any]]:
        """Drop rows where the assistant reply shows signs of hesitation or dead-loop.

        Three independent signals, any one of which triggers the filter:
          1. High hesitation-marker density (>5 per 1000 chars)
          2. Self-correction cascade (≥5 signals within an 800-char window)
          3. High n-gram repetition ratio (>45% of 8-grams are duplicates)
        """
        out = []
        for row in rows:
            messages = row.get('messages') or []
            asst_msgs = [
                m for m in messages
                if isinstance(m, dict) and m.get('role') == 'assistant'
            ]
            if not asst_msgs:
                out.append(row)
                continue
            if not any(_is_stuck((m.get('content') or '').strip()) for m in asst_msgs):
                out.append(row)
            else:
                continue
        return out
