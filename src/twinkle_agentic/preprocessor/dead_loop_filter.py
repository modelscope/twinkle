# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import List, Dict, Any

from twinkle.preprocessor import Preprocessor

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


def _has_correction_cascade_with_threshold(text: str, threshold: int, window: int = 800) -> bool:
    matches = [m.start() for m in _CASCADE_RE.finditer(text)]
    if len(matches) < threshold:
        return False
    for i in range(len(matches) - threshold + 1):
        if matches[i + threshold - 1] - matches[i] <= window:
            return True
    return False


def _high_repetition_with_threshold(text: str, threshold: float, ngram_size: int = 8, ngram_min_words: int = 30) -> bool:
    words = text.split()
    if len(words) < ngram_min_words:
        return False
    ngrams = [' '.join(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return (1.0 - unique_ratio) > threshold


def _is_stuck(
    text: str,
    hesitation_density_threshold: float = 5.0,
    cascade_window: int = 800,
    cascade_threshold: int = 5,
    repetition_threshold: float = 0.45,
    ngram_size: int = 8,
    ngram_min_words: int = 30,
    think_hesitation_density_threshold: float = 15.0,
    think_cascade_threshold: int = 20,
    think_repetition_threshold: float = 0.65,
) -> bool:
    """Return True if the text exhibits signs of a hesitation / dead-loop."""
    import re as _re
    think_match = _re.search(r'<think>(.*?)</think>', text, _re.DOTALL)
    if think_match:
        think_part = think_match.group(1)
        response_part = text[think_match.end():]
        think_stuck = (
            _hesitation_density(think_part) > think_hesitation_density_threshold
            or _has_correction_cascade_with_threshold(think_part, think_cascade_threshold, cascade_window)
            or _high_repetition_with_threshold(think_part, think_repetition_threshold, ngram_size, ngram_min_words)
        )
        response_stuck = response_part.strip() and (
            _hesitation_density(response_part) > hesitation_density_threshold
            or _has_correction_cascade_with_threshold(response_part, cascade_threshold, cascade_window)
            or _high_repetition_with_threshold(response_part, repetition_threshold, ngram_size, ngram_min_words)
        )
        return think_stuck or response_stuck
    return (
        _hesitation_density(text) > hesitation_density_threshold
        or _has_correction_cascade_with_threshold(text, cascade_threshold, cascade_window)
        or _high_repetition_with_threshold(text, repetition_threshold, ngram_size, ngram_min_words)
    )


# ── Preprocessor ─────────────────────────────────────────────────────────────

class DeadLoopFilter(Preprocessor):

    def __init__(
        self,
        hesitation_density_threshold: float = 5.0,
        cascade_window: int = 800,
        cascade_threshold: int = 5,
        repetition_threshold: float = 0.45,
        ngram_size: int = 8,
        ngram_min_words: int = 30,
        think_hesitation_density_threshold: float = 15.0,
        think_cascade_threshold: int = 20,
        think_repetition_threshold: float = 0.65,
    ) -> None:
        super().__init__()
        self._hesitation_density_threshold = hesitation_density_threshold
        self._cascade_window = cascade_window
        self._cascade_threshold = cascade_threshold
        self._repetition_threshold = repetition_threshold
        self._ngram_size = ngram_size
        self._ngram_min_words = ngram_min_words
        self._think_hesitation_density_threshold = think_hesitation_density_threshold
        self._think_cascade_threshold = think_cascade_threshold
        self._think_repetition_threshold = think_repetition_threshold

    def __call__(self, rows) -> List[Dict[str, Any]]:
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
            stuck = any(
                _is_stuck(
                    (m.get('content') or '').strip(),
                    hesitation_density_threshold=self._hesitation_density_threshold,
                    cascade_window=self._cascade_window,
                    cascade_threshold=self._cascade_threshold,
                    repetition_threshold=self._repetition_threshold,
                    ngram_size=self._ngram_size,
                    ngram_min_words=self._ngram_min_words,
                    think_hesitation_density_threshold=self._think_hesitation_density_threshold,
                    think_cascade_threshold=self._think_cascade_threshold,
                    think_repetition_threshold=self._think_repetition_threshold,
                )
                for m in asst_msgs
            )
            if not stuck:
                out.append(row)
        return out
