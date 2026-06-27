# Copyright (c) ModelScope Contributors. All rights reserved.
"""Drop assistant messages that exhibit hesitation / dead-loop patterns."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from twinkle.preprocessor import Preprocessor
from .utils import cjk_ratio, is_agent_row, msg_content_text

# ── Hesitation-marker regexes ─────────────────────────────────────────────────
#
# Match SURFACE FORM, not semantic meaning, to avoid false positives on normal
# explanatory language.

_EN_HESITATE = re.compile(
    r'\b('
    r'wait[,\s]*\.{2,}|wait[,\s]+(wait|no|actually|hmm|let)|'
    r'no\s+wait|oh\s+wait|but\s+wait|'
    r'hmm+[,\s]*\.{0,3}|uh+m*[,\s]*\.{0,3}|'
    r'actually[,\s]+no|actually[,\s]+wait|actually[,\s]+i\s+was|'
    r'no[,\s]+actually[,\s]+(that|this|i)|'
    r'let\s+me\s+(re-?think|try\s+again|start\s+over|reconsider)|'
    r'i\'?ll\s+(start\s+over|try\s+again|redo\s+this)|'
    r'i\'?m\s+(getting\s+confused|going\s+in\s+circles|lost\s+here|not\s+sure\s+where)|'
    r'this\s+is\s+(getting|becoming)\s+(messy|complicated\s+fast|circular)|'
    r'i\s+keep\s+(making|getting)\s+(the\s+same\s+)?error|'
    r'i\s+(made|keep\s+making)\s+(the\s+same\s+)?(mistake|error)\s+again'
    r')\b',
    re.IGNORECASE,
)

# '哦' excluded (95%+ sentence-final particle, e.g. "拍拍我哦"); single '嗯' excluded
# (often affirmation, e.g. "嗯，好的") — only repeated '嗯{2,}' counts.
# '等一下' excluded — overwhelmingly polite '稍等一下', not self-hesitation.
_ZH_HESITATE = re.compile(
    r'('
    r'等等[，,。\s]*\.{0,3}|哦等等|不不不+|'
    r'嗯{2,}[，,。\s]*\.{0,3}|呃+[，,。\s]*\.{0,3}|'
    r'不对[，,。]?[，,\s]?(等等|重新|让我)|错了[，,。]?\s*让我|'
    r'让我(重新|再次?)(想|试|来|考虑|计算)|'
    r'我(再|重新)(想想|试试|来一次|考虑)|'
    r'我(越来越|有点)?(搞不清楚?|不确定|迷糊了?|乱了?)|'
    r'这(变得|太|越来越)(复杂|乱|难以?理清)|'
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

_HESITATE_PATTERNS = (_EN_HESITATE, _ZH_HESITATE, _JA_HESITATE, _KO_HESITATE)

# 'let me' deliberately excluded — canonical agent-prelude phrasing
# ("Let me read the file...") and would over-fire on long agent trajectories.
_CASCADE_RE = re.compile(
    r'\b(wait|actually|hmm|no\s+wait|oh\s+wait|'
    r'i\s+was\s+wrong|i\s+made\s+an?\s+(error|mistake))\b|'
    r'(等等|不对|重新|错了|嗯{2,}|让我再)',
    re.IGNORECASE | re.UNICODE,
)

# Cover both `<think>` and `<thinking>` block forms.
_THINK_BLOCK_RE = re.compile(r'<think(?:ing)?>(.*?)</think(?:ing)?>', re.DOTALL | re.IGNORECASE)

# ── Detection helpers ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _StuckThresholds:
    hesitation_density: float
    cascade_threshold: int
    cascade_window: int
    repetition_threshold: float
    ngram_size: int
    ngram_min_words: int


def _hesitation_density(text: str) -> float:
    """Hesitation markers per 1000 chars across all language patterns."""
    count = sum(len(p.findall(text)) for p in _HESITATE_PATTERNS)
    return count / max(len(text), 1) * 1000


def _has_correction_cascade(text: str, threshold: int, window: int) -> bool:
    """True iff ``threshold`` cascade markers fall within any ``window``-char span."""
    starts = [m.start() for m in _CASCADE_RE.finditer(text)]
    if len(starts) < threshold:
        return False
    return any(starts[i + threshold - 1] - starts[i] <= window for i in range(len(starts) - threshold + 1))


def _high_repetition(text: str, threshold: float, ngram_size: int, ngram_min_words: int) -> bool:
    if cjk_ratio(text[:500]) > 0.3:
        tokens = list(text.replace(' ', '').replace('\n', ''))
        min_tokens = ngram_min_words * ngram_size
    else:
        tokens = text.split()
        min_tokens = ngram_min_words
    if len(tokens) < min_tokens:
        return False
    ngrams = [tuple(tokens[i:i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]
    if not ngrams:
        return False
    return (1.0 - len(set(ngrams)) / len(ngrams)) > threshold


def _is_segment_stuck(text: str, t: _StuckThresholds) -> bool:
    if not text:
        return False
    return (_hesitation_density(text) > t.hesitation_density
            or _has_correction_cascade(text, t.cascade_threshold, t.cascade_window)
            or _high_repetition(text, t.repetition_threshold, t.ngram_size, t.ngram_min_words))


def _split_think(text: str) -> Tuple[str, str]:
    """Return (think_block_inner, post_think_response). Pre-think text is treated as response."""
    m = _THINK_BLOCK_RE.search(text)
    if not m:
        return '', text
    return m.group(1), text[m.end():]


# ── Preprocessor ─────────────────────────────────────────────────────────────


class DeadLoopFilter(Preprocessor):

    def __init__(
        self,
        hesitation_density_threshold: float = 7.0,
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
        # Two threshold profiles: laxer inside <think> reasoning (free to ramble),
        # stricter on the visible response.
        self._response_th = _StuckThresholds(
            hesitation_density=hesitation_density_threshold,
            cascade_threshold=cascade_threshold,
            cascade_window=cascade_window,
            repetition_threshold=repetition_threshold,
            ngram_size=ngram_size,
            ngram_min_words=ngram_min_words,
        )
        self._think_th = _StuckThresholds(
            hesitation_density=think_hesitation_density_threshold,
            cascade_threshold=think_cascade_threshold,
            cascade_window=cascade_window,
            repetition_threshold=think_repetition_threshold,
            ngram_size=ngram_size,
            ngram_min_words=ngram_min_words,
        )

    def _is_stuck(self, text: str, reasoning: str = '') -> bool:
        think_part, response_part = _split_think(text)
        if reasoning and not think_part:
            think_part = reasoning
        return (_is_segment_stuck(think_part, self._think_th)
                or _is_segment_stuck(response_part.strip(), self._response_th))

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for row in rows:
            messages = row.get('messages') or []
            if is_agent_row(messages):
                out.append(row)
                continue
            asst_msgs = [m for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']
            if not asst_msgs:
                out.append(row)
                continue
            if any(
                    self._is_stuck(
                        msg_content_text(m).strip(),
                        (m.get('reasoning_content') or m.get('thinking') or '').strip(),
                    ) for m in asst_msgs):
                dropped.append(dict(row, drop_reason='dead_loop'))
            else:
                out.append(row)
        return out, dropped
