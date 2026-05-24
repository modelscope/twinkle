# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Dict, List

from twinkle.preprocessor import Preprocessor

# Only inspect the opening window of the first assistant reply;
# refusals almost always appear in the first sentence(s).
_CHECK_WINDOW = 600

# ── English refusal patterns ──────────────────────────────────────────────────
#
# Design principle: require a SELF-REFERENTIAL subject (I/we) + a task-directed
# inability/refusal verb.  This avoids false positives on:
#   "I cannot stress enough…"  "I cannot find the bug…"
#   "The API cannot handle null"  "You cannot use this without auth"

# Core: I/we + modal inability + task verb
_EN_CORE = re.compile(
    r'\b(i|we)\b.{0,25}\b('
    r"can'?t|cannot|am\s+not\s+able|are\s+not\s+able|"
    r"won'?t|will\s+not|am\s+unable|are\s+unable|"
    r"must\s+decline|have\s+to\s+decline|"
    r"decline\s+to|refuse\s+to|"
    r"am\s+not\s+(allowed|permitted|authorized|comfortable)\s+to|"
    r"are\s+not\s+(allowed|permitted|authorized)"
    r")\b.{0,60}\b("
    r'help|assist|answer|respond|provide|generate|create|produce|'
    r'fulfill|comply|address|process|complete|handle|discuss|support'
    r')\b',
    re.IGNORECASE | re.DOTALL,
)

# Apology opener + refusal: "I'm sorry, but I can't…" / "Unfortunately I cannot…"
_EN_APOLOGY = re.compile(
    r'\b(i\'?m\s+sorry|i\s+apologize|unfortunately|i\s+regret)\b.{0,80}'
    r'\b(can\'?t|cannot|unable|won\'?t|will\s+not|must\s+decline|have\s+to\s+decline|'
    r'not\s+(allowed|able|comfortable|appropriate))\b',
    re.IGNORECASE | re.DOTALL,
)

# Policy / content violation signal
_EN_POLICY = re.compile(
    r'\b(this|that|your|the)\s+(request|question|prompt|content|topic|task)\b.{0,60}'
    r'\b(violates?|goes?\s+against|is\s+(inappropriate|not\s+(appropriate|allowed|permitted|'
    r'something\s+i\s+can)))\b',
    re.IGNORECASE | re.DOTALL,
)

# Standalone declarative refusal phrases
_EN_STANDALONE = re.compile(
    r'\b(i|we)\s+(must|have\s+to|am\s+going\s+to|need\s+to)\s+(decline|refuse)\b|'
    r'\b(i|we)\s+(decline|refuse)\s+(this|your|to)\b|'
    r'\bthis\s+(falls\s+outside|is\s+outside|is\s+beyond)\s+(what\s+i|my)\b|'
    r'\bas\s+an\s+ai[,.]?\s+i\s+(can\'?t|cannot|am\s+not\s+able|won\'?t)\b',
    re.IGNORECASE,
)

_EN_PATTERNS = (_EN_CORE, _EN_APOLOGY, _EN_POLICY, _EN_STANDALONE)

# ── Chinese refusal patterns ──────────────────────────────────────────────────

# Apology + inability (高精确：抱歉/对不起 + 无法/不能 near start)
_ZH_APOLOGY = re.compile(
    r'(非常|十分|很|极为)?抱歉[，,。\s]{0,5}.{0,40}(无法|不能|不可以|不便|没有办法)|'
    r'对不起[，,。\s]{0,5}.{0,40}(无法|不能|不可以|不便)',
    re.UNICODE,
)

# Self-referential: 我 + refusal + task object
_ZH_SELF = re.compile(
    r'我(无法|不能|不可以|没有办法|不便|不适合|不被允许|不被授权)'
    r'.{0,30}(帮|回答|提供|生成|处理|协助|完成|执行|回复|解答|协|帮助)',
    re.UNICODE,
)

# Request-level violation
_ZH_VIOLATION = re.compile(
    r'(您的|这个|该)(请求|问题|内容|话题).{0,20}(违反|不当|不合适|超出了?我)',
    re.UNICODE,
)

# AI identity + refusal
_ZH_AI_ID = re.compile(
    r'作为(AI|人工智能|语言模型|大模型)[，,].{0,30}(无法|不能|不便|不应该|不适合)',
    re.UNICODE,
)

_ZH_PATTERNS = (_ZH_APOLOGY, _ZH_SELF, _ZH_VIOLATION, _ZH_AI_ID)

# ── Japanese refusal patterns ─────────────────────────────────────────────────

_JA_PATTERNS = (
    re.compile(r'(申し訳|恐れ入り)ます(が|けれど).{0,40}(できません|お答えできません|対応できません)', re.UNICODE),
    re.compile(r'(回答|対応|お答え)(する|いたす)ことは?できません', re.UNICODE),
    re.compile(r'ご要望には?お(応え|答え)できません', re.UNICODE),
    re.compile(r'(その|この)(リクエスト|質問|依頼).{0,20}(お断り|辞退|対応できません)', re.UNICODE),
)

# ── Korean refusal patterns ───────────────────────────────────────────────────

_KO_PATTERNS = (
    re.compile(r'(죄송하지만|유감스럽게도).{0,40}(드릴 수 없|없습니다|못합니다)', re.UNICODE),
    re.compile(r'(답변|도움|처리|제공)(드리기|하기)\s*(어렵|불가|할 수 없)', re.UNICODE),
    re.compile(r'(요청|질문|내용).{0,20}(거절|거부|응할 수 없)', re.UNICODE),
)

_ALL_PATTERNS = _EN_PATTERNS + _ZH_PATTERNS + _JA_PATTERNS + _KO_PATTERNS


# ── Core helper ───────────────────────────────────────────────────────────────

def _is_refusal(text: str) -> bool:
    """Return True if the text contains a self-referential refusal signal."""
    window = text[:_CHECK_WINDOW]
    return any(p.search(window) for p in _ALL_PATTERNS)


# ── Preprocessor ─────────────────────────────────────────────────────────────

class RefuseFilter(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.refuse_filter(rows)
        rows = self.map_row_to_col(rows)
        return rows

    def refuse_filter(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop rows where the first assistant reply expresses a refusal or inability."""
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
            first_reply = (asst_msgs[0].get('content') or '').strip()
            if not _is_refusal(first_reply):
                out.append(row)
        return out
