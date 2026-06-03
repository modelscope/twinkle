# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Dict, List

from twinkle.preprocessor import Preprocessor

# ── Language detection ────────────────────────────────────────────────────────

_CJK_RE = re.compile(
    r'[\u4e00-\u9fff'        # CJK Unified Ideographs (Chinese)
    r'\u3040-\u309f'         # Hiragana
    r'\u30a0-\u30ff'         # Katakana
    r'\uac00-\ud7a3]',       # Hangul Syllables
    re.UNICODE,
)


def _cjk_ratio(text: str) -> float:
    return len(_CJK_RE.findall(text)) / max(len(text), 1)


# ── English simple-query patterns ─────────────────────────────────────────────

_EN_GREETING_RE = re.compile(
    r'^(h+e+l+l+o+|h+i+|hey+|yo+|howdy|greetings|'
    r'good\s+(morning|afternoon|evening|night|day)|'
    r'what\'?s\s+up|how\'?s\s+it\s+going|how\s+are\s+you)'
    r'[\s,!.?]*$',
    re.IGNORECASE,
)

_EN_SIMPLE_RE = re.compile(
    r'^('
    # bare wh-question: interrogative word + short tail
    r'(what|who|where|when|why|how)\s+(is|are|was|were|does|do|did|has|have|can|could|would|should)\b.{0,30}|'
    r'(what|who|where|when|why|how)\'s\b.{0,30}|'
    # polar question opener
    r'(is|are|was|were|do|does|did|can|could|would|should|may|might)\s+(it|this|that|you|there|they|he|she)\b.{0,30}|'
    # imperative with no body
    r'(tell\s+me(\s+(about|more))?|explain(\s+to\s+me)?|define|describe|list|summarize|give\s+me)\b.{0,20}|'
    # help-me opener (no task detail)
    r'(please\s+)?(help\s+me|assist\s+me)\b.{0,20}'
    r')\s*[?!.]?$',
    re.IGNORECASE | re.DOTALL,
)

# ── Chinese simple-query patterns ─────────────────────────────────────────────

_ZH_GREETING_RE = re.compile(
    r'^(你好+|您好+|早上好|下午好|晚上好|大家好|嗨+|哈+喽+|哈+|喂+|hello+|hi+)'
    r'[\s,，！!。.]*$',
    re.UNICODE,
)

_ZH_SIMPLE_RE = re.compile(
    r'^('
    # "X是什么" / "什么是X" / "X怎么样"
    r'.{0,7}(是什么|是啥|啥意思|是何|什么意思|怎么样|如何|为什么|为啥)[？?。]?|'
    r'(什么|啥|哪|谁|何|怎么|怎样|为什么|为啥|几|多少|何时|何地).{0,7}[？?。]?|'
    # single-verb imperative with no substantive object
    r'(介绍|解释|说明|告诉我|帮我说说|请问|能说说|讲讲).{0,5}|'
    # short open-ended knowledge prompt with no substantive body
    r'(请\s*(给出|介绍|解释|说明|提供|列举|讲讲|阐述|描述|概述|举例|分析|说一下)|能否\s*(给出|设计|提供|介绍|解释|说明)).{0,10}'
    r')\s*[？?！!。]?$',
    re.UNICODE,
)

# ── Japanese simple-query patterns ────────────────────────────────────────────

_JA_GREETING_RE = re.compile(
    r'^(こんにちは+|こんばんは+|おはよう(ございます)?|やあ+|どうも+|はじめまして|よろしく(おねがいします)?)'
    r'[\s！!。.]*$',
    re.UNICODE,
)

_JA_SIMPLE_RE = re.compile(
    r'^('
    r'.{0,7}(とは何ですか|って何|とはなんですか|について教えて(ください)?|はどうですか|ですか)[？?]?|'
    r'(何|なに|どこ|いつ|誰|だれ|なぜ|どうして|どう|どれ|どの).{0,7}[？?。]?'
    r')\s*[？?！!。]?$',
    re.UNICODE,
)

# ── Korean simple-query patterns ──────────────────────────────────────────────

_KO_GREETING_RE = re.compile(
    r'^(안녕(하세요|하십니까)?|좋은\s*(아침|오후|저녁)|반갑습니다|여보세요)'
    r'[\s！!.]*$',
    re.UNICODE,
)

_KO_SIMPLE_RE = re.compile(
    r'^('
    r'.{0,7}(이?란\s*무엇|는\s*무엇|은\s*무엇|이?\s*뭐|가\s*뭐)[인가요까요]?[？?]?|'
    r'(무엇|뭐|어디|언제|누가|왜|어떻게).{0,7}[？?]?|'
    r'.{0,7}(에\s*대해|에\s*관해)\s*(알려주|설명해)[세요주십시오]?'
    r')\s*[？?！!]?$',
    re.UNICODE,
)


# ── Core helpers ──────────────────────────────────────────────────────────────

def _is_simple_query(text: str, min_user_chars: int = 10, min_user_chars_cjk: int = 6) -> bool:
    """Return True if ``text`` is a greeting or trivially simple question."""
    t = text.strip()
    if not t:
        return True

    if _cjk_ratio(t) >= 0.3:
        if len(t) < min_user_chars_cjk:
            return True
        return bool(
            _ZH_GREETING_RE.match(t) or _ZH_SIMPLE_RE.match(t) or
            _JA_GREETING_RE.match(t) or _JA_SIMPLE_RE.match(t) or
            _KO_GREETING_RE.match(t) or _KO_SIMPLE_RE.match(t)
        )

    if len(t) < min_user_chars:
        return True
    return bool(_EN_GREETING_RE.match(t) or _EN_SIMPLE_RE.match(t))


def _has_thinking(msg: Dict[str, Any]) -> bool:
    """Return True if an assistant message carries a non-empty thinking chain."""
    thinking = msg.get('thinking') or msg.get('reasoning_content') or ''
    return bool(thinking.strip()) if isinstance(thinking, str) else bool(thinking)


# ── Preprocessor ─────────────────────────────────────────────────────────────

class HardFilter(Preprocessor):

    def __init__(
        self,
        min_user_chars: int = 10,
        min_user_chars_cjk: int = 6,
        min_assistant_chars_2turn: int = 80,
        allow_incomplete_role: bool = False,
    ) -> None:
        super().__init__()
        self._min_user_chars = min_user_chars
        self._min_user_chars_cjk = min_user_chars_cjk
        self._min_assistant_chars_2turn = min_assistant_chars_2turn
        self.allow_incomplete_role = allow_incomplete_role

    def __call__(self, rows) -> List[Dict[str, Any]]:
        """Drop rows that are trivially low-quality by two rules:

        Rule 1 — Single-turn simple query:
            Only one user message AND that message is a greeting or bare simple question.

        Rule 2 — Two-turn shallow assistant reply:
            Exactly one user + one assistant turn, assistant reply is shorter than
            _MIN_ASSISTANT_CHARS_2TURN, and the assistant message has no thinking chain.
        """
        out = []
        for row in rows:
            messages = row.get('messages') or []
            if not isinstance(messages, list):
                continue

            user_msgs = [m for m in messages if isinstance(m, dict) and m.get('role') == 'user']
            asst_msgs = [m for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']

            if not user_msgs:
                if self.allow_incomplete_role:
                    out.append(row)
                continue

            # Rule 1: single-turn trivial query
            if len(user_msgs) == 1:
                user_text = (user_msgs[0].get('content') or '').strip()
                if _is_simple_query(user_text, self._min_user_chars, self._min_user_chars_cjk):
                    continue

            # Rule 2: two-turn shallow reply without thinking
            if len(user_msgs) == 1 and len(asst_msgs) == 1:
                asst = asst_msgs[0]
                asst_text = (asst.get('content') or '').strip()
                if len(asst_text) < self._min_assistant_chars_2turn and not _has_thinking(asst):
                    continue

            out.append(row)
        return out
