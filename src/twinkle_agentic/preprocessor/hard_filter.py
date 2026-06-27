# Copyright (c) ModelScope Contributors. All rights reserved.
"""Hard rule-based row filter (greetings, shallow replies, deny-listed system, length caps)."""
import re
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from .utils import cjk_ratio, msg_content_text, msg_has_media

# ── Language detection ────────────────────────────────────────────────────────

# ── Simple-query patterns ─────────────────────────────────────────────────────

_EN_GREETING_RE = re.compile(
    r'^(h+e+l+l+o+|h+i+|hey+|yo+|howdy|greetings|'
    r'good\s+(morning|afternoon|evening|night|day)|'
    r'what\'?s\s+up|how\'?s\s+it\s+going|how\s+are\s+you)'
    r'[\s,!.?]*$',
    re.IGNORECASE,
)

_EN_SIMPLE_RE = re.compile(
    r'^('
    r'(what|who|where|when|why|how)\s+(is|are|was|were|does|do|did|has|have|can|could|would|should)\b.{0,30}|'
    r'(what|who|where|when|why|how)\'s\b.{0,30}|'
    r'(is|are|was|were|do|does|did|can|could|would|should|may|might)\s+(it|this|that|you|there|they|he|she)\b.{0,30}|'
    r'(tell\s+me(\s+(about|more))?|explain(\s+to\s+me)?|define|describe|list|summarize|give\s+me)\b.{0,20}|'
    r'(please\s+)?(help\s+me|assist\s+me)\b.{0,20}'
    r')\s*[?!.]?$',
    re.IGNORECASE | re.DOTALL,
)

_ZH_GREETING_RE = re.compile(
    r'^(你好+|您好+|早上好|下午好|晚上好|大家好|嗨+|哈+喽+|哈+|喂+|hello+|hi+)'
    r'[\s,，！!。.]*$',
    re.UNICODE,
)

_ZH_SIMPLE_RE = re.compile(
    r'^('
    r'.{0,7}(是什么|是啥|啥意思|是何|什么意思|怎么样|如何|为什么|为啥)[？?。]?|'
    r'(什么|啥|哪|谁|何|怎么|怎样|为什么|为啥|几|多少|何时|何地).{0,7}[？?。]?|'
    r'(介绍|解释|说明|告诉我|帮我说说|请问|能说说|讲讲).{0,5}|'
    r'(请\s*(给出|介绍|解释|说明|提供|列举|讲讲|阐述|描述|概述|举例|分析|说一下)|'
    r'能否\s*(给出|设计|提供|介绍|解释|说明)).{0,10}'
    r')\s*[？?！!。]?$',
    re.UNICODE,
)

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

_CJK_SIMPLE_REGEXES = (_ZH_GREETING_RE, _ZH_SIMPLE_RE, _JA_GREETING_RE, _JA_SIMPLE_RE, _KO_GREETING_RE, _KO_SIMPLE_RE)
_LATIN_SIMPLE_REGEXES = (_EN_GREETING_RE, _EN_SIMPLE_RE)

# ── Content helpers ──────────────────────────────────────────────────────────


def _has_tool_calls(msg: Dict[str, Any]) -> bool:
    """Truthy ``tool_calls`` excluding the empty-array sentinels '' / '[]' / []."""
    tc = msg.get('tool_calls')
    if not tc:
        return False
    if isinstance(tc, str):
        s = tc.strip()
        return bool(s) and s != '[]'
    return bool(tc)


def _is_simple_query(text: str, min_user_chars: int, min_user_chars_cjk: int) -> bool:
    """True if ``text`` is a greeting or trivially simple question."""
    t = text.strip()
    if not t:
        return True
    cjk = cjk_ratio(t) >= 0.3
    threshold = min_user_chars_cjk if cjk else min_user_chars
    if len(t) < threshold:
        return True
    regexes = _CJK_SIMPLE_REGEXES if cjk else _LATIN_SIMPLE_REGEXES
    return any(r.match(t) for r in regexes)


def _has_thinking(msg: Dict[str, Any], min_chars: int) -> bool:
    """True if an assistant message carries a sufficiently long thinking chain."""
    thinking = msg.get('thinking') or msg.get('reasoning_content') or ''
    if isinstance(thinking, str):
        return len(thinking.strip()) >= min_chars
    return bool(thinking)


# ── Preprocessor ─────────────────────────────────────────────────────────────


class HardFilter(Preprocessor):

    def __init__(
        self,
        min_user_chars: int = 10,
        min_user_chars_cjk: int = 6,
        min_assistant_chars_2turn: int = 80,
        min_thinking_chars: int = 200,
        allow_incomplete_role: bool = False,
        system_deny_keywords: Optional[List[str]] = None,
        max_chars_per_round: Optional[int] = None,
        max_total_chars: Optional[int] = None,
        max_rounds: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._min_user_chars = min_user_chars
        self._min_user_chars_cjk = min_user_chars_cjk
        self._min_assistant_chars_2turn = min_assistant_chars_2turn
        self._min_thinking_chars = min_thinking_chars
        self.allow_incomplete_role = allow_incomplete_role
        self._system_deny_re = (
            re.compile('|'.join(re.escape(k)
                                for k in system_deny_keywords), re.IGNORECASE) if system_deny_keywords else None)
        self._max_chars_per_round = max_chars_per_round
        self._max_total_chars = max_total_chars
        self._max_rounds = max_rounds

    def _drop_reason(self, row: Dict[str, Any], messages: List[Any]) -> Optional[str]:
        """Apply rules in order; return first matching drop_reason, or None to keep."""
        if not isinstance(messages, list):
            return 'invalid_messages'

        user_msgs = [m for m in messages if isinstance(m, dict) and m.get('role') == 'user']
        asst_msgs = [m for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']

        if not user_msgs:
            return None if self.allow_incomplete_role else 'no_user'

        # Rule 1: single-turn trivial query (only meaningful when user content is plain text).
        if len(user_msgs) == 1:
            user_content = user_msgs[0].get('content')
            if isinstance(user_content, str) and _is_simple_query(user_content, self._min_user_chars,
                                                                  self._min_user_chars_cjk):
                if not asst_msgs or not _has_thinking(asst_msgs[0], self._min_thinking_chars):
                    return 'trivial_single_turn'

        # Rule 2: two-turn shallow reply without thinking.
        if len(user_msgs) == 1 and len(asst_msgs) == 1:
            asst = asst_msgs[0]
            if (len(msg_content_text(asst)) < self._min_assistant_chars_2turn
                    and not _has_thinking(asst, self._min_thinking_chars)):
                return 'shallow_reply'

        # Rule 3: every assistant turn is content-empty, has no thinking, and has no tool_calls.
        # Multimodal non-text parts (images etc.) also count as substantive.
        if asst_msgs and all(not msg_content_text(m).strip() and not msg_has_media(m)
                             and not _has_thinking(m, self._min_thinking_chars) and not _has_tool_calls(m)
                             for m in asst_msgs):
            return 'all_empty_assistant'

        # Rule 4: system prompt matches deny keywords.
        if self._system_deny_re:
            sys_text = next(
                (msg_content_text(m) for m in messages if isinstance(m, dict) and m.get('role') == 'system'), '')
            if self._system_deny_re.search(sys_text):
                return 'system_deny_keyword'

        # Rule 5: per-round character length limit (counted on textual projection).
        if self._max_chars_per_round and any(
                len(msg_content_text(m)) > self._max_chars_per_round for m in messages if isinstance(m, dict)):
            return 'round_too_long'

        # Rule 6: total conversation character length limit.
        if self._max_total_chars:
            total = sum(len(msg_content_text(m)) for m in messages if isinstance(m, dict))
            if total > self._max_total_chars:
                return 'total_too_long'

        # Rule 7: max rounds (user-assistant pairs).
        if self._max_rounds and len(asst_msgs) > self._max_rounds:
            return 'too_many_rounds'

        return None

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for row in rows:
            reason = self._drop_reason(row, row.get('messages') or [])
            if reason is None:
                out.append(row)
            else:
                dropped.append(dict(row, drop_reason=reason))
        return out, dropped
