# Copyright (c) ModelScope Contributors. All rights reserved.
"""Drop rows whose first assistant reply is a self-referential refusal."""
import re
from typing import Any, Dict, List, Tuple

from twinkle.preprocessor import Preprocessor

# ── English refusal patterns ──────────────────────────────────────────────────
#
# Design principle: require a SELF-REFERENTIAL subject (I/we) + a task-directed
# inability/refusal verb. This avoids false positives on:
#   "I cannot stress enough…"  "I cannot find the bug…"
#   "The API cannot handle null"  "You cannot use this without auth"

# I/we + modal inability + task verb.
_EN_CORE = re.compile(
    r'\b(i|we)\b.{0,25}\b('
    r"can'?t|cannot|am\s+not\s+able|are\s+not\s+able|"
    r"won'?t|will\s+not|am\s+unable|are\s+unable|"
    r'must\s+decline|have\s+to\s+decline|'
    r'decline\s+to|refuse\s+to|'
    r'am\s+not\s+(allowed|permitted|authorized|comfortable)\s+to|'
    r'are\s+not\s+(allowed|permitted|authorized)'
    r')\b.{0,60}\b('
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

# Policy / content violation signal.
_EN_POLICY = re.compile(
    r'\b(this|that|your|the)\s+(request|question|prompt|content|topic|task)\b.{0,60}'
    r'\b(violates?|goes?\s+against|is\s+(inappropriate|not\s+(appropriate|allowed|permitted|'
    r'something\s+i\s+can)))\b',
    re.IGNORECASE | re.DOTALL,
)

# Standalone declarative refusals. The trailing task-verb gate attaches only to the
# "as an ai…" alternative — the first three alternatives are sufficiently specific
# on their own ("I must decline", "this falls outside what I", "I refuse to").
_EN_STANDALONE = re.compile(
    r'\b(i|we)\s+(must|have\s+to|am\s+going\s+to|need\s+to)\s+(decline|refuse)\b'
    r'|\b(i|we)\s+(decline|refuse)\s+(this|your|to)\b'
    r'|\bthis\s+(falls\s+outside|is\s+outside|is\s+beyond)\s+(what\s+i|my)\b'
    r'|\bas\s+an\s+ai[,.]?\s+i\s+(can\'?t|cannot|am\s+not\s+able|won\'?t)\b'
    r'.{0,40}\b(help|assist|answer|respond|provide|generate|create|fulfill|comply|'
    r'address|process|complete|handle|discuss|support)\b',
    re.IGNORECASE | re.DOTALL,  # DOTALL: refusal phrase + task verb may straddle a newline.
)

# ── Chinese refusal patterns ──────────────────────────────────────────────────

_ZH_APOLOGY = re.compile(
    r'(非常|十分|很|极为)?抱歉[，,。\s]{0,5}.{0,40}(无法|不能|不可以|不便|没有办法)|'
    r'对不起[，,。\s]{0,5}.{0,40}(无法|不能|不可以|不便)',
    re.UNICODE | re.DOTALL,
)

_ZH_SELF = re.compile(
    r'我(无法|不能|不可以|没有办法|不便|不适合|不被允许|不被授权)'
    r'.{0,30}(帮|回答|提供|生成|处理|协助|完成|执行|回复|解答|协|帮助)',
    re.UNICODE | re.DOTALL,
)

_ZH_VIOLATION = re.compile(
    r'(您的|这个|该)(请求|问题|内容|话题).{0,20}(违反|不当|不合适|超出了?我)',
    re.UNICODE | re.DOTALL,
)

_ZH_AI_ID = re.compile(
    r'作为(AI|人工智能|语言模型|大模型)[，,].{0,30}(无法|不能|不便|不应该|不适合)'
    r'.{0,20}(帮|回答|提供|生成|处理|协助|完成|执行|回复|解答|讨论|参与|评论|创作|输出)',
    re.UNICODE | re.DOTALL,
)

# ── Japanese refusal patterns ─────────────────────────────────────────────────

_JA_PATTERNS = (
    re.compile(r'(申し訳|恐れ入り)ます(が|けれど).{0,40}(できません|お答えできません|対応できません)', re.UNICODE | re.DOTALL),
    re.compile(r'(回答|対応|お答え)(する|いたす)ことは?できません', re.UNICODE),
    re.compile(r'ご要望には?お(応え|答え)できません', re.UNICODE),
    re.compile(r'(その|この)(リクエスト|質問|依頼).{0,20}(お断り|辞退|対応できません)', re.UNICODE | re.DOTALL),
)

# ── Korean refusal patterns ───────────────────────────────────────────────────

_KO_PATTERNS = (
    re.compile(r'(죄송하지만|유감스럽게도).{0,40}(드릴 수 없|없습니다|못합니다)', re.UNICODE | re.DOTALL),
    re.compile(r'(답변|도움|처리|제공)(드리기|하기)\s*(어렵|불가|할 수 없)', re.UNICODE),
    re.compile(r'(요청|질문|내용).{0,20}(거절|거부|응할 수 없)', re.UNICODE | re.DOTALL),
)

_ALL_PATTERNS = (_EN_CORE, _EN_APOLOGY, _EN_POLICY, _EN_STANDALONE, _ZH_APOLOGY, _ZH_SELF, _ZH_VIOLATION,
                 _ZH_AI_ID) + _JA_PATTERNS + _KO_PATTERNS

# Strip both `<think>` and `<thinking>` blocks before scanning so reasoning-trace
# refusal-like phrasing doesn't get mistaken for a real user-facing refusal.
_THINK_BLOCK_RE = re.compile(r'<think(?:ing)?>.*?</think(?:ing)?>\s*', re.DOTALL | re.IGNORECASE)

# ── Continuation exemption ────────────────────────────────────────────────────
#
# A genuine refusal is TERMINAL — the assistant stops helping. In agent / coding
# traces the model very often states a local, technical inability and then
# immediately pivots to an alternative action:
#   "I can't write to E:\…. I'll need to use exec to create the directory…"
#   "I can't read files outside the sandbox. Let me use exec to …"
# These are NOT refusals of the user's request. If a pivot-to-action cue appears
# anywhere in the scanned window, we exempt the row.
_EN_CONTINUE = re.compile(
    r"\b(let\s+me|let'?s|i'?ll|i\s+will|i'?m\s+going\s+to|i\s+need\s+to|i'?ll\s+need\s+to|"
    r'instead|so\s+i(\'?ll|\s+will)?|so\s+let|try\s+(again|another)|as\s+an\s+alternative|'
    r'alternatively|workaround|work\s+around|use\s+(exec|the\s+\w+\s+tool)|'
    r'run\s+the|call\s+the|switch\s+to|fall\s+back)\b',
    re.IGNORECASE | re.DOTALL,
)
_ZH_CONTINUE = re.compile(
    r'(让我|我来|我先|我会|我将|我需要|改用|换用|换个|换成|试试|尝试|再试|退而|作为替代|'
    r'替代方案|变通|绕过|所以我|因此我|接下来我|那我|改为|改成|使用工具|调用工具|执行命令)',
    re.UNICODE | re.DOTALL,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _text(content: Any) -> str:
    """Project content to plain text. Multimodal list → concat of text parts only."""
    if isinstance(content, list):
        return ''.join(p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text')
    return content if isinstance(content, str) else ''


# Patterns that signal a *soft* technical inability rather than a hard refusal of
# the user's request. These are the ones prone to false positives on agents that
# state a constraint and keep working, so they are subject to the continuation
# exemption. The remaining patterns (apology-decline, policy/violation, AI-identity
# refusal, standalone "I refuse to") are terminal and never exempted.
_SOFT_INABILITY = frozenset({id(_EN_CORE), id(_ZH_SELF)})


def _is_refusal(text: str, check_window: int = 600) -> bool:
    """Return True if the text contains a self-referential refusal signal.

    ``check_window <= 0`` scans the whole text (no truncation). A soft technical
    inability ("I can't write to X") that is immediately followed by a pivot to an
    alternative action ("let me use exec…") is exempted — that is an agent working
    around a constraint, not refusing the user's request.
    """
    window = text if check_window <= 0 else text[:check_window]
    pivots = None  # lazily computed only when a soft-inability pattern hits
    for p in _ALL_PATTERNS:
        if not p.search(window):
            continue
        if id(p) in _SOFT_INABILITY:
            if pivots is None:
                pivots = bool(_EN_CONTINUE.search(window) or _ZH_CONTINUE.search(window))
            if pivots:
                continue  # constraint-then-pivot: not a refusal
        return True
    return False


# ── Preprocessor ─────────────────────────────────────────────────────────────


class RefuseFilter(Preprocessor):
    """Drop rows whose assistant reply is a self-referential refusal.

    Args:
        check_window: chars scanned per assistant message (0 = whole message).
        scan_all_assistants: scan every assistant turn, not just the first — a
            multi-turn conversation may only refuse in a later turn.
        scan_reasoning: also scan ``reasoning_content``/``thinking`` fields.
            Default False: reasoning traces often rehearse refusal-like phrasing
            that the model then overrides, so scanning them raises false positives.
    """

    def __init__(self, check_window: int = 600, *, scan_all_assistants: bool = True,
                 scan_reasoning: bool = False) -> None:
        super().__init__()
        self._check_window = check_window
        self._scan_all = bool(scan_all_assistants)
        self._scan_reasoning = bool(scan_reasoning)

    def _is_refusal_row(self, row: Dict[str, Any]) -> bool:
        messages = row.get('messages') or []
        asst_msgs = [m for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']
        if not self._scan_all:
            asst_msgs = asst_msgs[:1]
        for m in asst_msgs:
            reply = _THINK_BLOCK_RE.sub('', _text(m.get('content'))).strip()
            if reply and _is_refusal(reply, self._check_window):
                return True
            if self._scan_reasoning:
                reasoning = (m.get('reasoning_content') or m.get('thinking') or '').strip()
                if reasoning and _is_refusal(reasoning, self._check_window):
                    return True
        return False

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for row in rows:
            if self._is_refusal_row(row):
                dropped.append(dict(row, drop_reason='refusal'))
            else:
                out.append(row)
        return out, dropped
