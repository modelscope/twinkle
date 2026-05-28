# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from twinkle.preprocessor import Preprocessor
from twinkle.utils import get_logger

from .llm_backend import LLMBackend, OpenAIBackend

logger = get_logger(only_local_master=False)

# ── Intent categories ─────────────────────────────────────────────────────────
INTENT_TOOL_CALL = 'tool_call'
INTENT_CODE = 'code'
INTENT_MATH = 'math'
INTENT_COMPLEX_LOGIC = 'complex_logic'
INTENT_USER_DISSATISFACTION = 'user_dissatisfaction'
INTENT_OTHER = 'other'

_ALL_INTENTS = (
    INTENT_TOOL_CALL, INTENT_CODE, INTENT_MATH,
    INTENT_COMPLEX_LOGIC, INTENT_USER_DISSATISFACTION, INTENT_OTHER,
)

# ── Heuristic patterns ────────────────────────────────────────────────────────
_CODE_BLOCK_RE = re.compile(r'```[\s\S]{20,}?```')
_CODE_KEYWORD_RE = re.compile(
    r'\b(def |class |import |function |const |let |var |return |if \(|for \(|while \(|'
    r'#include|public class|private |protected )\b'
)

_MATH_LATEX_RE = re.compile(
    r'(\$\$.+?\$\$|\$[^$\n]+?\$|'
    r'\\frac|\\sum|\\int|\\lim|\\begin\{(equation|align|matrix)|'
    r'\\mathbb|\\partial|\\nabla|\\sqrt|\\overline|'
    r'\\\[.+?\\\])',
    re.DOTALL,
)

_DISSATISFACTION_ZH_RE = re.compile(
    r'(不[满好对行]|太[差慢烂]|重[做来新]|错了|又错|有问题|没用|答非所问|'
    r'别瞎|你在说什么|这是什么|离谱|搞什么|质量太|胡说|瞎编)',
)
_DISSATISFACTION_EN_RE = re.compile(
    r'\b(wrong|incorrect|useless|terrible|awful|bad answer|redo|try again|'
    r'not what i asked|disappointed|frustrat|unacceptable|nonsense|garbage)\b',
    re.IGNORECASE,
)

_LLM_CLASSIFY_PROMPT = """You are a trajectory intent classifier. Given a multi-turn conversation, classify its PRIMARY intent into exactly one category.

Categories:
- complex_logic: Requires multi-step reasoning, planning, logical deduction, or strategic thinking (NOT code/math).
- user_dissatisfaction: The user expresses dissatisfaction, complaints, or frustration with previous responses.
- other: General Q&A, creative writing, translation, chitchat, or anything not fitting the above.

Reply with EXACTLY one word from: complex_logic, user_dissatisfaction, other"""

_LLM_ROUND_CONFIRM_PROMPT = """You are a conversation round classifier. Given a (user, assistant) pair, confirm whether the round matches the proposed category.

Categories:
- code: The round is primarily about writing, debugging, or explaining code.
- math: The round is primarily about mathematical derivation or computation.
- user_dissatisfaction: The user expresses dissatisfaction or frustration.
- complex_logic: Requires multi-step reasoning or planning.
- no: The proposed category does NOT match this round.

Reply with EXACTLY one word from: code, math, user_dissatisfaction, complex_logic, no"""

_DEFAULT_TIMEOUT = 60.0


# ── Heuristic detectors ───────────────────────────────────────────────────────

def _msg_text(msg: Dict[str, Any]) -> str:
    """Extract plain text from a single message."""
    c = msg.get('content')
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ' '.join(
            p.get('text', '') for p in c
            if isinstance(p, dict) and p.get('type') == 'text'
        )
    return ''


def _extract_text(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        parts.append(_msg_text(m))
    return '\n'.join(parts)


def _has_tool_calls(messages: List[Dict[str, Any]]) -> bool:
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get('role') == 'tool':
            return True
        if m.get('tool_calls'):
            return True
    return False


def _is_code_heavy(text: str) -> bool:
    blocks = _CODE_BLOCK_RE.findall(text)
    if len(blocks) >= 2:
        return True
    if blocks and _CODE_KEYWORD_RE.search(text):
        return True
    kw_hits = len(_CODE_KEYWORD_RE.findall(text))
    return kw_hits >= 5


def _is_math_heavy(text: str) -> bool:
    hits = _MATH_LATEX_RE.findall(text)
    return len(hits) >= 2


def _is_dissatisfied(text: str) -> bool:
    return bool(_DISSATISFACTION_ZH_RE.search(text) or _DISSATISFACTION_EN_RE.search(text))


def _has_dissatisfaction_signal(messages: List[Dict[str, Any]]) -> bool:
    """Check user messages for dissatisfaction keywords."""
    for m in messages:
        if not isinstance(m, dict) or m.get('role') != 'user':
            continue
        c = m.get('content', '')
        if not isinstance(c, str):
            continue
        if _is_dissatisfied(c):
            return True
    return False


def _detect_msg_signal(text: str) -> Optional[str]:
    """Detect heuristic signal from a single message's text. Returns intent or None."""
    if _is_code_heavy(text):
        return INTENT_CODE
    if _is_math_heavy(text):
        return INTENT_MATH
    if _is_dissatisfied(text):
        return INTENT_USER_DISSATISFACTION
    return None


# ── LLM classification ────────────────────────────────────────────────────────

def _format_conversation(messages: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    parts = []
    total = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role', 'unknown')
        content = (m.get('content') or '')
        if isinstance(content, list):
            content = ' '.join(
                p.get('text', '') for p in content
                if isinstance(p, dict) and p.get('type') == 'text'
            )
        content = content.strip()[:800]
        line = f'[{role}]: {content}'
        if total + len(line) > max_chars:
            parts.append('[... truncated ...]')
            break
        parts.append(line)
        total += len(line)
    return '\n'.join(parts)


def _llm_classify_one(
    backend: LLMBackend,
    messages: List[Dict[str, Any]],
) -> str:
    """Call LLM to classify a single trajectory. Returns intent string."""
    conversation_text = _format_conversation(messages)
    choices = backend.chat(
        [{'role': 'system', 'content': _LLM_CLASSIFY_PROMPT},
         {'role': 'user', 'content': f'Classify this conversation:\n\n{conversation_text}'}],
        temperature=0.0, max_tokens=16,
    )
    if not choices:
        return INTENT_OTHER
    text = choices[0].get('content', '').strip().lower()
    for intent in (INTENT_COMPLEX_LOGIC, INTENT_USER_DISSATISFACTION, INTENT_OTHER):
        if intent in text:
            return intent
    return INTENT_OTHER


def _llm_confirm_round(
    backend: LLMBackend,
    user_text: str,
    assistant_text: str,
    proposed: str,
) -> Optional[str]:
    """Ask LLM to confirm whether a (user, assistant) pair matches the proposed intent."""
    prompt = (f'Proposed category: {proposed}\n\n'
              f'[user]: {user_text[:1500]}\n[assistant]: {assistant_text[:1500]}')
    choices = backend.chat(
        [{'role': 'system', 'content': _LLM_ROUND_CONFIRM_PROMPT},
         {'role': 'user', 'content': prompt}],
        temperature=0.0, max_tokens=16,
    )
    if not choices:
        return None
    text = choices[0].get('content', '').strip().lower()
    if 'no' in text:
        return None
    for intent in (INTENT_CODE, INTENT_MATH, INTENT_USER_DISSATISFACTION, INTENT_COMPLEX_LOGIC):
        if intent in text:
            return intent
    return None


# ── Preprocessor ──────────────────────────────────────────────────────────────

class IntentClassifier(Preprocessor):
    """Annotate each trajectory with its primary intent category.

    Detection strategy:
    - tool_call: role='tool' or assistant has tool_calls field (heuristic)
    - code: fenced code blocks + language keywords (heuristic)
    - math: LaTeX formulas (heuristic)
    - complex_logic: multi-step reasoning (LLM)
    - user_dissatisfaction: user complaints (heuristic + LLM)
    - other: fallback

    Adds an 'intent' field (str) to each row.
    """

    def __init__(
        self,
        backend: LLMBackend = None,
        max_workers: int = 8,
        intent_field: str = 'intent',
        # Legacy params (used to create OpenAIBackend if backend is None)
        api_endpoint: str = '',
        model: str = 'default',
        api_key: str = '',
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__()
        self._intent_field = intent_field
        self._max_workers = max_workers
        self._backend: Optional[LLMBackend] = None

        if backend is not None:
            self._backend = backend
        elif api_endpoint:
            self._backend = OpenAIBackend(
                endpoint=api_endpoint, model=model, api_key=api_key, timeout=timeout)

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = self.classify_intent(rows)
        return self.map_row_to_col(rows)

    def classify_intent(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate each row with intent label and key_rounds in user_data."""
        if not rows:
            return rows

        # Phase 1: per-round heuristic signal detection
        # Each entry: (row_idx, assistant_idx, user_text, asst_text, proposed_intent)
        candidates: List[tuple] = []
        row_intents: Dict[int, str] = {}
        confirmed_rounds: Dict[int, List[int]] = {}  # row_idx → list of assistant indices
        round_intents: Dict[int, Dict[int, str]] = {}  # row_idx → {asst_idx: intent}

        for ri, row in enumerate(rows):
            messages = row.get('messages')
            if not isinstance(messages, list) or not messages:
                row_intents[ri] = INTENT_OTHER
                continue

            # tool_call is definitive — mark assistants with tool_calls as key rounds
            if _has_tool_calls(messages):
                row_intents[ri] = INTENT_TOOL_CALL
                for idx, m in enumerate(messages):
                    if isinstance(m, dict) and m.get('role') == 'assistant' and m.get('tool_calls'):
                        confirmed_rounds.setdefault(ri, []).append(idx)
                        round_intents.setdefault(ri, {})[idx] = INTENT_TOOL_CALL
                continue

            # Scan each message for signals
            found_any = False
            for idx, m in enumerate(messages):
                if not isinstance(m, dict):
                    continue
                role = m.get('role')
                text = _msg_text(m)
                if not text:
                    continue
                signal = _detect_msg_signal(text)
                if not signal:
                    continue

                # Determine (user, assistant) pair based on where signal is
                if role == 'user':
                    # Find next assistant
                    asst_idx = None
                    for j in range(idx + 1, len(messages)):
                        if isinstance(messages[j], dict) and messages[j].get('role') == 'assistant':
                            asst_idx = j
                            break
                    if asst_idx is None:
                        continue
                    user_text = text
                    asst_text = _msg_text(messages[asst_idx])
                    candidates.append((ri, asst_idx, user_text, asst_text, signal))
                    found_any = True
                elif role == 'assistant':
                    # Find previous user
                    user_idx = None
                    for j in range(idx - 1, -1, -1):
                        if isinstance(messages[j], dict) and messages[j].get('role') == 'user':
                            user_idx = j
                            break
                    if user_idx is None:
                        continue
                    user_text = _msg_text(messages[user_idx])
                    asst_text = text
                    candidates.append((ri, idx, user_text, asst_text, signal))
                    found_any = True

            if not found_any:
                # No heuristic signal → needs full-trajectory LLM classification
                row_intents.setdefault(ri, None)  # mark for LLM

        # Phase 2: LLM confirmation for candidates (per-round pairs)
        # Deduplicate candidates by (row_idx, assistant_idx) — keep first signal
        seen_pairs: set = set()
        deduped_candidates: List[tuple] = []
        for c in candidates:
            pair = (c[0], c[1])  # (ri, asst_idx)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                deduped_candidates.append(c)
        candidates = deduped_candidates

        if candidates and self._backend:
            n_workers = min(self._max_workers, len(candidates))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_cand = {
                    pool.submit(
                        _llm_confirm_round,
                        self._backend,
                        c[2], c[3], c[4],
                    ): c
                    for c in candidates
                }
                for future in as_completed(future_to_cand):
                    cand = future_to_cand[future]
                    ri, asst_idx, _, _, proposed = cand
                    try:
                        confirmed = future.result()
                    except Exception:
                        confirmed = None
                    if confirmed:
                        confirmed_rounds.setdefault(ri, []).append(asst_idx)
                        round_intents.setdefault(ri, {})[asst_idx] = confirmed
        elif candidates:
            # No LLM — trust heuristic directly
            for ri, asst_idx, _, _, proposed in candidates:
                confirmed_rounds.setdefault(ri, []).append(asst_idx)
                round_intents.setdefault(ri, {})[asst_idx] = proposed

        # Phase 3: full-trajectory LLM for rows without any heuristic signal
        needs_full_llm = [ri for ri, v in row_intents.items() if v is None]
        if needs_full_llm and self._backend:
            n_workers = min(self._max_workers, len(needs_full_llm))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_idx = {
                    pool.submit(
                        _llm_classify_one,
                        self._backend,
                        rows[ri].get('messages') or [],
                    ): ri
                    for ri in needs_full_llm
                }
                for future in as_completed(future_to_idx):
                    ri = future_to_idx[future]
                    try:
                        row_intents[ri] = future.result()
                    except Exception:
                        row_intents[ri] = INTENT_OTHER
        else:
            for ri in needs_full_llm:
                messages = rows[ri].get('messages') or []
                if _has_dissatisfaction_signal(messages):
                    row_intents[ri] = INTENT_USER_DISSATISFACTION
                else:
                    row_intents[ri] = INTENT_OTHER

        # Phase 3.5: generate key_rounds for full-LLM rows (mark last assistant)
        for ri in needs_full_llm:
            intent = row_intents.get(ri, INTENT_OTHER)
            if intent == INTENT_OTHER:
                continue
            if ri in confirmed_rounds:
                continue
            messages = rows[ri].get('messages') or []
            last_asst = None
            for idx in range(len(messages) - 1, -1, -1):
                if isinstance(messages[idx], dict) and messages[idx].get('role') == 'assistant':
                    last_asst = idx
                    break
            if last_asst is not None:
                confirmed_rounds.setdefault(ri, []).append(last_asst)
                round_intents.setdefault(ri, {})[last_asst] = intent

        # Phase 4: determine primary intent from key_rounds for candidate rows
        for ri in confirmed_rounds:
            if ri not in row_intents or row_intents.get(ri) == INTENT_TOOL_CALL:
                continue
            # Primary = most common confirmed intent
            ri_intents = round_intents.get(ri, {})
            intents = [ri_intents.get(idx, INTENT_OTHER) for idx in confirmed_rounds[ri]]
            from collections import Counter
            most_common = Counter(intents).most_common(1)[0][0]
            row_intents[ri] = most_common

        # For candidate rows with no confirmed rounds, fall back to other
        for ri, row in enumerate(rows):
            if ri not in row_intents:
                row_intents[ri] = INTENT_OTHER

        # Phase 5: annotate output
        out = []
        for i, row in enumerate(rows):
            row = dict(row)
            row[self._intent_field] = row_intents.get(i, INTENT_OTHER)
            # Store key rounds in user_data
            if i in confirmed_rounds and confirmed_rounds[i]:
                user_data = dict(row.get('user_data') or {})
                user_data['key_rounds'] = confirmed_rounds[i]
                if i in round_intents:
                    user_data['intents'] = round_intents[i]
                row['user_data'] = user_data
            out.append(row)

        from collections import Counter
        dist = Counter(r[self._intent_field] for r in out)
        logger.info(f'[IntentClassifier] distribution: {dict(dist)}')

        return out
