import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

from twinkle.preprocessor import Preprocessor
from .utils import msg_content_text

_SYSTEM_INJECTION_RE = re.compile(r'^<(?:system-reminder|system_reminder|context|user_info|attached_files)[ >]',
                                  re.IGNORECASE)


def _is_real_user(msg: Dict[str, Any]) -> bool:
    if msg.get('role') != 'user':
        return False
    text = msg_content_text(msg).strip()
    if not text:
        return False
    return not _SYSTEM_INJECTION_RE.match(text)


def _head_tail(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n * 2:
        return text
    return text[:n] + text[-n:]


def _prefix_signature(messages: List[Dict[str, Any]], user_chars: int, asst_chars: int) -> str:
    """Hash of the first real user turn (head+tail) + its first assistant reply (head+tail).

    Skips system messages and system-injected user messages so the signature reflects the
    actual conversation prefix, not template boilerplate. Falls back to the first two
    non-empty assistant contents when no real user is present.
    """
    user_text = ''
    asst_text = ''
    seen_user = False
    for msg in messages:
        if not seen_user:
            if _is_real_user(msg):
                user_text = _head_tail(msg_content_text(msg), user_chars)
                seen_user = True
            continue
        if msg.get('role') == 'assistant':
            t = msg_content_text(msg).strip()
            if t:
                asst_text = _head_tail(t, asst_chars)
                break
    if not seen_user:
        parts: List[str] = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                t = msg_content_text(msg).strip()
                if t:
                    parts.append(_head_tail(t, asst_chars))
                    if len(parts) == 2:
                        break
        user_text = parts[0] if parts else ''
        asst_text = parts[1] if len(parts) > 1 else ''
    return hashlib.md5(json.dumps([user_text, asst_text], ensure_ascii=False).encode()).hexdigest()


def _full_hash(messages: List[Dict[str, Any]]) -> str:
    return hashlib.md5(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class DedupFilter(Preprocessor):
    """Global longest-wins deduplication over a fully materialized row collection.

    Contract:
        - Pure in-memory single pass. No state files, no locks, no shared cross-process state,
          no cross-call memory. Same input → same output, every time.
        - Must see the entire dataset in ONE __call__. NOT a per-batch pipeline step:
          do not place inside QualityPreprocessor (which calls steps per Dataset.map batch
          — per-batch state cannot express a global longest-wins decision).
        - Run on List[Dict] before or after the QP pipeline; the caller is responsible for
          materializing the dataset and re-wrapping the kept rows.

    Semantics:
        - Signature = first real user (head+tail) + first assistant reply (head+tail).
          System and system-injected user messages are skipped.
        - Within a signature group, the row with the most messages wins; exact-content
          duplicates (matching full-hash) are silently collapsed; ties on message count
          but different content keep the first-seen row.
        - All non-winners are returned as dropped with drop_reason='duplicate'.
    """

    def __init__(self, prefix_chars: int = 100, asst_chars: int = 100):
        super().__init__()
        self._prefix = prefix_chars
        self._asst_chars = asst_chars

    def __call__(self, rows) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        # sig -> {'idx': winner index in `rows`, 'n': msg count, 'fh': full-message hash}
        best: Dict[str, Dict[str, Any]] = {}
        keep: List[bool] = [False] * len(rows)
        dropped: List[Dict[str, Any]] = []

        for i, row in enumerate(rows):
            msgs = row.get('messages') or []
            sig = _prefix_signature(msgs, self._prefix, self._asst_chars)
            n = len(msgs)
            fh = _full_hash(msgs)
            cur = best.get(sig)
            if cur is None:
                best[sig] = {'idx': i, 'n': n, 'fh': fh}
                keep[i] = True
            elif fh == cur['fh']:
                dropped.append(dict(row, drop_reason='duplicate'))
            elif n > cur['n']:
                # Longer version wins — demote the previous winner
                keep[cur['idx']] = False
                dropped.append(dict(rows[cur['idx']], drop_reason='duplicate'))
                best[sig] = {'idx': i, 'n': n, 'fh': fh}
                keep[i] = True
            else:
                dropped.append(dict(row, drop_reason='duplicate'))

        kept = [rows[i] for i, k in enumerate(keep) if k]
        return kept, dropped
