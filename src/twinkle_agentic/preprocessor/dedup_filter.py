import hashlib
import json
import os
import uuid
from typing import Any, Dict, List, Optional

from twinkle.preprocessor import Preprocessor
from twinkle.utils.parallel import PosixFileLock


def _conversation_sig(row: Dict[str, Any], prefix_chars: int = 200, asst_chars: int = 150) -> str:
    """Hash of system + first user + first assistant as conversation-level fingerprint."""
    msgs = row.get('messages') or []
    sys_text = ''
    user_text = ''
    asst_text = ''
    for m in msgs:
        role = m.get('role', '')
        content = m.get('content') or ''
        if role == 'system' and not sys_text:
            sys_text = content[:prefix_chars]
        elif role == 'user' and not user_text:
            user_text = content[:prefix_chars]
        elif role == 'assistant' and not asst_text:
            asst_text = content[:asst_chars]
            break
    raw = f'{sys_text}||{user_text}||{asst_text}'
    return hashlib.md5(raw.encode()).hexdigest()


class DedupFilter(Preprocessor):
    """Conversation-level near-dedup: keep at most max_per_sig rows per signature.

    Uses file-backed state + PosixFileLock for multi-process safety.
    """

    def __init__(self, max_per_sig: int = 1, prefix_chars: int = 200, asst_chars: int = 150,
                 state_file: Optional[str] = None):
        self._max = max_per_sig
        self._prefix = prefix_chars
        self._asst_chars = asst_chars
        # Unique per instantiation — no stale-file cleanup needed, no race.
        self._state_file = state_file or f'/tmp/dedup_filter_{uuid.uuid4().hex[:12]}.json'
        self._lock_file = self._state_file + '.lock'

    def __call__(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = self.map_col_to_row(rows)
        with PosixFileLock(self._lock_file):
            seen = {}
            if os.path.exists(self._state_file):
                with open(self._state_file) as f:
                    seen = json.load(f)
            out = []
            for r in rows:
                sig = _conversation_sig(r, self._prefix, self._asst_chars)
                count = seen.get(sig, 0)
                if count < self._max:
                    seen[sig] = count + 1
                    out.append(r)
            with open(self._state_file, 'w') as f:
                json.dump(seen, f)
        return out
