# Copyright (c) ModelScope Contributors. All rights reserved.
"""Persist kept task statements and select similar references with 3-gram Jaccard overlap."""
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ['TaskBank', 'jaccard_3gram', 'grams']


def grams(text: str, n: int = 3) -> Set[Tuple[str, ...]]:
    words = re.findall(r'[a-z0-9_]+', (text or '').lower())
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def jaccard_3gram(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TaskBank:
    """Statements from previous iterations, plus the ones this run adds.

    Args:
        path: the JSONL file. A missing file is an empty bank, not an error -- the
            first iteration has nothing to compare against and must still run.
        refs: how many stored statements :meth:`references` returns.
    """

    def __init__(self, path: str, refs: int = 5):
        self.path = path
        self.refs = max(0, refs)
        self._statements: List[str] = []
        self._grams: List[Set[Tuple[str, ...]]] = []
        # Which group each statement came from. Kept because the computable
        # novelty score calibrates its clustering threshold on the similarity
        # between siblings, and the siblings are only recoverable from this.
        self._group_ids: List[Any] = []
        self._seen: Set[str] = set()
        self._lock = threading.Lock()
        self.n_loaded = 0
        self.n_added = 0
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A half-written last line from a killed run. Skipped rather
                    # than fatal: losing one reference is not worth failing a run,
                    # and the count below says how many were read.
                    continue
                statement = (rec.get('statement') or '').strip()
                if statement and statement not in self._seen:
                    self._seen.add(statement)
                    self._statements.append(statement)
                    self._grams.append(grams(statement))
                    self._group_ids.append(rec.get('group_id'))
        self.n_loaded = len(self._statements)

    def __len__(self) -> int:
        return len(self._statements)

    def statements(self) -> List[str]:
        with self._lock:
            return list(self._statements)

    def sibling_groups(self, minimum: int = 2) -> List[List[str]]:
        """Return stored groups with at least minimum siblings for novelty calibration."""
        with self._lock:
            by_group: Dict[Any, List[str]] = {}
            for group_id, statement in zip(self._group_ids, self._statements):
                by_group.setdefault(group_id, []).append(statement)
        return [members for members in by_group.values() if len(members) >= minimum]

    def references(self, statement: str, extra: Sequence[str] = ()) -> List[str]:
        """Prepend extra sibling statements, then return the closest historical references."""
        with self._lock:
            pairs = list(zip(self._statements, self._grams))
        target = grams(statement)
        scored = [(jaccard_3gram(target, g), s) for s, g in pairs if s != statement]
        scored.sort(key=lambda p: -p[0])
        out = [s for s in extra if s and s != statement]
        out.extend(s for _, s in scored[:self.refs])
        return out

    def add(self, statement: str, check: str = '', **fields: Any) -> bool:
        """Append a non-empty, previously unseen statement immediately; return whether it was added."""
        statement = (statement or '').strip()
        if not statement:
            return False
        with self._lock:
            if statement in self._seen:
                return False
            self._seen.add(statement)
            self._statements.append(statement)
            self._grams.append(grams(statement))
            self.n_added += 1
            self._group_ids.append(fields.get('group_id'))
            if self.path:
                rec: Dict[str, Any] = {'statement': statement, 'check': check}
                rec.update(fields)
                os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return True

    def ingest_recorder(self, save_dir: str, outcomes: Sequence[str] = ('kept', )) -> int:
        """Ingest selected outcomes from save_dir/trajs/index.jsonl.

        Defaults to kept proposals. Repeated calls deduplicate statements, so startup,
        per-iteration and final ingestion can share this path. Returns new statements.
        """
        index_path = os.path.join(save_dir, 'trajs', 'index.jsonl')
        if not os.path.isfile(index_path):
            return 0
        wanted = set(outcomes)
        added = 0
        with open(index_path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A half-written last line from a killed run -- the recorder
                    # appends while the loop is live. Skipped, as in _load.
                    continue
                if record.get('outcome') not in wanted:
                    continue
                if self.add(
                        record.get('statement') or '', record.get('check_script') or '',
                        group_id=record.get('group_id')):
                    added += 1
        return added

    def stats(self) -> Dict[str, Optional[int]]:
        return {'path': self.path, 'loaded': self.n_loaded, 'added': self.n_added, 'total': len(self._statements)}
