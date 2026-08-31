# Copyright (c) ModelScope Contributors. All rights reserved.
"""A file of the tasks earlier iterations already produced, to compare new ones against.

Novelty is meaningless without something to be novel against. Within one run the
proposals of a group can be compared to each other, but the failure this is for is
slower than that: iteration k+1 re-proposing what iteration k already trained on. That
needs a file that outlives a run, which is what this is -- one JSON object per line,
appended by :meth:`add`, read back by the next run.

The similarity used to pick which stored tasks to show the judge is 3-gram Jaccard over
the statement. It is a weak measure and known to be: measured over run_clean9's 188
statements the closest pair scored 0.060, so ranking by it is nearly ranking at random,
and it cannot see that two tasks with no shared wording are both 'write the given files
verbatim, then derive one from them'. It is used only to CHOOSE the handful of tasks the
judge reads, never to score novelty -- the judging is
:mod:`twinkle_agentic.verifier.rubric_score`, whose criteria compare task shapes. Even a
near-random pick gives the judge real tasks from the same generator to compare against,
which is what the criteria need.
"""
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
        self.n_loaded = len(self._statements)

    def __len__(self) -> int:
        return len(self._statements)

    def references(self, statement: str, extra: Sequence[str] = ()) -> List[str]:
        """The stored statements most similar to ``statement``, closest first.

        ``extra`` is prepended and never dropped -- it is how the proposals of the
        current group get in front of the judge. Without them a whole group can be
        scored identically novel against history while being eight versions of one
        idea, and GRPO subtracts the group mean, so an identical term across the
        group produces no gradient at all.
        """
        with self._lock:
            pairs = list(zip(self._statements, self._grams))
        target = grams(statement)
        scored = [(jaccard_3gram(target, g), s) for s, g in pairs if s != statement]
        scored.sort(key=lambda p: -p[0])
        out = [s for s in extra if s and s != statement]
        out.extend(s for _, s in scored[:self.refs])
        return out

    def add(self, statement: str, check: str = '', **fields: Any) -> bool:
        """Append one task. Returns False if the statement is already stored.

        Appended immediately rather than at the end of the run: a run that crashes
        after 60 of 80 tasks should still contribute those 60, or the bank silently
        under-reports what has been trained on.
        """
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
            if self.path:
                rec: Dict[str, Any] = {'statement': statement, 'check': check}
                rec.update(fields)
                os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return True

    def stats(self) -> Dict[str, Optional[int]]:
        return {'path': self.path, 'loaded': self.n_loaded, 'added': self.n_added,
                'total': len(self._statements)}
