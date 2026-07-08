# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trajectory segmentation for segment-level rubric scoring.

A long agent trajectory is split into a list of *segments*; each segment is a
self-contained sub-trajectory (``{'messages': [...], 'tools': [...]}``) that can
be fed directly to a :class:`~twinkle_agentic.verifier.Verifier`.

Two layers, matching the literature:
- **Structural (per-turn)** — free, deterministic. A "turn" is one assistant
  message plus the tool result messages it triggered (Web-Shepherd / AgentPRM
  style turn-level MDP). See :class:`TurnSegmenter`.
- **Sub-goal (LLM)** — compress each turn to a one-line intent gist, then make
  ONE LLM pass over the whole (short) gist list to group turns into a few
  coarse sub-goals (Web-Shepherd / MiRA style), then reassemble segments from
  the ORIGINAL messages by index. See :class:`LlmSegmenter`.

Design notes:
- The leading ``system`` message and the first ``user`` message form the
  trajectory *preamble*; it is not itself a scorable segment, but every segment
  carries the preamble (system + original user query) so the verifier keeps the
  task context. This mirrors ``RubricVerifier._infer_query``.
- A new ``user`` message mid-trajectory is a hard boundary (a new turn starts).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Turn:
    """One structural turn: an assistant message + its tool results.

    ``indices`` are positions into the original ``messages`` list so a segment
    can be reassembled verbatim from the source (never from a summary).
    """
    indices: List[int]
    messages: List[dict] = field(default_factory=list)
    role_kind: str = 'assistant'  # 'assistant' | 'user' | 'other'


class Segmenter(ABC):
    """Split a trajectory into scorable sub-trajectories.

    Subclasses implement :meth:`segment`. The base class provides the shared
    structural turn splitter and segment assembly so LLM-based subclasses only
    decide *how turns are grouped*.
    """

    def __call__(self, trajectory: dict, **kwargs) -> List[dict]:
        return self.segment(trajectory, **kwargs)

    @abstractmethod
    def segment(self, trajectory: dict, **kwargs) -> List[dict]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # shared: preamble + structural turn splitting + assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _split_preamble(messages: List[dict]) -> Tuple[List[dict], int]:
        """Return (preamble_messages, start_index).

        Preamble = leading system message(s) + the first user message. Segments
        start at ``start_index`` (the message after the first user turn).
        """
        preamble: List[dict] = []
        i = 0
        n = len(messages)
        while i < n and messages[i].get('role') == 'system':
            preamble.append(messages[i])
            i += 1
        if i < n and messages[i].get('role') == 'user':
            preamble.append(messages[i])
            i += 1
        return preamble, i

    @classmethod
    def split_turns(cls, messages: List[dict], start: int = 0) -> List[Turn]:
        """Group messages[start:] into structural turns.

        A turn begins at an ``assistant`` message and absorbs the following
        ``tool`` messages. A mid-trajectory ``user`` message becomes its own
        boundary turn (role_kind='user'). Stray leading non-assistant messages
        are attached to the first turn.
        """
        turns: List[Turn] = []
        cur: Optional[Turn] = None
        for idx in range(start, len(messages)):
            role = messages[idx].get('role')
            if role == 'assistant':
                cur = Turn(indices=[idx], messages=[messages[idx]], role_kind='assistant')
                turns.append(cur)
            elif role == 'user':
                # hard boundary: user re-prompt starts a fresh turn
                cur = Turn(indices=[idx], messages=[messages[idx]], role_kind='user')
                turns.append(cur)
            else:  # tool / other -> attach to current turn, or open a new one
                if cur is None:
                    cur = Turn(indices=[idx], messages=[messages[idx]], role_kind='other')
                    turns.append(cur)
                else:
                    cur.indices.append(idx)
                    cur.messages.append(messages[idx])
        return turns

    @staticmethod
    def _assemble(trajectory: dict, preamble: List[dict], turns_slice: List[Turn]) -> dict:
        """Build a segment sub-trajectory from preamble + a slice of turns.

        Messages are taken from the ORIGINAL trajectory (verbatim), so the
        verifier scores real content, not any compressed gist.
        """
        seg_messages: List[dict] = list(preamble)
        for t in turns_slice:
            seg_messages.extend(t.messages)
        seg: Dict[str, Any] = {'messages': seg_messages}
        if trajectory.get('tools'):
            seg['tools'] = list(trajectory['tools'])
        if trajectory.get('user_data'):
            seg['user_data'] = list(trajectory['user_data'])
        return seg

    @classmethod
    def _segments_from_groups(
        cls,
        trajectory: dict,
        preamble: List[dict],
        turns: List[Turn],
        groups: List[List[int]],
    ) -> List[dict]:
        """Assemble segments given a grouping of turn-indices.

        ``groups`` is a list of lists of indices into ``turns``. Robust to
        gaps/overlaps: see :meth:`_normalize_groups`.
        """
        groups = cls._normalize_groups(groups, len(turns))
        return [cls._assemble(trajectory, preamble, [turns[i] for i in grp])
                for grp in groups if grp]

    @staticmethod
    def _normalize_groups(groups: List[List[int]], n_turns: int) -> List[List[int]]:
        """Repair LLM-proposed groupings into a clean partition of 0..n_turns-1.

        - drop out-of-range indices
        - dedupe (first occurrence wins; later duplicates dropped)
        - sort each group; sort groups by their first index
        - assign any uncovered turns to the nearest preceding group (or the
          first group), so every turn lands in exactly one segment
        """
        if n_turns <= 0:
            return []
        seen: set = set()
        cleaned: List[List[int]] = []
        for grp in groups:
            g = []
            for i in grp:
                if isinstance(i, bool):  # guard: bools are ints in python
                    continue
                if isinstance(i, int) and 0 <= i < n_turns and i not in seen:
                    seen.add(i)
                    g.append(i)
            if g:
                cleaned.append(sorted(g))
        cleaned.sort(key=lambda g: g[0])

        # cover missing turns
        missing = [i for i in range(n_turns) if i not in seen]
        if missing:
            if not cleaned:
                cleaned = [missing]
            else:
                for i in missing:
                    # nearest preceding group by first-index
                    target = cleaned[0]
                    for grp in cleaned:
                        if grp[0] <= i:
                            target = grp
                        else:
                            break
                    target.append(i)
                for grp in cleaned:
                    grp.sort()
            cleaned.sort(key=lambda g: g[0])
        return cleaned


class TurnSegmenter(Segmenter):
    """Structural, LLM-free segmenter.

    ``granularity='turn'``: one segment per turn (finest; per-tool-call level).
    ``granularity='cluster'``: merge consecutive tool-using assistant turns into
        one segment, closing the cluster on a turn that produces a user-facing
        text answer with no tool calls (sub-task attempt level). A ``user`` turn
        always starts a new cluster.
    """

    def __init__(self, granularity: str = 'cluster'):
        if granularity not in ('turn', 'cluster'):
            raise ValueError("granularity must be 'turn' or 'cluster'")
        self.granularity = granularity

    def segment(self, trajectory: dict, **kwargs) -> List[dict]:
        messages = list(trajectory.get('messages', []) or [])
        preamble, start = self._split_preamble(messages)
        turns = self.split_turns(messages, start)
        if not turns:
            return [self._assemble(trajectory, preamble, [])] if preamble else []

        if self.granularity == 'turn':
            groups = [[i] for i in range(len(turns))]
        else:
            groups = self._cluster_groups(turns)
        return self._segments_from_groups(trajectory, preamble, turns, groups)

    @staticmethod
    def _cluster_groups(turns: List[Turn]) -> List[List[int]]:
        groups: List[List[int]] = []
        cur: List[int] = []
        for i, t in enumerate(turns):
            if t.role_kind == 'user':
                if cur:
                    groups.append(cur)
                    cur = []
                groups.append([i])  # user re-prompt as its own boundary segment
                continue
            cur.append(i)
            has_tool_call = any(m.get('role') == 'assistant' and m.get('tool_calls')
                                for m in t.messages)
            if not has_tool_call:
                # a text-only assistant answer closes the current sub-task cluster
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        return groups
