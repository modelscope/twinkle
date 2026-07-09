# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compress-then-segment: per-turn action gist + one-pass LLM sub-goal grouping.

Pipeline (all LLM calls distilled via ``llm_backup``):
    1. Split the trajectory into structural turns (base class, free).
    2. Compress each turn to a one-line intent gist with an
       :class:`ActionSummarizer` (optional; falls back to a truncated
       structural render when no summarizer is given).
    3. Make ONE LLM pass over the numbered gist list to group turns into a few
       coarse sub-goals — the segmenter LLM sees the whole (short) trajectory at
       once, which is what makes segmentation global yet cheap.
    4. Reassemble segments from the ORIGINAL messages by turn index (scoring
       always uses verbatim content, never the gist).

The grouping call is wrapped with ``@llm_backup`` (student sub-goal model with
teacher fallback + progressive distillation), keyed by ``query`` so confidence
is tracked per task family. Index alignment is validated/repaired in pure code
by the base class ``_normalize_groups`` (contiguous, non-overlapping, full
coverage).
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, List, Optional

from twinkle_agentic.utils.llm_backup import llm_backup

from .base import Segmenter, Turn

if TYPE_CHECKING:
    from twinkle.data_format import SamplingParams  # noqa: F401
    from twinkle.sampler.base import Sampler  # noqa: F401
    from twinkle_agentic.summarizer.action_summarizer import ActionSummarizer  # noqa: F401


_SEG_SYSTEM = """\
You segment an AI agent's trajectory into a FEW coarse sub-goals for later \
evaluation. You are given the task query and a numbered list of one-line turn \
gists (one per turn). Group CONSECUTIVE turns that jointly pursue the same \
sub-goal.

Guidelines:
- Prefer {min_g}-{max_g} sub-goals total. Each sub-goal spans a contiguous
  range of turns; ranges must not overlap and must cover every turn.
- Group by MEANINGFUL task progress, not by exact actions: e.g. several
  consecutive searches that gather info for one purpose form ONE sub-goal.
- A user re-prompt starts a new sub-goal.

Output ONLY a JSON array, each item: {"goal": "<short label>", "start": <first turn #>, "end": <last turn #>}
Turn numbers are 0-based and inclusive. No prose, no markdown fence."""

_SEG_USER = """\
## Task / query
{query}

## Turn gists (numbered)
{gists}

Now output the JSON array of sub-goals covering turns 0..{last}."""


class LlmSegmenter(Segmenter):
    """Sub-goal segmenter via compress-then-one-pass-LLM grouping.

    Args:
        sampler: Student model sampler for the grouping call. If ``None``, every
            grouping call is served by the teacher via ``llm_backup``.
        action_summarizer: Optional :class:`ActionSummarizer` used to compress
            each turn. If ``None``, a truncated structural render is used as the
            gist (no per-turn LLM cost).
        min_subgoals / max_subgoals: Target sub-goal count window.
        sampling_params: Sampling params for the grouping call.
        lora_path: LoRA adapter for the sub-goal student model.
        max_gist_chars: Truncation for the structural-render fallback gist.
    """

    def __init__(
        self,
        sampler: Optional['Sampler'] = None,
        *,
        action_summarizer: Optional['ActionSummarizer'] = None,
        min_subgoals: int = 3,
        max_subgoals: int = 6,
        sampling_params: Optional['SamplingParams'] = None,
        lora_path: Optional[str] = None,
        max_gist_chars: int = 200,
    ):
        if max_subgoals < min_subgoals:
            raise ValueError('max_subgoals must be >= min_subgoals')
        if min_subgoals < 1:
            raise ValueError('min_subgoals must be >= 1')
        self.sampler = sampler
        self.action_summarizer = action_summarizer
        self.min_subgoals = int(min_subgoals)
        self.max_subgoals = int(max_subgoals)
        self.sampling_params = sampling_params
        self.lora_path = lora_path or None
        self.max_gist_chars = int(max_gist_chars)

    # ------------------------------------------------------------------
    def segment(self, trajectory: dict, *, query: Optional[str] = None, **kwargs) -> List[dict]:
        messages = list(trajectory.get('messages', []) or [])
        preamble, start = self._split_preamble(messages)
        turns = self.split_turns(messages, start)
        if not turns:
            return [self._assemble(trajectory, preamble, [])] if preamble else []
        # Too few turns to bother segmenting -> one segment.
        if len(turns) <= self.min_subgoals:
            return self._segments_from_groups(
                trajectory, preamble, turns, [[i] for i in range(len(turns))])

        from .base import TurnSegmenter

        # No LLM available at all (no student sampler AND no teacher configured)
        # -> degrade to free structural clustering instead of crashing.
        if not self._llm_available():
            groups = TurnSegmenter._cluster_groups(turns)
            return self._segments_from_groups(trajectory, preamble, turns, groups)

        query = query or self._infer_query(messages)
        gists = [self._turn_gist(t, query) for t in turns]
        gist_block = '\n'.join(f'[{i}] {g}' for i, g in enumerate(gists))

        raw = self._group(
            trajectory=self._group_trajectory(query, gist_block, len(turns)),
            sampling_params=self._group_sampling_params(),
            query=query)
        groups = self._parse_groups(raw, len(turns))
        if not groups:
            # LLM produced nothing usable -> fall back to structural clustering.
            groups = TurnSegmenter._cluster_groups(turns)
        return self._segments_from_groups(trajectory, preamble, turns, groups)

    def _llm_available(self) -> bool:
        """True if a student sampler exists or a teacher API is configured.

        Mirrors the env vars ``llm_backup`` uses for its teacher; when neither a
        student nor a teacher is present we must not attempt an LLM call.
        """
        if self.sampler is not None:
            return True
        import os
        return bool(os.environ.get('LLM_BACKUP_API_KEY')
                    or os.environ.get('OPENAI_API_KEY')
                    or os.environ.get('LLM_BACKUP_BASE_URL'))

    # ------------------------------------------------------------------
    # the distilled grouping call
    # ------------------------------------------------------------------
    @llm_backup(key_params=['query'], comparator=lambda a, b: _grouping_similar(a, b))
    def _group(self, trajectory, sampling_params, query: str = None) -> str:
        if self.sampler is None:
            return ''
        sample_kwargs: dict[str, Any] = {'sampling_params': sampling_params}
        if self.lora_path is None:
            sample_kwargs['use_base_model'] = True
        else:
            sample_kwargs['adapter_path'] = self.lora_path
        responses = self.sampler.sample([trajectory], **sample_kwargs)
        resp = list(responses)[0] if responses else None
        if resp is None:
            return ''
        seqs = getattr(resp, 'sequences', None) or []
        return (getattr(seqs[0], 'decoded', None) or '') if seqs else ''

    # ------------------------------------------------------------------
    # per-turn gist
    # ------------------------------------------------------------------
    def _turn_gist(self, turn: Turn, query: str) -> str:
        rendered = self._render_turn(turn)
        if self.action_summarizer is not None:
            try:
                gist = self.action_summarizer(rendered, query=query)
                if isinstance(gist, str) and gist.strip():
                    return ' '.join(gist.split())[:self.max_gist_chars]
            except Exception:
                pass
        # fallback: truncated structural render (no LLM)
        return ' '.join(rendered.split())[:self.max_gist_chars]

    @staticmethod
    def _render_turn(turn: Turn) -> str:
        parts: List[str] = []
        for m in turn.messages:
            role = m.get('role', '?')
            content = m.get('content')
            if isinstance(content, list):
                content = '\n'.join(p.get('text', '') for p in content
                                    if isinstance(p, dict) and p.get('type') == 'text')
            content = content or ''
            tool_calls = m.get('tool_calls') or []
            if tool_calls:
                names = ', '.join((tc.get('function') or {}).get('name', '?')
                                  for tc in tool_calls if isinstance(tc, dict))
                parts.append(f'{role}: {content} [tool_calls: {names}]'.strip())
            else:
                parts.append(f'{role}: {content}'.strip())
        return ' | '.join(p for p in parts if p)

    # ------------------------------------------------------------------
    # prompt / sampling plumbing
    # ------------------------------------------------------------------
    def _group_trajectory(self, query: str, gist_block: str, n_turns: int) -> dict:
        system = (_SEG_SYSTEM
                  .replace('{min_g}', str(self.min_subgoals))
                  .replace('{max_g}', str(self.max_subgoals)))
        user = (_SEG_USER
                .replace('{query}', query)
                .replace('{gists}', gist_block)
                .replace('{last}', str(n_turns - 1)))
        return {'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ]}

    def _group_sampling_params(self):
        if self.sampling_params is not None:
            return self.sampling_params
        from twinkle.data_format.sampling import SamplingParams
        return SamplingParams(temperature=0.0, max_tokens=512)

    # ------------------------------------------------------------------
    # parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_query(messages: List[dict]) -> str:
        for m in messages:
            if m.get('role') == 'user':
                c = m.get('content')
                if isinstance(c, str) and c.strip():
                    return c.strip()
        return '(no explicit query)'

    _JSON_ARRAY_RE = re.compile(r'\[.*\]', re.DOTALL)

    @classmethod
    def _parse_groups(cls, raw: str, n_turns: int) -> List[List[int]]:
        """Parse the sub-goal JSON into a list of turn-index groups.

        Accepts ``[{"goal":..,"start":i,"end":j}, ...]``. Falls back to empty
        on unparseable output (caller then uses structural clustering).
        """
        if not raw:
            return []
        text = raw.strip()
        m = cls._JSON_ARRAY_RE.search(text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        groups: List[List[int]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            s, e = item.get('start'), item.get('end')
            if not isinstance(s, int) or not isinstance(e, int):
                continue
            if e < s:
                s, e = e, s
            s = max(0, s)
            e = min(n_turns - 1, e)
            grp = list(range(s, e + 1))
            if grp:
                groups.append(grp)
        return groups


# ---------------------------------------------------------------------------
# comparator for llm_backup
# ---------------------------------------------------------------------------
_JSON_ARRAY_RE = re.compile(r'\[.*\]', re.DOTALL)


def _boundaries(raw: str) -> Optional[List[int]]:
    m = _JSON_ARRAY_RE.search((raw or '').strip())
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    starts = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get('start'), int):
            starts.append(item['start'])
    return sorted(starts) if starts else None


def _grouping_similar(a: str, b: str) -> bool:
    """Two segmentations match when they have a similar number of sub-goals and
    near-identical boundaries (not byte-identical labels/text)."""
    ba, bb = _boundaries(a), _boundaries(b)
    if ba is None or bb is None:
        return (a or '').strip() == (b or '').strip()
    if abs(len(ba) - len(bb)) > 1:
        return False
    # Jaccard-ish agreement on boundary start positions
    sa, sb = set(ba), set(bb)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union >= 0.6
