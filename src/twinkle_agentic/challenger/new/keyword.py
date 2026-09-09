# Copyright (c) ModelScope Contributors. All rights reserved.
"""Keywords per direction: generate, de-duplicate, store, read back.

A keyword is a *topic* to build a task around, not a task statement, which is
why over-length replies are dropped rather than stored.
"""
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from twinkle.data_format import SamplingParams, Trajectory
from twinkle.utils import get_logger
from twinkle_agentic.rollout import MultiTurnRollout
from twinkle_agentic.utils.code_utils import strip_reasoning
from twinkle_agentic.utils.message_utils import assistant_text

logger = get_logger()

__all__ = ['KEYWORD_MAX_LEN', 'KeywordGenerator']

KEYWORD_MAX_LEN = 60


class KeywordGenerator:
    """Keyword combinations drawn from one list per direction.

    ``keywords_group_size`` of the directions are active at a time and one draw
    takes a keyword from each. What a draw spends is the *combination*, not the
    keywords: a group only has to differ from every group already handed out, so
    three directions holding ``num_keywords`` each are worth their product in
    draws rather than just ``num_keywords``. A direction that has produced
    ``num_keywords`` is retired and the next unused one takes its slot, which is
    why more directions than a group needs is the normal case. De-duplication of
    the keywords themselves is flat, so a keyword one direction produced is never
    handed to another.

    Args:
        query: what the keywords have to satisfy -- one entry per direction. Must
            be at least ``keywords_group_size`` of them.
        backend: an API client or a sampler; driven through ``MultiTurnRollout``.
        path: JSONL cache. Empty means in-memory only.
        num_keywords: a direction's budget; past it, it is retired.
        keywords_group_size: how many keywords one draw combines.
        system_prompt: overrides the built-in one.
        recycle: once every direction is spent, hand out the same combinations
            again instead of returning None.
        rollout_kwargs: passed to ``MultiTurnRollout``. ``template`` is required;
            API request options belong in ``api_kwargs``.
    """

    # How many known keywords the 'do not repeat these' line may quote. A cap in
    # both directions: too few and a second round says the same things again, too
    # many and the model runs out of room to obey.
    _avoid_max = 100
    _avoid_lead = '\nDo NOT repeat any of these: '

    # A default prompt to use to generate the keywords
    _default_prompt = (
        'You brainstorm topics. Reply with a JSON array of short noun phrases '
        f'(at most {KEYWORD_MAX_LEN} characters each) and nothing else. '
        'Each phrase names a subject to build a task around, never a task statement.')

    _user_prompt = 'Give {k} distinct topics that satisfy:\n{query}'

    def __init__(
        self,
        query: Sequence[str],
        backend: Any,
        path: str,
        *,
        num_keywords: int = 64,
        keywords_group_size: int = 3,
        system_prompt: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None,
        recycle: bool = False,
        **rollout_kwargs: Any,
    ):
        self.query = list(query)
        if keywords_group_size < 1:
            raise ValueError(f'keywords_group_size must be >= 1, got {keywords_group_size}')
        if len(self.query) < keywords_group_size:
            raise ValueError(f'{len(self.query)} query(ies) cannot fill a group of '
                             f'{keywords_group_size}')
        self.path = path
        self.num_keywords = num_keywords
        self.keywords_group_size = keywords_group_size
        self.recycle = recycle
        self.system_prompt = system_prompt or self._default_prompt
        # Built on the first call rather than here, so a fully cached run needs no backend.
        self._backend = backend
        self._rollout_kwargs = dict(rollout_kwargs, sampling_params=sampling_params, max_turns=1)
        self._rollout: Optional[Any] = None
        self._cached_keywords: Dict[str, List[str]] = self.load_keywords()
        # Flat: one keyword belongs to one direction, whichever produced it first.
        self._seen = {kw.lower() for kws in self._cached_keywords.values() for kw in kws}
        # The active slots, the next direction to promote, which slot retires
        # next, and the mixed-radix counter walking the active buckets. Drawn
        # combinations are remembered because a bucket growing mid-run shifts the
        # counter's order and would otherwise let it land on an old group again.
        self._active = list(self.query[:keywords_group_size])
        self._next_query = keywords_group_size
        self._retire_slot = 0
        self._odometer = [0] * keywords_group_size
        self._drawn: Set[Tuple[str, ...]] = set()
        self._recycled = False

    # ------------------------------------------------------------------- get

    def get_keywords(self, num_groups: int = 1) -> Optional[List[List[str]]]:
        """Up to ``num_groups`` combinations of ``keywords_group_size`` keywords each.

        Fewer than asked for when the directions run dry mid-way -- a partial
        batch is still usable -- and None when not even one group could be
        filled, which is the caller's signal to stop.
        """
        if num_groups < 1:
            raise ValueError(f'num_groups must be >= 1, got {num_groups}')
        groups: List[List[str]] = []
        for _ in range(num_groups):
            group = self._draw_group()
            if group is None:
                break
            groups.append(group)
        return groups or None

    def _draw_group(self) -> Optional[List[str]]:
        """The next combination nobody has been handed, widening the pool to find one."""
        while True:
            group = self._step()
            if group is not None:
                return group
            if not self._grow_or_retire():
                return None

    def _step(self) -> Optional[List[str]]:
        """One sweep of the odometer for an undrawn combination. None once there is none."""
        buckets = [self._cached_keywords.get(q, []) for q in self._active]
        total = 1
        for bucket in buckets:
            total *= len(bucket)
        for _ in range(total):
            combo = tuple(bucket[i] for bucket, i in zip(buckets, self._odometer))
            self._advance(buckets)
            if combo not in self._drawn:
                self._drawn.add(combo)
                self._recycled = False
                return list(combo)
        return None

    def _advance(self, buckets: Sequence[Sequence[str]]) -> None:
        """Odometer +1, last slot first, carrying into the one before it."""
        for slot in reversed(range(len(buckets))):
            self._odometer[slot] += 1
            if self._odometer[slot] < len(buckets[slot]):
                return
            self._odometer[slot] = 0

    def _grow_or_retire(self) -> bool:
        """Widen the combination space: more keywords, else a new direction.

        False once neither is left. Growing comes first because it multiplies what
        the current slots are worth, while retiring gives up on a direction.
        """
        short = [q for q in self._active
                 if len(self._cached_keywords.get(q, [])) < self.num_keywords]
        # A round that adds nothing means the model has run out of distinct ideas
        # for these directions, so asking again would only spend calls.
        if short and self.generate(short):
            return True
        # Round-robin, so the surplus queries are spent evenly across the slots.
        slot = self._retire_slot
        self._retire_slot = (slot + 1) % self.keywords_group_size
        return self._retire(slot)

    def _retire(self, slot: int) -> bool:
        """Promote the next unused direction into ``slot``. False once nothing is left to serve."""
        if self._next_query < len(self.query):
            self._active[slot] = self.query[self._next_query]
            self._next_query += 1
            self._odometer = [0] * self.keywords_group_size
            return True
        # Recycling twice without a group in between would spin forever, so it is
        # allowed only once per exhaustion -- ``_step`` clears the flag on success.
        if self._recycled or not self.recycle or not any(self._cached_keywords.values()):
            logger.warning(f'all {len(self.query)} query(ies) are spent; '
                           f'pass recycle=True to hand out the same groups again')
            return False
        self._drawn.clear()
        self._active = list(self.query[:self.keywords_group_size])
        self._next_query = self.keywords_group_size
        self._odometer = [0] * self.keywords_group_size
        self._recycled = True
        logger.info(f'[{type(self).__name__}] every query spent -> recycling the combinations')
        return True

    # -------------------------------------------------------------- generate

    def generate(self, query: Optional[Sequence[str]] = None) -> int:
        """Ask every direction (or just ``query``) for more. Returns how many landed.

        Callable as often as wanted: each round tells the model what that
        direction already holds, so the lists grow instead of repeating.
        """
        query = list(query if query is not None else self.query)
        added = self._add_to_cached(query, self._generate_keywords(query))
        if added:
            self.save_keywords()
        return added

    def _generate_keywords(self, query: Sequence[str]) -> List[List[str]]:
        """One model call per direction, in a single batch; replies stay aligned with ``query``."""
        prompts: List[Trajectory] = [{
            'messages': [{'role': 'system', 'content': self.system_prompt},
                         {'role': 'user', 'content': self._build_user_prompt(q)}],
        } for q in query]
        if self._rollout is None:
            self._rollout = MultiTurnRollout(self._backend, **self._rollout_kwargs)
        return [self._parse_keywords_from_response(assistant_text(t))
                for t in self._rollout(prompts)]

    def _build_user_prompt(self, query: str) -> str:
        """The ask for one direction, plus what it already holds as an avoid list."""
        known = self._cached_keywords.get(query, [])
        want = max(1, self.num_keywords - len(known))
        user = self._user_prompt.format(k=want, query=query)
        if known:
            user += self._avoid_lead + ', '.join(known[-self._avoid_max:])
        return user

    @staticmethod
    def _parse_keywords_from_response(text: str) -> List[str]:
        """The JSON array in ``text``, over-length and non-string entries dropped."""
        body = strip_reasoning(text)
        start, end = body.find('['), body.rfind(']')
        if start < 0 or end <= start:
            return []
        try:
            arr = json.loads(body[start:end + 1])
        except (ValueError, TypeError):
            return []
        return [s for s in (x.strip() for x in arr if isinstance(x, str))
                if 0 < len(s) <= KEYWORD_MAX_LEN]

    # ----------------------------------------------------------------- store

    def _add_to_cached(self, query: Sequence[str],
                       keywords: Sequence[Sequence[str]]) -> int:
        """Append each direction's new keywords, case-insensitively. Returns how many landed."""
        added = 0
        for q, kws in zip(query, keywords):
            bucket = self._cached_keywords.setdefault(q, [])
            for kw in kws:
                if kw.lower() in self._seen:
                    continue
                self._seen.add(kw.lower())
                bucket.append(kw)
                added += 1
        if not added:
            # Silence here would read as a model that simply produced less.
            logger.warning(f'no new keyword for {len(query)} direction(s); '
                           f'everything generated was already known')
        return added

    def load_keywords(self) -> Dict[str, List[str]]:
        """Read the cache back, one direction per line. An unreadable line is skipped."""
        cached: Dict[str, List[str]] = {}
        if not (self.path and os.path.exists(self.path)):
            return cached
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(r.get('query'), str) and isinstance(r.get('keywords'), list):
                    cached[r['query']] = [kw for kw in r['keywords'] if isinstance(kw, str)]
        return cached

    def save_keywords(self) -> None:
        """Write the cache out atomically, so a crash mid-write cannot truncate it."""
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            for q, kws in self._cached_keywords.items():
                f.write(json.dumps({'query': q, 'keywords': kws}, ensure_ascii=False) + '\n')
        os.replace(tmp, self.path)
