# Copyright (c) ModelScope Contributors. All rights reserved.
"""The keyword bank: what the next task gets built about.

Every challenger here proposes from a topic rather than from a fixed prompt,
because a fixed prompt collapses onto a handful of archetypes within a few
hundred proposals. The cycle that prevents it is the same one everywhere -- draw
a combination, refill whichever category ran dry, ask for more of whatever the
solver could not solve -- so it lives here once, in :class:`KeywordBank`, and a
proposer *holds* one rather than inheriting it. Nothing in this module knows what
a task looks like: it deals in short strings and in the prompts its owner supplies,
which is why the two challengers in this package and the RSI drivers in
``cookbook/rsi`` can all share it.

Two failures shaped the file, both from real runs, both recorded where they hit:
a refill that returns nothing has to be loud (a silent one leaves every proposal
falling back to the from-scratch prompt while the run looks healthy), and a
keyword that arrives written as a sentence has to be counted rather than dropped
in silence -- see :func:`split_keyword_list`.
"""
import json
import os
import random
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from twinkle.data_format import SamplingParams, Trajectory, user_data_get
from twinkle.utils import get_logger
from twinkle_agentic.utils.code_utils import strip_reasoning
from twinkle_agentic.utils.message_utils import assistant_text
from .base import Explorer, KeywordPrompts, map_parallel

logger = get_logger()

__all__ = [
    'KEYWORD_MAX_LEN', 'KeywordBank', 'KeywordPrompts', 'KeywordStore',
    'parse_keyword_list', 'split_keyword_list',
]

# A keyword is a topic to build a task around, not a task statement. Past this many
# characters the model has written the second thing, and storing it makes the next
# prompt ask for a variation on a sentence rather than on a subject.
KEYWORD_MAX_LEN = 60


def split_keyword_list(text: str) -> Tuple[List[str], List[str]]:
    """Extract a JSON array of short strings; return (kept, dropped for length).

    The dropped half exists because it used to be discarded inside a list
    comprehension. A refill that returned eight well-formed keywords, all of them
    written out as sentences, reached the caller as an empty list and was recorded
    as ``n_parsed: 0`` -- the same three characters a garbled reply, a timeout and
    an over-length reply all produce, so the log could not tell them apart. One
    iteration lost 27% of its keywords that way and the cause was found by
    re-parsing the stored replies by hand.

    The bias is the reason to count rather than only to log: length correlates with
    specificity, so the filter removes "Compute the critical path delay through a
    gate-level netlist with annotated cell delays" and keeps whatever was vague
    enough to be short. That is the opposite of what the bank is for.
    """
    body = strip_reasoning(text)
    start, end = body.find('['), body.rfind(']')
    if start < 0 or end <= start:
        return [], []
    try:
        arr = json.loads(body[start:end + 1])
    except (ValueError, TypeError):
        return [], []
    kept: List[str] = []
    dropped: List[str] = []
    for x in arr:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        (kept if len(s) <= KEYWORD_MAX_LEN else dropped).append(s)
    return kept, dropped


def parse_keyword_list(text: str) -> List[str]:
    """The kept half of :func:`split_keyword_list`, for callers with nothing to record."""
    return split_keyword_list(text)[0]


# ── keyword bank ───────────────────────────────────────────────────────────
class KeywordStore:
    """Persistent keyword bank with usage tracking, one bucket per category.

    Keywords exist to stop the challenger collapsing onto a handful of
    archetypes. They are consumed rather than sampled with replacement, so a
    run keeps reaching for topics it has not used; when a bucket runs dry the
    caller refills it from the model, and recycles only if the model has run out
    of distinct ideas.

    On-disk format (one JSON per line)::

        {"category", "text", "used": bool, "used_count": int,
         "source": "gen"|"expand", "parent": <keyword or null>}

    De-duplicates case-insensitively within a category, so re-runs never
    conflict with the bank on disk.
    """

    def __init__(self, path: str, categories: Sequence[str]):
        if not categories:
            raise ValueError('KeywordStore needs at least one category')
        self.path = path
        self.categories = tuple(categories)
        self.items: Dict[str, List[Dict[str, Any]]] = {c: [] for c in self.categories}
        self._seen: Dict[str, set] = {c: set() for c in self.categories}
        if path and os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    c, t = r.get('category'), r.get('text')
                    if c in self.items and isinstance(t, str) and t.strip():
                        key = t.strip().lower()
                        if key not in self._seen[c]:
                            self._seen[c].add(key)
                            self.items[c].append(r)

    def save(self) -> None:
        """Write the bank out, atomically. A bank without a path is in-memory only."""
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            for c in self.categories:
                for r in self.items[c]:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        os.replace(tmp, self.path)

    def add(self, category: str, texts: Sequence[str], source: str = 'gen',
            parent: Optional[str] = None) -> int:
        added = 0
        for t in texts:
            key = t.strip().lower()
            if not key or key in self._seen[category]:
                continue
            self._seen[category].add(key)
            self.items[category].append({'category': category, 'text': t.strip(),
                                         'used': False, 'used_count': 0,
                                         'source': source, 'parent': parent})
            added += 1
        return added

    def unused(self, category: str) -> List[Dict[str, Any]]:
        return [r for r in self.items[category] if not r.get('used')]

    def texts(self, category: str) -> List[str]:
        return [r['text'] for r in self.items[category]]

    def take(self, category: str, rng: random.Random) -> Optional[str]:
        """Consume one unused keyword from ``category``; None if it is dry."""
        un = self.unused(category)
        if not un:
            return None
        r = rng.choice(un)
        r['used'] = True
        r['used_count'] = r.get('used_count', 0) + 1
        return r['text']

    def recycle(self, category: str) -> None:
        """Mark every keyword unused again (safety valve when the model is tapped out)."""
        for r in self.items[category]:
            r['used'] = False


class KeywordBank:
    """The draw / refill / expand cycle over a :class:`KeywordStore`.

    Held by whoever proposes rather than inherited: the two challengers in this
    package share this entire cycle and almost nothing else, and the RSI drivers
    in ``cookbook/rsi`` are not challengers at all yet need exactly the same
    thing. Safe to drive from several threads -- see :meth:`draw`.

    Args:
        store: the bank this works on.
        prompts: the :class:`KeywordPrompts` this bank sends.
        category_desc: category -> description, shown when asking for more. Must
            cover every category in the store, or a dry one could not be refilled.
        explorer: batch-in / batch-out generation, used for keyword calls only.
            Worth keeping separate from the proposing explorer: brainstorming a
            list is a text round, so a tool-calling rollout both wastes turns and
            may take a bracketed list in the reply for a tool call.
        rng: shared with the owner, so one seed reproduces the whole run.
        name: what log lines call this bank; normally the owner's class name.
        sampling_params: params for keyword calls. None sends the explorer's own.
        sink: called once per keyword call with the prompt, the reply and both
            halves of the parse. The one question such a dump exists to answer --
            did the model disobey the format, or does the parser reject what it
            produced -- cannot be answered from a count.
        combo_arity: ``'triple'`` draws one keyword per category; ``'mix'`` draws
            a random 1..len(categories) subset.
        arity_weights: sampling weights for the ``'mix'`` subset size.
        single_kw_prob: in ``'triple'`` mode, the chance of using one category
            instead of all of them.
        refill_target: how many new keywords one refill aims for.
        gen_calls: how many model calls it may spend on that.
        refill_concurrency: how many of those go out together. At 1 every call is
            told what the ones before it produced, which is what the avoid list is
            for; raising it is faster and comes back with more synonyms.
        refill_tries: refills to attempt before recycling a tapped-out category.
        min_batch: smallest batch worth sending -- a sampler shards a batch over
            its data-parallel workers, and a smaller one leaves some with nothing
            to do. Set it to the number of sampler workers.
        expand_per_kw / expand_max_kws: size of the :meth:`expand_hard` ask.
    """

    # How many phrases the 'do not repeat these' line may quote in total. There
    # has to be a ceiling in both directions: too few and a serial refill stops
    # seeing what it just said, too many and the model runs out of room to obey.
    # Measured on armA2ser, where this refill's own output went in uncapped: with
    # 130 quoted the eighth call was still answering normally, with 150 it started
    # inventing -- 'îRAPIÓN holistic replace', 'ซะ subspace cutter map limit', 10
    # of 480 phrases that run. 100 sits below where that began.
    _AVOID_TOTAL = 100
    _AVOID_LEAD = '\nDo NOT repeat any of these already-used topics: '

    def __init__(
        self,
        store: KeywordStore,
        *,
        prompts: KeywordPrompts,
        category_desc: Dict[str, str],
        explorer: Explorer,
        rng: random.Random,
        name: str = 'keywords',
        sampling_params: Optional[SamplingParams] = None,
        sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        combo_arity: str = 'triple',
        arity_weights: Optional[Sequence[float]] = None,
        single_kw_prob: float = 0.1,
        refill_target: int = 128,
        gen_calls: int = 8,
        refill_concurrency: int = 1,
        refill_tries: int = 2,
        min_batch: int = 1,
        expand_per_kw: int = 8,
        expand_max_kws: int = 32,
    ):
        if combo_arity not in ('triple', 'mix'):
            raise ValueError(f"combo_arity must be 'triple' or 'mix', got {combo_arity!r}")
        if refill_concurrency < 1:
            raise ValueError(f'refill_concurrency must be >= 1, got {refill_concurrency}')
        missing = [c for c in store.categories if not (category_desc or {}).get(c)]
        if missing:
            raise ValueError(f'category_desc is missing a description for {missing}; '
                             f'a dry category could not be refilled.')
        self.store = store
        self.prompts = prompts
        self.category_desc = dict(category_desc)
        self.explorer = explorer
        self.rng = rng
        self.name = name
        self.sampling_params = sampling_params
        self.sink = sink
        self.combo_arity = combo_arity
        self.arity_weights = list(arity_weights) if arity_weights else None
        self.single_kw_prob = single_kw_prob
        self.refill_target = refill_target
        self.gen_calls = gen_calls
        self.refill_concurrency = refill_concurrency
        self.refill_tries = refill_tries
        self.min_batch = max(1, min_batch)
        self.expand_per_kw = expand_per_kw
        self.expand_max_kws = expand_max_kws
        # One draw at a time; see :meth:`draw` for why the whole draw and not just
        # the store access.
        self._draw_lock = threading.Lock()
        # Held while the rng, the nonce, the bank or the hard list are touched, and
        # never across a model call. Separate from the sink lock, which waits on
        # disk: a refill running in another thread must not queue behind a write.
        self._state_lock = threading.Lock()
        self._sink_lock = threading.Lock()
        # Perturbs prompts so two calls are never byte-identical.
        self._nonce = 0
        # (category, keyword) behind whatever nobody could solve, for expand_hard.
        self._hard: List[Tuple[str, str]] = []

    # -------------------------------------------------------------- drawing

    @property
    def categories(self) -> Tuple[str, ...]:
        return self.store.categories

    @staticmethod
    def block(picks: Sequence[Tuple[str, str]]) -> str:
        """The drawn keywords as the line block a prompt's ``{keywords}`` takes."""
        return '\n'.join(f'- {c}: {t}' for c, t in picks)

    def draw(self) -> List[Tuple[str, str]]:
        """Consume one keyword combination, refilling whatever ran dry first.

        Serialised as a whole rather than per bank access: a refill is a batch of
        model calls whose prompts quote what the calls before them produced, and
        two draws overlapping would each refill without seeing the other's
        keywords -- exactly what the avoid list exists to prevent.
        """
        with self._draw_lock:
            cats = self._pick_categories()
            # Refill every dry category at once rather than as each one is reached:
            # they are independent model calls that used to run one after another
            # (20s each at the start of a run) and they touch separate buckets.
            dry = [c for c in cats if not self.store.unused(c)]
            if dry:
                map_parallel(self.refill, dry)
            picks: List[Tuple[str, str]] = []
            for category in cats:
                with self._state_lock:
                    text = self.store.take(category, self.rng)
                if text is not None:
                    picks.append((category, text))
            return picks

    def _pick_categories(self) -> List[str]:
        """Which categories one draw covers, per ``combo_arity``."""
        categories = self.store.categories
        with self._state_lock:
            if self.combo_arity == 'mix':
                if self.arity_weights and len(self.arity_weights) == len(categories):
                    k = self.rng.choices(range(1, len(categories) + 1),
                                         weights=self.arity_weights)[0]
                else:
                    k = self.rng.randint(1, len(categories))
                return self.rng.sample(list(categories), k)
            if self.rng.random() < self.single_kw_prob:
                return [self.rng.choice(categories)]
        return list(categories)

    # ------------------------------------------------------------- refilling

    def refill(self, category: str) -> None:
        """Ask the model for more keywords in ``category``; recycle if it is tapped out.

        Says so when it comes back empty. A silent no-op here is the worst outcome
        available: :meth:`draw` then hands out no keywords, every proposal quietly
        falls back to the from-scratch prompt, and the run looks normal while
        producing one identical prompt over and over. That is exactly what happened
        for whole runs when the prompt asked for one keyword per line and the parser
        wanted a JSON array.
        """
        tries = 0
        while not self.store.unused(category):
            new = self._generate(category, self.refill_target)
            with self._state_lock:
                added = self.store.add(category, new, source='gen')
                if added:
                    # Saved now rather than at the end of the run: a refill costs a
                    # batch of model calls, and a run that crashes later should not
                    # have to spend them again -- the next iteration reads this file
                    # to know what was already used.
                    self.store.save()
            tries += 1
            if added:
                logger.info(f'[{self.name}] keyword category {category!r} refilled '
                            f'+{added} (try {tries})')
                continue
            logger.warning(
                f'[{self.name}] keyword refill for {category!r} produced nothing on try '
                f'{tries}: {len(new)} parsed, 0 new. Proposals will run without keywords '
                f'unless this recovers -- pass a keyword sink to see the replies.')
            if tries >= self.refill_tries:
                with self._state_lock:
                    # Every keyword marked unused again. The alternative is a
                    # category that can never be drawn from, which stops the run: a
                    # repeat draw is worse than no run only if diversity matters
                    # more than collecting anything at all.
                    n_recycled = len(self.store.items[category])
                    if n_recycled:
                        self.store.recycle(category)
                        self.store.save()
                if n_recycled:
                    logger.info(f'[{self.name}] keyword category {category!r} exhausted '
                                f'-> recycled {n_recycled} topics')
                break

    def _generate(self, category: str, n_want: int) -> List[str]:
        """Up to ``n_want`` keywords the bank does not already hold."""
        if n_want <= 0:
            return []
        with self._state_lock:
            known = self.store.texts(category)
        n_calls = max(self.gen_calls, self.min_batch)
        per_call = max(1, -(-n_want // n_calls) + 4)   # ceil(n/calls) + margin
        seen = {t.strip().lower() for t in known}
        out: List[str] = []
        n_long = 0
        for start in range(0, n_calls, self.refill_concurrency):
            group = range(start, min(start + self.refill_concurrency, n_calls))
            # Every call in a group is built before any of them runs, so they all
            # carry the same avoid list -- which is exactly the batched behaviour,
            # and why a group of one is what lets call k+1 see call k.
            users = [(self.prompts.user.format(
                k=per_call, desc=self.category_desc[category])
                + self._avoid_note(known, out)
                + f'\n(batch {self._next_nonce()}-{i})') for i in group]
            for user, reply in zip(users, self._explore(users)):
                text = assistant_text(reply)
                parsed, dropped_long = split_keyword_list(text)
                n_long += len(dropped_long)
                fresh = [kw for kw in parsed if kw.lower() not in seen]
                seen.update(kw.lower() for kw in fresh)
                out.extend(fresh)
                # Full text, both sides: the question this dump answers is whether
                # the model disobeyed the format or the parser rejected what it
                # produced, and a count cannot say which.
                self._record({
                    'category': category, 'prompt': user, 'reply': text,
                    'stop_reason': reply.get('stop_reason'),
                    'truncated': bool(reply.get('truncated')),
                    'parsed': parsed, 'n_parsed': len(parsed), 'n_new': len(fresh),
                    # Which backend answered, for an explorer that has more than
                    # one -- an API with a local fallback is the case this exists
                    # for, and nothing else here could know which one ran.
                    'via': reply.get('via'),
                    # The two fields that make the sentence above true. Without them
                    # ``n_parsed: 0`` reads the same whether the reply was garbled,
                    # empty, or eight usable keywords written at sentence length --
                    # and the third is the one that happened.
                    'dropped_long': dropped_long, 'n_dropped_long': len(dropped_long),
                })
            if len(out) >= n_want:
                # The surplus is dropped below, so further calls would buy nothing.
                break
        if n_long and self.sink is None:
            # Without a dump to write to, the count has to be said out loud or the
            # refill looks like the model simply produced less.
            logger.warning(f'[{self.name}] dropped {n_long} keyword(s) over '
                           f'{KEYWORD_MAX_LEN} chars while refilling; the prompt is '
                           f'asking for task statements rather than topics')
        with self._state_lock:
            self.rng.shuffle(out)
        return out[:n_want]

    def _avoid_note(self, older: List[str], fresh: List[str]) -> str:
        """The 'do not repeat these' line, newest first, capped at ``_AVOID_TOTAL``.

        What this refill has just produced comes first and evicts older entries
        rather than the reverse -- the calls run one at a time so that each can
        avoid what the ones before it said, and dropping those would undo it. Past
        the cap the oldest of *this refill's* phrases are what falls off, which is
        also the least costly thing to drop: the model has already moved away from
        them.
        """
        fresh_shown = list(fresh)[-self._AVOID_TOTAL:]
        room = max(0, self._AVOID_TOTAL - len(fresh_shown))
        with self._state_lock:
            shown = older if len(older) <= room else self.rng.sample(older, room)
        avoid = fresh_shown + list(shown)
        return self._AVOID_LEAD + ', '.join(avoid) if avoid else ''

    # -------------------------------------------------------------- feedback

    def remember_hard(self, picks: Iterable[Sequence[str]]) -> None:
        """Note the (category, keyword) pairs behind something nobody solved.

        De-duplicated case-insensitively and kept in arrival order. This is the
        only feedback the bank gets from difficulty; without it, it drifts wherever
        the refill prompt happens to go.
        """
        with self._state_lock:
            seen = {(c, t.lower()) for c, t in self._hard}
            for pick in picks or ():
                if not (isinstance(pick, (list, tuple)) and len(pick) >= 2):
                    continue
                category, text = pick[0], pick[1]
                if (category, text.lower()) not in seen:
                    seen.add((category, text.lower()))
                    self._hard.append((category, text))

    def remember_unsolved(self, candidates: Sequence[Trajectory], max_pass: int = 0) -> None:
        """:meth:`remember_hard` for measured candidates at or below ``max_pass``.

        Each candidate carries ``n_pass`` and its keyword draw in ``user_data``, so
        this is the whole of what a difficulty round feeds back to the bank.
        """
        for task in candidates:
            data = task.get('user_data')
            if user_data_get(data, 'n_pass', 0) > max_pass:
                continue
            self.remember_hard(user_data_get(data, 'keywords', []) or [])

    def expand_hard(self) -> int:
        """Brainstorm more topics in the families that produced the hardest tasks.

        Called by whoever drives the proposer, after a round, so the bank drifts
        toward material the solver actually struggles with. Returns how many new
        keywords were added.
        """
        if not self._hard or self.expand_per_kw <= 0:
            return 0
        template = self.prompts.expand_user.strip()
        if not template:
            raise ValueError(f'[{self.name}] measured {len(self._hard)} hard keyword(s) to '
                             f'expand on, but the prompts carry no expand_user.')
        with self._state_lock:
            hard = self._hard[:self.expand_max_kws]
            self.rng.shuffle(hard)
        # Cycle a short list so the batch still covers every sampler worker.
        reqs = list(hard)
        while len(reqs) < self.min_batch:
            reqs.append(hard[len(reqs) % len(hard)])
        # ``desc`` is offered alongside ``kw``/``m``: a template that does not ask
        # for it ignores it, and one that does gets the category's rules with it.
        users = [(template.format(kw=kw, m=self.expand_per_kw,
                                  desc=self.category_desc.get(category, ''))
                  + f'\n(batch {self._next_nonce()}-{i})')
                 for i, (category, kw) in enumerate(reqs)]
        added = 0
        n_long = 0
        for (category, kw), user, reply in zip(reqs, users, self._explore(users)):
            text = assistant_text(reply)
            parsed, dropped_long = split_keyword_list(text)
            n_long += len(dropped_long)
            with self._state_lock:
                added += self.store.add(category, parsed, source='expand', parent=kw)
            # The prompt goes in whole, as the refill path already does. This used
            # to record the literal string 'expand', which left the dump unable to
            # answer the one question it gets asked -- whether a change to the
            # expansion prompt was live in a given iteration.
            self._record({
                'category': category, 'parent': kw, 'prompt': user, 'reply': text,
                'stop_reason': reply.get('stop_reason'),
                'truncated': bool(reply.get('truncated')),
                'parsed': parsed, 'n_parsed': len(parsed),
                'dropped_long': dropped_long, 'n_dropped_long': len(dropped_long),
                'via': reply.get('via'),
            })
        if added:
            with self._state_lock:
                self.store.save()
        if n_long and self.sink is None:
            logger.warning(f'[{self.name}] dropped {n_long} expanded keyword(s) over '
                           f'{KEYWORD_MAX_LEN} chars; expansion follows the parent, so a '
                           f'wordy parent produces wordy children')
        logger.info(f'[{self.name}] expanded {len(hard)} hard keyword(s) -> '
                    f'+{added} same-domain topics')
        return added

    def save(self) -> None:
        """Write the bank out. Refills and expansions already do; this is for the end."""
        with self._state_lock:
            self.store.save()

    # --------------------------------------------------------------- private

    def _next_nonce(self) -> int:
        """A number no other call gets, so two prompts are never byte-identical.

        Shared across categories, which refill at the same time: two threads
        reading the counter together would send the same prompt twice and halve the
        diversity with nothing to show that it happened.
        """
        with self._state_lock:
            self._nonce += 1
            return self._nonce

    def _explore(self, users: Sequence[str]) -> List[Trajectory]:
        """One batch of keyword calls, one per user message.

        Sampling params are only forwarded when set, so a plain callable explorer
        that takes nothing else keeps working.
        """
        if not users:
            return []
        prompts: List[Trajectory] = [{
            'messages': [{'role': 'system', 'content': self.prompts.system},
                         {'role': 'user', 'content': user}],
        } for user in users]
        if self.sampling_params is None:
            return self.explorer(prompts)
        return self.explorer(prompts, sampling_params=self.sampling_params)

    def _record(self, record: Dict[str, Any]) -> None:
        """Hand one keyword call to the sink, if there is one."""
        if self.sink is None:
            return
        with self._sink_lock:
            self.sink(record)
