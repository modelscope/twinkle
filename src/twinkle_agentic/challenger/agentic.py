# Copyright (c) ModelScope Contributors. All rights reserved.
"""Agentic challenger: invent tasks by doing them first.

The approach mirrors how the code challenger works, adapted to tool-using
agents. Instead of writing a problem statement and hoping it is achievable,
the model first *does* something interesting in a sandbox (round 1), then a
second call writes check assertions that verify the end state, and a third
call writes the problem statement someone else would need to reproduce it.

Steps for one candidate:

    1. Choose direction + keywords. Optionally start from a seed trajectory.
    2. Round 1 (explore, multi-turn with tools): model acts in a clean sandbox,
       producing a tool-call chain and a final workspace state.
    3. Round 2a (explore, single-turn): model sees the trajectory and writes a
       python check script that asserts properties of the end state.
    4. Verify: run the check script in the sandbox (must pass).
    5. Round 2b (explore, single-turn): model sees trajectory + checks and
       writes a problem statement.
    6. Difficulty filter: reset workspace, let the solver do the task N times,
       run checks, keep only "sometimes pass" tasks.

Because every round-1 episode needs a clean workspace and because episodes
share a single long-lived sandbox, round 1 is **serial** -- one proposal at a
time with a workspace reset in between. Rounds 2a/2b are text-only generation
and can be batched.

Prompt text is not here. Every string the model sees arrives in
:class:`AgenticPrompts`, built by whoever runs the challenger -- see
``cookbook/rsi/agentic/prompts.py``.
"""
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format import SamplingParams, Trajectory, user_data_get
from twinkle.utils import get_logger
from .base import Challenger, Explorer, assistant_text, attach_user_data
from .code import KeywordStore, parse_keyword_list

logger = get_logger()

__all__ = [
    'AgenticChallenger',
    'AgenticPrompts',
    'parse_check_script',
    'parse_problem_statement',
]

_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)


# ── parsing ───────────────────────────────────────────────────────────────

def parse_check_script(text: str) -> Optional[str]:
    """Extract a python check script from the model's reply.

    Looks for the last fenced python code block after ``</think>``.
    Returns ``None`` when nothing usable is found.
    """
    body = text or ''
    idx = body.rfind('</think>')
    if idx >= 0:
        body = body[idx + len('</think>'):]
    blocks = _FENCE_RE.findall(body)
    if not blocks:
        return None
    script = blocks[-1].strip()
    return script if script else None


def parse_problem_statement(text: str) -> Optional[str]:
    """Extract a problem statement from the model's reply.

    The model is asked to return the problem in prose (not code). We take
    everything after ``</think>`` with code fences stripped as the statement.
    Returns ``None`` when the result is empty.
    """
    body = text or ''
    idx = body.rfind('</think>')
    if idx >= 0:
        body = body[idx + len('</think>'):]
    # Strip any fenced blocks (those are code, not prose)
    body = _FENCE_RE.sub('', body).strip()
    # Strip json fences too
    body = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', body, flags=re.I).strip()
    return body if body else None


def _trajectory_summary(trajectory: Trajectory) -> str:
    """A compact text representation of a trajectory for prompting.

    Shows each message as role: content (truncated for tool results).
    """
    parts = []
    for msg in trajectory.get('messages') or []:
        role = msg.get('role', '?')
        content = msg.get('content') or ''
        if role == 'tool' and len(content) > 500:
            content = content[:500] + '...[truncated]'
        parts.append(f'[{role}] {content}')
    return '\n'.join(parts)


# The fields a local rollout splices into a trajectory, and the only ones a
# later GRPO step needs: ``labels`` marks which of ``input_ids`` are trainable
# (-100 elsewhere) and ``logprobs`` holds one entry per trainable token, taken
# from the policy that actually generated it.
_TRAINABLE_KEYS = ('input_ids', 'labels', 'logprobs')


def _propose_round(stage: str, trajectory: Trajectory) -> Dict[str, Any]:
    """One proposing round, reduced to what a later training step would read.

    ``messages`` comes along for reading by humans; it is redundant with
    ``input_ids`` and is not what a trainer should encode from.
    """
    record: Dict[str, Any] = {
        'stage': stage,
        'messages': [dict(m) for m in trajectory.get('messages') or []],
    }
    for key in _TRAINABLE_KEYS:
        value = trajectory.get(key)
        if value is not None:
            record[key] = value
    return record


# ── prompts ────────────────────────────────────────────────────────────────

@dataclass
class AgenticPrompts:
    """Every string an :class:`AgenticChallenger` sends.

    All fields are injected by the caller (no defaults with real text here).
    Placeholder validation happens at construction time.
    """

    # Round 1: model acts in sandbox
    system: str
    from_scratch: str
    from_seed: str = ''
    from_keywords: str = ''
    from_seed_keywords: str = ''

    # Round 2a: write check script
    check_system: str = ''
    check_user: str = ''

    # Round 2b: write problem statement
    problem_system: str = ''
    problem_user: str = ''

    # Keyword generation (same structure as code side)
    keyword_system: str = ''
    keyword_user: str = ''
    keyword_expand_user: str = ''

    _REQUIRED_FIELDS = {
        'from_seed': ('seed',),
        'from_keywords': ('keywords',),
        'from_seed_keywords': ('seed', 'keywords'),
        'check_user': ('trajectory', 'final_state'),
        'problem_user': ('trajectory', 'checks'),
        'keyword_user': ('k', 'desc'),
        'keyword_expand_user': ('kw', 'm'),
    }

    def __post_init__(self):
        for name in ('system', 'from_scratch', 'check_system', 'check_user',
                     'problem_system', 'problem_user'):
            if not getattr(self, name).strip():
                raise ValueError(f'AgenticPrompts.{name} is required')
        for name, placeholders in self._REQUIRED_FIELDS.items():
            text = getattr(self, name)
            if not text:
                continue
            for placeholder in placeholders:
                if '{' + placeholder + '}' not in text:
                    raise ValueError(f'AgenticPrompts.{name} must contain '
                                     f'{{{placeholder}}}')

    def require(self, *names: str) -> None:
        """Raise unless every named prompt was supplied."""
        missing = [n for n in names if not getattr(self, n).strip()]
        if missing:
            raise ValueError(f'this configuration needs AgenticPrompts.'
                             f'{", AgenticPrompts.".join(missing)}')


# ── challenger ─────────────────────────────────────────────────────────────

class AgenticChallenger(Challenger):
    """Propose tool-using tasks by first doing them, then describing them.

    Args:
        prompts: every string sent to the model.
        explorer: batch-in / batch-out generation with sandbox tools (multi-turn).
        seeds: optional pool of seed trajectories (dicts with a ``query`` key),
            drawn with replacement.
        keyword_store: optional bank for diversity control.
        category_desc: category -> description for keyword generation.
        seed_mix_prob: chance a proposal carries a seed.
        reset_fn: called before each round-1 episode to clean the sandbox
            workspace. Must be synchronous and leave the workspace empty.
        run_check_fn: run a python script in the sandbox's current state.
            Signature: ``(source: str) -> (exit_code: int, output: str)``.
        workspace_snapshot_fn: after round 1, return a text summary of the
            workspace state (e.g. ``find . -type f``). If None, a default
            that lists messages is used.
        combo_arity: ``'triple'`` or ``'mix'``, as in :class:`.CodeChallenger`.
        arity_weights: weights for the ``'mix'`` subset size.
        single_kw_prob: chance of using one category in ``'triple'`` mode.
        keyword_refill_target / keyword_gen_calls / keyword_refill_tries /
        keyword_params: keyword bank refill parameters.
        min_batch: smallest batch worth sending to the explorer.
        problem_max_chars: reject problem statements longer than this.
        reject_sink: called with a dict for every rejected proposal.
        propose_sink: called once per proposal attempt -- kept, rejected while
            building, or dropped by the difficulty band alike -- with the
            token-level record of the rounds that produced it. This is the only
            way the proposing rounds survive: they are generation like any
            other, so they carry ``input_ids`` / ``labels`` / ``logprobs`` and
            could later be trained on, but nothing downstream of ``build``
            looks at them and without a sink they are dropped on the floor.
            Rejects are included on purpose: they are the zero-reward half of a
            GRPO group, so a set of kept-only records has no variance to learn
            from. Requires a local sampler -- an API explorer returns text only.
    """

    def __init__(
        self,
        prompts: AgenticPrompts,
        explorer: Explorer,
        *,
        seeds: Sequence[Dict[str, Any]] = (),
        keyword_store: Optional[KeywordStore] = None,
        category_desc: Optional[Dict[str, str]] = None,
        seed_mix_prob: float = 0.5,
        reset_fn: Callable[[], None],
        run_check_fn: Callable[[str], Tuple[int, str]],
        workspace_snapshot_fn: Optional[Callable[[], str]] = None,
        combo_arity: str = 'triple',
        arity_weights: Optional[Sequence[float]] = None,
        single_kw_prob: float = 0.1,
        keyword_refill_target: int = 128,
        keyword_gen_calls: int = 8,
        keyword_refill_tries: int = 2,
        keyword_params: Optional[SamplingParams] = None,
        min_batch: int = 1,
        problem_max_chars: int = 8192,
        reject_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        propose_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        **challenger_kwargs: Any,
    ):
        super().__init__(explorer, system=prompts.system, **challenger_kwargs)
        if combo_arity not in ('triple', 'mix'):
            raise ValueError(f"combo_arity must be 'triple' or 'mix', got {combo_arity!r}")
        if keyword_store is not None:
            desc = category_desc or {}
            missing_cats = [c for c in keyword_store.categories if not desc.get(c)]
            if missing_cats:
                raise ValueError(f'category_desc is missing a description for '
                                 f'{missing_cats}; a dry category could not be refilled.')
            prompts.require('keyword_system', 'keyword_user', 'from_keywords')
        self.prompts = prompts
        self.seeds = list(seeds)
        self.store = keyword_store
        self.category_desc = dict(category_desc or {})
        self.seed_mix_prob = seed_mix_prob
        self.reset_fn = reset_fn
        self.run_check_fn = run_check_fn
        self.workspace_snapshot_fn = workspace_snapshot_fn
        self.combo_arity = combo_arity
        self.arity_weights = list(arity_weights) if arity_weights else None
        self.single_kw_prob = single_kw_prob
        self.keyword_refill_target = keyword_refill_target
        self.keyword_gen_calls = keyword_gen_calls
        self.keyword_refill_tries = keyword_refill_tries
        self.keyword_params = keyword_params
        self.min_batch = max(1, min_batch)
        self.problem_max_chars = problem_max_chars
        self.reject_sink = reject_sink
        self.propose_sink = propose_sink
        if self.seeds:
            prompts.require('from_seed')
            if self.store is not None:
                prompts.require('from_seed_keywords')
        self._nonce = 0
        self.stats: Dict[str, int] = {
            'round1_done': 0, 'check_parse_fail': 0, 'check_run_fail': 0,
            'problem_parse_fail': 0, 'too_long': 0, 'parsed': 0,
        }
        self._hard: List[Tuple[str, str]] = []

    # ------------------------------------------------------------- proposing

    def propose(self, count: int) -> List[Trajectory]:
        """Build ``count`` prompt trajectories for round 1.

        Each carries a direction + keywords + optional seed. The explorer will
        run these multi-turn in the sandbox.
        """
        proposals: List[Trajectory] = []
        for _ in range(count):
            picks = self._draw_keywords()
            body = '\n'.join(f'- {c}: {t}' for c, t in picks)
            use_seed = bool(self.seeds) and self.rng.random() < self.seed_mix_prob
            seed = self.rng.choice(self.seeds) if use_seed else None
            if use_seed and picks:
                user = self.prompts.from_seed_keywords.format(
                    seed=seed['query'], keywords=body)
            elif use_seed:
                user = self.prompts.from_seed.format(seed=seed['query'])
            elif picks:
                user = self.prompts.from_keywords.format(keywords=body)
            else:
                user = self.prompts.from_scratch
            proposal: Trajectory = {
                'messages': [{'role': 'system', 'content': self.prompts.system},
                             {'role': 'user', 'content': user}],
            }
            proposals.append(attach_user_data(
                proposal, keywords=picks, seeded=use_seed, keyword_block=body))
        return proposals

    # ------------------------------------------------------------- building

    def build(self, explored: List[Trajectory]) -> List[Optional[Trajectory]]:
        """Satisfy the abstract method; not usable outside ``_round``.

        ``_build_one`` requires the sandbox to hold the episode's workspace state,
        which is only guaranteed inside the serial ``_round`` loop. Calling this
        method directly will produce wrong results because the sandbox state does
        not match the trajectory being processed.
        """
        raise RuntimeError(
            f'{type(self).__name__}.build() must not be called directly; '
            f'the serial _round() loop calls _build_one() per episode instead.')

    def _build_one(self, explored: Trajectory) -> Optional[Trajectory]:
        """Process one round-1 result: write checks, verify, write problem.

        Called while the sandbox still holds this episode's workspace state.
        """
        summary = _trajectory_summary(explored)
        snapshot = self.workspace_snapshot_fn() if self.workspace_snapshot_fn else summary
        keywords = user_data_get(explored.get('user_data'), 'keywords', [])
        seeded = user_data_get(explored.get('user_data'), 'seeded', False)
        # Every round this proposal generates, in order. Handed to propose_sink
        # with whatever verdict the proposal ends up with, so a rejected attempt
        # is recorded as fully as a kept one.
        rounds = [_propose_round('explore', explored)]

        # Round 2a: write check script
        # NOTE: This goes through the same explorer (with tool schemas visible).
        # The prompt must clearly instruct the model to output ONLY a code block
        # and not call tools, otherwise tool calls would corrupt the sandbox state
        # before verification. The check_system prompt enforces this.
        check_prompt: Trajectory = {
            'messages': [
                {'role': 'system', 'content': self.prompts.check_system},
                {'role': 'user', 'content': self.prompts.check_user.format(
                    trajectory=summary, final_state=snapshot)},
            ],
        }
        check_reply = self.explore([check_prompt])
        rounds.append(_propose_round('check', check_reply[0]))
        script = parse_check_script(assistant_text(check_reply[0]))
        if script is None:
            self.stats['check_parse_fail'] += 1
            self._reject_record(explored, 'check_parse_fail')
            self._emit_propose(rounds, 'check_parse_fail', keywords=keywords, seeded=seeded)
            return None

        # Verify: run check script in current sandbox state (must pass)
        exit_code, output = self.run_check_fn(script)
        if exit_code != 0:
            self.stats['check_run_fail'] += 1
            self._reject_record(explored, 'check_run_fail',
                                detail=f'exit {exit_code}: {output[-200:]}')
            self._emit_propose(rounds, 'check_run_fail', keywords=keywords, seeded=seeded)
            return None

        # Round 2b: write problem statement
        problem_prompt: Trajectory = {
            'messages': [
                {'role': 'system', 'content': self.prompts.problem_system},
                {'role': 'user', 'content': self.prompts.problem_user.format(
                    trajectory=summary, checks=script)},
            ],
        }
        problem_reply = self.explore([problem_prompt])
        rounds.append(_propose_round('problem', problem_reply[0]))
        statement = parse_problem_statement(assistant_text(problem_reply[0]))
        if statement is None:
            self.stats['problem_parse_fail'] += 1
            self._reject_record(explored, 'problem_parse_fail')
            self._emit_propose(rounds, 'problem_parse_fail', keywords=keywords, seeded=seeded)
            return None
        if len(statement) > self.problem_max_chars:
            self.stats['too_long'] += 1
            self._reject_record(explored, 'too_long')
            self._emit_propose(rounds, 'too_long', keywords=keywords, seeded=seeded)
            return None

        self.stats['parsed'] += 1
        task: Trajectory = {
            'messages': [{'role': 'user', 'content': statement}],
        }
        task = attach_user_data(task, check_script=script, keywords=keywords, seeded=seeded)
        # Carried, not emitted: the verdict this proposal earns depends on the
        # difficulty measurement, which has not run yet. A plain top-level key
        # rather than user_data, which json-encodes every value on each update.
        task['propose_rounds'] = rounds
        return task

    def _reject_record(self, traj: Trajectory, reason: str, detail: str = '') -> None:
        if self.reject_sink is not None:
            payload: Dict[str, Any] = {'reason': reason}
            if detail:
                payload['detail'] = detail
            payload['last_assistant'] = assistant_text(traj)[:500]
            self.reject_sink(payload)

    def _emit_propose(self, rounds: Optional[List[Dict[str, Any]]], outcome: str, *,
                      keywords: Any = (), seeded: bool = False,
                      n_pass: Optional[int] = None) -> None:
        """Hand one proposal attempt's rounds to ``propose_sink``.

        ``pass_rate`` is the raw fraction of solver attempts that succeeded. It
        is left as the measurement rather than mapped onto a difficulty score:
        the target rate and its tolerance are training decisions, and baking a
        guess at them into the dump would make it look like they had been
        settled.
        """
        if self.propose_sink is None or not rounds:
            return
        rollouts = self.solver_rollouts or None
        self.propose_sink({
            'outcome': outcome,
            'n_pass': n_pass,
            'n_rollouts': rollouts,
            'pass_rate': (n_pass / rollouts) if (n_pass is not None and rollouts) else None,
            'keywords': list(keywords or ()),
            'seeded': bool(seeded),
            'rounds': rounds,
        })

    def _take_rounds(self, task: Trajectory) -> Optional[List[Dict[str, Any]]]:
        """Detach a task's proposing rounds. Popped even with no sink attached:
        token ids for a whole agentic episode are large, and a kept task is held
        until the caller's batch is full.
        """
        return task.pop('propose_rounds', None)

    # ------------------------------------------------------------ revised _round

    def _round(self, missing: int) -> Optional[List[Trajectory]]:
        """One cycle: serial round-1 episodes, inline build, then difficulty filter."""
        count = min(self._estimate(missing), self.max_proposals_per_round)
        proposals = self.propose(count)
        if not proposals:
            return None

        usable: List[Trajectory] = []
        for proposal in proposals:
            # Reset workspace, run round 1
            self.reset_fn()
            result = self.explore([proposal])
            if not result:
                continue
            explored = result[0]
            self.stats['round1_done'] += 1
            # Workspace still holds this episode's state → build inline
            task = self._build_one(explored)
            if task is not None:
                usable.append(task)

        kept = self._filter_difficulty(usable) if self.solver_rollouts else usable
        if not self.solver_rollouts:
            # No difficulty stage, so the verdict is final as soon as it is built.
            for task in usable:
                self._emit_propose(self._take_rounds(task), 'kept',
                                   keywords=user_data_get(task.get('user_data'), 'keywords', []),
                                   seeded=user_data_get(task.get('user_data'), 'seeded', False))
        self.n_proposed += len(proposals)
        self.n_kept += len(kept)
        band = (f', in difficulty band {len(kept)}' if self.solver_rollouts else '')
        logger.info(f'[{type(self).__name__}] proposed {len(proposals)}, usable '
                    f'{len(usable)}{band} (cumulative {self.n_kept}/{self.n_proposed})')
        return kept

    # ------------------------------------------------------------ difficulty

    def _filter_difficulty(self, tasks: List[Trajectory]) -> List[Trajectory]:
        """Override: each solver attempt needs a clean workspace, so run serially."""
        if not tasks:
            return []
        passes = [0] * len(tasks)
        for i, task in enumerate(tasks):
            prompt = self.solver_prompt(task)
            for _ in range(self.solver_rollouts):
                self.reset_fn()
                attempts = self._solver_explore(
                    [dict(prompt)], sampling_params=self.solver_params)
                if attempts and self.judge_attempt(task, attempts[0]):
                    passes[i] += 1

        measured = [
            attach_user_data(task, n_pass=passes[i], n_rollouts=self.solver_rollouts)
            for i, task in enumerate(tasks)
        ]
        self.on_difficulty_measured(measured)
        high = self.solver_rollouts - self.keep_max_pass_margin
        in_band = [self.keep_min_pass <= n <= high for n in passes]
        # Emit here, not in _round: this is where a proposal's verdict is
        # decided, and both sides of the band are worth keeping -- a task nobody
        # solved and one everybody solved are the two failure modes the
        # proposer would need to learn to avoid.
        for task, n, kept_flag in zip(measured, passes, in_band):
            self._emit_propose(self._take_rounds(task),
                               'kept' if kept_flag else 'outside_band',
                               keywords=user_data_get(task.get('user_data'), 'keywords', []),
                               seeded=user_data_get(task.get('user_data'), 'seeded', False),
                               n_pass=n)
        return [t for t, kept_flag in zip(measured, in_band) if kept_flag]

    def solver_prompt(self, task: Trajectory) -> Trajectory:
        """The task statement, nothing else -- the solver's own harness adds the system."""
        return {'messages': [dict(m) for m in task.get('messages') or []]}

    def judge_attempt(self, task: Trajectory, attempt: Trajectory) -> bool:
        """Run the check script against the sandbox's current state."""
        script = user_data_get(task.get('user_data'), 'check_script', '')
        if not script:
            return False
        exit_code, _ = self.run_check_fn(script)
        return exit_code == 0

    def on_difficulty_measured(self, candidates: List[Trajectory]) -> None:
        """Remember keywords behind candidates nobody solved."""
        if self.store is None:
            return
        seen = {(c, t.lower()) for c, t in self._hard}
        for task in candidates:
            data = task.get('user_data')
            if user_data_get(data, 'n_pass', 0) > 0:
                continue
            for pick in user_data_get(data, 'keywords', []) or []:
                if isinstance(pick, (list, tuple)) and len(pick) >= 2:
                    c, t = pick[0], pick[1]
                    if (c, t.lower()) not in seen:
                        seen.add((c, t.lower()))
                        self._hard.append((c, t))

    # ------------------------------------------------------------ keywords

    def _draw_keywords(self) -> List[Tuple[str, str]]:
        """Consume one keyword combination from the bank; [] without a bank."""
        if self.store is None:
            return []
        categories = self.store.categories
        if self.combo_arity == 'mix':
            if self.arity_weights and len(self.arity_weights) == len(categories):
                k = self.rng.choices(range(1, len(categories) + 1),
                                     weights=self.arity_weights)[0]
            else:
                k = self.rng.randint(1, len(categories))
            cats = self.rng.sample(list(categories), k)
        elif self.rng.random() < self.single_kw_prob:
            cats = [self.rng.choice(categories)]
        else:
            cats = list(categories)
        picks: List[Tuple[str, str]] = []
        for c in cats:
            if not self.store.unused(c):
                self._refill(c)
            text = self.store.take(c, self.rng)
            if text is not None:
                picks.append((c, text))
        return picks

    def _refill(self, category: str) -> None:
        """Ask the model for more keywords in ``category``."""
        tries = 0
        while not self.store.unused(category):
            new = self._generate_keywords(category, self.keyword_refill_target)
            added = self.store.add(category, new, source='gen')
            tries += 1
            if added == 0 and tries >= self.keyword_refill_tries:
                if self.store.items[category]:
                    self.store.recycle(category)
                    logger.info(f'[AgenticChallenger] keyword category {category!r} '
                                f'exhausted -> recycled {len(self.store.items[category])} topics')
                break

    def _generate_keywords(self, category: str, n_want: int) -> List[str]:
        """Up to ``n_want`` keywords the bank does not already hold."""
        if n_want <= 0:
            return []
        known = self.store.texts(category)
        n_calls = max(self.keyword_gen_calls, self.min_batch)
        per_call = max(1, -(-n_want // n_calls) + 4)
        avoid_note = ''
        if known:
            shown = known if len(known) <= 40 else self.rng.sample(known, 40)
            avoid_note = ('\nDo NOT repeat any of these already-used topics: '
                          + ', '.join(shown))
        base = self.prompts.keyword_user.format(
            k=per_call, desc=self.category_desc[category]) + avoid_note
        self._nonce += 1
        prompts = [{
            'messages': [{'role': 'system', 'content': self.prompts.keyword_system},
                         {'role': 'user', 'content': f'{base}\n(batch {self._nonce}-{i})'}],
        } for i in range(n_calls)]
        seen = {t.strip().lower() for t in known}
        out: List[str] = []
        for reply in self.explore(prompts, sampling_params=self.keyword_params):
            for kw in parse_keyword_list(assistant_text(reply)):
                key = kw.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(kw)
        self.rng.shuffle(out)
        return out[:n_want]

    # ------------------------------------------------------------ feedback

    def expand_hard_keywords(self) -> int:
        """Brainstorm more topics in families that produced the hardest tasks."""
        if self.store is None or not self._hard or not hasattr(self.prompts, 'keyword_expand_user'):
            return 0
        self.prompts.require('keyword_expand_user')
        hard = self._hard[:32]
        self.rng.shuffle(hard)
        reqs = list(hard)
        while len(reqs) < self.min_batch:
            reqs.append(hard[len(reqs) % len(hard)])
        self._nonce += 1
        prompts = [{
            'messages': [
                {'role': 'system', 'content': self.prompts.keyword_system},
                {'role': 'user',
                 'content': self.prompts.keyword_expand_user.format(kw=kw, m=8)
                 + f'\n(batch {self._nonce}-{i})'},
            ],
        } for i, (_c, kw) in enumerate(reqs)]
        added = 0
        for (cat, kw), reply in zip(reqs, self.explore(prompts,
                                                       sampling_params=self.keyword_params)):
            added += self.store.add(cat, parse_keyword_list(assistant_text(reply)),
                                    source='expand', parent=kw)
        logger.info(f'[AgenticChallenger] expanded {len(hard)} hard keyword(s) -> '
                    f'+{added} same-domain topics')
        return added
