# Copyright (c) ModelScope Contributors. All rights reserved.
"""Challenger: turn raw material into training tasks.

A challenger invents the problems a solver will later be trained on. The three
things that vary between deployments are all injected:

* **what to ask for** -- the system prompt, and the parser that reads the
  answer back. They are one contract, so they are passed together.
* **how to explore** -- an :class:`Explorer`, i.e. anything that takes a batch
  of trajectories and returns them with the model's reply appended. Both
  rollouts in :mod:`twinkle_agentic.rollout` have that signature already, so a
  challenger can explore *with tools* -- running code, reading files -- while
  it invents, over a local sampler or over an HTTP endpoint alike.
  :func:`twinkle_agentic.rollout.build_rollout` picks the right one for the
  backend at hand.
* **what counts as a keeper** -- subclasses decide, in :meth:`Challenger.build`.
* **how hard is hard enough** -- optional. Ask for ``solver_rollouts`` attempts per
  candidate and only tasks the model solves *sometimes* are kept: a task every
  attempt gets right, or none does, gives GRPO a zero gradient, so it costs a
  training slot and teaches nothing. Counting the attempts is the same work in
  every domain and lives here; deciding whether one attempt was right is not,
  and is left to :meth:`Challenger.judge_attempt`.

Everything a strategy needs beyond that (seed examples, keyword banks) goes in
``__init__``; :meth:`Challenger.__call__` only says how many tasks you want per
batch. It is a generator that yields *full* batches: challengers throw away
most of what they propose -- keep rates of a few percent are normal once the
difficulty filter runs -- so the alternative is a caller that has to cope with
ragged batches for reasons that have nothing to do with it.
"""
import math
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from twinkle.data_format import SamplingParams, Trajectory, attach_user_data
from twinkle.utils import get_logger
from twinkle_agentic.envs import Env

logger = get_logger()

__all__ = ['Challenger', 'Explorer', 'KeywordPrompts', 'PromptSet']

# A batch of trajectories in, the same trajectories with the model's reply
# appended out. Both MultiTurnRollout and APIMultiTurnRollout satisfy this
# as-is; build_rollout() returns whichever fits the backend. Both also accept a
# per-call ``sampling_params=`` keyword, which is how the difficulty stage asks
# for its own temperature and length budget without a second explorer.
Explorer = Callable[[List[Trajectory]], List[Trajectory]]


@dataclass
class KeywordPrompts:
    """The three strings a :class:`~.keywords.KeywordBank` sends, and nothing else.

    Its own type rather than the caller's prompt object: the challengers here
    carry a dozen other prompts, the RSI drivers in ``cookbook/rsi`` keep theirs
    as module constants, and a bank that reached into either by attribute name
    would be coupled to both spellings. Building one of these is how a caller
    says which of its strings are the keyword ones -- see
    :meth:`PromptSet.keyword_prompts` for the challengers' answer.

    Lives here rather than beside the bank so that :class:`PromptSet` can produce
    one without importing it.

    Args:
        system: the system message every keyword call carries.
        user: asks for ``{k}`` topics in a category described by ``{desc}``.
        expand_user: asks for ``{m}`` more topics like ``{kw}``, optionally with
            the category's ``{desc}``. Only :meth:`.KeywordBank.expand_hard`
            needs it.
    """
    system: str
    user: str
    expand_user: str = ''

    def __post_init__(self):
        missing = [f for f in ('system', 'user') if not getattr(self, f).strip()]
        if missing:
            raise ValueError(f'KeywordPrompts needs {" and ".join(missing)}: a dry '
                             f'category could not be refilled without it.')


class PromptSet:
    """Base for a challenger's bundle of prompts: what is required, and validation.

    Every challenger here is a dataclass of strings plus the same three questions
    -- are the mandatory ones filled in, do the optional ones carry the
    placeholders they will be formatted with, and does this configuration have
    the ones it needs. Answering them once means a missing placeholder is caught
    at construction in every domain, rather than as a ``KeyError`` mid-run in
    whichever domain remembered to check.

    Subclasses declare:

    * ``_REQUIRED`` -- fields that must carry text.
    * ``_REQUIRED_FIELDS`` -- field -> placeholders its text must contain.
    """

    _REQUIRED: Tuple[str, ...] = ()
    _REQUIRED_FIELDS: Dict[str, Sequence[str]] = {}

    def __post_init__(self):
        name = type(self).__name__
        for field in self._REQUIRED:
            if not getattr(self, field).strip():
                raise ValueError(f'{name}.{field} is required')
        for field, placeholders in self._REQUIRED_FIELDS.items():
            text = getattr(self, field)
            if not text:
                continue
            for placeholder in placeholders:
                if '{' + placeholder + '}' not in text:
                    raise ValueError(f'{name}.{field} must contain {{{placeholder}}}')

    def require(self, *names: str) -> None:
        """Raise unless every named prompt was supplied.

        For what only a configuration knows: drawing from a keyword bank needs the
        keyword prompts, seeds need the seed prompt, and a challenger asks for the
        ones its arguments imply.
        """
        missing = [n for n in names if not getattr(self, n).strip()]
        if missing:
            name = type(self).__name__
            separator = f', {name}.'
            raise ValueError(f'this configuration needs {name}.'
                             f'{separator.join(missing)}')

    def keyword_prompts(self) -> KeywordPrompts:
        """The keyword subset, for the bank. Validated by :meth:`require` first."""
        self.require('keyword_system', 'keyword_user')
        return KeywordPrompts(system=self.keyword_system, user=self.keyword_user,
                              expand_user=self.keyword_expand_user)


class Challenger(ABC):
    """Base class: propose, explore, keep, repeat until the batch is full.

    Args:
        explorer: takes a batch of trajectories and returns them with the
            model's reply appended -- a rollout from
            :func:`twinkle_agentic.rollout.build_rollout`, over a local sampler
            or over an API endpoint.
        system: system prompt handed to the model. It carries the output
            contract, which is why ``build`` -- the code that reads that output
            back -- lives in the same subclass.
        envs: the environments this challenger works in, one per slot. A slot is
            owned whole for as long as a job needs it, because the workspace
            lives inside it, so ``len(envs)`` is also how many jobs may run at
            once. Both halves take the same parameter and reach it the same way
            (:meth:`env`), which is what lets one caller decide where everything
            it runs is executed and graded: ``[LocalEnv()]`` keeps judgement on
            the training host and costs milliseconds, sandbox slots trade that
            for isolation. Empty is allowed for a challenger that executes
            nothing; :meth:`env` then says so rather than raising IndexError.
        max_proposals_per_round: ceiling on how many proposals one round may
            request. Without it a low keep rate makes the estimator ask for an
            unbounded batch after the first round.
        solver_rollouts: attempts per candidate in the difficulty stage. ``0``
            skips the stage entirely; any other value requires the subclass to
            implement :meth:`solver_prompt` and :meth:`judge_attempt`.
        keep_pass_band: ``(low, high)`` attempt counts, inclusive on both ends:
            keep a candidate only if that many of its ``solver_rollouts``
            attempts succeeded. Required whenever the stage runs, and has no
            default because the counts are absolute -- ``(1, 7)`` reads as "hard
            but solvable" against eight rollouts and as something far stricter
            against sixteen, so it has to be written by whoever chose the
            rollout count.
        solver_params: sampling params for the difficulty stage only, passed to
            the explorer per call. ``None`` reuses whatever the explorer was
            built with -- which is usually the proposing temperature, and that
            is higher than a solver should get.
        solver_explorer: optional separate explorer for the difficulty stage.
            ``None`` reuses the main explorer. Useful when the solver needs a
            different configuration (e.g. sandbox tools, more turns) than the
            proposer.
        seed: RNG seed for whatever sampling a subclass does. ``None`` leaves
            the RNG unseeded.
    """

    def __init__(
        self,
        explorer: Explorer,
        *,
        system: str,
        envs: Sequence[Env] = (),
        max_proposals_per_round: int = 512,
        solver_rollouts: int = 0,
        keep_pass_band: Optional[Tuple[int, int]] = None,
        solver_params: Optional[SamplingParams] = None,
        solver_explorer: Optional[Explorer] = None,
        seed: Optional[int] = None,
    ):
        if not system:
            raise ValueError('Challenger needs a system prompt: it carries the output '
                             'contract that build() parses back.')
        if solver_rollouts < 0:
            raise ValueError(f'solver_rollouts must be >= 0, got {solver_rollouts}')
        if solver_rollouts:
            # Checked here rather than at first use: the stage runs after a full
            # round of generation, and finding out then that this challenger
            # cannot grade an attempt wastes the whole round.
            missing = [
                name for name in ('solver_prompt', 'judge_attempt')
                if getattr(type(self), name) is getattr(Challenger, name)
            ]
            if missing:
                raise NotImplementedError(
                    f'solver_rollouts={solver_rollouts} needs {type(self).__name__} to '
                    f'implement {", ".join(missing)}; pass solver_rollouts=0 to skip the '
                    f'difficulty stage.')
            if keep_pass_band is None:
                raise ValueError(f'solver_rollouts={solver_rollouts} needs '
                                 f'keep_pass_band=(low, high): the band is in attempt '
                                 f'counts, so what it asks for depends on how many '
                                 f'attempts were run.')
            if len(keep_pass_band) != 2:
                raise ValueError(f'keep_pass_band is (low, high) in attempt counts, got '
                                 f'{keep_pass_band}')
            low, high = keep_pass_band
            if not 0 <= low <= high <= solver_rollouts:
                raise ValueError(
                    f'keep_pass_band must satisfy 0 <= low <= high <= solver_rollouts, '
                    f'got {keep_pass_band} against solver_rollouts={solver_rollouts}')
        elif keep_pass_band is not None:
            raise ValueError('keep_pass_band has nothing to filter while '
                             'solver_rollouts=0 leaves the difficulty stage off; pass '
                             'the rollout count too, or drop the band.')
        self.explorer = explorer
        self.system = system
        self.envs = list(envs)
        self.max_proposals_per_round = max_proposals_per_round
        self.solver_rollouts = solver_rollouts
        self.keep_pass_band = keep_pass_band
        self.solver_params = solver_params
        self.solver_explorer = solver_explorer
        self.rng = random.Random(seed)
        # Running tally, used to size the next round and worth logging: a keep
        # rate near zero means the prompt or the filter is miscalibrated, not
        # that the model is bad.
        self.n_proposed = 0
        self.n_kept = 0

    # ----------------------------------------------------------------- envs

    @property
    def n_slots(self) -> int:
        """How many jobs may run at once: one per environment."""
        return len(self.envs)

    def env(self, slot: int = 0) -> Env:
        """The environment for ``slot``.

        Fetched per use rather than held in a local, so a slot that had to be
        rebuilt underneath is picked up on the next call instead of being used
        dead. ``slot=0`` is the default because a challenger with nothing to run
        concurrently -- one script, no state to share -- has only one.
        """
        if not self.envs:
            raise RuntimeError(
                f'{type(self).__name__} was given no envs, so there is nowhere to run '
                f'anything: pass envs=[LocalEnv()] to execute on the training host, or '
                f'sandbox slots to execute in one.')
        return self.envs[slot]

    # ------------------------------------------------------------- subclass

    @abstractmethod
    def propose(self, count: int) -> List[Trajectory]:
        """Build ``count`` prompt trajectories to hand to the explorer.

        Returning fewer than asked is allowed and means the source material ran
        out; :meth:`__call__` stops once a round proposes nothing.
        """

    @abstractmethod
    def build(self, explored: List[Trajectory]) -> List[Optional[Trajectory]]:
        """Turn explored proposals into finished tasks.

        Returns one entry per input, ``None`` for anything rejected -- failed
        parse, failed verification, wrong difficulty. Positional so a subclass
        can line rejects up against what produced them.
        """

    def solver_prompt(self, task: Trajectory) -> Trajectory:
        """The trajectory to hand a solver attempting ``task``.

        Only called when ``solver_rollouts`` is non-zero. It must return a
        prompt for every task: a task that cannot be attempted has no measurable
        difficulty and should have been rejected in :meth:`build` instead.
        """
        raise NotImplementedError()

    def judge_attempt(self, task: Trajectory, attempt: Trajectory) -> bool:
        """Did this solver attempt solve ``task``?

        ``attempt`` is the explored :meth:`solver_prompt` trajectory, so the
        model's answer is its last assistant message. Program checks only: a
        judgement that drifts between rounds turns the difficulty band into
        noise.
        """
        raise NotImplementedError()

    def on_difficulty_measured(self, candidates: List[Trajectory]) -> None:
        """Called once per round with every measured candidate, before filtering.

        Each carries ``n_pass`` / ``n_rollouts`` in its ``user_data``. This is
        the only place that sees the candidates the band is about to drop, which
        is what a strategy adapting to difficulty needs -- an all-fail task says
        more about its source material than a kept one does.
        """

    # ---------------------------------------------------------------- public

    def __call__(self, batch_size: int, total: Optional[int] = None) -> Iterator[List[Trajectory]]:
        """Yield batches of exactly ``batch_size`` finished tasks.

        Args:
            batch_size: tasks per yielded batch.
            total: stop after this many tasks. ``None`` runs until the source
                material is exhausted, which for a from-scratch challenger
                means forever -- pass a total or break out of the loop.

        The final batch is short only when the source runs out or ``total`` is
        not a multiple of ``batch_size``.
        """
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        pending: List[Trajectory] = []
        produced = 0
        while total is None or produced < total:
            want = batch_size if total is None else min(batch_size, total - produced)
            while len(pending) < want:
                kept = self._round(want - len(pending))
                if kept is None:
                    # Source exhausted: hand back whatever is left rather than
                    # spinning, and let the caller see a short final batch.
                    if pending:
                        yield pending
                    return
                pending.extend(kept)
            yield pending[:want]
            produced += want
            pending = pending[want:]

    # --------------------------------------------------------------- private

    def _round(self, missing: int) -> Optional[List[Trajectory]]:
        """One propose/explore/build/measure cycle. ``None`` means the source is dry."""
        count = min(self._estimate(missing), self.max_proposals_per_round)
        proposals = self.propose(count)
        if not proposals:
            return None
        explored = self.explore(proposals)
        built = self.build(explored)
        usable = [t for t in built if t is not None]
        kept = self._filter_difficulty(usable) if self.solver_rollouts else usable
        self.n_proposed += len(proposals)
        self.n_kept += len(kept)
        band = (f', in difficulty band {len(kept)}' if self.solver_rollouts else '')
        logger.info(f'[{type(self).__name__}] proposed {len(proposals)}, usable '
                    f'{len(usable)}{band} (cumulative {self.n_kept}/{self.n_proposed})')
        return kept

    def explore(
        self,
        trajectories: List[Trajectory],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs: Any,
    ) -> List[Trajectory]:
        """Run the explorer over a batch, optionally overriding its sampling params.

        The override is only forwarded when asked for, so a plain callable
        explorer keeps working; both rollouts in
        :mod:`twinkle_agentic.rollout` accept it. Anything else in ``kwargs`` is
        passed straight through for the same reason -- a caller that needs a
        rollout-specific hook (``followup_fn``) says so per call, and an explorer
        that does not take it fails loudly instead of silently ignoring it.
        """
        if not trajectories:
            return []
        if sampling_params is not None:
            kwargs['sampling_params'] = sampling_params
        if not kwargs:
            return self.explorer(trajectories)
        return self.explorer(trajectories, **kwargs)

    def _solver_explore(
        self,
        trajectories: List[Trajectory],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs: Any,
    ) -> List[Trajectory]:
        """Run solver attempts through the solver explorer, or fall back to the main one.

        Subclasses that need per-attempt isolation (e.g. sandbox workspace reset)
        override this rather than the whole difficulty filter. Extra kwargs are
        forwarded, which is how such a subclass says which sandbox each attempt
        runs in (``tool_manager`` as a list, one entry per trajectory).
        """
        if self.solver_explorer is not None:
            if sampling_params is None:
                return self.solver_explorer(trajectories, **kwargs)
            return self.solver_explorer(trajectories, sampling_params=sampling_params, **kwargs)
        return self.explore(trajectories, sampling_params=sampling_params, **kwargs)

    def _filter_difficulty(self, tasks: List[Trajectory]) -> List[Trajectory]:
        """Attempt each task ``solver_rollouts`` times; keep the ones in the band.

        All attempts for the whole batch go out in one explorer call: on the
        sampler path that is one batched generate, and the alternative -- a call
        per task -- would leave the GPUs idle between them.
        """
        if not tasks:
            return []
        prompts: List[Trajectory] = []
        owners: List[int] = []
        for i, task in enumerate(tasks):
            prompt = self.solver_prompt(task)
            for _ in range(self.solver_rollouts):
                prompts.append(dict(prompt))
                owners.append(i)

        attempts = self._solver_explore(prompts, sampling_params=self.solver_params)
        if len(attempts) != len(prompts):
            # Counting a partial return would silently understate every affected
            # task's pass count, i.e. report tasks as harder than they are.
            raise RuntimeError(f'explorer returned {len(attempts)} attempts for '
                              f'{len(prompts)} solver prompts; expected one per prompt.')

        passes = [0] * len(tasks)
        for owner, attempt in zip(owners, attempts):
            if self.judge_attempt(tasks[owner], attempt):
                passes[owner] += 1

        measured = [
            attach_user_data(task, n_pass=passes[i], n_rollouts=self.solver_rollouts)
            for i, task in enumerate(tasks)
        ]
        self.on_difficulty_measured(measured)
        low, high = self.keep_pass_band
        return [t for t, n in zip(measured, passes) if low <= n <= high]

    def _estimate(self, missing: int) -> int:
        """How many proposals to make for ``missing`` keepers.

        The first round has nothing to go on and asks for exactly what is
        missing; after that the measured keep rate scales the request. A round
        that kept nothing leaves the rate at its last non-zero estimate rather
        than dividing by zero.
        """
        if self.n_kept <= 0:
            return missing
        rate = self.n_kept / max(1, self.n_proposed)
        return max(missing, math.ceil(missing / rate))

    # ------------------------------------------------------------- utilities

    def prompt_trajectory(self, user: str, **extra: Any) -> Trajectory:
        """A two-message trajectory carrying this challenger's system prompt."""
        trajectory: Trajectory = {
            'messages': [
                {'role': 'system', 'content': self.system},
                {'role': 'user', 'content': user},
            ],
        }
        trajectory.update(extra)
        return trajectory

    @staticmethod
    def draw(rng: random.Random, pool: Sequence[Any], count: int) -> List[Any]:
        """Draw ``count`` items with replacement; ``[]`` for an empty pool."""
        return [rng.choice(pool) for _ in range(count)] if pool else []


def map_parallel(fn: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
    """Map ``fn`` over ``items`` at once, results in input order.

    Every use of this is waiting on a sandbox or on a model call, not computing,
    so the thread pool is the point. One item runs inline: a pool for a single
    call only adds a thread, and it keeps a serial configuration on exactly the
    code path it had before.
    """
    items = list(items)
    if len(items) <= 1:
        return [fn(item) for item in items]
    out: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def sampling_params_of(explorer: Any) -> Optional[SamplingParams]:
    """The sampling params an explorer was built with, when it exposes them.

    Only used for logging what a run actually asked for; both explorer kinds
    keep the field under the same name.
    """
    params: Optional[SamplingParams] = getattr(explorer, 'sampling_params', None)
    return params


def as_dict(trajectory: Trajectory) -> Dict[str, Any]:
    """A plain dict copy, for writing a trajectory to jsonl."""
    return {k: v for k, v in trajectory.items()}
