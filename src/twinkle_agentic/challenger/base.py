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
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from twinkle.data_format import SamplingParams, Trajectory, pack_user_data
from twinkle.utils import get_logger

logger = get_logger()

__all__ = ['Challenger', 'Explorer', 'assistant_text', 'attach_user_data']

# A batch of trajectories in, the same trajectories with the model's reply
# appended out. Both MultiTurnRollout and APIMultiTurnRollout satisfy this
# as-is; build_rollout() returns whichever fits the backend. Both also accept a
# per-call ``sampling_params=`` keyword, which is how the difficulty stage asks
# for its own temperature and length budget without a second explorer.
Explorer = Callable[[List[Trajectory]], List[Trajectory]]


def attach_user_data(trajectory: Trajectory, **values: Any) -> Trajectory:
    """Return ``trajectory`` with ``values`` merged into its packed ``user_data``.

    ``user_data`` is a list of ``(key, json_string)`` pairs rather than a dict,
    so it cannot be updated in place with ``update()``; going through
    :func:`pack_user_data` keeps it in the one shape readers understand.
    """
    merged: Dict[str, Any] = {}
    for entry in trajectory.get('user_data') or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            merged[entry[0]] = entry[1]
    merged.update(values)
    out = dict(trajectory)
    out['user_data'] = pack_user_data(merged)
    return out


def assistant_text(trajectory: Trajectory) -> str:
    """The last assistant message's text, or '' if the model produced none.

    Explorers differ in what else they attach -- token ids, logprobs, tool
    turns -- but every one of them leaves the reply as an assistant message,
    so this is the one field a parser can rely on.
    """
    for message in reversed(trajectory.get('messages') or []):
        if isinstance(message, dict) and message.get('role') == 'assistant':
            return message.get('content') or ''
    return ''


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
        max_proposals_per_round: ceiling on how many proposals one round may
            request. Without it a low keep rate makes the estimator ask for an
            unbounded batch after the first round.
        solver_rollouts: attempts per candidate in the difficulty stage. ``0``
            skips the stage entirely; any other value requires the subclass to
            implement :meth:`solver_prompt` and :meth:`judge_attempt`.
        keep_min_pass: keep a candidate only if at least this many attempts
            succeeded. The default drops tasks nobody solved.
        keep_max_pass_margin: keep a candidate only if at most
            ``solver_rollouts - keep_max_pass_margin`` attempts succeeded. The
            default drops tasks everybody solved.
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
        max_proposals_per_round: int = 512,
        solver_rollouts: int = 0,
        keep_min_pass: int = 1,
        keep_max_pass_margin: int = 1,
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
            if keep_min_pass > solver_rollouts - keep_max_pass_margin:
                raise ValueError(
                    f'difficulty band is empty: keep_min_pass={keep_min_pass} > '
                    f'solver_rollouts - keep_max_pass_margin = '
                    f'{solver_rollouts - keep_max_pass_margin}')
        self.explorer = explorer
        self.system = system
        self.max_proposals_per_round = max_proposals_per_round
        self.solver_rollouts = solver_rollouts
        self.keep_min_pass = keep_min_pass
        self.keep_max_pass_margin = keep_max_pass_margin
        self.solver_params = solver_params
        self.solver_explorer = solver_explorer
        self.rng = random.Random(seed)
        # Running tally, used to size the next round and worth logging: a keep
        # rate near zero means the prompt or the filter is miscalibrated, not
        # that the model is bad.
        self.n_proposed = 0
        self.n_kept = 0

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
    ) -> List[Trajectory]:
        """Run the explorer over a batch, optionally overriding its sampling params.

        The override is only forwarded when asked for, so a plain callable
        explorer keeps working; both rollouts in
        :mod:`twinkle_agentic.rollout` accept it.
        """
        if not trajectories:
            return []
        if sampling_params is None:
            return self.explorer(trajectories)
        return self.explorer(trajectories, sampling_params=sampling_params)

    def _solver_explore(
        self,
        trajectories: List[Trajectory],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[Trajectory]:
        """Run solver attempts through the solver explorer, or fall back to the main one.

        Subclasses that need per-attempt isolation (e.g. sandbox workspace reset)
        override this rather than the whole difficulty filter.
        """
        if self.solver_explorer is not None:
            if sampling_params is None:
                return self.solver_explorer(trajectories)
            return self.solver_explorer(trajectories, sampling_params=sampling_params)
        return self.explore(trajectories, sampling_params=sampling_params)

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
        high = self.solver_rollouts - self.keep_max_pass_margin
        return [t for t, n in zip(measured, passes) if self.keep_min_pass <= n <= high]

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
