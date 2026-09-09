# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI self-play, code half, as one iteration of the resident loop.

``code/challenge.py`` generates problems and writes them to jsonl for a separate
training run to pick up later. This does the same generation and hands the result
straight to the step, in the process that owns the weights -- the arrangement
rsi.py's docstring argues for, and the reason a code problem and an agentic task
now land in one ``trajs/index.jsonl`` under one ``side`` field.

What one problem contributes is one GRPO group: the ``solver_rollouts`` attempts
the difficulty stage already made at it, each with a binary reward. Nothing is
sampled twice. The band that decides whether a problem is worth keeping --
``1 <= n_pass <= 7`` of 8 by default -- is the same band that guarantees the
group has a gradient, so selection and grouping are one decision rather than two
that can disagree. Attempts reach here through ``CodeChallenger``'s
``solver_sink``; the agentic half has had the same hook for the same reason.

Difficulty judgements do not take a sandbox slot. One is a subprocess running the
problem's asserts, milliseconds, and the stage makes ``candidates x rollouts`` of
them per round -- through a microVM that would be the dominant cost of the
iteration, and the 32 slots are worth more to the agentic half, whose episodes
cannot run anywhere else. That choice is one argument: the ``envs`` this half is
built with are :class:`~twinkle_agentic.envs.local.LocalEnv`, and handing it
``sandbox.open_pool``'s slots instead is the whole change if the trade ever does.
"""
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from twinkle import get_logger
from twinkle.data_format import SamplingParams, Trajectory, user_data_get
from twinkle_agentic.challenger import CodeChallenger, KeywordStore, load_seeds
from twinkle_agentic.envs import LocalEnv
from twinkle_agentic.rollout import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager

# Appended, not prepended: rsi.py imports this half into the process that already
# owns the agentic one, and the two directories both hold a challenge.py. Putting
# this one in front would decide that name for everybody who imports afterwards.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)
from challenge_prompts import CATEGORIES, CATEGORY_DESC, code_prompts  # noqa: E402

logger = get_logger()


class CollectingChallenger(CodeChallenger):
    """A CodeChallenger that keeps the attempts its difficulty stage makes.

    The base class measures a candidate by sampling it ``solver_rollouts`` times
    and then reports one number, and those rollouts are what the solving side
    trains on. Holding on to all of them would cost a gigabyte a round -- most
    candidates fall outside the band -- so they are dropped as soon as the number
    they produced says the candidate is not a keeper.

    The dropping reads ``keep_pass_band`` off self, i.e. the same tuple the base
    class applies one line later, so this is not a second filter with its own
    opinion. It runs in :meth:`on_difficulty_measured`, which the base class calls
    with every candidate of the round after they are measured and before they are
    filtered -- the only moment where both the counts and the attempts are in hand.
    """

    def __init__(self, *args: Any,
                 attempt_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
                 **kwargs: Any):
        super().__init__(*args, solver_sink=self._keep, **kwargs)
        if not self.solver_rollouts:
            raise ValueError('CollectingChallenger has nothing to train on with the '
                             'difficulty stage off: the attempts it collects ARE the '
                             'solving side. Pass solver_rollouts and keep_pass_band.')
        self.attempt_sink = attempt_sink
        # check_script -> the attempts at the problem it verifies. Keyed on the
        # script because that is the one field a task carries unchanged from the
        # judgement to the batch it is yielded in; the task dict itself is copied
        # on the way through attach_user_data.
        self._attempts: Dict[str, List[Dict[str, Any]]] = {}

    def _keep(self, record: Dict[str, Any]) -> None:
        """``solver_sink``: file the verdict, hold on to the trajectory."""
        if self.attempt_sink is not None:
            # Without the trajectory: every attempt that ends up trained on is
            # written in full to index.jsonl anyway, and the ones that do not are
            # here for the question of why a problem measured 0 of 8, which the
            # verdict and the interpreter's complaint answer.
            self.attempt_sink({k: v for k, v in record.items() if k != 'attempt'})
        self._attempts.setdefault(record['check_script'], []).append(record)

    def on_difficulty_measured(self, candidates: List[Trajectory]) -> None:
        super().on_difficulty_measured(candidates)
        low, high = self.keep_pass_band
        for task in candidates:
            data = task.get('user_data')
            if not low <= user_data_get(data, 'n_pass', 0) <= high:
                self._attempts.pop(user_data_get(data, 'check_script', '') or '', None)

    def take(self, check_script: str) -> List[Dict[str, Any]]:
        """The attempts at one kept problem, removed from the store."""
        return self._attempts.pop(check_script, [])


def build_challenger(args, sampler, template, *, recorder=None) -> CollectingChallenger:
    """The code half wired to the loop's live sampler, ready for one iteration.

    ``sampler`` and ``template`` belong to the caller and outlive this: an
    iteration must propose and solve with the weights the last step produced, so
    building an engine here would be building the wrong one. The template has to
    be the caller's object too -- the rollout continues a conversation by splicing
    token ids, so the ids it appends must come from the same encoder the agentic
    half is using on the same sampler.
    """
    params = SamplingParams(max_tokens=args.code_propose_max_tokens, num_samples=1,
                            logprobs=1, temperature=args.code_propose_temp, top_p=0.95)
    # One rollout for proposing and, through solver_params, for solving. max_turns=1
    # because a code answer is one message: there is nothing for a second turn to
    # react to until the asserts have run, and running them is the next stage.
    explorer = MultiTurnRollout(sampler, template=template,
                                tool_manager=ToolManager([]), max_turns=1,
                                sampling_params=params)
    store = None
    if args.code_keywords_n > 0:
        store = KeywordStore(args.code_keyword_db, CATEGORIES)
        logger.info('[collect_code] keyword bank: '
                    + ', '.join(f'{c}={len(store.items[c])}' for c in CATEGORIES))
    seeds = load_seeds(args.code_seed_file)
    logger.info(f'[collect_code] seeds: {len(seeds)} from {args.code_seed_file!r}')
    return CollectingChallenger(
        code_prompts(),
        explorer,
        # This half's slot: a check is a self-contained program over its own
        # asserts, run here in a throwaway directory. See the module docstring
        # for why it is not one of the agentic half's sandboxes.
        envs=[LocalEnv()],
        seeds=seeds,
        keyword_store=store,
        category_desc=CATEGORY_DESC if store else None,
        seed_mix_prob=args.code_seed_mix_prob,
        two_step=not args.code_no_two_step,
        keyword_refill_target=args.code_keywords_n,
        keyword_params=SamplingParams(max_tokens=1024, num_samples=1, logprobs=1,
                                      temperature=1.3, top_p=0.98),
        # A batch under the sampler's data-parallel width leaves workers idle.
        min_batch=args.sampler_gpus,
        problem_max_chars=args.code_problem_max_chars,
        max_checks=args.code_max_checks,
        sandbox_timeout=args.code_script_timeout,
        max_proposals_per_round=args.code_max_proposals_per_round,
        solver_rollouts=args.code_solver_rollouts,
        keep_pass_band=tuple(args.code_keep_pass_band),
        solver_params=SamplingParams(max_tokens=args.code_solver_max_tokens,
                                     num_samples=1, logprobs=1,
                                     temperature=args.code_solver_temp, top_p=0.95),
        seed=args.random_seed,
        reject_sink=(recorder.rejected if recorder is not None else None),
        attempt_sink=(recorder.attempt if recorder is not None else None),
    )


def collect(args, challenger: CollectingChallenger, recorder, *,
            group_id_base: int = 0) -> Dict[str, Any]:
    """Generate problems until ``--code-keep-target``, writing groups as they land.

    ``group_id_base`` offsets the ids so two task sources sharing one recorder
    cannot collide. train.py groups on ``(side, group_id)`` and ``side`` already
    separates the halves, so this is belt and braces -- and it is what makes the
    ids in index.jsonl still mean something when read by hand.
    """
    started = time.time()
    counts: Dict[str, int] = {'kept': 0, 'groups': 0, 'trajectories': 0,
                              'no_attempts': 0, 'ungrouped': 0}
    pass_dist: Dict[int, int] = {}
    batch_size = args.code_batch_size or args.code_keep_target
    for batch in challenger(batch_size=batch_size, total=args.code_keep_target):
        for task in batch:
            counts['kept'] += 1
            data = task.get('user_data')
            check_script = user_data_get(data, 'check_script', '') or ''
            n_pass = user_data_get(data, 'n_pass', 0)
            pass_dist[n_pass] = pass_dist.get(n_pass, 0) + 1
            records = challenger.take(check_script)
            if len(records) < 2:
                # A group of one has an advantage of the reward minus itself, and
                # none at all is reachable only if two problems ended up with
                # byte-identical asserts and the first yielded took both sets.
                # Counted rather than ignored: either would otherwise read as a
                # quiet shortfall in how much the iteration trained on.
                counts['no_attempts' if not records else 'ungrouped'] += 1
                continue
            group_id = group_id_base + counts['groups']
            counts['groups'] += 1
            recorder.task({
                'side': 'code', 'group_id': group_id,
                'statement': records[0].get('statement', ''),
                'check_script': check_script,
                'setup_script': user_data_get(data, 'setup_script', '') or '',
                # The challenger's own passing code, for OPSD and for reading a
                # group back: an attempt is only judgeable against a solution.
                'solution': user_data_get(data, 'solution', ''),
                'entry': user_data_get(data, 'entry', ''),
                'n_pass': n_pass,
                'n_rollouts': user_data_get(data, 'n_rollouts', 0),
                'keywords': user_data_get(data, 'keywords', []),
                'seeded': user_data_get(data, 'seeded', False),
                'two_step': user_data_get(data, 'two_step', False),
            })
            for idx, record in enumerate(records):
                counts['trajectories'] += 1
                recorder.trajectory(
                    record['attempt'], side='code', group_id=group_id,
                    # One problem is one group, so there is no proposal to index
                    # within it. Written anyway, at 0, because index.jsonl is read
                    # by one loader for both halves.
                    proposal_idx=0,
                    reward=1.0 if record['passed'] else 0.0,
                    attempt_idx=idx, passed=record['passed'],
                    n_pass=n_pass, check_output=record.get('output', ''))
        logger.info(f'[collect_code] {counts["kept"]}/{args.code_keep_target} problems, '
                    f'{counts["groups"]} groups, {counts["trajectories"]} trajectories; '
                    f'proposal stats {challenger.stats}')

    if challenger.keywords is not None:
        # After the loop: what it adds is for the next iteration, so a crash in
        # collection does not also cost the bank.
        challenger.expand_hard_keywords()
        challenger.keywords.save()
    metrics = {
        'scalars': {
            'code_problems': counts['kept'],
            'code_groups': counts['groups'],
            'code_trajectories': counts['trajectories'],
            'code_proposed': challenger.n_proposed,
            'code_seconds': round(time.time() - started, 1),
        },
        'counts': {**counts, 'proposals': dict(challenger.stats),
                   'pass_dist': dict(sorted(pass_dist.items()))},
    }
    logger.info(f'[collect_code] done in {metrics["scalars"]["code_seconds"]}s: '
                f'{metrics["scalars"]}; pass counts {metrics["counts"]["pass_dist"]}')
    return metrics
