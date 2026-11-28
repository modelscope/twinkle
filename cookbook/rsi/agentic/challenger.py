# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compose the sandbox RSI curriculum with Twinkle's challenger lifecycle."""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import replace

from twinkle.data_format import attach_user_data, user_data_get
from twinkle.utils import get_logger
from twinkle_agentic.challenger import AgenticChallenger

from . import prompts as P
from . import topic_profile as TP
from .novelty_metric import NoveltyScorer
from .profiles import default_profile_registry
from .verification import verify_check

logger = get_logger()


def novelty_factor(novelty, floor):
    """Missing measurements leave difficulty unchanged; measured scores are clipped."""
    if novelty is None:
        return 1.0
    return floor + (1.0 - floor) * min(1.0, max(0.0, float(novelty)))


def overlong_penalty(trajectory, budget, buffer_tokens, scale):
    """Charge generated tokens in the last buffer_tokens of a nonzero budget."""
    if budget <= 0 or buffer_tokens <= 0:
        return 0.0
    labels = trajectory.get('labels')
    if labels is None:
        return 0.0
    generated = sum(1 for label in labels if label != -100)
    slack = budget - generated
    if slack >= buffer_tokens:
        return 0.0
    return min(1.0, (buffer_tokens - slack) / buffer_tokens) * scale


def environment_block(profile_id):
    """Describe the actual profile's inputs without disclosing retrieved answers."""
    if not profile_id:
        return ''
    profile = default_profile_registry().get(profile_id)
    state = TP.starting_state(profile_id)
    if state == TP.START_HOST:
        starting = 'The workspace starts empty; task data is reached through the host tools.'
        note = ('Query or retrieve the real data and produce a checkable result in the workspace. '
                'Require the solver to retrieve it too; do not invent or paste the answers.\n')
    elif state == TP.START_SEEDED:
        starting = 'The workspace already contains given inputs, restored before each attempt.'
        note = (f'Given data is planted by: {", ".join(TP.seed_steps_of(profile_id))}. '
                'Read it and build on it; do not overwrite it or restate its contents.\n')
    else:
        starting = 'The workspace starts empty.'
        note = ('Create the material the task needs now. Later state the filenames and verbatim '
                'contents of inputs the solver must create.\n')
    return P.ENVIRONMENT_BLOCK.format(
        profile_id=profile_id, starting_state=starting, tools=', '.join(profile.allowed_tools) or '(none)', note=note)


class RsiChallenger(AgenticChallenger):
    """Route whole proposal groups and their solver attempts to the same profile.

    Twinkle owns scheduling, rollout, grading and group admission. This subclass
    supplies the experiment's prompts, verifier checks and novelty reward.
    """

    def __init__(self,
                 backend,
                 *,
                 knowledge_seeder=None,
                 topic_profiles=None,
                 profile_ids=(),
                 task_bank=None,
                 novelty_source='computed',
                 novelty_floor=1.0,
                 novelty_cluster_quantile=0.75,
                 novelty_tries=3,
                 novelty_workers=8,
                 novelty_model=None,
                 novelty_judge=None,
                 overlong_budget=0,
                 overlong_buffer=4096,
                 overlong_max_penalty=1.0,
                 max_build_files=4,
                 **kwargs):
        self.validate_options(locals())
        self.knowledge_seeder = knowledge_seeder
        self.topic_profiles = topic_profiles
        self.profile_ids = tuple(profile_ids)
        self.bank = task_bank
        self.novelty_source = novelty_source
        self.novelty_floor = novelty_floor
        self.novelty_cluster_quantile = novelty_cluster_quantile
        self.novelty_tries = novelty_tries
        self.novelty_workers = novelty_workers
        self.novelty_model = novelty_model
        self.novelty_judge = novelty_judge
        self._novelty_extra_body = (kwargs.get('api_kwargs') or {}).get('extra_body')
        self.overlong_budget = overlong_budget
        self.overlong_buffer = overlong_buffer
        self.overlong_max_penalty = overlong_max_penalty
        self._routing_index = 0
        self._audit_lock = threading.Lock()
        save_dir = kwargs.get('save_dir')
        self._novelty_path = os.path.join(save_dir, 'novelty_scores.jsonl') if save_dir else ''
        kwargs.setdefault('system_prompt', P.SYSTEM +
                          (P.BUILD_SIZE_CAP.format(n=max_build_files) if max_build_files > 0 else ''))
        kwargs.setdefault('from_scratch_prompt', P.OPEN_DIRECTION)
        kwargs.setdefault('from_seed_prompt', '{seed}')
        kwargs.setdefault('check_followup_prompt', P.CHECK_FOLLOWUP)
        kwargs.setdefault('check_retry_followup_prompt', P.CHECK_RETRY_FOLLOWUP)
        kwargs.setdefault('stop_after_stuck_turns', 2)
        kwargs.setdefault('api_appended_as', 'context')
        super().__init__(backend, **kwargs)

    @staticmethod
    def validate_options(options):
        """Check experiment configuration before distributed/model initialization."""
        novelty_source = options.get('novelty_source', 'computed')
        novelty_floor = options.get('novelty_floor', 1.0)
        novelty_cluster_quantile = options.get('novelty_cluster_quantile', 0.75)
        novelty_tries = options.get('novelty_tries', 3)
        novelty_workers = options.get('novelty_workers', 8)
        overlong_budget = options.get('overlong_budget', 0)
        overlong_buffer = options.get('overlong_buffer', 4096)
        overlong_max_penalty = options.get('overlong_max_penalty', 1.0)
        if options.get('max_build_files', 4) < 0:
            raise ValueError('max_build_files must be non-negative')
        if options.get('stop_after_stuck_turns', 2) < 0:
            raise ValueError('stop_after_stuck_turns must be non-negative')
        token_limit = options.get('max_trajectory_tokens')
        if token_limit is not None and token_limit < 1:
            raise ValueError('max_trajectory_tokens must be positive or unset')
        if novelty_source not in ('computed', 'judge', 'off'):
            raise ValueError('novelty_source must be computed, judge or off')
        if not math.isfinite(novelty_floor) or not 0 <= novelty_floor <= 1:
            raise ValueError('novelty_floor must be in [0, 1]')
        if not math.isfinite(novelty_cluster_quantile) or not 0 <= novelty_cluster_quantile <= 1:
            raise ValueError('novelty_cluster_quantile must be in [0, 1]')
        if novelty_tries < 1 or novelty_workers < 1:
            raise ValueError('novelty tries and workers must be positive')
        if overlong_budget < 0 or (overlong_budget and not 0 < overlong_buffer < overlong_budget):
            raise ValueError('overlong buffer must be positive and below its budget')
        if not 0 < overlong_max_penalty <= 1:
            raise ValueError('overlong_max_penalty must be in (0, 1]')
        if novelty_source == 'computed':
            import importlib.util
            if importlib.util.find_spec('sklearn') is None:
                raise ValueError('computed novelty requires pip install -r cookbook/rsi/requirements.txt')

    def _build_challenge_prompt(self):
        prompt = super()._build_challenge_prompt()
        route = self.knowledge_seeder.last_route if self.knowledge_seeder is not None else None
        profile_id = None
        if route is not None and self.topic_profiles is not None:
            anchor = getattr(route, 'anchor', None)
            if anchor is not None:
                profile_id = self.topic_profiles.for_card(anchor)
        if profile_id is None and self.profile_ids:
            # A fallback is fixed for the group, including all its solver jobs.
            anchor = getattr(route, 'anchor', None)
            fallback_index = anchor.index if anchor is not None else self._routing_index
            profile_id = self.profile_ids[fallback_index % len(self.profile_ids)]
        self._routing_index += 1
        prompt['messages'][-1]['content'] += environment_block(profile_id)
        return attach_user_data(
            prompt, profile_id=profile_id or '', provider=route.metadata() if route is not None else None)

    @staticmethod
    def _prepare_profile(trajectory, env):
        profile_id = user_data_get(trajectory.get('user_data'), 'profile_id', '')
        if profile_id and getattr(env, 'profile_id', '') != profile_id:
            env.switch_profile(profile_id)

    def _episode(self, prompt, env):
        self._prepare_profile(prompt, env)
        result = super()._episode(prompt, env)
        metadata = {
            'profile_id': getattr(env, 'profile_id', ''),
            'provider': user_data_get(prompt.get('user_data'), 'provider'),
        }
        result.trajectory = attach_user_data(result.trajectory, **metadata)
        if result.task is not None:
            result.task = attach_user_data(result.task, **metadata)
        return result

    def _solve(self, unit, proposal, env):
        # Select before the base method constructs the tool manager and prompt.
        try:
            self._prepare_profile(proposal.task, env)
        except Exception:
            self._job_done(unit)
            raise
        return super()._solve(unit, proposal, env)

    def _solver_prompt(self, task, env):
        prompt = super()._solver_prompt(task, env)
        return attach_user_data(
            prompt,
            profile_id=getattr(env, 'profile_id', ''),
            provider=user_data_get(task.get('user_data'), 'provider'))

    def _verify_check(self, state, script):
        result, error = verify_check(state['env'], script)
        state['verification'] = result
        return (1, error) if error else (0, '')

    def _build_test_case(self, state, reply):
        followup = super()._build_test_case(state, reply)
        if state.get('checked') and followup is not None:
            profile_id = getattr(state['env'], 'profile_id', '')
            starting = {
                TP.START_HOST: P.START_HOST_SERVICE,
                TP.START_SEEDED: P.START_SEEDED,
            }.get(TP.starting_state(profile_id), P.START_EMPTY)
            return P.PROBLEM_FOLLOWUP.format(starting_state=starting)
        return followup

    def _build_query(self, state, explored):
        task = super()._build_query(state, explored)
        if task is None:
            return None
        verification = state.get('verification') or {}
        profile = getattr(state['env'], 'profile', None)
        artifacts = verification.get('artifact_profile', [])
        derived = set(verification.get('derived_paths', []))
        deliverables = [
            dict(
                path=item['path'],
                media_type=item['media_type'],
                role='derived' if item['path'] in derived else 'supporting') for item in artifacts
        ]
        workspace = {key: verification.get(key) for key in ('snapshot_sha256', 'artifact_profile')}
        return attach_user_data(
            task,
            verification=verification,
            profile_digest=profile.digest if profile is not None else '',
            deliverables=deliverables,
            workspace=workspace)

    @staticmethod
    def _statement(proposal):
        return next((m.get('content', '') for m in proposal.task.get('messages', [])
                     if m.get('role') == 'user'), '') if proposal.task is not None else ''

    def _audit_novelty(self, record):
        if self._novelty_path:
            with self._audit_lock:
                with open(self._novelty_path, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _judge_novelty(self, statements, members, group_id):
        if self.bank is None:
            return [None] * len(members)
        from twinkle_agentic.verifier import score_tasks
        judge = self.novelty_judge or score_tasks
        scores = [None] * len(members)
        for attempt in range(self.novelty_tries):
            pending = [i for i, value in enumerate(scores) if value is None]
            if not pending:
                break
            payload = [{
                'statement':
                statements[i],
                'check':
                user_data_get(members[i].task.get('user_data'), 'check_script', ''),
                'references':
                self.bank.references(statements[i], extra=[text for j, text in enumerate(statements) if i != j]),
            } for i in pending]
            try:
                results = judge(
                    payload,
                    workers=self.novelty_workers,
                    model=self.novelty_model,
                    extra_body=self._novelty_extra_body)
            except Exception as error:
                logger.warning(f'[rsi] novelty judge attempt {attempt + 1} failed: {error}')
                continue
            for i, item, result in zip(pending, payload, results):
                scores[i] = result.scores.get('novelty')
                self._audit_novelty(
                    dict(
                        group_id=group_id,
                        source='judge',
                        attempt=attempt + 1,
                        scores=result.scores,
                        error=result.error,
                        **item))
        return scores

    def _score(self, unit):
        members = [p for p in unit.proposals if p is not None and self._statement(p)]
        statements = [self._statement(p) for p in members]
        scores = [None] * len(members)
        if self.novelty_source == 'computed' and members:
            try:
                if len(members) == 1:
                    scores = [1.0]
                else:
                    corpus = (self.bank.statements() if self.bank is not None else []) + statements
                    groups = (self.bank.sibling_groups() if self.bank is not None else []) + [statements]
                    scorer = NoveltyScorer(
                        corpus, calibration_groups=groups, cluster_quantile=self.novelty_cluster_quantile)
                    scores = scorer.score(statements)
                self._audit_novelty(
                    dict(group_id=unit.group_id, source='computed', statements=statements, scores=scores))
            except Exception as error:
                logger.warning(f'[rsi] computed novelty failed: {error}; using judge fallback')
                scores = self._judge_novelty(statements, members, unit.group_id)
        elif self.novelty_source == 'judge' and members:
            scores = self._judge_novelty(statements, members, unit.group_id)
            if self.bank is not None and any(score is None for score in scores):
                # An incomplete judge group must not train with partial reward terms.
                for proposal in unit.proposals:
                    if proposal is not None:
                        proposal.outcome = 'novelty_unscored'
                        if proposal.task is not None and self.num_solver_rollouts:
                            proposal.n_pass = sum(passed for _, passed in proposal.attempts)
                with self._tally:
                    self.n_proposed += sum(proposal is not None for proposal in unit.proposals)
                self._record_proposals([p for p in unit.proposals if p is not None])
                self._complete([], [])
                return
        for proposal, score in zip(members, scores):
            proposal.trajectory = attach_user_data(proposal.trajectory, novelty=score)
        super()._score(unit)

    def _proposal_reward(self, proposal):
        novelty = user_data_get(proposal.trajectory.get('user_data'), 'novelty')
        return super()._proposal_reward(proposal) * novelty_factor(novelty, self.novelty_floor)

    def _penalize(self, trajectory):
        reward = float(trajectory.get('rewards') or 0.0)
        penalty = overlong_penalty(trajectory, self.overlong_budget, self.overlong_buffer,
                                   self.overlong_max_penalty) if reward else 0.0
        result = attach_user_data(trajectory, task_reward=reward, length_penalty=penalty)
        result['rewards'] = max(0.0, reward - penalty)
        return result

    @staticmethod
    def _chosen(proposals):
        return max((p for p in proposals if p.outcome == 'kept' and p.task is not None and p.attempts),
                   key=lambda p: p.reward,
                   default=None)

    def _record_proposals(self, proposals):
        chosen = self._chosen(proposals)
        recorded = []
        for proposal in proposals:
            trajectory = attach_user_data(proposal.trajectory, selected=proposal is chosen)
            trajectory['rewards'] = proposal.reward
            trajectory = self._penalize(trajectory)
            if proposal.task is not None:
                for key in ('verification', 'profile_digest', 'deliverables', 'workspace'):
                    trajectory = attach_user_data(trajectory,
                                                  **{key: user_data_get(proposal.task.get('user_data'), key)})
            recorded.append(replace(proposal, trajectory=trajectory, reward=trajectory['rewards']))
        super()._record_proposals(recorded)

    def _groups(self, proposals):
        proposing, solving = super()._groups(proposals)
        proposing = [[self._penalize(t) for t in group] for group in proposing]
        solving = [[self._penalize(t) for t in group] for group in solving]
        chosen = self._chosen(proposals)
        if self._recorder is not None and chosen is not None:
            for group in solving:
                for episode in group:
                    data = episode.get('user_data')
                    passed = user_data_get(data, 'passed', False)
                    self._recorder.write(
                        episode,
                        side='solve',
                        group_id=user_data_get(data, 'group_id'),
                        proposal_index=user_data_get(data, 'proposal_index'),
                        passed=passed,
                        outcome='passed' if passed else 'failed',
                        reward=episode['rewards'],
                        statement=self._statement(chosen))
        if chosen is not None and self.bank is not None:
            fields = {
                key: user_data_get(chosen.trajectory.get('user_data'), key)
                for key in ('provider', 'profile_id', 'novelty')
            }
            fields.update({
                key: user_data_get(chosen.task.get('user_data'), key)
                for key in ('verification', 'deliverables', 'workspace', 'profile_digest')
            })
            self.bank.add(
                self._statement(chosen),
                user_data_get(chosen.task.get('user_data'), 'check_script', ''),
                group_id=chosen.group_id,
                n_pass=chosen.n_pass,
                **fields)
        return proposing, solving
