# Copyright (c) ModelScope Contributors. All rights reserved.
"""Agentic challenger: act in a sandbox, verify the result, then describe it."""
import math
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format import SamplingParams, Trajectory, attach_user_data, user_data_get
from twinkle.data_format.sampling import SampledSequence, SampleResponse
from twinkle.utils import get_logger
from twinkle_agentic.envs import Env
from twinkle_agentic.protocol.api_sampler import APISampler
from twinkle_agentic.rollout import MultiTurnRollout
from twinkle_agentic.utils.code_utils import parse_fenced_code, strip_reasoning
from twinkle_agentic.utils.message_utils import assistant_text
from .base import Challenger
from .recorder import RolloutRecorder

__all__ = ['AgenticChallenger', 'parse_problem_statement']

logger = get_logger()

_FENCED_BLOCK_RE = re.compile(r'```[^\r\n]*\r?\n(.*?)```', re.S)


def parse_problem_statement(text: str) -> Optional[str]:
    """Return the statement after removing reasoning and one outer fence."""
    body = strip_reasoning(text).strip()
    whole = _FENCED_BLOCK_RE.fullmatch(body)
    if whole:
        body = whole.group(1).strip()
    return body or None


def _sample_one(sampler: Any, input_feature: Dict[str, Any], sampling_params: Optional[SamplingParams],
                adapter_kwargs: Dict[str, Any]) -> SampledSequence:
    responses = sampler.sample([input_feature], sampling_params=sampling_params, **adapter_kwargs)
    if not isinstance(responses, list):
        raise TypeError(f'expected List[SampleResponse] from sampler.sample, got '
                        f'{type(responses).__name__}')
    if len(responses) != 1:
        raise RuntimeError(f'sampler returned {len(responses)} responses for a single request; '
                           'expected exactly one')
    response = responses[0]
    if not isinstance(response, SampleResponse):
        raise TypeError(f'expected SampleResponse from sampler.sample, got '
                        f'{type(response).__name__}')
    if len(response.sequences) != 1:
        raise RuntimeError(f'SampleResponse contains {len(response.sequences)} sequences; '
                           'expected exactly one')
    sequence = response.sequences[0]
    if not isinstance(sequence, SampledSequence):
        raise TypeError(f'expected SampledSequence, got {type(sequence).__name__}')
    return sequence


def _api_followup_response(
    sampler: Any,
    api: Optional[APISampler],
    sampling_params: Optional[SamplingParams],
    *,
    input_feature: Dict[str, Any],
    adapter_kwargs: Dict[str, Any],
    followups: int,
    **kwargs: Any,
) -> SampledSequence:
    """Use the API for appended stages and the primary backend otherwise."""
    if followups:
        if api is None:
            raise ValueError('a follow-up stage was routed to the API but none was configured')
        return api(input_feature, sampling_params, **adapter_kwargs)
    if sampler is not None:
        return _sample_one(sampler, input_feature, sampling_params, adapter_kwargs)
    if api is not None:
        return api(input_feature, sampling_params, **adapter_kwargs)
    raise ValueError('AgenticChallenger has neither a sampler nor an API backend')


@dataclass
class _ProposalResult:
    trajectory: Trajectory
    group_id: str = ''
    task: Optional[Trajectory] = None
    reason: str = ''
    detail: str = ''
    outcome: str = ''
    n_pass: Optional[int] = None
    reward: float = 0.0
    attempts: List[Tuple[Trajectory, bool]] = field(default_factory=list)


@dataclass
class _Unit:
    """One prompt's worth of work: its proposals, and the attempts they earn.

    Held together by a count of jobs rather than by a barrier, because its jobs do
    not start together: a proposal that lands early has its attempts queued while
    its siblings are still proposing. The last job to finish, of either kind, is
    the one that scores the unit.
    """

    group_id: str
    proposals: List[Optional[_ProposalResult]]
    pending: int
    lock: threading.Lock = field(default_factory=threading.Lock)


class AgenticChallenger(Challenger):
    """Invent tool-using tasks by doing, checking, and describing them.

    ``backend`` drives exploration and solver attempts. When an ``api`` backend is
    given, it generates only the appended check-script and problem-statement turns;
    those turns retain the masking semantics selected by ``api_appended_as`` in
    ``rollout_kwargs``.

    What a unit of work proposes *about* is not this class's business: ``seed_fn``
    is asked once per unit and whatever it returns is appended to the opening
    instruction. So a run adds a kind of variety -- a keyword pool, earlier
    trajectories, a difficulty ladder -- by passing a different callable, not by
    growing a parameter here per kind. Nothing back means propose from scratch,
    which is also what no ``seed_fn`` at all means.

    ``solver_rollout`` decides how an attempt is *run*. Left out, attempts go
    through the same loop the proposing side uses: this class writes the opening,
    generates, dispatches the environment's tools, appends the results. Given one,
    that whole job is handed over -- to :class:`~..rollout.external.ExternalRollout`
    for an agent that ships as its own program, or to anything else that answers
    ``(trajectories, env=..., tool_manager=...) -> List[Trajectory]`` and returns
    one trained episode per prompt. The proposing side never uses it: proposing is
    a training-only role, and its loop is this class's own by design.
    """

    _system = ('You invent tasks for another agent to solve. You have a sandbox and '
               'tools. Work in it first: build something real, then you will be asked '
               'to verify it and to describe it.')
    _from_scratch = ('Choose a task worth doing in this sandbox and do it now, using '
                     'your tools. Do not describe it yet.')
    # The seed is appended rather than woven in: it says what to build around, this
    # says what to do with it, and neither has to know how the other is phrased.
    _from_seed = ('Choose a task worth doing in this sandbox and do it now, using your '
                  'tools. Do not describe it yet.\n\n{seed}')
    _check_followup = ('Stop working. This is the workspace you produced:\n\n{final_state}\n\n'
                       'Write a {language} script that verifies this end state, as a fenced '
                       '{language} code block and nothing else. It must exit with a non-zero status '
                       'if the work was not done. Check what can be read out of the files -- their '
                       'structure and the values inside them. NEVER check a file size in bytes, a '
                       'checksum, or the full source text of a script: correct solutions differ '
                       'there, and such a check only its own author can pass.')
    _check_retry_followup = ('Your check script did not pass:\n\n{error}\n\nThe workspace is:\n\n'
                             '{final_state}\n\nReturn a corrected script as a fenced {language} code block '
                             'and nothing else.')
    _check_parse_error = ('Could not read a check script from your reply: it was not a '
                          'fenced {language} code block. Do not wrap it in a tool call and '
                          'do not add prose -- return ONLY a fenced {language} code block.')
    _problem_followup = ('Now write the task statement: what someone starting from an empty workspace '
                         'would have to be told to produce what you produced, and nothing about how you '
                         'did it. Name the files to create and quote any input data verbatim. Do not '
                         'reveal values your check script computes. Reply with the statement only.')
    # The statement says what to produce, not that producing it is the job. Without
    # this a solver answers with a description of the work and the check script,
    # reading a workspace nobody touched, fails it.
    _solver_system = ('You solve tasks in a workspace using your tools. Do the work -- create the '
                      'files the task asks for. Do not just describe what you would do.')

    def __init__(
        self,
        backend: Any,
        *,
        api: Optional[Any] = None,
        seed_fn: Optional[Callable[[], Optional[str]]] = None,
        system_prompt: Optional[str] = None,
        from_scratch_prompt: Optional[str] = None,
        from_seed_prompt: Optional[str] = None,
        check_followup_prompt: Optional[str] = None,
        check_retry_followup_prompt: Optional[str] = None,
        check_parse_error_prompt: Optional[str] = None,
        problem_followup_prompt: Optional[str] = None,
        solver_system_prompt: Optional[str] = None,
        check_retries: int = 1,
        problem_max_chars: int = 8192,
        check_language: str = 'python',
        parse_check_fn: Optional[Callable[[str], Optional[str]]] = None,
        brittle_check_fn: Optional[Callable[[str], Optional[str]]] = None,
        pass_rate_target: float = 0.2,
        envs: Sequence[Env] = (),
        solver_rollout: Optional[Any] = None,
        num_challenger_rollouts: int = 8,
        num_solver_rollouts: int = 8,
        pass_band: Tuple[float, float] = (1.0, 7.0),
        pass_rate_width: float = 0.3,
        max_empty_rounds: int = 0,
        followup_params: Optional[SamplingParams] = None,
        checker: Optional[Callable[[Trajectory], bool]] = None,
        save_dir: Optional[str] = None,
        save_failed_rollouts: bool = True,
        **rollout_kwargs: Any,
    ):
        super().__init__(
            envs=envs,
            num_challenger_rollouts=num_challenger_rollouts,
            num_solver_rollouts=num_solver_rollouts,
            pass_band=pass_band,
            max_empty_rounds=max_empty_rounds,
        )
        if check_retries < 0:
            raise ValueError(f'check_retries must be >= 0, got {check_retries}')
        if problem_max_chars <= 0:
            raise ValueError(f'problem_max_chars must be positive, got {problem_max_chars}')
        if not check_language.strip():
            raise ValueError('check_language must not be empty')
        if not 0 <= pass_rate_target <= 1:
            raise ValueError(f'pass_rate_target must be in [0, 1], got {pass_rate_target}')
        if pass_rate_width <= 0:
            raise ValueError(f'pass_rate_width must be positive, got {pass_rate_width}')
        if api is not None and rollout_kwargs.get('response_callback') is not None:
            raise ValueError('api= routes the appended turns and cannot be combined with response_callback')
        self.seed_fn = seed_fn
        self._system = self._system if system_prompt is None else system_prompt
        self._from_scratch = self._from_scratch if from_scratch_prompt is None else from_scratch_prompt
        self._from_seed = self._from_seed if from_seed_prompt is None else from_seed_prompt
        self._check_followup = self._check_followup if check_followup_prompt is None else check_followup_prompt
        self._check_retry_followup = (
            self._check_retry_followup if check_retry_followup_prompt is None else check_retry_followup_prompt)
        self._check_parse_error = (
            self._check_parse_error if check_parse_error_prompt is None else check_parse_error_prompt)
        self._problem_followup = (
            self._problem_followup if problem_followup_prompt is None else problem_followup_prompt)
        self._solver_system = self._solver_system if solver_system_prompt is None else solver_system_prompt
        self._check_retries = check_retries
        self._problem_max_chars = problem_max_chars
        self._check_language = check_language.strip().lower()
        self._parse_check_fn = parse_check_fn
        self._brittle_check_fn = brittle_check_fn
        self._pass_rate_target = pass_rate_target
        self._pass_rate_width = pass_rate_width
        self.checker = checker
        self.followup_params = followup_params
        self.save_failed_rollouts = save_failed_rollouts
        self._recorder = RolloutRecorder(save_dir) if save_dir else None
        kwargs = dict(rollout_kwargs)
        # A separate API backend takes only the appended check/statement turns;
        # the acting turns stay on the policy. No api, no split -- the default
        # callback keeps every turn on the backend.
        if api is not None:
            kwargs['api'] = api
            kwargs['response_callback'] = _api_followup_response
        # Every job shares one rollout, built here rather than on first use:
        # building it needs nothing a job has, so building it up front spares the
        # jobs a race over who gets to -- one they would all lose but one.
        self._rollout = MultiTurnRollout(backend, **kwargs)
        # Attempts run through the same loop unless a caller handed one over.
        # Not owned either way: a rollout passed in was built by the caller and is
        # the caller's to close, and the one built here is closed as itself.
        self._solver_rollout = solver_rollout if solver_rollout is not None else self._rollout
        self._tally = threading.Lock()

    def _tool_manager(self, env: Env) -> Optional[Any]:
        return env.tool_manager() if env.tools() else None

    @staticmethod
    def _with_tools(prompt: Trajectory, env: Env) -> Trajectory:
        """A copy of ``prompt`` advertising the tools this environment executes.

        Read off the environment in hand, per job, rather than once at
        construction: an environment that stands its tool runtime up on first use
        has nothing to report before it is leased, and taking the schemas from
        the side that will run them is what keeps the contract in the prompt and
        the code behind it from drifting apart.
        """
        tools = env.tools()
        if not tools:
            return prompt
        prompt = dict(prompt)
        prompt['tools'] = list(tools)
        return prompt

    def _build_challenge_prompt(self) -> Optional[Trajectory]:
        """The opening turn of a unit of work, with whatever the seeder offered appended.

        Never None: a seeder with nothing left to offer costs this unit a plainer
        prompt, not the run. What ends a run is ``max_empty_rounds``, which counts
        units that produced nothing trainable -- the honest measure, since a seed
        is inspiration and a unit can succeed without one.
        """
        seed = (self.seed_fn() if self.seed_fn is not None else None) or ''
        seed = seed.strip()
        user = self._from_seed.format(seed=seed) if seed else self._from_scratch
        prompt: Trajectory = {
            'messages': [
                {
                    'role': 'system',
                    'content': self._system
                },
                {
                    'role': 'user',
                    'content': user
                },
            ],
        }
        return attach_user_data(prompt, seed=seed)

    def _launch(self) -> bool:
        """Queue one prompt's proposing episodes."""
        prompt = self._build_challenge_prompt()
        if prompt is None:
            return False
        unit = _Unit(
            group_id=uuid.uuid4().hex,
            proposals=[None] * self.num_challenger_rollouts,
            pending=self.num_challenger_rollouts)
        for index in range(self.num_challenger_rollouts):
            self._submit(lambda env, i=index: self._propose(unit, i, prompt, env))
        return True

    def _propose(self, unit: _Unit, index: int, prompt: Trajectory, env: Env) -> None:
        """One proposing episode, and the attempts it earns by producing a task.

        The attempts are queued from here rather than once the unit has finished
        proposing: a task can be solved the moment it exists, and waiting for its
        siblings is what leaves environments idle at the end of every round.
        """
        try:
            result = self._episode(prompt, env)
            result.group_id = unit.group_id
            unit.proposals[index] = result
            if result.task is not None and self.num_solver_rollouts:
                # Counted in before this job is counted out, or the unit reads as
                # finished with its attempts not yet asked for.
                with unit.lock:
                    unit.pending += self.num_solver_rollouts
                for _ in range(self.num_solver_rollouts):
                    self._submit(lambda solver_env, r=result: self._solve(unit, r, solver_env))
        finally:
            self._job_done(unit)

    def _episode(self, prompt: Trajectory, env: Env) -> _ProposalResult:
        prompt = self._with_tools(prompt, env)
        state: Dict[str, Any] = {'env': env}
        kwargs: Dict[str, Any] = {
            'followup_fn': lambda trajectory, n_before: self._followup(state, trajectory, n_before),
        }
        manager = self._tool_manager(env)
        if manager is not None:
            kwargs['tool_manager'] = manager
        explored = self._rollout([prompt], **kwargs)
        if not explored:
            self._reject(state, 'rollout_no_output')
            return _ProposalResult(dict(prompt), reason='rollout_no_output')
        trajectory = explored[0]
        task = self._build_query(state, trajectory)
        reason, detail = state.get('reject', ('', ''))
        return _ProposalResult(trajectory, task=task, reason=reason, detail=detail)

    def _solve(self, unit: _Unit, proposal: _ProposalResult, env: Env) -> None:
        """One attempt at one task, graded in the environment that made it.

        Both the environment and its tool manager go to the rollout, and which of
        the two it reads is its own business: a loop driven here dispatches through
        the manager, an agent that runs as a program is handed the environment to
        run in. Neither has to be told which kind it is talking to.
        """
        try:
            kwargs: Dict[str, Any] = {'env': env}
            manager = self._tool_manager(env)
            if manager is not None:
                kwargs['tool_manager'] = manager
            attempts = self._solver_rollout([self._solver_prompt(proposal.task, env)], **kwargs)
            if attempts:
                passed = self._judge(proposal.task, env)
                with unit.lock:
                    proposal.attempts.append((attempts[0], passed))
        finally:
            self._job_done(unit)

    def _job_done(self, unit: _Unit) -> None:
        """Count one job out, and score the unit if it was the last one."""
        with unit.lock:
            unit.pending -= 1
            if unit.pending:
                return
        self._score(unit)

    def _followup(self, state: Dict[str, Any], trajectory: Trajectory,
                  n_before: int) -> Optional[Tuple[str, Optional[SamplingParams]]]:
        if state.get('checked'):
            return None
        reply = None if n_before == 0 else assistant_text(trajectory)
        followup = self._build_test_case(state, reply)
        if followup is None:
            return None
        return followup, self.followup_params

    def _build_test_case(self, state: Dict[str, Any], reply: Optional[str]) -> Optional[str]:
        env: Env = state['env']
        if reply is None:
            snapshot, error = env.snapshot()
            state['snapshot'] = snapshot
            if not snapshot.strip():
                state['reject'] = ('snapshot_unavailable' if error else 'empty_workspace', error)
                return None
            return self._check_followup.format(final_state=snapshot, language=self._check_language)

        attempt = state.get('check_attempts', 0) + 1
        state['check_attempts'] = attempt
        script = (
            self._parse_check_fn(reply) if self._parse_check_fn is not None else parse_fenced_code(
                reply, language_tags=None))
        if script is None:
            if attempt <= self._check_retries:
                return self._check_retry_followup.format(
                    error=self._check_parse_error.format(language=self._check_language),
                    final_state=state.get('snapshot', ''),
                    language=self._check_language,
                )
            state['reject'] = ('check_parse_fail', reply)
            return None
        state['script'] = script
        # Read off the script before it is run, because passing on the author's own
        # workspace is exactly what hides this defect: a check that pins a file's
        # size or quotes a script's source passes for its author and fails every
        # correct reproduction. The reason goes back the way a failed assertion
        # does, since it is the same kind of fault.
        brittle = self._brittle_check_fn(script) if self._brittle_check_fn is not None else None
        exit_code, output = (1, brittle) if brittle else env.run_script(script, interpreter=self._check_language)
        if exit_code == 0:
            state['checked'] = True
            return self._problem_followup
        after = env.snapshot()[0]
        state.setdefault('attempts', []).append(f'--- attempt {attempt}: exit {exit_code} ---\n{output}\n'
                                                f'--- check script ---\n{script}')
        if attempt <= self._check_retries:
            return self._check_retry_followup.format(
                error=output,
                final_state=after or state.get('snapshot', ''),
                language=self._check_language,
            )
        state['reject'] = ('check_run_fail', '\n'.join(state['attempts']))
        return None

    def _build_query(self, state: Dict[str, Any], explored: Trajectory) -> Optional[Trajectory]:
        if state.get('reject'):
            return self._reject(state, *state['reject'])
        if not state.get('checked'):
            return self._reject(
                state,
                'episode_cut_short',
                f"stop_reason={explored.get('stop_reason')} "
                f"truncated={bool(explored.get('truncated'))} "
                f"turns={explored.get('turns')}",
            )
        statement = parse_problem_statement(assistant_text(explored))
        if statement is None:
            return self._reject(state, 'problem_parse_fail')
        if len(statement) > self._problem_max_chars:
            return self._reject(state, 'too_long', f'{len(statement)} chars')
        task: Trajectory = attach_user_data(
            {'messages': [{
                'role': 'user',
                'content': statement
            }]},
            check_script=state['script'],
            seed=user_data_get(explored.get('user_data'), 'seed', ''),
        )
        if self.checker is not None and not self.checker(task):
            return self._reject(state, 'rejected_by_checker')
        return task

    def _reject(self, state: Dict[str, Any], reason: str, detail: str = '') -> Optional[Trajectory]:
        state['reject'] = (reason, detail)
        logger.info(f'[{type(self).__name__}] rejected: {reason}'
                    f"{f' -- {detail[:400]}' if detail else ''}")
        return None

    def _solver_prompt(self, task: Trajectory, env: Env) -> Trajectory:
        """The opening one attempt starts from: the statement, plus how to read it.

        The tools come from the environment, since the schemas that mean anything
        are the ones the environment will honour. A rollout that brings its own
        agent brings its own opening too and reads only the statement out of this,
        which costs it the unused keys and nothing else.
        """
        statement = next(
            (message.get('content', '')
             for message in task.get('messages') or [] if isinstance(message, dict) and message.get('role') == 'user'),
            '')
        messages: List[Dict[str, Any]] = [{'role': 'user', 'content': statement}]
        if self._solver_system:
            messages.insert(0, {'role': 'system', 'content': self._solver_system})
        return self._with_tools({'messages': messages}, env)

    def _judge(self, task: Trajectory, env: Env) -> bool:
        script = user_data_get(task.get('user_data'), 'check_script', '')
        if not script:
            return False
        return env.run_script(script, interpreter=self._check_language)[0] == 0

    def challenger_reward(self, n_pass: Optional[int]) -> float:
        """Reward tasks near the target solver pass rate; unmeasured failures score zero."""
        if n_pass is None or not self.num_solver_rollouts or n_pass <= 0:
            return 0.0
        gap = n_pass / self.num_solver_rollouts - self._pass_rate_target
        variance = 2.0 * self._pass_rate_width**2
        return math.exp(-(gap * gap) / variance)

    def _record_proposals(self, proposals: List[_ProposalResult]) -> None:
        if self._recorder is None:
            return
        for index, proposal in enumerate(proposals):
            if proposal.task is None and not self.save_failed_rollouts:
                continue
            trajectory = dict(proposal.trajectory)
            trajectory['rewards'] = proposal.reward
            task_data = proposal.task.get('user_data') if proposal.task is not None else None
            statement = ''
            if proposal.task is not None:
                statement = next((message.get('content', '') for message in proposal.task.get('messages') or []
                                  if isinstance(message, dict) and message.get('role') == 'user'), '')
            self._recorder.write(
                trajectory,
                side='propose',
                group_id=proposal.group_id,
                proposal_index=index,
                outcome=proposal.outcome or ('rejected' if proposal.reason else 'kept'),
                reason=proposal.reason,
                detail=proposal.detail,
                reward=proposal.reward,
                n_pass=proposal.n_pass,
                n_rollouts=(self.num_solver_rollouts if proposal.n_pass is not None else None),
                pass_rate=(proposal.n_pass / self.num_solver_rollouts
                           if proposal.n_pass is not None and self.num_solver_rollouts else None),
                statement=statement,
                check_script=user_data_get(task_data, 'check_script', ''),
                seed=user_data_get(proposal.trajectory.get('user_data'), 'seed', ''),
            )

    def _score(self, unit: _Unit) -> None:
        """Grade a finished unit and hand over whatever is trainable in it.

        Difficulty is counted here rather than measured: the attempts have already
        run, each in an environment of its own, and how many of them passed is what
        puts a task inside the band or outside it.
        """
        proposals = [proposal for proposal in unit.proposals if proposal is not None]
        low, high = self.pass_band
        kept = 0
        for proposal in proposals:
            if proposal.task is None:
                continue
            if not self.num_solver_rollouts:
                proposal.outcome = 'kept'
                kept += 1
                continue
            n_pass = sum(1 for _, passed in proposal.attempts if passed)
            proposal.task = attach_user_data(proposal.task, n_pass=n_pass, n_rollouts=self.num_solver_rollouts)
            proposal.n_pass = n_pass
            proposal.reward = self.challenger_reward(n_pass)
            if low <= n_pass <= high:
                proposal.outcome = 'kept'
                kept += 1
            else:
                proposal.outcome = 'outside_band'
        verified = sum(1 for proposal in proposals if proposal.task is not None)
        with self._tally:
            self.n_proposed += len(proposals)
            self.n_kept += kept
            logger.info(f'[{type(self).__name__}] {len(proposals)} episodes, {verified} verified, '
                        f'{kept} in band (cumulative {self.n_kept}/{self.n_proposed})')
        self._record_proposals(proposals)
        self._complete(*self._groups(proposals))

    def _groups(self, proposals: List[_ProposalResult]) -> Tuple[List[List[Trajectory]], List[List[Trajectory]]]:
        """One proposing group, and the attempts on at most one of the tasks it kept.

        Every proposal is in the proposing group, the rejected ones included: their
        zero reward is what the rest of the group is measured against. Only one task
        per unit hands over its attempts, the one whose pass rate landed closest to
        the target -- the tasks of a unit share a seed and a prompt, so training on
        several of them buys correlated data with a batch slot that another unit's
        would have filled better. A unit whose proposals all fell outside the band
        hands over none.
        """
        challenger: List[Trajectory] = []
        solver: List[Tuple[float, List[Trajectory]]] = []
        for index, proposal in enumerate(proposals):
            trajectory = dict(proposal.trajectory)
            trajectory['rewards'] = proposal.reward
            challenger.append(
                attach_user_data(
                    trajectory,
                    side='propose',
                    group_id=proposal.group_id,
                    proposal_index=index,
                    outcome=proposal.outcome or ('rejected' if proposal.reason else 'kept'),
                    n_pass=proposal.n_pass))
            if proposal.outcome != 'kept':
                continue
            group: List[Trajectory] = []
            for attempt, passed in proposal.attempts:
                episode = dict(attempt)
                episode['rewards'] = 1.0 if passed else 0.0
                group.append(
                    attach_user_data(
                        episode,
                        side='solve',
                        group_id=f'{proposal.group_id}:{index}',
                        proposal_index=index,
                        passed=passed))
            if group:
                solver.append((proposal.reward, group))
        best = max(solver, key=lambda entry: entry[0], default=None)
        return [challenger] if challenger else [], [best[1]] if best is not None else []

    def close(self) -> None:
        """Workers and environments first, then a solver rollout that holds something.

        In that order because ``super().close()`` is what waits the jobs out, and
        a rollout must not have its endpoint pulled while a job could still be
        driving an agent against it.
        """
        super().close()
        if self._solver_rollout is not self._rollout and hasattr(self._solver_rollout, 'close'):
            self._solver_rollout.close()
