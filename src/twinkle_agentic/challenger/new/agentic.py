# Copyright (c) ModelScope Contributors. All rights reserved.
"""Agentic challenger: act in a sandbox, verify the result, then describe it."""
import math
import random
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format import SamplingParams, Trajectory, attach_user_data, user_data_get
from twinkle.data_format.sampling import SampledSequence, SampleResponse
from twinkle.utils import get_logger
from twinkle_agentic.envs import Env
from twinkle_agentic.protocol.base import API
from twinkle_agentic.rollout import APISampler, MultiTurnRollout
from twinkle_agentic.summarizer import Summarizer
from twinkle_agentic.utils.code_utils import parse_fenced_code, strip_reasoning
from twinkle_agentic.utils.message_utils import assistant_text, msg_content_text, normalize_tool_calls
from .base import Challenger, _parallel
from .keyword import KeywordGenerator
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
            raise ValueError('use_api=True requires an API backend')
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


class AgenticChallenger(Challenger):
    """Invent tool-using tasks by doing, checking, and describing them.

    ``backend`` drives exploration and solver attempts. When ``use_api`` is true,
    ``api`` generates only the appended check-script and problem-statement turns;
    those turns retain the masking semantics selected by ``api_appended_as`` in
    ``rollout_kwargs``.
    """

    _system = ('You invent tasks for another agent to solve. You have a sandbox and '
               'tools. Work in it first: build something real, then you will be asked '
               'to verify it and to describe it.')
    _from_scratch = ('Choose a task worth doing in this sandbox and do it now, using '
                     'your tools. Do not describe it yet.')
    _from_keywords = ('Choose a task around these topics and do it now, using your '
                      'tools. Do not describe it yet.\n\nTopics: {keywords}')
    _from_seed = ('Here is an earlier task:\n\n{seed}\n\nDo something in the same '
                  'spirit but different, using your tools now. Do not describe it yet.')
    _from_seed_keywords = ('Here is an earlier task:\n\n{seed}\n\nDo something in the same '
                           'spirit but different, may be more complex and interesting and meaningful, '
                           'around these topics, using your tools now. Do not describe it yet.\n\n'
                           'Topics: {keywords}')
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


    def __init__(
        self,
        backend: Any,
        *,
        api: Optional[Any] = None,
        use_api: bool = False,
        keyword_generator: Optional[KeywordGenerator] = None,
        trajectory_seed: Optional[List[Trajectory]] = None,
        summarizer: Optional[Summarizer] = None,
        system_prompt: Optional[str] = None,
        from_scratch_prompt: Optional[str] = None,
        from_keywords_prompt: Optional[str] = None,
        from_seed_prompt: Optional[str] = None,
        from_seed_keywords_prompt: Optional[str] = None,
        check_followup_prompt: Optional[str] = None,
        check_retry_followup_prompt: Optional[str] = None,
        check_parse_error_prompt: Optional[str] = None,
        problem_followup_prompt: Optional[str] = None,
        check_retries: int = 1,
        problem_max_chars: int = 8192,
        check_language: str = 'python',
        parse_check_fn: Optional[Callable[[str], Optional[str]]] = None,
        pass_rate_target: float = 0.2,
        envs: Sequence[Env] = (),
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
        if use_api and rollout_kwargs.get('response_callback') is not None:
            raise ValueError('use_api=True cannot be combined with response_callback')
        backend_is_api = isinstance(backend, (API, APISampler))
        if use_api and api is None and not backend_is_api:
            raise ValueError('use_api=True requires api= when backend is a sampler')
        self.keyword_generator = keyword_generator
        self.trajectory_seed = list(trajectory_seed or ())
        self.summarizer = summarizer
        self._system = self._system if system_prompt is None else system_prompt
        self._from_scratch = self._from_scratch if from_scratch_prompt is None else from_scratch_prompt
        self._from_keywords = self._from_keywords if from_keywords_prompt is None else from_keywords_prompt
        self._from_seed = self._from_seed if from_seed_prompt is None else from_seed_prompt
        self._from_seed_keywords = (self._from_seed_keywords if from_seed_keywords_prompt is None else
                                    from_seed_keywords_prompt)
        self._check_followup = self._check_followup if check_followup_prompt is None else check_followup_prompt
        self._check_retry_followup = (self._check_retry_followup if check_retry_followup_prompt is None else
                                      check_retry_followup_prompt)
        self._check_parse_error = (self._check_parse_error if check_parse_error_prompt is None else
                                   check_parse_error_prompt)
        self._problem_followup = (self._problem_followup if problem_followup_prompt is None else
                                  problem_followup_prompt)
        self._check_retries = check_retries
        self._problem_max_chars = problem_max_chars
        self._check_language = check_language.strip().lower()
        self._parse_check_fn = parse_check_fn
        self._pass_rate_target = pass_rate_target
        self._pass_rate_width = pass_rate_width
        self.checker = checker
        self.followup_params = followup_params
        self.rng = random.Random()
        self.use_api = use_api
        self.save_failed_rollouts = save_failed_rollouts
        self._recorder = RolloutRecorder(save_dir) if save_dir else None
        self._round_proposals: List[_ProposalResult] = []
        self._backend = backend
        self._rollout_kwargs = dict(rollout_kwargs)
        if api is not None:
            self._rollout_kwargs['api'] = api
        if use_api:
            self._rollout_kwargs['response_callback'] = _api_followup_response
        self._rollout: Optional[MultiTurnRollout] = None
        self._tool_schemas = self.env().tools() or None

    def _rollout_instance(self) -> MultiTurnRollout:
        if self._rollout is None:
            self._rollout = MultiTurnRollout(self._backend, **self._rollout_kwargs)
        return self._rollout

    def _tool_manager(self, slot: int) -> Optional[Any]:
        env = self.env(slot)
        return env.tool_manager() if env.tools() else None

    def _summary(self, trajectory: Trajectory) -> str:
        turns: List[str] = []
        for message in trajectory.get('messages') or []:
            if not isinstance(message, dict):
                continue
            role = message.get('role') or ''
            if role == 'system':
                continue
            parts = [msg_content_text(message).strip()]
            for call in normalize_tool_calls(message) or ():
                fn = call.get('function') or {}
                if isinstance(fn, dict) and fn.get('name'):
                    parts.append(f"calls {fn['name']}({fn.get('arguments') or ''})")
            body = '\n'.join(part for part in parts if part)
            if body:
                turns.append(f'{role}: {body}')
        text = '\n'.join(turns)
        if not text:
            return ''
        return self.summarizer(text) if self.summarizer is not None else text

    def _build_challenge_prompt(self) -> Optional[Trajectory]:
        keywords: List[str] = []
        if self.keyword_generator is not None:
            groups = self.keyword_generator.get_keywords(1)
            if not groups:
                return None
            keywords = groups[0]
        seed = ''
        if self.trajectory_seed:
            seed = self._summary(self.rng.choice(self.trajectory_seed))
        block = ', '.join(keywords)
        if seed and keywords:
            user = self._from_seed_keywords.format(seed=seed, keywords=block)
        elif seed:
            user = self._from_seed.format(seed=seed)
        elif keywords:
            user = self._from_keywords.format(keywords=block)
        else:
            user = self._from_scratch
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
        if self._tool_schemas:
            prompt['tools'] = self._tool_schemas
        return attach_user_data(prompt, keywords=keywords, seeded=bool(seed))

    def _explore(self, prompt: Trajectory) -> List[Trajectory]:
        group_id = uuid.uuid4().hex
        proposals: List[_ProposalResult] = []
        remaining = self.num_challenger_rollouts
        while remaining > 0:
            wave = min(self.n_slots, remaining)
            proposals.extend(_parallel(lambda slot: self._run_episode(prompt, slot), wave))
            remaining -= wave
        for proposal in proposals:
            proposal.group_id = group_id
        self._round_proposals = proposals
        return [proposal.task for proposal in proposals if proposal.task is not None]

    def _run_episode(self, prompt: Trajectory, slot: int) -> _ProposalResult:
        self.env(slot).clear()
        state: Dict[str, Any] = {'slot': slot}
        kwargs: Dict[str, Any] = {
            'followup_fn': lambda trajectory, n_before: self._followup(state, trajectory, n_before),
        }
        manager = self._tool_manager(slot)
        if manager is not None:
            kwargs['tool_manager'] = manager
        explored = self._rollout_instance()([prompt], **kwargs)
        if not explored:
            self._reject(state, 'rollout_no_output')
            return _ProposalResult(dict(prompt), reason='rollout_no_output')
        trajectory = explored[0]
        task = self._build_query(state, trajectory)
        reason, detail = state.get('reject', ('', ''))
        return _ProposalResult(trajectory, task=task, reason=reason, detail=detail)

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
        slot = state['slot']
        if reply is None:
            snapshot, error = self.env(slot).snapshot()
            state['snapshot'] = snapshot
            if not snapshot.strip():
                state['reject'] = ('snapshot_unavailable' if error else 'empty_workspace', error)
                return None
            return self._check_followup.format(final_state=snapshot, language=self._check_language)

        attempt = state.get('check_attempts', 0) + 1
        state['check_attempts'] = attempt
        script = (self._parse_check_fn(reply) if self._parse_check_fn is not None else
                  parse_fenced_code(reply, language_tags=None))
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
        exit_code, output = self.env(slot).run_script(script, interpreter=self._check_language)
        if exit_code == 0:
            state['checked'] = True
            return self._problem_followup
        after = self.env(slot).snapshot()[0]
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
            keywords=user_data_get(explored.get('user_data'), 'keywords', []),
            seeded=user_data_get(explored.get('user_data'), 'seeded', False),
        )
        if self.checker is not None and not self.checker(task):
            return self._reject(state, 'rejected_by_checker')
        return task

    def _reject(self, state: Dict[str, Any], reason: str, detail: str = '') -> Optional[Trajectory]:
        state['reject'] = (reason, detail)
        logger.info(f'[{type(self).__name__}] rejected: {reason}'
                    f"{f' -- {detail[:400]}' if detail else ''}")
        return None

    def _solver_prompt(self, task: Trajectory) -> Trajectory:
        prompt: Trajectory = {'messages': [dict(message) for message in task.get('messages') or []]}
        if self._tool_schemas:
            prompt['tools'] = self._tool_schemas
        return prompt

    def _judge(self, task: Trajectory, slot: int) -> bool:
        script = user_data_get(task.get('user_data'), 'check_script', '')
        if not script:
            return False
        return self.env(slot).run_script(script, interpreter=self._check_language)[0] == 0

    def challenger_reward(self, n_pass: Optional[int]) -> float:
        """Reward tasks near the target solver pass rate; unmeasured failures score zero."""
        if n_pass is None or not self.num_solver_rollouts or n_pass <= 0:
            return 0.0
        gap = n_pass / self.num_solver_rollouts - self._pass_rate_target
        variance = 2.0 * self._pass_rate_width**2
        return math.exp(-(gap * gap) / variance)

    def _record_proposals(self) -> None:
        proposals, self._round_proposals = self._round_proposals, []
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
                keywords=user_data_get(proposal.trajectory.get('user_data'), 'keywords', []),
                seeded=user_data_get(proposal.trajectory.get('user_data'), 'seeded', False),
            )

    def _filter_difficulty(self, tasks: List[Trajectory]) -> List[Trajectory]:
        successful = [proposal for proposal in self._round_proposals if proposal.task is not None]
        if len(successful) != len(tasks):
            raise RuntimeError('proposal/task alignment failed before difficulty filtering')
        if not tasks or not self.num_solver_rollouts:
            for proposal in successful:
                proposal.outcome = 'kept'
            self._record_proposals()
            return tasks

        passes = [0] * len(tasks)
        plan = [i for i in range(len(tasks)) for _ in range(self.num_solver_rollouts)]
        rollout = self._rollout_instance()
        for start in range(0, len(plan), self.n_slots):
            wave = plan[start:start + self.n_slots]
            _parallel(lambda slot: self.env(slot).clear(), len(wave))
            prompts = [self._solver_prompt(tasks[i]) for i in wave]
            kwargs: Dict[str, Any] = {}
            managers = [self._tool_manager(slot) for slot in range(len(wave))]
            if any(manager is not None for manager in managers):
                kwargs['tool_manager'] = managers
            attempts = rollout(prompts, **kwargs)
            if len(attempts) != len(prompts):
                raise RuntimeError(f'rollout returned {len(attempts)} attempts for '
                                   f'{len(prompts)} prompts; expected one per prompt')
            verdicts = _parallel(lambda slot: self._judge(tasks[wave[slot]], slot), len(wave))
            for slot, passed in enumerate(verdicts):
                if passed:
                    passes[wave[slot]] += 1

        low, high = self.pass_band
        measured = [
            attach_user_data(task, n_pass=n_pass, n_rollouts=self.num_solver_rollouts)
            for task, n_pass in zip(tasks, passes)
        ]
        kept: List[Trajectory] = []
        for proposal, task, n_pass in zip(successful, measured, passes):
            proposal.task = task
            proposal.n_pass = n_pass
            proposal.reward = self.challenger_reward(n_pass)
            if low <= n_pass <= high:
                proposal.outcome = 'kept'
                kept.append(task)
            else:
                proposal.outcome = 'outside_band'
        self._record_proposals()
        return kept
