# Copyright (c) ModelScope Contributors. All rights reserved.
"""Code challenger: invent Python problems whose ground truth was executed.

The task is built backwards. The model writes a problem *and* a reference
solution; the solution is run to capture what each check expression actually
returns, and those captured values become the asserts. So the answer exists
before the question does, and no external labelling is involved. Two gates carry
over from earlier runs, both from real failures:

* the reference solution must pass its own asserts, or the ground truth is noise;
* output capture uses a sentinel marker plus the exit status, never the last
  stdout line, so a startup banner can never be read as a result.

Prompt text is not here. Every string the model sees arrives in
:class:`CodePrompts`, built by whoever runs the challenger -- see
``cookbook/rsi/code/challenge_prompts.py``. Neither is execution: every script
runs in an :class:`~twinkle_agentic.envs.base.Env`, which is the same interface
the agentic half verifies through, so where a task gets graded is a decision made
once by the caller rather than twice by the two halves. What stays here is the
machinery that cannot be restated in a prompt: the assert capture, the
constant-answer check, and how a proposal becomes a task. Neither is the keyword
bank -- drawing, refilling and expanding topics is the same cycle on both halves,
so it lives once in :mod:`.keywords` and this challenger holds one.
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format import SamplingParams, Trajectory, user_data_get
from twinkle.utils import get_logger
from twinkle_agentic.envs import Env
from twinkle_agentic.utils.code_utils import strip_reasoning, unwrap_code
from twinkle_agentic.utils.message_utils import assistant_text
from .base import Challenger, Explorer, PromptSet, attach_user_data
from .keywords import KeywordBank, KeywordStore

logger = get_logger()

__all__ = [
    'CodeChallenger', 'CodePrompts', 'build_asserts', 'is_constant_answer',
    'load_seeds', 'parse_challenge', 'run_asserts', 'run_check_script',
]

# Isolates a captured value from anything else the script prints.
_MARK = '__RSI_GT__'
_JSON_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.I)


def run_check_script(code: str, check_script: str, env: Env,
                     timeout: int = 30) -> Tuple[bool, str]:
    """Run ``code`` against ``check_script`` in ``env``; True when it exits 0.

    One script, one exit status -- the same judgement the agentic half makes, so
    a task from either half is graded the same way. The run's output comes back
    too: a verdict that only says "wrong" leaves a second attempt nothing to go
    on.
    """
    if not code.strip():
        return False, 'no code was produced'
    if not check_script.strip():
        return False, 'no check script was produced'
    rc, out = env.run_script(f'{code}\n\n{check_script}', timeout=timeout)
    return rc == 0, out


def run_asserts(code: str, setup: str, asserts: List[str], env: Env,
                timeout: int = 30) -> bool:
    """True when every assert passes (exit status 0).

    For callers holding a list of asserts rather than one check script -- a
    tests file, say. The list plus the setup *is* the check script.
    """
    parts = [setup] if (setup or '').strip() else []
    parts.extend(asserts or ())
    return run_check_script(code, '\n\n'.join(parts), env, timeout)[0]


def build_asserts(solution: str, checks: List[str], env: Env, timeout: int = 30,
                  max_checks: int = 6) -> Optional[List[str]]:
    """Run the reference solution once to capture each check's repr, then form
    ``assert <check> == <captured>``.

    Returns None if the solution crashed or produced no usable output -- the
    caller drops that problem. The marker plus the exit status is what makes the
    capture trustworthy: a crash or a banner line can never become a value.
    """
    checks = [c for c in checks if isinstance(c, str) and c.strip()][:max_checks]
    if not checks:
        return None
    lines = [solution, '']
    for i, c in enumerate(checks):
        # repr on its own line, tagged with index; a check that raises makes the
        # whole script exit non-zero -> we drop the problem. Pure f-string (no %%
        # formatting) so a check expression containing '%' (modulo/percent) is safe.
        lines.append(f'print("{_MARK}{i}=" + repr({c}))')
    rc, out = env.run_script('\n'.join(lines), timeout=timeout)
    if rc != 0:
        return None
    captured: Dict[int, str] = {}
    for line in out.splitlines():
        if line.startswith(_MARK):
            try:
                idx_str, val = line[len(_MARK):].split('=', 1)
                idx = int(idx_str)
            except (ValueError, IndexError):
                continue
            if idx in captured:
                # Two lines claiming the same check. The script prints each one
                # exactly once, so a second one came from the solution itself --
                # stderr is part of the output now, and a solution that can
                # redefine what its own check returned is not ground truth.
                return None
            captured[idx] = val
    if len(captured) != len(checks):
        return None
    # The captured text is a repr, so it is a valid literal to compare against.
    return [f'assert ({c}) == ({captured[i]})' for i, c in enumerate(checks)]


def _split_top_eq(s: str) -> Optional[tuple]:
    """Split on the first top-level ``==``, ignoring anything inside brackets or quotes."""
    depth = 0
    quote = ''
    i = 0
    while i < len(s) - 1:
        c = s[i]
        if quote:
            if c == quote:
                quote = ''
        elif c in '\'"':
            quote = c
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif depth == 0 and c == '=' and s[i + 1] == '=':
            return s[:i].strip(), s[i + 2:].strip()
        i += 1
    return None


def _expected_of(assert_line: str) -> Optional[str]:
    """The value the solver actually has to produce for one assert.

    :func:`build_asserts` emits ``assert (<check>) == (<repr>)``, but a check may
    itself be a comparison, giving ``assert (f(x) == 3) == (True)``. Reading the
    outer side there would report 'True' and make such a problem look
    constant-answer, so the inner right-hand side is used instead. An outer
    ``False`` pins nothing down at all and is reported as unknown.
    """
    m = re.match(r'^\s*assert\s*\((.*)\)\s*==\s*\((.*)\)\s*$', assert_line.strip())
    if not m:
        return None
    lhs, rhs = m.group(1).strip(), m.group(2).strip()
    inner = _split_top_eq(lhs)
    if rhs in ('True', 'False') and inner is not None:
        return inner[1] if rhs == 'True' else None
    return rhs


def is_constant_answer(asserts: List[str]) -> bool:
    """Would ``return <one constant>`` satisfy every assert?

    Such a problem pays full reward for ignoring its own statement, so it
    actively teaches the solver not to read the input. Requires at least two
    asserts with a readable expectation: a single assert is trivially
    'constant', and one unreadable assert must not hide a constant set.
    """
    vals = [_expected_of(a) for a in asserts]
    if any(v is None for v in vals) or len(vals) < 2:
        return False
    return len(set(vals)) == 1


# ── parsing ────────────────────────────────────────────────────────────────
def parse_challenge(text: str, require_solution: bool = True) -> Optional[Dict[str, Any]]:
    """Pull the ``{problem, solution, entry, checks}`` object out of a completion.

    ``require_solution=False`` is for the two-step flow, whose second call is
    told the solution is already known and returns only the statement.
    """
    body = strip_reasoning(text)
    body = _JSON_FENCE_RE.sub('', body.strip()).strip()
    # Grab the outermost {...} if there is leading/trailing prose.
    start, end = body.find('{'), body.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(body[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    problem, solution, checks = obj.get('problem'), obj.get('solution'), obj.get('checks')
    if not (isinstance(problem, str) and problem.strip()
            and isinstance(checks, list) and checks):
        return None
    if require_solution:
        if not (isinstance(solution, str) and solution.strip()):
            return None
    else:
        # Told not to include a solution; if it did anyway, ignore it -- the
        # caller overwrites with the code that actually ran.
        solution = solution if isinstance(solution, str) else ''
    if solution and '```' in solution:
        solution = unwrap_code(solution)
    return {'problem': problem.strip(), 'solution': (solution or '').strip(),
            'entry': str(obj.get('entry') or '').strip(), 'checks': checks}


def load_seeds(path: str) -> List[Dict[str, str]]:
    """Read seed problems from a jsonl: dicts with ``query`` and maybe ``code``.

    A seed without ``code`` cannot take the two-step path (there is no reference
    solution to build on top of) and falls back to the single-call prompt.
    """
    if not path or not os.path.exists(path):
        return []
    seeds: List[Dict[str, str]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            q = row.get('query') or row.get('problem') or row.get('prompt')
            if isinstance(q, dict):
                q = q.get('content')
            if not q:
                msgs = row.get('messages') or []
                q = next((m.get('content') for m in msgs if m.get('role') == 'user'), None)
            if isinstance(q, str) and q.strip():
                seeds.append({'query': q.strip(), 'code': (row.get('code') or '').strip()})
    return seeds


# ── prompts (text supplied by the caller) ──────────────────────────────────
@dataclass
class CodePrompts(PromptSet):
    """Every string a :class:`CodeChallenger` sends, and nothing else.

    Deliberately without defaults for the always-needed fields: a prompt is the
    experiment, so a run has to state which one it used rather than inherit a
    library's idea of it. Optional groups stay empty until the feature that
    needs them is switched on, and the constructor says so if one is missing.

    Placeholders are checked at construction: a typo'd ``{keywords}`` would
    otherwise surface as a KeyError halfway through a generation run. That
    checking, and the keyword subset a bank is given, are :class:`.PromptSet`.
    """

    system: str
    from_scratch: str
    solver_system: str
    solver_user: str
    from_seed: str = ''
    from_keywords: str = ''
    from_seed_keywords: str = ''
    two_step_system: str = ''
    two_step_solution: str = ''
    two_step_problem: str = ''
    keyword_system: str = ''
    keyword_user: str = ''
    keyword_expand_user: str = ''

    #: fields that must carry text.
    _REQUIRED = ('system', 'from_scratch', 'solver_system', 'solver_user')
    #: field -> placeholders it must contain.
    _REQUIRED_FIELDS = {
        'solver_user': ('problem', ),
        'from_seed': ('seed', ),
        'from_keywords': ('keywords', ),
        'from_seed_keywords': ('seed', 'keywords'),
        'two_step_solution': ('seed', 'code', 'keywords'),
        'two_step_problem': ('code', 'seed', 'keywords'),
        'keyword_user': ('k', 'desc'),
        'keyword_expand_user': ('kw', 'm'),
    }


class CodeChallenger(Challenger):
    """Propose code problems, execute them for ground truth, keep the graded ones.

    One class rather than several because 'from scratch', 'from a seed problem',
    'from keywords' and the two-step build differ only in which prompt the
    proposal carries: parsing, execution, the self-check and the difficulty
    band are the same afterwards. Which path a proposal takes is decided per
    proposal, so one run mixes them.

    Args:
        prompts: every string sent to the model.
        explorer: batch-in / batch-out generation, see :class:`.base.Explorer`.
        seeds: optional pool from :func:`load_seeds`, drawn with replacement.
        keyword_store: optional bank; without it proposals carry no topics.
        category_desc / combo_arity / arity_weights / single_kw_prob /
        keyword_refill_target / keyword_gen_calls / keyword_refill_tries /
        keyword_params / min_batch / expand_per_kw / expand_max_kws: handed to the
            :class:`.keywords.KeywordBank` this challenger holds, which is where
            they are documented -- they behave the same on the agentic half.
        seed_mix_prob: chance a proposal also carries a seed problem, when a
            pool was given.
        two_step: allow the two-call path (write a harder solution on top of the
            seed's reference code, then describe the problem it answers). Needs
            a seed carrying ``code`` and at least one keyword, so it is skipped
            silently for proposals that have neither.
        problem_max_chars: reject statements longer than this. Rambling
            non-problems, and they would also crowd out the solver's context.
        max_checks / sandbox_timeout: passed to :func:`build_asserts`.
        drop_constant_answer: reject problems where one constant satisfies every
            assert.
        low_pass_expand: a candidate solved this many times or fewer counts as
            hard, and its topics are fed back through
            :meth:`expand_hard_keywords`.
        reject_sink: called with a dict for every rejected proposal. The caller
            decides whether that goes to a file; nothing here writes one.
        solver_sink: called once per solver attempt in the difficulty stage, with
            the check script, the attempt and the verdict. Two things need it:
            ``n_pass=0`` reads the same whether the problem is impossible or the
            statement withholds a value its asserts demand, and the attempts are
            the trainable half of this challenger's output -- see
            :meth:`judge_attempt`. Requires a local sampler; an API explorer
            returns text without token fields.
        keyword_sink: called once per keyword-generation call, with the prompt,
            the reply and both halves of the parse.
    """

    def __init__(
        self,
        prompts: CodePrompts,
        explorer: Explorer,
        *,
        seeds: Sequence[Dict[str, str]] = (),
        keyword_store: Optional[KeywordStore] = None,
        category_desc: Optional[Dict[str, str]] = None,
        seed_mix_prob: float = 0.5,
        two_step: bool = True,
        combo_arity: str = 'triple',
        arity_weights: Optional[Sequence[float]] = None,
        single_kw_prob: float = 0.1,
        keyword_refill_target: int = 128,
        keyword_gen_calls: int = 8,
        keyword_refill_tries: int = 2,
        keyword_params: Optional[SamplingParams] = None,
        min_batch: int = 1,
        problem_max_chars: int = 4000,
        max_checks: int = 6,
        sandbox_timeout: int = 30,
        drop_constant_answer: bool = True,
        low_pass_expand: int = 0,
        expand_per_kw: int = 8,
        expand_max_kws: int = 32,
        reject_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        solver_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        keyword_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        **challenger_kwargs: Any,
    ):
        super().__init__(explorer, system=prompts.system, **challenger_kwargs)
        if not self.envs:
            raise ValueError('envs is empty: there is nowhere to run a check, and every '
                             'proposal would be rejected for a ground truth that never ran.')
        if keyword_store is not None:
            prompts.require('from_keywords')
        self.prompts = prompts
        self.seeds = list(seeds)
        # The whole keyword cycle -- draw, refill, expand -- is one object shared
        # with the agentic challenger rather than a second copy of it here. None
        # means no bank was configured, and proposals then carry no topics.
        self.keywords: Optional[KeywordBank] = None if keyword_store is None else KeywordBank(
            keyword_store, prompts=prompts.keyword_prompts(),
            category_desc=category_desc or {},
            explorer=explorer, rng=self.rng, name=type(self).__name__,
            sampling_params=keyword_params, sink=keyword_sink, combo_arity=combo_arity,
            arity_weights=arity_weights, single_kw_prob=single_kw_prob,
            refill_target=keyword_refill_target, gen_calls=keyword_gen_calls,
            refill_tries=keyword_refill_tries, min_batch=min_batch,
            expand_per_kw=expand_per_kw, expand_max_kws=expand_max_kws)
        self.seed_mix_prob = seed_mix_prob
        self.two_step = two_step
        self.problem_max_chars = problem_max_chars
        self.max_checks = max_checks
        self.sandbox_timeout = sandbox_timeout
        self.drop_constant_answer = drop_constant_answer
        self.low_pass_expand = low_pass_expand
        self.reject_sink = reject_sink
        self.solver_sink = solver_sink
        if self.seeds:
            # Both are reachable with a bank configured: a proposal draws no
            # keywords when every category is dry, and then falls back to the
            # seed-only prompt.
            prompts.require('from_seed')
            if self.keywords is not None:
                prompts.require('from_seed_keywords')
        if two_step:
            prompts.require('two_step_system', 'two_step_solution', 'two_step_problem')
        # Why proposals died, for the caller to log; the shape a run is judged on.
        self.stats: Dict[str, int] = {
            'parsed': 0, 'parse_fail': 0, 'stage1_no_code': 0, 'too_long': 0,
            'gt_fail': 0, 'selfcheck_fail': 0, 'constant_answer': 0,
        }

    # ------------------------------------------------------------- proposing

    def propose(self, count: int) -> List[Trajectory]:
        proposals: List[Trajectory] = []
        for _ in range(count):
            picks = self.keywords.draw() if self.keywords else []
            body = KeywordBank.block(picks)
            use_seed = bool(self.seeds) and self.rng.random() < self.seed_mix_prob
            seed = self.rng.choice(self.seeds) if use_seed else None
            two = bool(use_seed and self.two_step and picks and seed and seed.get('code'))
            if two:
                system = self.prompts.two_step_system
                user = self.prompts.two_step_solution.format(
                    seed=seed['query'], code=seed['code'], keywords=body)
            elif use_seed and picks:
                system = self.prompts.system
                user = self.prompts.from_seed_keywords.format(seed=seed['query'], keywords=body)
            elif use_seed:
                system = self.prompts.system
                user = self.prompts.from_seed.format(seed=seed['query'])
            elif picks:
                system = self.prompts.system
                user = self.prompts.from_keywords.format(keywords=body)
            else:
                system = self.prompts.system
                user = self.prompts.from_scratch
            proposal: Trajectory = {
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': user}],
            }
            # Carried through the explorer so build() knows which path this
            # proposal took and what the second call has to be told.
            proposals.append(attach_user_data(
                proposal, keywords=picks, seeded=use_seed, two_step=two,
                seed_query=(seed['query'] if two else ''), keyword_block=body))
        return proposals

    # ---------------------------------------------------------------- building

    def build(self, explored: List[Trajectory]) -> List[Optional[Trajectory]]:
        """Parse, execute, self-check; None for every proposal that did not survive.

        The second call of the two-step path happens here rather than in
        :meth:`propose`, because it needs the code the first call produced. It
        goes out as one batch for the whole round, so the extra call costs one
        more generate, not one per proposal.
        """
        objs: List[Optional[Dict[str, Any]]] = [None] * len(explored)
        # Proposals that died before parsing: they must not also be counted as a
        # parse failure, because the cause -- and the fix -- is a different one.
        dead: List[bool] = [False] * len(explored)
        stage2_idx: List[int] = []
        stage2_prompts: List[Trajectory] = []
        for i, traj in enumerate(explored):
            text = assistant_text(traj)
            if not user_data_get(traj.get('user_data'), 'two_step', False):
                objs[i] = parse_challenge(text)
                continue
            code = unwrap_code(text)
            if not code.strip():
                # Usually a truncated completion: there is no solution to
                # describe, so this proposal ends here.
                self.stats['stage1_no_code'] += 1
                dead[i] = True
                continue
            objs[i] = {'_stage1_code': code}
            stage2_idx.append(i)
            stage2_prompts.append({
                'messages': [
                    {'role': 'system', 'content': self.prompts.system},
                    {'role': 'user', 'content': self.prompts.two_step_problem.format(
                        code=code,
                        seed=user_data_get(explored[i].get('user_data'), 'seed_query', ''),
                        keywords=user_data_get(explored[i].get('user_data'),
                                               'keyword_block', ''))},
                ],
            })
        if stage2_prompts:
            logger.info(f'[CodeChallenger] two-step stage 2: {len(stage2_prompts)} problem '
                        f'writes ({self.stats["stage1_no_code"]} first calls had no code)')
            for i, reply in zip(stage2_idx, self.explore(stage2_prompts)):
                stage1_code = objs[i]['_stage1_code']
                obj = parse_challenge(assistant_text(reply), require_solution=False)
                if obj is not None:
                    # Ground truth is the code that actually ran, never the one
                    # the second call may have re-imagined.
                    obj['solution'] = stage1_code
                objs[i] = obj

        return [None if dead[i] else self._finish(explored[i], obj)
                for i, obj in enumerate(objs)]

    def _finish(self, proposal: Trajectory, obj: Optional[Dict[str, Any]]) -> Optional[Trajectory]:
        """One parsed proposal -> a task, or None with a reason recorded."""
        if obj is None:
            self.stats['parse_fail'] += 1
            return None
        self.stats['parsed'] += 1

        def _reject(reason: str, **extra: Any) -> None:
            self.stats[reason] += 1
            if self.reject_sink is not None:
                self.reject_sink({'reason': reason, **extra, **obj})

        if len(obj['problem']) > self.problem_max_chars:
            _reject('too_long')
            return None
        asserts = build_asserts(obj['solution'], obj['checks'], self.env(),
                                timeout=self.sandbox_timeout, max_checks=self.max_checks)
        if not asserts:
            _reject('gt_fail')
            return None
        check_script = '\n'.join(asserts)
        if not self._check(obj['solution'], check_script)[0]:
            # A reference solution that fails its own asserts is not ground
            # truth, whatever the statement says.
            _reject('selfcheck_fail', asserts=asserts)
            return None
        if self.drop_constant_answer and is_constant_answer(asserts):
            _reject('constant_answer', asserts=asserts)
            return None

        user_data = proposal.get('user_data')
        # The task the solver is trained on: the statement alone, exactly as the
        # difficulty stage will present it, with no instructions from the
        # challenger's own prompt leaking in.
        task: Trajectory = {
            'messages': [{'role': 'system', 'content': self.prompts.solver_system},
                         {'role': 'user', 'content': obj['problem']}],
        }
        return attach_user_data(
            task,
            # One script rather than a list of asserts, named as the agentic half
            # names it: a consumer that trains on both halves then reads the
            # verifier the same way. setup_script is where that half puts the
            # part that runs before the checks; nothing here needs one.
            check_script=check_script,
            setup_script='',
            solution=obj['solution'],
            entry=obj['entry'],
            keywords=user_data_get(user_data, 'keywords', []),
            seeded=user_data_get(user_data, 'seeded', False),
            two_step=user_data_get(user_data, 'two_step', False))

    # -------------------------------------------------------------- difficulty

    def solver_prompt(self, task: Trajectory) -> Trajectory:
        problem = next((m['content'] for m in reversed(task.get('messages') or [])
                        if m.get('role') == 'user'), '')
        return {
            'messages': [{'role': 'system', 'content': self.prompts.solver_system},
                         {'role': 'user',
                          'content': self.prompts.solver_user.format(problem=problem)}],
        }

    def _check(self, code: str, check_script: str) -> Tuple[bool, str]:
        """Did ``code`` pass ``check_script``? With the output, for feedback.

        Slot 0 always: a code judgement is one script with no state to share, and
        this half runs them one at a time -- see the note in
        :meth:`judge_attempt` -- so the slots the agentic half needs for its
        concurrent episodes have nothing to do here.
        """
        return run_check_script(code, check_script, self.env(), self.sandbox_timeout)

    def judge_attempt(self, task: Trajectory, attempt: Trajectory) -> bool:
        """Did this attempt's code pass the task's asserts?

        Also hands the whole attempt to ``solver_sink`` when one is given. The
        stage otherwise reduces each task to one number and drops the attempts,
        and they are exactly what a solver trains on: a task kept at 3 of 8 is
        one prompt answered eight times with a binary reward, which is a GRPO
        group already measured to have a gradient. Sampling them again after the
        band has been applied pays for the same tokens twice and can still land
        the group at 0 or 8, where the advantage is the reward minus itself.
        """
        check_script = user_data_get(task.get('user_data'), 'check_script', '') or ''
        passed, output = self._check(unwrap_code(assistant_text(attempt)), check_script)
        if self.solver_sink is not None:
            # No lock: the difficulty stage judges attempts one at a time, in the
            # loop that counts them, unlike the agentic half where each judgement
            # is a sandbox round trip worth running concurrently.
            self.solver_sink({
                'statement': next((m.get('content', '') for m in task.get('messages') or []
                                   if m.get('role') == 'user'), ''),
                # What the caller groups on: two problems with byte-identical
                # asserts are the same problem, and a kept task carries this
                # field through unchanged.
                'check_script': check_script,
                'passed': passed,
                'output': output,
                'truncated': bool((attempt or {}).get('truncated')),
                'attempt': attempt,
            })
        return passed

    def on_difficulty_measured(self, candidates: List[Trajectory]) -> None:
        """Remember the topics behind the candidates nobody solved."""
        if self.keywords is not None:
            self.keywords.remember_unsolved(candidates, self.low_pass_expand)

    # ------------------------------------------------------------- feedback

    def expand_hard_keywords(self) -> int:
        """Brainstorm more topics in the families that produced the hardest tasks.

        Called by whoever drives the challenger, after generating, so the bank
        drifts toward material the solver actually struggles with. Returns how
        many new keywords were added.
        """
        return self.keywords.expand_hard() if self.keywords is not None else 0
