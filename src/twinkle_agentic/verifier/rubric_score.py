# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rubric scores for a proposed task: how new it is, what it is worth, how hard it is.

:mod:`result_check` scores what a solver *did*, with ordinary programs, which is why it
is stable. This module scores the *task itself*, which no program can read: whether a
statement asks for something the pool does not already contain, whether the thing it
asks for resembles work anyone does, and how much reasoning it takes.

The shape is taken from the rubric verifier this repo used before (deleted in 5175833;
readable at ``git show 5175833^:src/twinkle_agentic/verifier/rubric_verifier.py``), for
the reason that made it work there: **the judge never emits a score.** It emits PASS or
FAIL per criterion and the number is computed here. Asking a model for "3 out of 4"
spends most of its resolution on distinctions it cannot make twice in a row; asking
"does this task need more than one command" is a question it answers the same way on a
re-run. A dimension's value is therefore a weighted pass fraction over 3 binary
judgements, not a level the judge chose.

Four more things carried over from that file, with its constants:

* ``[Hard Rule]`` criteria weigh 3, ``[Principle]`` 1 (``hard_weight=3.0``,
  ``principle_weight=1.0``). A hard rule fails unless unambiguously satisfied.
* One vote is normally enough. A second and third are spent only when the first is
  undecided -- when a dimension lands within ``margin`` of the middle -- so cost tracks
  difficulty rather than volume (``margin_threshold=0.25``).
* Criteria are fixed and generic here, naming no file, value or domain from the task
  being judged. Letting a model invent the criteria per task was named in
  ``rubric_library.py`` as the main source of score jitter.
* Anything a program can decide does not go to the judge. Whether the statement quotes
  the values the check compares against is already computed by
  ``derived_check_literals`` (challenger/agentic.py) and is deliberately NOT a criterion
  below, so the two never disagree.

Those four constants are inherited, not re-measured for this use. What has to be
measured before any number here is used: how often a re-run flips a criterion.

What this is NOT: part of the reward. ``AgenticChallenger.challenger_reward`` is the
pass-rate term alone. Ornith-1.5 multiplies its difficulty term by a novelty term
(``R = V x D x N``, ornith.ai/ornith_1_5.html) and ``novelty`` below is the obvious
candidate, but wiring it in needs one more fact first: GRPO subtracts the group mean
(``GRPOAdvantage(scale='group')``), so a term that is near-constant across the eight
proposals sharing a keyword direction contributes no gradient however sensible it looks
per task.
"""
import os
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    'Criterion',
    'RubricResult',
    'CRITERIA',
    'DIMENSIONS',
    'build_rubric_prompt',
    'parse_verdicts',
    'score_task',
    'score_tasks',
]

# Inherited from the deleted rubric_verifier.py (ARROW's 3 / 1) -- not re-derived here.
HARD_WEIGHT = 3.0
PRINCIPLE_WEIGHT = 1.0
# The old verifier escalated an undecided result to 3 votes. Measured on 188 tasks from
# run_clean9 that buys nothing here: tasks whose first pass was decisive repeated to
# within 0.043, tasks that spent all 3 votes to within 0.051 -- no better, at 3x the
# calls (92 of 188 tasks escalated). So one vote, and the spread is reported rather
# than voted away. Raise MAX_VOTES to bring the escalation back; the threshold still
# controls when it triggers.
MARGIN_THRESHOLD = 0.25
MAX_VOTES = 1


@dataclass
class Criterion:
    """One yes/no question about the task.

    Args:
        dimension: which score it contributes to.
        text: the question, phrased so that PASS is the good direction. A criterion
            whose PASS means "this task is bad" inverts the aggregate silently.
        is_hard: objectively checkable from the statement and check as written, so a
            FAIL is not a matter of taste. Weighed ``HARD_WEIGHT``.
        needs_references: skipped when no comparison set was supplied.
    """
    dimension: str
    text: str
    is_hard: bool
    needs_references: bool = False


# Every criterion below is phrased so PASS is the good direction, and every one was
# either kept or replaced on evidence from a first run over run_clean9's 188 tasks
# (.tmp_analysis/rubric_run_clean9.json, kept as rubric_v1.json). What that run showed:
#
#   * The first two novelty criteria agreed on 188 of 188 tasks -- one of them was
#     free. Both are gone, replaced by three that ask about different things: the
#     shape of the task, the machinery it needs, and the form of its end state. Those
#     are the three axes a labelling pass over the same pool found the collapse in
#     (54% of tasks were 'write a script that simulates a process').
#   * The judge was deciding novelty by DOMAIN, not by what the task does: a task whose
#     skeleton was identical to the ones it scored 0.0 got 1.0 because it was about
#     PCIe rather than about log files. The shapes are therefore enumerated, and the
#     criterion says outright that a different domain is not a different task.
#   * 'Reaching the end state takes more than one command' passed 80% of the time and
#     passed on 16 tasks that all eight solvers then solved. Replaced by whether the
#     obvious untested attempt fails, which is what 'hard' has to mean here.
#   * The two soft usefulness criteria barely moved the dimension (it tracked its hard
#     criterion: 0.22 mean when that failed, 0.93 when it passed), so both were
#     replaced. One of the replacements -- whether the input data looks like a real
#     sample -- then passed 7% of the time, i.e. decided nothing, and what it was
#     reaching for is countable without a model anyway: 30% of these statements paste
#     .py source in as an "input file", and those tasks are the easy ones (n_pass 5.7
#     vs 4.4). That belongs in a regex, not in a rubric, so the criterion now asks the
#     part a regex cannot: whether the statement dictates the code to write.
CRITERIA: List[Criterion] = [
    # -- novelty: three independent axes, judged only against the reference set ----
    Criterion(
        'novelty',
        'This task has a different SHAPE from every reference task. Shapes: (a) write '
        'given input files verbatim, then produce a derived file from them; (b) write '
        'a script that demonstrates a defect and a second that fixes it; (c) build a '
        'database or structured store and populate it; (d) run something and report '
        'timings or counts; (e) parse a log or config and summarise it; (f) anything '
        'not in this list. Two tasks of the same shape are the same task here EVEN IF '
        'they are about different subject matter -- a different domain, file format or '
        'vocabulary does not make a different shape',
        is_hard=True, needs_references=True),
    Criterion(
        'novelty',
        'Solving this needs machinery that no reference task needs -- a different one '
        'of: plain text handling, tabular data, binary formats, a database, threads or '
        'processes, subprocesses, sockets, the filesystem layout itself, timing',
        is_hard=False, needs_references=True),
    Criterion(
        'novelty',
        'The FORM of the end state differs from every reference: one text file, several '
        'files, a database file, a program that must run correctly, or a directory tree',
        is_hard=False, needs_references=True),
    # -- usefulness: the hard criterion kept as-is, it separated 29 from 159 and the
    #    calls held up on inspection.
    Criterion(
        'usefulness',
        'The end state is something a person would want for its own sake, not only as '
        'an exercise',
        is_hard=True),
    Criterion(
        'usefulness',
        'The statement says what the end state must be and leaves how to reach it to '
        'the solver, rather than dictating the code or commands to write',
        is_hard=False),
    Criterion(
        'usefulness',
        'The task would still be worth doing if the input were a thousand times larger',
        is_hard=False),
    # -- complexity: the hard criterion asks for a countable property of the task.
    #    Asking instead whether 'the obvious untested attempt would fail' made the
    #    judge guess at a counterfactual and it flipped on 13% of re-runs -- the worst
    #    of the nine, and it carries weight 3.
    Criterion(
        'complexity',
        'Reaching the end state takes at least three steps that depend on each other, '
        'where a later step needs the result of an earlier one',
        is_hard=True),
    Criterion(
        'complexity',
        'Reaching a passing state means choosing between at least two plausible '
        'approaches, of which at least one does not work',
        is_hard=False),
    Criterion(
        'complexity',
        'Passing requires computing something: writing the expected output as a '
        'literal would not satisfy the check',
        is_hard=False),
]

DIMENSIONS = ('novelty', 'usefulness', 'complexity')


@dataclass
class RubricResult:
    """One task's scores plus the verdicts they were computed from.

    ``scores[dim]`` is the weighted PASS fraction in [0, 1], or ``None`` when the
    dimension was not judged -- an unparseable reply, or novelty with no references.
    ``None`` rather than 0.0 so an unjudged task drops out of a mean instead of
    dragging it down.
    """
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    verdicts: List[Optional[bool]] = field(default_factory=list)
    pass_rates: List[Optional[float]] = field(default_factory=list)
    n_votes: int = 0
    raw: List[str] = field(default_factory=list)
    error: str = ''

    @property
    def ok(self) -> bool:
        return not self.error and any(v is not None for v in self.scores.values())

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {k: self.scores.get(k) for k in DIMENSIONS}
        out['n_votes'] = self.n_votes
        out['verdicts'] = list(self.verdicts)
        if self.error:
            out['error'] = self.error
        return out


# Criterion 1's shape list stays INSIDE the criterion. Moving it to its own prompt
# section, so the criterion read 'the shapes listed below', made the judge less stable
# rather than more: verdict flips between two runs went 5% -> 9% and novelty's run-to-run
# spread 0.041 -> 0.100 over the same 60 tasks. What actually stopped the judge from
# answering criterion 1 with a shape name ('1: f', which cost 2 of 188 tasks their
# novelty score) is the paragraph below forbidding it.
_SYSTEM = (
    'You judge a programming task that was generated automatically, before it is used '
    'to train a model.\n\n'
    'The task has two parts. The STATEMENT is everything a solver sees: it starts in an '
    'empty directory, cannot ask questions, and never sees the check. The CHECK is a '
    'python script run against the solver\'s directory afterwards, where exit 0 means '
    'passed. The check is shown to you because it is what the task really demands, '
    'which the statement can understate.\n\n'
    'For each numbered criterion output one line:\n\n'
    '    <index>: PASS   or   <index>: FAIL\n\n'
    'PASS and FAIL are the only two words you may write after the index. Some criteria '
    'list categories to compare by; those are there to define the question, never to be '
    'answered with -- naming a category instead of a verdict makes the line unusable.\n\n'
    'Judge every criterion independently and literally, against this task only. A '
    '[Hard Rule] is FAIL unless it is unambiguously satisfied. Do not explain, do not '
    'restate the criterion, output only the verdict lines in order and then stop.\n')


def _applicable(references: Sequence[str],
                criteria: Sequence[Criterion] = CRITERIA) -> List[Criterion]:
    return [c for c in criteria if references or not c.needs_references]


def build_rubric_prompt(
    statement: str,
    check: str = '',
    references: Sequence[str] = (),
    criteria: Sequence[Criterion] = CRITERIA,
    reference_chars: int = 600,
) -> List[Dict[str, str]]:
    """The messages sent to the judge, and the criterion order the reply must follow.

    All three dimensions go in one call: the judge reads the task once, and nine
    yes/no lines cost about what one dimension would. The cost is that one dimension
    can colour another -- if the scores turn out to move together, splitting into one
    call per dimension is the fix, and the correlation is measurable from the dumps.

    References are cut to ``reference_chars`` each. What a task asks for is in its
    first paragraph; sending statements whole would spend the context on input data
    quoted verbatim, which is the bulk of a statement here.
    """
    items = _applicable(references, criteria)
    lines = [f'{i + 1}. {c.text} [{"Hard Rule" if c.is_hard else "Principle"}]'
             for i, c in enumerate(items)]
    parts = ['## Criteria\n' + '\n'.join(lines) + '\n']
    if references:
        parts.append('\n## Reference tasks (for the novelty criteria only)\n')
        for i, ref in enumerate(references):
            parts.append(f'[{i}] {(ref or "")[:reference_chars]}\n')
    parts.append('\n## Statement\n' + (statement or '') + '\n')
    if check:
        parts.append('\n## Check\n' + check + '\n')
    parts.append(f'\nNow output {len(items)} verdict lines, in order.')
    return [{'role': 'system', 'content': _SYSTEM},
            {'role': 'user', 'content': ''.join(parts)}]


# Same tolerant form the previous verifier parsed, so a reply written as '1) yes' or
# '1. FAIL' is read rather than thrown away.
_VERDICT_RE = re.compile(r'^\s*(\d+)\s*[:.)]\s*(pass|fail|true|false|yes|no|1|0)\b',
                         re.IGNORECASE)
_TRUE = {'pass', 'true', 'yes', '1'}


def parse_verdicts(raw: str, n: int) -> List[Optional[bool]]:
    """Read ``n`` PASS/FAIL verdicts. A line that is missing stays ``None``.

    Indexed by the number the judge wrote rather than by position, because a reply
    that skips a criterion would otherwise shift every later verdict onto the wrong
    question -- and the scores would still come out as numbers.
    """
    out: List[Optional[bool]] = [None] * n
    for line in (raw or '').splitlines():
        match = _VERDICT_RE.match(line)
        if not match:
            continue
        idx = int(match.group(1)) - 1
        if 0 <= idx < n:
            out[idx] = match.group(2).lower() in _TRUE
    return out


def _aggregate(items: Sequence[Criterion],
               rates: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Weighted PASS fraction per dimension; ``None`` when nothing was judged."""
    totals: Dict[str, List[float]] = {}
    for crit, rate in zip(items, rates):
        if rate is None:
            continue
        weight = HARD_WEIGHT if crit.is_hard else PRINCIPLE_WEIGHT
        got, tot = totals.setdefault(crit.dimension, [0.0, 0.0])
        totals[crit.dimension] = [got + weight * rate, tot + weight]
    return {dim: (totals[dim][0] / totals[dim][1] if dim in totals else None)
            for dim in DIMENSIONS}


def _undecided(scores: Dict[str, Optional[float]], margin: float) -> bool:
    """Is any dimension close enough to the middle that another vote could move it?"""
    return any(v is not None and margin < v < 1.0 - margin for v in scores.values())


_client = None
_client_lock = threading.Lock()


def _get_client(model: Optional[str] = None):
    """The judge API, from the same environment variables llm_backup.py reads.

    Default model is ``qwen3.8-max``, the same one that writes the check scripts and
    problem statements, so a task is judged by the model that phrased it.

    Note for comparing numbers: every rubric measurement on file -- the criterion
    flip rates, the per-dimension spreads, the 4-in-940 rate of replies with no
    usable verdict -- was taken with ``qwen3-max``, which was the default until now.
    Those are not a baseline for this judge.
    """
    global _client
    if model is None and _client is not None:
        return _client
    from twinkle_agentic.protocol.openai import OpenAI
    client = OpenAI(
        model=model or os.environ.get('RUBRIC_MODEL')
        or os.environ.get('LLM_BACKUP_MODEL', 'qwen3.8-max'),
        api_key=os.environ.get('LLM_BACKUP_API_KEY'),
        base_url=os.environ.get('LLM_BACKUP_BASE_URL'),
        client_kwargs={'timeout': float(os.environ.get('LLM_BACKUP_TIMEOUT', '120')),
                       'max_retries': int(os.environ.get('LLM_BACKUP_MAX_RETRIES', '2'))},
    )
    if model is None:
        with _client_lock:
            _client = client
    return client


def score_task(
    statement: str,
    check: str = '',
    references: Sequence[str] = (),
    *,
    criteria: Sequence[Criterion] = CRITERIA,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    margin: float = MARGIN_THRESHOLD,
    max_votes: int = MAX_VOTES,
    extra_body: Optional[Dict[str, Any]] = None,
    client: Any = None,
) -> RubricResult:
    """Score one task. Never raises: an API failure comes back in ``error``.

    Votes past the first are spent only on an undecided result, and they are sampled
    (temperature 1.0) whatever ``temperature`` says -- repeating a temperature-0 call
    would mostly repeat its answer, which reads as agreement without being any.
    ``max_tokens`` is small because the reply is nine short lines; a judge that starts
    explaining gets cut off, and the verdict lines it already wrote are still read.

    ``extra_body`` is forwarded on every call, and on a reasoning judge it is what
    makes the call finish. Measured on one real payload (2044 prompt tokens, 9
    criteria, 6 references) against ``qwen3.8-max``: left alone the judge spent 3757
    reasoning tokens and 93 seconds to write 38 tokens of verdicts, and
    ``LLM_BACKUP_TIMEOUT`` at its default of 120s cut off about half of a 27-call
    batch. ``max_tokens`` does not bound this -- it bounds the visible answer only,
    which is why 256 neither truncated a verdict nor prevented a timeout. With
    ``{'thinking_budget': 512}`` the same payload came back in 11 seconds with the
    same nine verdicts.
    """
    from twinkle.data_format.sampling import SamplingParams
    items = _applicable(references, criteria)
    messages = build_rubric_prompt(statement, check, references, criteria)
    api = client or _get_client(model)
    result = RubricResult()
    votes: List[List[Optional[bool]]] = []

    for attempt in range(max(1, max_votes)):
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if attempt == 0 else 1.0,
            top_p=0.95,
            num_samples=1)
        try:
            message = api({'messages': messages}, params,
                          **({'extra_body': extra_body} if extra_body else {}))
        except Exception as e:  # noqa
            if not votes:
                result.error = f'{type(e).__name__}: {e}'
                return result
            break
        if isinstance(message, list):
            message = message[0] if message else {}
        content = message.get('content', '') if isinstance(message, dict) else ''
        result.raw.append(content)
        votes.append(parse_verdicts(content, len(items)))
        result.n_votes = len(votes)

        # Mean over the votes cast so far, per criterion, then aggregate. Voting on
        # each criterion separately rather than on the final number is what keeps one
        # flipped criterion from moving the whole dimension.
        rates: List[Optional[float]] = []
        for i in range(len(items)):
            seen = [v[i] for v in votes if v[i] is not None]
            rates.append(statistics.fmean(1.0 if s else 0.0 for s in seen) if seen else None)
        result.pass_rates = rates
        result.verdicts = [None if r is None else r >= 0.5 for r in rates]
        result.scores = _aggregate(items, rates)
        if not _undecided(result.scores, margin):
            break

    if all(v is None for v in result.scores.values()):
        result.error = result.error or 'no usable verdict in reply'
    # Novelty is absent rather than zero when there was nothing to compare against.
    for crit in criteria:
        if crit.needs_references and not references:
            result.scores.setdefault(crit.dimension, None)
    return result


def score_tasks(
    tasks: Sequence[Dict[str, Any]],
    *,
    workers: int = 8,
    **kwargs,
) -> List[RubricResult]:
    """Score ``{statement, check, references}`` dicts, order preserved.

    Concurrency is over API calls only; nothing here touches a GPU or a sandbox.
    """

    def _one(task: Dict[str, Any]) -> RubricResult:
        return score_task(task.get('statement') or '', task.get('check') or '',
                          task.get('references') or (), **kwargs)

    if workers <= 1:
        return [_one(t) for t in tasks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, tasks))
