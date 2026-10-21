# Copyright (c) ModelScope Contributors. All rights reserved.
"""Program-checked outcomes for agentic episodes.

An agentic episode ends with *state*, not with a string: files written, a
command that now succeeds, an answer stated in the final turn. This module
scores that end state with ordinary programs -- no judge model, so the same
trajectory always earns the same reward and difficulty filtering stays stable.

A task declares a list of :class:`Check`; :func:`run_checks` evaluates them and
returns a :class:`CheckReport` whose ``score`` is the reward.

Checks that need to *run* something (``shell`` / ``python``) run inside the
episode's :class:`~twinkle_agentic.envs.base.Env`, so they see exactly the state
the agent left behind -- hand over the sandbox the episode acted in. Without one
they fall back to a :class:`~twinkle_agentic.envs.localenv.LocalEnv` over
``workspace``, which is only correct when the episode itself ran locally.
"""
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # importing the env package for a type would cost every caller
    from ..envs.base import Env  # a second of import time -- see _local_env.

__all__ = [
    'Check',
    'CheckOutcome',
    'CheckReport',
    'CheckContext',
    'run_checks',
    'checks_from_dicts',
]

DEFAULT_TIMEOUT = int(os.environ.get('RESULT_CHECK_TIMEOUT', 60))
# Cap a runaway check so one bad task cannot take the trainer down with it.
_MEM_LIMIT_GB = 2.0

_KINDS = (
    'file_exists',
    'file_absent',
    'file_contains',
    'file_equals',
    'file_json',
    'shell',
    'python',
    'hidden_python',
    'answer_contains',
    'answer_equals',
    'answer_one_of',
    'answer_number',
    'answer_regex',
    'sql_query',
    'web_receipt',
    'lean',
)

_NUMBER_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


@dataclass
class Check:
    """One assertion about the end state.

    Args:
        kind: one of :data:`_KINDS`.
        path: workspace-relative file for the ``file_*`` kinds.
        value: expected substring / exact text / JSON value, per kind.
        pattern: regex alternative to ``value`` where the kind allows it.
        code: shell command (``shell``) or python source (``python``).
        key: dotted path into the document for ``file_json``, e.g. ``a.b.0.c``.
        expect_exit: required exit status for ``shell`` / ``python``.
        weight: contribution to the score; defaults to 1.0.
        timeout: per-check seconds for the running kinds.
        description: shown in the report so a failure is readable.
    """
    kind: str
    path: str = ''
    value: Any = None
    pattern: str = ''
    code: str = ''
    key: str = ''
    expect_exit: int = 0
    weight: float = 1.0
    timeout: int = DEFAULT_TIMEOUT
    description: str = ''
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    unit: str = ''
    query: str = ''
    db_path: str = ''
    expect_rows: Any = None
    ordered: bool = False
    columns: Any = None
    # web_receipt: one entry per url the answer cites. Each must have been served
    # by web_search (it appeared in a result list) or fetched by fetch_page.
    min_published: str = ''
    # lean: the .lean file the episode is expected to have written, and the
    # declaration that must exist in it. `path` holds the file.
    theorem: str = ''
    lean_project: str = ''

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f'unknown check kind {self.kind!r}; expected one of {_KINDS}')
        if self.weight <= 0:
            raise ValueError(f'check weight must be positive, got {self.weight}')
        if self.kind == 'answer_one_of':
            if (not isinstance(self.value, (list, tuple)) or not self.value
                    or any(not isinstance(item, (str, int, float, bool)) for item in self.value)):
                raise ValueError('answer_one_of value must be a non-empty list of scalar answers')
        if self.kind == 'sql_query':
            # A SQL check is only meaningful if the query is well-formed and the
            # expectation is stated. Validated here rather than at run time so a
            # malformed task fails at build, next to the author who can fix it.
            if not isinstance(self.query, str) or not self.query.strip():
                raise ValueError('sql_query requires a non-empty query')
            if not isinstance(self.db_path, str) or not self.db_path.strip():
                raise ValueError('sql_query requires a db_path')
            if self.expect_rows is None:
                raise ValueError('sql_query requires expect_rows (a list of rows)')
            if not isinstance(self.expect_rows, (list, tuple)):
                raise ValueError('sql_query expect_rows must be a list of rows')
            for row in self.expect_rows:
                if not isinstance(row, (list, tuple)):
                    raise ValueError('sql_query expect_rows entries must be lists/rows')
        if self.kind == 'lean':
            # A lean check without a named declaration would pass on an empty
            # file, so the name is required rather than defaulted.
            if not isinstance(self.path, str) or not self.path.strip():
                raise ValueError('lean requires path (the .lean file to grade)')
            if not isinstance(self.theorem, str) or not self.theorem.strip():
                raise ValueError('lean requires theorem (the declaration that must exist)')
        if self.kind == 'answer_number':
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError('answer_number value must be numeric')
            if not math.isfinite(float(self.value)):
                raise ValueError('answer_number value must be finite')
            for name in ('abs_tol', 'rel_tol'):
                tolerance = getattr(self, name)
                if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                        or not math.isfinite(float(tolerance)) or tolerance < 0):
                    raise ValueError(f'answer_number {name} must be finite and non-negative')
            if not isinstance(self.unit, str):
                raise ValueError('answer_number unit must be a string')


@dataclass
class CheckOutcome:
    check: Check
    passed: bool
    detail: str = ''


@dataclass
class CheckReport:
    """Result of scoring one episode."""
    score: float
    n_passed: int
    n_total: int
    outcomes: List[CheckOutcome] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.n_total > 0 and self.n_passed == self.n_total

    def failures(self) -> List[str]:
        return [(o.check.description or o.check.kind) + ': ' + o.detail for o in self.outcomes if not o.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'n_passed': self.n_passed,
            'n_total': self.n_total,
            'failures': self.failures(),
        }


@dataclass
class CheckContext:
    """What the checks are allowed to look at.

    Args:
        workspace: directory the episode wrote into.
        final_answer: text of the last assistant turn, for the ``answer_*`` kinds.
        env: where the ``shell`` / ``python`` kinds run -- the environment the
            episode acted in. ``None`` runs them locally in ``workspace``.
    """
    workspace: str = ''
    final_answer: str = ''
    env: Optional['Env'] = None
    # What the search tools actually served this episode, in order, as recorded
    # by the host that answered them. Empty for a task that used no web tools,
    # which is why `web_receipt` fails rather than passes when it is asked for
    # with nothing here: an absent log is not evidence of a retrieval.
    receipts: Sequence[Dict[str, Any]] = ()


def _local_env(workspace: str) -> 'Env':
    """Run checks in ``workspace`` on this machine.

    The fallback for a :class:`CheckContext` with no env. It is a
    :class:`~twinkle_agentic.envs.localenv.LocalEnv`, so a check that falls back to
    here and a check that runs in a sandbox go through one interface -- and the
    process isolation (own session, killpg on timeout, capped address space)
    lives in one place instead of being restated by every caller that needs it.
    """
    # Imported here, not at module scope: the env package pulls in twinkle's
    # remote-class machinery, and a task declaring only file_* checks should not
    # pay a second of import time for an environment it never runs anything in.
    from ..envs.localenv import LocalEnv
    return LocalEnv(workspace=workspace or '.', command_timeout=DEFAULT_TIMEOUT, memory_limit_gb=_MEM_LIMIT_GB)


def checks_from_dicts(raw: Sequence[Dict[str, Any]]) -> List[Check]:
    """Build checks from the plain dicts a task file carries."""
    return [Check(**dict(item)) for item in raw or []]


def _resolve(workspace: str, path: str) -> str:
    """Resolve a task-declared path inside the workspace.

    Rejects escapes: a task must not be able to assert on files outside the
    episode's own directory, or one episode could pass by reading another's.
    """
    root = os.path.realpath(workspace or '.')
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f'check path {path!r} escapes the workspace')
    return target


def _read_text(workspace: str, path: str) -> Tuple[Optional[str], str]:
    try:
        full = _resolve(workspace, path)
    except ValueError as e:
        return None, str(e)
    if not os.path.isfile(full):
        return None, f'{path} does not exist'
    try:
        with open(full, encoding='utf-8', errors='replace') as f:
            return f.read(), ''
    except OSError as e:
        return None, f'cannot read {path}: {e}'


def _dig(doc: Any, key: str) -> Tuple[bool, Any]:
    """Walk a dotted path; integer segments index into lists."""
    cur = doc
    for seg in [s for s in key.split('.') if s]:
        if isinstance(cur, dict):
            if seg not in cur:
                return False, None
            cur = cur[seg]
        elif isinstance(cur, list):
            if not seg.lstrip('-').isdigit():
                return False, None
            idx = int(seg)
            if not -len(cur) <= idx < len(cur):
                return False, None
            cur = cur[idx]
        else:
            return False, None
    return True, cur


def _norm(text: Any) -> str:
    return str(text if text is not None else '').strip()


def _eval_one(check: Check, ctx: CheckContext) -> CheckOutcome:
    kind = check.kind

    if kind in ('file_exists', 'file_absent'):
        try:
            full = _resolve(ctx.workspace, check.path)
        except ValueError as e:
            return CheckOutcome(check, False, str(e))
        there = os.path.exists(full)
        want = (kind == 'file_exists')
        return CheckOutcome(
            check, there == want, '' if there == want else
            (f'{check.path} does not exist' if want else f'{check.path} should not exist'))

    if kind in ('file_contains', 'file_equals', 'file_json'):
        text, err = _read_text(ctx.workspace, check.path)
        if text is None:
            return CheckOutcome(check, False, err)
        if kind == 'file_contains':
            if check.pattern:
                ok = re.search(check.pattern, text, re.S) is not None
                return CheckOutcome(check, ok, '' if ok else f'{check.path} does not match /{check.pattern}/')
            ok = _norm(check.value) in text
            return CheckOutcome(check, ok, '' if ok else f'{check.path} does not contain {_norm(check.value)!r}')
        if kind == 'file_equals':
            ok = text.strip() == _norm(check.value)
            return CheckOutcome(
                check, ok, '' if ok else f'{check.path} is {text.strip()[:120]!r}, '
                f'expected {_norm(check.value)[:120]!r}')
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            return CheckOutcome(check, False, f'{check.path} is not valid JSON: {e}')
        found, got = _dig(doc, check.key)
        if not found:
            return CheckOutcome(check, False, f'{check.path} has no key {check.key!r}')
        ok = got == check.value if not isinstance(check.value, str) else _norm(got) == _norm(check.value)
        return CheckOutcome(check, ok, '' if ok else f'{check.path}:{check.key} is {got!r}, expected {check.value!r}')

    if kind in ('shell', 'python', 'hidden_python'):
        env = ctx.env or _local_env(ctx.workspace)
        interpreter = 'shell' if kind == 'shell' else 'python'
        try:
            code, out = env.run_script(check.code, interpreter, check.timeout)
        except Exception as e:  # noqa
            return CheckOutcome(check, False, f'{type(env).__name__} raised {type(e).__name__}: {e}')
        if code != check.expect_exit:
            return CheckOutcome(check, False, f'exit {code} (expected {check.expect_exit}); output: {out[-300:]}')
        if check.pattern and re.search(check.pattern, out or '', re.S) is None:
            return CheckOutcome(check, False, f'output does not match /{check.pattern}/')
        if check.value is not None and _norm(check.value) not in (out or ''):
            return CheckOutcome(check, False, f'output does not contain {_norm(check.value)!r}')
        return CheckOutcome(check, True)

    if kind == 'sql_query':
        try:
            full = _resolve(ctx.workspace, check.db_path)
        except ValueError as e:
            return CheckOutcome(check, False, str(e))
        if not os.path.isfile(full):
            return CheckOutcome(check, False, f'{check.db_path} does not exist')
        # Read-only, via URI, so a check can never mutate the artifact it grades.
        # A task that passes by writing during verification is not a solved task.
        try:
            conn = sqlite3.connect(f'file:{full}?mode=ro', uri=True)
        except sqlite3.Error as e:
            return CheckOutcome(check, False, f'{check.db_path} is not a readable SQLite file: {e}')
        try:
            try:
                got = conn.execute(check.query).fetchall()
            except sqlite3.Error as e:
                return CheckOutcome(check, False, f'query failed: {e}')
        finally:
            conn.close()
        want = [tuple(row) for row in check.expect_rows]
        got = [tuple(r) for r in got]
        if not check.ordered:
            # Set comparison by default: SQL without ORDER BY has no guaranteed row
            # order, so requiring one would fail correct solutions on an accident of
            # the planner. Compare sorted, but sort by the repr so mixed and NULL
            # values do not raise on comparison.
            got_cmp, want_cmp = sorted(map(repr, got)), sorted(map(repr, want))
            ok = got_cmp == want_cmp
        else:
            ok = got == want
        if ok:
            return CheckOutcome(check, True)
        return CheckOutcome(
            check, False, f'query returned {len(got)} row(s) {got[:3]!r}..., expected {len(want)} '
            f'row(s) {want[:3]!r}...' + ('' if check.ordered else ' (unordered)'))

    answer = ctx.final_answer or ''
    if kind == 'answer_contains':
        ok = _norm(check.value) in answer
        return CheckOutcome(check, ok, '' if ok else f'final answer does not contain {_norm(check.value)!r}')
    if kind == 'answer_equals':
        ok = answer.strip() == _norm(check.value)
        return CheckOutcome(
            check, ok, '' if ok else f'final answer is {answer.strip()[:120]!r}, '
            f'expected {_norm(check.value)[:120]!r}')
    if kind == 'answer_one_of':
        choices = [_norm(value) for value in check.value]
        ok = answer.strip() in choices
        return CheckOutcome(check, ok, '' if ok else f'final answer is {answer.strip()[:120]!r}, '
                            f'expected one of {choices!r}')
    if kind == 'answer_number':
        text = answer.strip()
        if check.unit:
            suffix = check.unit.strip()
            if not suffix or not text.endswith(suffix):
                return CheckOutcome(check, False, f'final answer does not use required unit {check.unit!r}')
            text = text[:-len(suffix)].strip()
        if not _NUMBER_RE.fullmatch(text):
            return CheckOutcome(check, False, f'final answer is not one unambiguous finite number: '
                                f'{answer.strip()[:120]!r}')
        got = float(text)
        if not math.isfinite(got):
            return CheckOutcome(check, False, 'final answer number is not finite')
        ok = math.isclose(got, float(check.value), rel_tol=float(check.rel_tol), abs_tol=float(check.abs_tol))
        return CheckOutcome(
            check, ok, '' if ok else f'final answer number is {got!r}, expected {check.value!r} '
            f'(abs_tol={check.abs_tol}, rel_tol={check.rel_tol})')
    if kind == 'web_receipt':
        return _check_web_receipt(check, ctx)

    if kind == 'lean':
        return _check_lean(check, ctx)

    ok = re.search(check.pattern, answer, re.S) is not None
    return CheckOutcome(check, ok, '' if ok else f'final answer does not match /{check.pattern}/')


def _check_lean(check: Check, ctx: CheckContext) -> CheckOutcome:
    """Compile in the episode environment, reject sorry warnings and require a named declaration."""
    if ctx.env is None:
        # Compiling needs the sandbox: the toolchain lives there, not here. A
        # check that silently ran locally would grade a different machine's PATH
        # and report a failure that has nothing to do with the submission.
        return CheckOutcome(check, False, 'lean checks require an environment to compile in')
    project = check.lean_project or '.'
    # `lean_env` is set by the image; the grader reads it the same way.
    theorem_pattern = r'(theorem|lemma)[[:space:]]+' + re.escape(check.theorem) + r'([[:space:]:]|$)'
    script = _LEAN_GRADER.format(
        project=_shquote(project), path=_shquote(check.path), theorem_pattern=_shquote(theorem_pattern))
    try:
        code, out = ctx.env.run_script(script, 'shell', check.timeout)
    except Exception as e:  # noqa: BLE001
        return CheckOutcome(check, False, f'{type(e).__name__}: {e}')
    if code == 2:
        return CheckOutcome(check, False, f'could not grade: {out.strip()[:300]}')
    if code != 0:
        return CheckOutcome(check, False, out.strip()[:400] or f'exit {code}')
    return CheckOutcome(check, True, '')


# Runs inside the sandbox. Kept as a template rather than inline f-string so the
# quoting of the three substituted values happens in one place.
_LEAN_GRADER = """
export PATH="${{LEAN_PREFIX:-/opt/lean}}/bin:$PATH"
cd {project} 2>/dev/null || {{ echo "lean project not found"; exit 2; }}
[ -f {path} ] || {{ echo "submission not found"; exit 2; }}
out=$(lake env lean {path} 2>&1); status=$?
# Checked before the exit code: sorry exits 0, so the exit code alone cannot
# distinguish a proof from a placeholder.
if printf '%s' "$out" | grep -qE 'declaration uses .sorry.|sorryAx'; then
  echo "uses sorry or admit, which proves nothing"; echo "$out" | head -5; exit 1
fi
[ $status -eq 0 ] || {{ echo "$out" | head -20; exit 1; }}
# A file can compile without ever stating the theorem; that is not a proof of it.
grep -qE {theorem_pattern} {path} \
  || {{ echo "compiles, but named declaration not found"; exit 1; }}
echo "accepted"
"""


def _shquote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def _check_web_receipt(check: Check, ctx: CheckContext) -> CheckOutcome:
    """Every cited url must appear in what the search tools actually served.

    This is the check that makes a citation mean something. Without it, an answer
    that names a plausible url is indistinguishable from one that read the page,
    and a policy learns that inventing a source is free. The evidence is the
    host's own log of the calls it answered -- not anything the policy wrote, so
    it cannot be manufactured from inside the episode.

    Three things are verified per cited url, and each fails for its own reason:

      * the url was served at all -- `web_search` returned it or `fetch_page`
        fetched it. A url the policy assembled from a site's naming scheme is not
        a retrieval.
      * if `min_published` is set, at least one receipt carrying that url also
        carried a publication date on or after it. This is what stops an answer
        built on a page the engine itself flagged as old.
      * the cited set is non-empty. An answer with no citations cannot have come
        from searching, so the check fails rather than passing vacuously.
    """
    cited = check.value
    if isinstance(cited, str):
        cited = [cited]
    if not isinstance(cited, (list, tuple)) or not cited:
        return CheckOutcome(check, False, 'web_receipt requires value = the cited url(s)')

    # url -> newest publication date seen for it, '' when only ever fetched.
    # Presence in this map is the retrieval evidence; the date is a separate
    # question. Conflating them was a bug: a url that was fetched but never
    # appeared in a dated result list vanished from the map entirely, and the
    # failure then said "never retrieved" about a page the episode had in fact
    # read -- an accusation that was both wrong and unhelpful to act on.
    served: Dict[str, str] = {}
    for receipt in ctx.receipts or ():
        for hit in (receipt or {}).get('urls') or ():
            url = (hit or {}).get('url') or ''
            if not url:
                continue
            published = (hit or {}).get('published') or ''
            if url not in served or published > served[url]:
                served[url] = published

    if not served:
        return CheckOutcome(
            check, False, f'the episode cites {len(cited)} url(s) but the search tools were '
            f'never used successfully; a citation with no retrieval behind it '
            f'is not evidence')

    missing = [url for url in cited if url not in served]
    if missing:
        return CheckOutcome(
            check, False, f'cited url(s) never came back from a search or fetch: {missing}. '
            f'Served urls were: {sorted(served)[:5]}')

    if check.min_published:
        undated = [url for url in cited if not served.get(url) or served[url] < check.min_published]
        if undated:
            return CheckOutcome(
                check, False, f'cited url(s) have no publication date on or after '
                f'{check.min_published}: {[(u, served.get(u) or "undated") for u in undated]}')

    return CheckOutcome(check, True, '')


def run_checks(
    checks: Sequence[Check],
    ctx: CheckContext,
    mode: str = 'fraction',
) -> CheckReport:
    """Score one episode against its checks.

    Args:
        checks: the task's assertions. An empty list scores 0.0 rather than a
            free 1.0, so a task that forgot to declare checks cannot look solved.
        ctx: workspace / final answer / environment.
        mode: ``fraction`` gives weighted partial credit, ``all_or_nothing``
            gives 1.0 only when every check passes.

    A check that raises is a failed check, never a failed batch: one malformed
    task must not abort scoring for the rest of the rollout group.
    """
    if mode not in ('fraction', 'all_or_nothing'):
        raise ValueError(f"mode must be 'fraction' or 'all_or_nothing', got {mode!r}")
    checks = list(checks or [])
    if not checks:
        return CheckReport(score=0.0, n_passed=0, n_total=0, outcomes=[])

    outcomes: List[CheckOutcome] = []
    for check in checks:
        try:
            outcomes.append(_eval_one(check, ctx))
        except Exception as e:  # noqa
            outcomes.append(CheckOutcome(check, False, f'{type(e).__name__}: {e}'))

    n_passed = sum(1 for o in outcomes if o.passed)
    if mode == 'all_or_nothing':
        score = 1.0 if n_passed == len(outcomes) else 0.0
    else:
        total_w = sum(o.check.weight for o in outcomes)
        score = sum(o.check.weight for o in outcomes if o.passed) / total_w
    return CheckReport(score=score, n_passed=n_passed, n_total=len(outcomes), outcomes=outcomes)
