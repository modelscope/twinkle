# Copyright (c) ModelScope Contributors. All rights reserved.
"""Program-checked outcomes for agentic episodes.

An agentic episode ends with *state*, not with a string: files written, a
command that now succeeds, an answer stated in the final turn. This module
scores that end state with ordinary programs -- no judge model, so the same
trajectory always earns the same reward and difficulty filtering stays stable.

A task declares a list of :class:`Check`; :func:`run_checks` evaluates them and
returns a :class:`CheckReport` whose ``score`` is the reward.

Checks that need to *run* something (``shell`` / ``python``) go through a
``runner`` so they execute wherever the episode ran -- pass the sandbox's
runner and the check sees exactly the state the agent left behind. Without one
they fall back to a local subprocess in ``workspace``, which is only correct
when the episode itself ran locally.
"""
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    'Check',
    'CheckOutcome',
    'CheckReport',
    'CheckContext',
    'run_checks',
    'checks_from_dicts',
    'local_runner',
]

# (exit_code, output) for one command run inside the episode's workspace.
Runner = Callable[[str, str], Tuple[int, str]]

DEFAULT_TIMEOUT = int(os.environ.get('RESULT_CHECK_TIMEOUT', 60))
# Cap a runaway check so one bad task cannot take the trainer down with it.
_MEM_LIMIT_BYTES = 2 * 1024**3

_KINDS = (
    'file_exists',
    'file_absent',
    'file_contains',
    'file_equals',
    'file_json',
    'shell',
    'python',
    'answer_contains',
    'answer_equals',
    'answer_regex',
)


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

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f'unknown check kind {self.kind!r}; expected one of {_KINDS}')
        if self.weight <= 0:
            raise ValueError(f'check weight must be positive, got {self.weight}')


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
        return [(o.check.description or o.check.kind) + ': ' + o.detail
                for o in self.outcomes if not o.passed]

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
        runner: executes a command in the episode's environment. ``None`` runs
            it locally in ``workspace``.
    """
    workspace: str = ''
    final_answer: str = ''
    runner: Optional[Runner] = None


def local_runner(workspace: str) -> Runner:
    """Run commands in ``workspace`` as a local subprocess.

    Uses ``start_new_session`` + ``killpg`` so a forking command cannot leave
    grandchildren behind on timeout, and caps address space at 2GB.
    """

    def _run(command: str, interpreter: str) -> Tuple[int, str]:
        return _local_exec(command, interpreter, workspace, DEFAULT_TIMEOUT)

    return _run


def _local_exec(source: str, interpreter: str, cwd: str, timeout: int) -> Tuple[int, str]:
    cwd = cwd or '.'
    os.makedirs(cwd, exist_ok=True)
    if interpreter == 'python':
        tmp = tempfile.mkdtemp(prefix='rescheck_')
        script = os.path.join(tmp, '_check.py')
        with open(script, 'w', encoding='utf-8') as f:
            f.write(source)
        argv = [sys.executable, script]
    else:
        tmp = None
        argv = ['/bin/bash', '-lc', source]

    env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false')
    env.pop('CUDA_VISIBLE_DEVICES', None)

    def _limit():
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))

    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors='replace',
                                start_new_session=True, preexec_fn=_limit)
        try:
            out, _ = proc.communicate(timeout=timeout)
            return proc.returncode, out or ''
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:  # noqa
                pass
            return 124, f'check did not finish within {timeout}s'
    except Exception as e:  # noqa
        return 1, f'{type(e).__name__}: {e}'
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


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
        return CheckOutcome(check, there == want,
                            '' if there == want else
                            (f'{check.path} does not exist' if want
                             else f'{check.path} should not exist'))

    if kind in ('file_contains', 'file_equals', 'file_json'):
        text, err = _read_text(ctx.workspace, check.path)
        if text is None:
            return CheckOutcome(check, False, err)
        if kind == 'file_contains':
            if check.pattern:
                ok = re.search(check.pattern, text, re.S) is not None
                return CheckOutcome(check, ok, '' if ok else
                                    f'{check.path} does not match /{check.pattern}/')
            ok = _norm(check.value) in text
            return CheckOutcome(check, ok, '' if ok else
                                f'{check.path} does not contain {_norm(check.value)!r}')
        if kind == 'file_equals':
            ok = text.strip() == _norm(check.value)
            return CheckOutcome(check, ok, '' if ok else
                                f'{check.path} is {text.strip()[:120]!r}, '
                                f'expected {_norm(check.value)[:120]!r}')
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            return CheckOutcome(check, False, f'{check.path} is not valid JSON: {e}')
        found, got = _dig(doc, check.key)
        if not found:
            return CheckOutcome(check, False, f'{check.path} has no key {check.key!r}')
        ok = got == check.value if not isinstance(check.value, str) else _norm(got) == _norm(check.value)
        return CheckOutcome(check, ok, '' if ok else
                            f'{check.path}:{check.key} is {got!r}, expected {check.value!r}')

    if kind in ('shell', 'python'):
        runner = ctx.runner or local_runner(ctx.workspace)
        try:
            code, out = runner(check.code, 'python' if kind == 'python' else 'shell')
        except Exception as e:  # noqa
            return CheckOutcome(check, False, f'runner raised {type(e).__name__}: {e}')
        if code != check.expect_exit:
            return CheckOutcome(check, False,
                                f'exit {code} (expected {check.expect_exit}); output: {out[-300:]}')
        if check.pattern and re.search(check.pattern, out or '', re.S) is None:
            return CheckOutcome(check, False, f'output does not match /{check.pattern}/')
        if check.value is not None and _norm(check.value) not in (out or ''):
            return CheckOutcome(check, False, f'output does not contain {_norm(check.value)!r}')
        return CheckOutcome(check, True)

    answer = ctx.final_answer or ''
    if kind == 'answer_contains':
        ok = _norm(check.value) in answer
        return CheckOutcome(check, ok, '' if ok else
                            f'final answer does not contain {_norm(check.value)!r}')
    if kind == 'answer_equals':
        ok = answer.strip() == _norm(check.value)
        return CheckOutcome(check, ok, '' if ok else
                            f'final answer is {answer.strip()[:120]!r}, '
                            f'expected {_norm(check.value)[:120]!r}')
    ok = re.search(check.pattern, answer, re.S) is not None
    return CheckOutcome(check, ok, '' if ok else
                        f'final answer does not match /{check.pattern}/')


def run_checks(
    checks: Sequence[Check],
    ctx: CheckContext,
    mode: str = 'fraction',
) -> CheckReport:
    """Score one episode against its checks.

    Args:
        checks: the task's assertions. An empty list scores 0.0 rather than a
            free 1.0, so a task that forgot to declare checks cannot look solved.
        ctx: workspace / final answer / runner.
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
