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
``cookbook/rsi/code/prompts.py``. What stays here is the machinery that cannot
be restated in a prompt: the sandbox, the assert capture, the constant-answer
check, the keyword bank, and how a proposal becomes a task.
"""
import json
import os
import random
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format import SamplingParams, Trajectory, user_data_get
from twinkle.utils import get_logger
from .base import Challenger, Explorer, assistant_text, attach_user_data

logger = get_logger()

__all__ = [
    'CodeChallenger', 'CodePrompts', 'KeywordStore', 'build_asserts',
    'extract_code', 'is_constant_answer', 'load_seeds', 'parse_challenge',
    'run_asserts',
]

# Isolates a captured value from anything else the script prints.
_MARK = '__RSI_GT__'
_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)
_JSON_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.I)


# ── sandbox ────────────────────────────────────────────────────────────────
def extract_code(text: str) -> str:
    """The last fenced code block after the thinking section, else the raw body."""
    idx = (text or '').rfind('</think>')
    body = text[idx + len('</think>'):] if idx >= 0 else (text or '')
    blocks = _FENCE_RE.findall(body)
    return (blocks[-1] if blocks else body).strip()


def _run_script(script: str, timeout: int) -> Tuple[int, str]:
    """Run a python script in an isolated dir, 2GB cap, killpg on timeout.

    Returns (returncode, stdout). returncode is -1 on timeout/spawn failure.
    """
    tmp = tempfile.mkdtemp(prefix='rsi_ch_')
    try:
        with open(os.path.join(tmp, '_run.py'), 'w', encoding='utf-8') as f:
            f.write(script + '\n')
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env.pop('CUDA_VISIBLE_DEVICES', None)

        def _limit():
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))

        proc = subprocess.Popen([sys.executable, '_run.py'], cwd=tmp, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, start_new_session=True, preexec_fn=_limit)
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
            except Exception:
                pass
            return -1, ''
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_asserts(code: str, setup: str, asserts: List[str], timeout: int = 30) -> bool:
    """True when every assert passes (exit status 0)."""
    if not code.strip() or not asserts:
        return False
    parts = [code]
    if (setup or '').strip():
        parts.append(setup)
    parts.extend(asserts)
    rc, _ = _run_script('\n\n'.join(parts), timeout)
    return rc == 0


def build_asserts(solution: str, checks: List[str], timeout: int = 30,
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
    rc, out = _run_script('\n'.join(lines), timeout)
    if rc != 0:
        return None
    captured: Dict[int, str] = {}
    for line in out.splitlines():
        if line.startswith(_MARK):
            try:
                idx_str, val = line[len(_MARK):].split('=', 1)
                captured[int(idx_str)] = val
            except (ValueError, IndexError):
                continue
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
    body = text
    idx = body.rfind('</think>')
    if idx >= 0:
        body = body[idx + len('</think>'):]
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
        solution = extract_code(solution)
    return {'problem': problem.strip(), 'solution': (solution or '').strip(),
            'entry': str(obj.get('entry') or '').strip(), 'checks': checks}


# A keyword is a topic to build a task around, not a task statement. Past this many
# characters the model has written the second thing, and storing it makes the next
# prompt ask for a variation on a sentence rather than on a subject.
KEYWORD_MAX_LEN = 60


def split_keyword_list(text: str) -> Tuple[List[str], List[str]]:
    """Extract a JSON array of short strings; return (kept, dropped for length).

    The dropped half exists because it used to be discarded inside a list
    comprehension. A refill that returned eight well-formed keywords, all of them
    written out as sentences, reached the caller as an empty list and was recorded
    as ``n_parsed: 0`` -- the same three characters a garbled reply, a timeout and
    an over-length reply all produce, so the log could not tell them apart. One
    iteration lost 27% of its keywords that way and the cause was found by
    re-parsing the stored replies by hand.

    The bias is the reason to count rather than only to log: length correlates with
    specificity, so the filter removes "Compute the critical path delay through a
    gate-level netlist with annotated cell delays" and keeps whatever was vague
    enough to be short. That is the opposite of what the bank is for.
    """
    body = text
    idx = body.rfind('</think>')
    if idx >= 0:
        body = body[idx + len('</think>'):]
    start, end = body.find('['), body.rfind(']')
    if start < 0 or end <= start:
        return [], []
    try:
        arr = json.loads(body[start:end + 1])
    except (ValueError, TypeError):
        return [], []
    kept: List[str] = []
    dropped: List[str] = []
    for x in arr:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        (kept if len(s) <= KEYWORD_MAX_LEN else dropped).append(s)
    return kept, dropped


def parse_keyword_list(text: str) -> List[str]:
    """The kept half of :func:`split_keyword_list`, for callers with nothing to record."""
    return split_keyword_list(text)[0]


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


# ── keyword bank ───────────────────────────────────────────────────────────
class KeywordStore:
    """Persistent keyword bank with usage tracking, one bucket per category.

    Keywords exist to stop the challenger collapsing onto a handful of
    archetypes. They are consumed rather than sampled with replacement, so a
    run keeps reaching for topics it has not used; when a bucket runs dry the
    caller refills it from the model, and recycles only if the model has run out
    of distinct ideas.

    On-disk format (one JSON per line)::

        {"category", "text", "used": bool, "used_count": int,
         "source": "gen"|"expand", "parent": <keyword or null>}

    De-duplicates case-insensitively within a category, so re-runs never
    conflict with the bank on disk.
    """

    def __init__(self, path: str, categories: Sequence[str]):
        if not categories:
            raise ValueError('KeywordStore needs at least one category')
        self.path = path
        self.categories = tuple(categories)
        self.items: Dict[str, List[Dict[str, Any]]] = {c: [] for c in self.categories}
        self._seen: Dict[str, set] = {c: set() for c in self.categories}
        if path and os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    c, t = r.get('category'), r.get('text')
                    if c in self.items and isinstance(t, str) and t.strip():
                        key = t.strip().lower()
                        if key not in self._seen[c]:
                            self._seen[c].add(key)
                            self.items[c].append(r)

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            for c in self.categories:
                for r in self.items[c]:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        os.replace(tmp, self.path)

    def add(self, category: str, texts: List[str], source: str = 'gen',
            parent: Optional[str] = None) -> int:
        added = 0
        for t in texts:
            key = t.strip().lower()
            if not key or key in self._seen[category]:
                continue
            self._seen[category].add(key)
            self.items[category].append({'category': category, 'text': t.strip(),
                                         'used': False, 'used_count': 0,
                                         'source': source, 'parent': parent})
            added += 1
        return added

    def unused(self, category: str) -> List[Dict[str, Any]]:
        return [r for r in self.items[category] if not r.get('used')]

    def texts(self, category: str) -> List[str]:
        return [r['text'] for r in self.items[category]]

    def take(self, category: str, rng: random.Random) -> Optional[str]:
        """Consume one unused keyword from ``category``; None if it is dry."""
        un = self.unused(category)
        if not un:
            return None
        r = rng.choice(un)
        r['used'] = True
        r['used_count'] = r.get('used_count', 0) + 1
        return r['text']

    def recycle(self, category: str) -> None:
        """Mark every keyword unused again (safety valve when the model is tapped out)."""
        for r in self.items[category]:
            r['used'] = False


# ── prompts (text supplied by the caller) ──────────────────────────────────
@dataclass
class CodePrompts:
    """Every string a :class:`CodeChallenger` sends, and nothing else.

    Deliberately without defaults for the always-needed fields: a prompt is the
    experiment, so a run has to state which one it used rather than inherit a
    library's idea of it. Optional groups stay empty until the feature that
    needs them is switched on, and the constructor says so if one is missing.

    Placeholders are checked at construction: a typo'd ``{keywords}`` would
    otherwise surface as a KeyError halfway through a generation run.
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

    def __post_init__(self):
        for name in ('system', 'from_scratch', 'solver_system', 'solver_user'):
            if not getattr(self, name).strip():
                raise ValueError(f'CodePrompts.{name} is required')
        for name, placeholders in self._REQUIRED_FIELDS.items():
            text = getattr(self, name)
            if not text:
                continue
            for placeholder in placeholders:
                if '{' + placeholder + '}' not in text:
                    raise ValueError(f'CodePrompts.{name} must contain '
                                     f'{{{placeholder}}}')

    def require(self, *names: str) -> None:
        """Raise unless every named prompt was supplied."""
        missing = [n for n in names if not getattr(self, n).strip()]
        if missing:
            raise ValueError(f'this configuration needs CodePrompts.'
                             f'{", CodePrompts.".join(missing)}')


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
        category_desc: category -> description used when asking for more
            keywords. Keys must cover the store's categories.
        seed_mix_prob: chance a proposal also carries a seed problem, when a
            pool was given.
        two_step: allow the two-call path (write a harder solution on top of the
            seed's reference code, then describe the problem it answers). Needs
            a seed carrying ``code`` and at least one keyword, so it is skipped
            silently for proposals that have neither.
        combo_arity: ``'triple'`` takes one keyword per category; ``'mix'``
            takes a random 1..len(categories) subset.
        arity_weights: sampling weights for the ``'mix'`` subset size.
        single_kw_prob: in ``'triple'`` mode, the chance of using one category
            instead of all of them.
        keyword_refill_target / keyword_gen_calls / keyword_refill_tries /
        keyword_params: how a dry category is refilled from the model.
        min_batch: smallest batch worth sending -- a sampler shards a batch over
            its data-parallel workers, and a batch smaller than that leaves some
            with nothing to do. Set it to the number of sampler workers.
        problem_max_chars: reject statements longer than this. Rambling
            non-problems, and they would also crowd out the solver's context.
        max_checks / sandbox_timeout: passed to :func:`build_asserts`.
        drop_constant_answer: reject problems where one constant satisfies every
            assert.
        low_pass_expand / expand_per_kw / expand_max_kws: feedback for
            :meth:`expand_hard_keywords`.
        reject_sink: called with a dict for every rejected proposal. The caller
            decides whether that goes to a file; nothing here writes one.
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
        **challenger_kwargs: Any,
    ):
        super().__init__(explorer, system=prompts.system, **challenger_kwargs)
        if combo_arity not in ('triple', 'mix'):
            raise ValueError(f"combo_arity must be 'triple' or 'mix', got {combo_arity!r}")
        if keyword_store is not None:
            desc = category_desc or {}
            missing = [c for c in keyword_store.categories if not desc.get(c)]
            if missing:
                raise ValueError(f'category_desc is missing a description for '
                                 f'{missing}; a dry category could not be refilled.')
            prompts.require('keyword_system', 'keyword_user', 'from_keywords')
        self.prompts = prompts
        self.seeds = list(seeds)
        self.store = keyword_store
        self.category_desc = dict(category_desc or {})
        self.seed_mix_prob = seed_mix_prob
        self.two_step = two_step
        self.combo_arity = combo_arity
        self.arity_weights = list(arity_weights) if arity_weights else None
        self.single_kw_prob = single_kw_prob
        self.keyword_refill_target = keyword_refill_target
        self.keyword_gen_calls = keyword_gen_calls
        self.keyword_refill_tries = keyword_refill_tries
        self.keyword_params = keyword_params
        self.min_batch = max(1, min_batch)
        self.problem_max_chars = problem_max_chars
        self.max_checks = max_checks
        self.sandbox_timeout = sandbox_timeout
        self.drop_constant_answer = drop_constant_answer
        self.low_pass_expand = low_pass_expand
        self.expand_per_kw = expand_per_kw
        self.expand_max_kws = expand_max_kws
        self.reject_sink = reject_sink
        if self.seeds:
            # Both are reachable with a bank configured: a proposal draws no
            # keywords when every category is dry, and then falls back to the
            # seed-only prompt.
            prompts.require('from_seed')
            if self.store is not None:
                prompts.require('from_seed_keywords')
        if two_step:
            prompts.require('two_step_system', 'two_step_solution', 'two_step_problem')
        # Perturbs refill prompts so a second ask does not repeat the first.
        self._nonce = 0
        # Why proposals died, for the caller to log; the shape a run is judged on.
        self.stats: Dict[str, int] = {
            'parsed': 0, 'parse_fail': 0, 'stage1_no_code': 0, 'too_long': 0,
            'gt_fail': 0, 'selfcheck_fail': 0, 'constant_answer': 0,
        }
        # (category, keyword) behind candidates nobody could solve, for feedback.
        self._hard: List[Tuple[str, str]] = []

    # ------------------------------------------------------------- proposing

    def propose(self, count: int) -> List[Trajectory]:
        proposals: List[Trajectory] = []
        for _ in range(count):
            picks = self._draw_keywords()
            body = '\n'.join(f'- {c}: {t}' for c, t in picks)
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

    def _draw_keywords(self) -> List[Tuple[str, str]]:
        """Consume one keyword combination from the bank; [] without a bank."""
        if self.store is None:
            return []
        categories = self.store.categories
        if self.combo_arity == 'mix':
            if self.arity_weights and len(self.arity_weights) == len(categories):
                k = self.rng.choices(range(1, len(categories) + 1),
                                     weights=self.arity_weights)[0]
            else:
                k = self.rng.randint(1, len(categories))
            cats = self.rng.sample(list(categories), k)
        elif self.rng.random() < self.single_kw_prob:
            cats = [self.rng.choice(categories)]
        else:
            cats = list(categories)
        picks: List[Tuple[str, str]] = []
        for c in cats:
            if not self.store.unused(c):
                self._refill(c)
            text = self.store.take(c, self.rng)
            if text is not None:
                picks.append((c, text))
        return picks

    def _refill(self, category: str) -> None:
        """Ask the model for more keywords in ``category``; recycle if it is tapped out."""
        tries = 0
        while not self.store.unused(category):
            new = self._generate_keywords(category, self.keyword_refill_target)
            added = self.store.add(category, new, source='gen')
            tries += 1
            if added == 0 and tries >= self.keyword_refill_tries:
                if self.store.items[category]:
                    self.store.recycle(category)
                    logger.info(f'[CodeChallenger] keyword category {category!r} exhausted '
                                f'-> recycled {len(self.store.items[category])} topics')
                break

    def _generate_keywords(self, category: str, n_want: int) -> List[str]:
        """Up to ``n_want`` keywords the bank does not already hold."""
        if n_want <= 0:
            return []
        known = self.store.texts(category)
        n_calls = max(self.keyword_gen_calls, self.min_batch)
        per_call = max(1, -(-n_want // n_calls) + 4)   # ceil(n/calls) + margin
        avoid_note = ''
        if known:
            shown = known if len(known) <= 40 else self.rng.sample(known, 40)
            avoid_note = ('\nDo NOT repeat any of these already-used topics: '
                          + ', '.join(shown))
        base = self.prompts.keyword_user.format(
            k=per_call, desc=self.category_desc[category]) + avoid_note
        self._nonce += 1
        prompts = [{
            'messages': [{'role': 'system', 'content': self.prompts.keyword_system},
                         {'role': 'user', 'content': f'{base}\n(batch {self._nonce}-{i})'}],
        } for i in range(n_calls)]
        seen = {t.strip().lower() for t in known}
        out: List[str] = []
        n_long = 0
        for reply in self.explore(prompts, sampling_params=self.keyword_params):
            kept, dropped = split_keyword_list(assistant_text(reply))
            n_long += len(dropped)
            for kw in kept:
                key = kw.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(kw)
        if n_long:
            # This path has no dump to write to, so the count has to be said out
            # loud or the refill looks like the model simply produced less.
            logger.warning(f'[CodeChallenger] dropped {n_long} keyword(s) over '
                           f'{KEYWORD_MAX_LEN} chars while refilling; the prompt is '
                           f'asking for task statements rather than topics')
        self.rng.shuffle(out)
        return out[:n_want]

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
            code = extract_code(text)
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
        asserts = build_asserts(obj['solution'], obj['checks'],
                                timeout=self.sandbox_timeout, max_checks=self.max_checks)
        if not asserts:
            _reject('gt_fail')
            return None
        if not run_asserts(obj['solution'], '', asserts, timeout=self.sandbox_timeout):
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
            asserts=asserts,
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

    def judge_attempt(self, task: Trajectory, attempt: Trajectory) -> bool:
        asserts = user_data_get(task.get('user_data'), 'asserts', []) or []
        return run_asserts(extract_code(assistant_text(attempt)), '', asserts,
                           timeout=self.sandbox_timeout)

    def on_difficulty_measured(self, candidates: List[Trajectory]) -> None:
        """Remember the topics behind the candidates nobody solved."""
        if self.store is None:
            return
        seen = {(c, t.lower()) for c, t in self._hard}
        for task in candidates:
            data = task.get('user_data')
            if user_data_get(data, 'n_pass', 0) > self.low_pass_expand:
                continue
            for pick in user_data_get(data, 'keywords', []) or []:
                c, t = pick[0], pick[1]
                if (c, t.lower()) not in seen:
                    seen.add((c, t.lower()))
                    self._hard.append((c, t))

    # ------------------------------------------------------------- feedback

    def expand_hard_keywords(self) -> int:
        """Brainstorm more topics in the families that produced the hardest tasks.

        Called by whoever drives the challenger, after generating, so the bank
        drifts toward material the solver actually struggles with. Returns how
        many new keywords were added.
        """
        if self.store is None or not self._hard or self.expand_per_kw <= 0:
            return 0
        self.prompts.require('keyword_expand_user')
        hard = self._hard[:self.expand_max_kws]
        self.rng.shuffle(hard)
        # Cycle a short list so the batch still covers every sampler worker.
        reqs = list(hard)
        while len(reqs) < self.min_batch:
            reqs.append(hard[len(reqs) % len(hard)])
        self._nonce += 1
        prompts = [{
            'messages': [
                {'role': 'system', 'content': self.prompts.keyword_system},
                {'role': 'user',
                 'content': self.prompts.keyword_expand_user.format(kw=kw, m=self.expand_per_kw)
                 + f'\n(batch {self._nonce}-{i})'},
            ],
        } for i, (_c, kw) in enumerate(reqs)]
        added = 0
        n_long = 0
        for (cat, kw), reply in zip(reqs, self.explore(prompts,
                                                      sampling_params=self.keyword_params)):
            kept, dropped = split_keyword_list(assistant_text(reply))
            n_long += len(dropped)
            added += self.store.add(cat, kept, source='expand', parent=kw)
        if n_long:
            logger.warning(f'[CodeChallenger] dropped {n_long} expanded keyword(s) over '
                           f'{KEYWORD_MAX_LEN} chars; expansion follows the parent, so a '
                           f'wordy parent produces wordy children')
        logger.info(f'[CodeChallenger] expanded {len(hard)} hard keyword(s) -> '
                    f'+{added} same-domain topics')
        return added
