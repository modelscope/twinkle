# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI step 0 — self-play data generation for code tasks.

A single model plays TWO roles (self-play, same Qwen3-4B weights, per the talk's
"出题者/做题者" setup):

  * CHALLENGER — writes a self-contained Python solution, we RUN it in the same
    sandbox rsi_rl uses to capture ground-truth outputs, and turn those outputs
    into asserts. The problem statement is what the solver will be shown; the
    executed reference solution is the GT. This is reverse construction: the
    answer exists first (we ran it), the problem is written around it, so a GT
    is available without any external labeling.

  * SOLVER (difficulty filter) — the SAME model then attempts each proposed
    problem N times from the problem statement alone. We run its code against
    the challenger's asserts and keep only problems whose pass count is strictly
    between 0 and N ("half-know" band): all-pass or all-fail rounds give GRPO a
    zero gradient (verified on MBPP), so they are dropped here.

Two safety gates carried over from earlier failures:
  * The challenger's OWN reference solution must pass its OWN asserts, or the
    problem is dropped (a GT that cannot pass its own tests is noise — the
    "standard answer that itself fails" pitfall).
  * Output capture uses a sentinel marker + returncode check, never the last
    stdout line, so an environment banner can never be mistaken for a result.

Optional seed dataset (RSI_CH_SEED): a jsonl whose rows carry a `query` and,
preferably, a `code` reference solution. When a row has `code` (and keywords are
enabled), that proposal takes the TWO-STEP path: call 1 writes a HARDER solution on
top of the reference code, call 2 describes the problem that solution answers, and
the stage-1 code becomes the ground truth. Rows without `code` fall back to the older
single-call "seed as inspiration" prompt. Set RSI_CH_TWO_STEP=0 to force that older
path everywhere. Without any seed the challenger invents problems from scratch.

Output (consumed directly by rsi_rl.py, no prepare/refine in between):
  RSI_CH_OUT_FLOWS   flows jsonl: {id, system, query, tools, rounds:[code round]}
  RSI_CH_OUT_TESTS   tests jsonl: {id, test_list, test_setup_code}  (-> RSI_TESTS)

Every knob is an env var (nothing hard-coded); see the config block below.
Run it as a Ray job just like rsi_rl.py (sampler-only, no trainer):

    RSI_CH_SEED=... RSI_CH_NUM_PROPOSE=2000 python -m twinkle_agentic.rsi.rsi_challenge
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
from typing import Any, Dict, List, Optional, Tuple

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ── config (all env; nothing hard-coded) ───────────────────────────────────
MODEL_ID = os.environ.get('RSI_CH_MODEL', 'ms://Qwen/Qwen3-4B')
TEMPLATE = os.environ.get('RSI_CH_TEMPLATE', 'Template')  # base text template for Qwen3-4B (text-only)
SAMPLER_GPUS = int(os.environ.get('RSI_CH_SAMPLER_GPUS', 4))

SEED_PATH = os.environ.get('RSI_CH_SEED', '')           # optional seed jsonl (empty = from scratch)
NUM_PROPOSE = int(os.environ.get('RSI_CH_NUM_PROPOSE', 2000))   # how many problems to attempt to create
PROPOSE_TEMP = float(os.environ.get('RSI_CH_PROPOSE_TEMP', 1.1))  # challenger temperature (higher = diverse)
PROPOSE_MAX_TOKENS = int(os.environ.get('RSI_CH_PROPOSE_MAX_TOKENS', 8192))  # raised: thinking+cross-domain is long
# Drop pathologically long problem statements (rambling / non-problems) before the solver
# stage, both for quality and so the solver input never exceeds the model context.
PROBLEM_MAX_CHARS = int(os.environ.get('RSI_CH_PROBLEM_MAX_CHARS', 4000))
MAX_MODEL_LEN = int(os.environ.get('RSI_CH_MAX_MODEL_LEN', 16384))

# Topic-keyword conditioning (from-scratch only): first brainstorm a pool of diverse
# coding topics at high temperature, then seed each proposal with a random keyword so
# the challenger stops collapsing onto a few archetypes (palindromes, brackets, ...).
# Topic-keyword conditioning (from-scratch only): keep a persistent, 3-category keyword
# bank on disk (algorithm / computer / non-computer). Each proposal is seeded with a
# cross-domain TRIPLE drawn WITHOUT replacement (keywords are consumed); when a category
# runs out we ask the model for more distinct ones, and if it can't, we recycle. Keywords
# whose problems the solver fails hardest are expanded into more same-domain topics.
KEYWORDS_N = int(os.environ.get('RSI_CH_KEYWORDS_N', 128))         # per-category target (0 = disable)
KEYWORD_GEN_CALLS = int(os.environ.get('RSI_CH_KEYWORD_GEN_CALLS', 8))  # sampler calls per generation batch
KEYWORD_TEMP = float(os.environ.get('RSI_CH_KEYWORD_TEMP', 1.3))   # high temp -> diverse keywords
KEYWORD_MAX_TOKENS = int(os.environ.get('RSI_CH_KEYWORD_MAX_TOKENS', 1024))
KEYWORD_DB = os.environ.get('RSI_CH_KEYWORD_DB', 'output/rsi/keywords.jsonl')  # persistent bank
KEYWORD_REFILL_TRIES = int(os.environ.get('RSI_CH_KEYWORD_REFILL_TRIES', 2))   # refill attempts before recycle
SINGLE_KW_PROB = float(os.environ.get('RSI_CH_SINGLE_KW_PROB', 0.1))  # chance a proposal uses 1 keyword, not a triple
# With a seed pool loaded, this is the chance a proposal gets a seed problem ON TOP of
# its keywords; the rest are keywords-only. Seeds are drawn WITH replacement, so a pool
# smaller than NUM_PROPOSE is fine. 1.0 reproduces the old seed-only behaviour when
# keywords are disabled (RSI_CH_KEYWORDS_N=0).
SEED_MIX_PROB = float(os.environ.get('RSI_CH_SEED_MIX_PROB', 0.5))
# Two-step seeded proposing (V4): a seeded proposal is produced by TWO sampler calls --
# first write a harder solution on top of the seed's reference code, then describe the
# problem that solution answers. Requires the drawn seed to have a non-empty `code` field
# AND keywords to be enabled (both prompts take a topic block, so RSI_CH_KEYWORDS_N=0
# silently keeps the single-call path). Keywords-only proposals and code-less seeds also
# keep the original single call. Costs one extra call per two-step proposal; the log line
# below reports how many proposals actually took it.
TWO_STEP = os.environ.get('RSI_CH_TWO_STEP', '1') == '1'

# Combination arity: 'triple' = one keyword per category (max diversity); 'mix' = random
# 1/2/3 categories per proposal (use with the audit dump to see which combos keep best).
COMBO_ARITY = os.environ.get('RSI_CH_COMBO_ARITY', 'triple').lower()
# Optional 'w1,w2,w3' sampling weights for arity 1/2/3 in 'mix' mode (empty = uniform).
# Diagnostic finding: keep-rate falls with arity (~17%/8%/2%), so favour 1-2 for yield.
ARITY_WEIGHTS = os.environ.get('RSI_CH_ARITY_WEIGHTS', '')
_ARITY_W: Optional[List[float]] = None
if ARITY_WEIGHTS:
    try:
        _ARITY_W = [float(x) for x in ARITY_WEIGHTS.split(',')]
    except ValueError:
        _ARITY_W = None
AUDIT_PATH = os.environ.get('RSI_CH_AUDIT', 'output/rsi/challenge_audit.jsonl')  # per-proposal outcome log
# Feedback: expand keywords whose problems the solver passed <= this many times (0 = all-fail).
LOW_PASS_EXPAND = int(os.environ.get('RSI_CH_LOW_PASS_EXPAND', 0))
EXPAND_PER_KW = int(os.environ.get('RSI_CH_EXPAND_PER_KW', 8))     # new topics per hard keyword
EXPAND_MAX_KWS = int(os.environ.get('RSI_CH_EXPAND_MAX_KWS', 32))  # cap hard keywords expanded per run

SOLVER_ROLLOUTS = int(os.environ.get('RSI_CH_SOLVER_ROLLOUTS', 8))    # N attempts per problem for difficulty
SOLVER_TEMP = float(os.environ.get('RSI_CH_SOLVER_TEMP', 1.0))
SOLVER_MAX_TOKENS = int(os.environ.get('RSI_CH_SOLVER_MAX_TOKENS', 2048))

KEEP_MIN_PASS = int(os.environ.get('RSI_CH_KEEP_MIN_PASS', 1))   # keep if pass in [MIN, N-KEEP_MAX_MARGIN]
# "drop all-pass / all-fail" == keep 0 < pass < N. Both bounds configurable.
KEEP_MAX_PASS_MARGIN = int(os.environ.get('RSI_CH_KEEP_MAX_MARGIN', 1))  # drop pass >= N - margin + 1

SANDBOX_TIMEOUT = int(os.environ.get('RSI_CH_SANDBOX_TIMEOUT', 30))
MAX_CHECKS = int(os.environ.get('RSI_CH_MAX_CHECKS', 6))         # asserts per problem cap
# Drop problems where every assert expects the SAME value: `return <that constant>`
# scores a perfect reward without reading the input, so the problem teaches nothing and
# actively rewards ignoring the task. Measured at 6.2% of kept problems on sp4_iter1.
DROP_CONSTANT_ANSWER = os.environ.get('RSI_CH_DROP_CONSTANT_ANSWER', '1') == '1'
SORT_BY_DIFFICULTY = os.environ.get('RSI_CH_SORT_BY_DIFFICULTY', '1') == '1'
# Cap how many kept problems to persist (0 = keep all). When set and exceeded,
# subsample EVENLY across the difficulty-sorted list so the stored set spans the
# whole difficulty range, not just the easiest end.
KEEP_TARGET = int(os.environ.get('RSI_CH_KEEP_TARGET', 0))
CH_SEED = int(os.environ.get('RSI_CH_RANDOM_SEED', 0))

OUT_FLOWS = os.environ.get('RSI_CH_OUT_FLOWS', 'output/rsi/challenge_flows.jsonl')
OUT_TESTS = os.environ.get('RSI_CH_OUT_TESTS', 'output/rsi/challenge_tests.jsonl')
DUMP_REJECTED = os.environ.get('RSI_CH_DUMP_REJECTED', 'output/rsi/challenge_rejected.jsonl')

CODE_SYSTEM = {'role': 'system', 'content': 'You are an expert Python programmer.'}

# ── challenger prompt (shown to the user for review; a brand-new prompt) ────
_MARK = '__RSI_GT__'   # sentinel isolating captured output from any banner/log

_CHALLENGER_SYS = (
    'You design self-contained Python coding problems for training another model.\n'
    'A good problem: (1) is solvable from its statement ALONE with no external files, '
    'network, images, or hidden context; (2) has ONE clear entry function; (3) is '
    'deterministic (same input -> same output), no randomness, no wall-clock, no threads; '
    '(4) is neither trivial nor impossible for a mid-size model.\n'
    'You will also write the reference solution. We will EXECUTE it to obtain the '
    'ground-truth outputs, so your solution must be correct and runnable as-is.\n'
    'Return ONLY one JSON object, no prose around it, with keys:\n'
    '  "problem":  the statement shown to the solver (describe the function name, its '
    'inputs and expected behavior; do NOT include the solution).\n'
    '  "solution": the reference implementation as plain Python source (no markdown fence).\n'
    '  "entry":    the entry function name.\n'
    '  "checks":   a list of 3-6 Python expressions calling the entry function on concrete '
    'inputs (e.g. "solve([1,2,3])"); each must be evaluable after running the solution. '
    'Do NOT write the expected value — we compute it by running your solution.'
)

_CHALLENGER_FROM_SCRATCH = (
    'Create ONE new Python coding problem now. Vary the topic freely '
    '(strings, arrays, math, greedy, DP, parsing, simulation ...).'
)

_CHALLENGER_FROM_SEED = (
    'Here is a seed problem. Create ONE NEW problem that is a meaningful VARIANT of it '
    '(change the twist, constraints, or data shape — not just renaming), keeping it '
    'self-contained and deterministic.\n\n[seed]\n{seed}'
)

# Keyword bank generation + keyword-conditioned proposing (diversity).
CATEGORIES = ('algorithm', 'computer', 'noncs')
_CATEGORY_DESC = {
    'algorithm': 'algorithmic techniques and paradigms (e.g. dynamic programming, binary '
                 'search, union-find, Dijkstra, backtracking, segment trees, greedy, '
                 'divide and conquer, sliding window ...)',
    'computer': 'computer-science / computing concepts that are NOT algorithms per se '
                '(e.g. hash maps, tries, LRU cache, bitsets, regular expressions, base '
                'conversion, finite state machines, serialization, parsing, memoization ...)',
    'noncs': 'real-world domains OUTSIDE computer science, used to give a problem flavor '
             '(e.g. biology, finance, chemistry, logistics, music, cooking, sports, '
             'astronomy, geography, linguistics ...)',
}
_KEYWORD_SYS = (
    'You brainstorm diverse topics for a Python coding-problem generator.'
)
_KEYWORD_CAT_USER = (
    'List {k} DISTINCT and SPECIFIC topics from this category: {desc}\n'
    'Be creative and concrete; avoid vague umbrella words. '
    'Return ONLY a JSON array of short strings, nothing else.'
)
_KEYWORD_EXPAND_USER = (
    'The topic "{kw}" turned out to seed genuinely HARD problems. List {m} MORE distinct, '
    'specific topics in the SAME family/domain as "{kw}" that could seed similarly '
    'challenging Python problems. Return ONLY a JSON array of short strings, nothing else.'
)
_CHALLENGER_FROM_KEYWORDS = (
    'Create ONE new Python coding problem now. Draw inspiration from the following '
    'topic(s) and combine them creatively into a single coherent problem:\n{keywords}\n'
    'You may use each topic directly or bend it loosely; combine with any data shape '
    '(strings, arrays, grids, trees, numbers, parsing, simulation ...). Make it require '
    'real thought, not a one-liner, and keep it self-contained and deterministic.'
)
# Seed AND keywords together. The seed is deliberately framed as inspiration only,
# not as something to produce a variant of: the point is to pull the generated
# problems toward the shape of public benchmark items (short statement, one plain
# task) while the keywords keep supplying topical variety.
_CHALLENGER_FROM_SEED_KEYWORDS = (
    'Create ONE new Python coding problem now. Use the problem below only as a '
    'STARTING POINT for inspiration — you do NOT have to keep its task, and the new '
    'problem does NOT need to be a variant of it.\n\n[inspiration]\n{seed}\n\n'
    'Also draw on the following topic(s), combining them into a single coherent '
    'problem:\n{keywords}\n'
    'Make it require real thought, not a one-liner, and keep it self-contained and '
    'deterministic.'
)

# ── two-step (V4) challenger: build the problem FROM a harder solution ───────
# Winning variant of the prompt bake-off (output/rsi/prompt_exp_mbpp_upgrade.py).
# The difficulty comes from adding a layer on top of a real, runnable reference
# solution, not from imagining a hard problem outright; splitting into two calls (write
# the harder code, THEN describe it) keeps the statement and the ground-truth solution
# consistent, which a single call does not. Measured on 40 MBPP seeds vs the single-call
# seed+keywords prompt: kept-rate 25% vs 15%, const-answer 4 vs 7, seed similarity 0.42.
#
# Stage 1: given the seed problem + its reference solution + topics, write a harder
# solution. Uses the plain code system prompt (CODE_SYSTEM), not _CHALLENGER_SYS, and
# returns raw code (extract_code parses it) rather than JSON.
_TWO_STEP_SOL = (
    'Below is a coding problem and its reference solution.\n\n'
    '[problem]\n{seed}\n\n[reference solution]\n{code}\n\n'
    'Write a MORE COMPLEX Python function that keeps the idea of the reference solution '
    'as one step and builds a harder computation around it (extra pass, different data '
    'structure, an added rule), in the direction of these topic(s):\n{keywords}\n'
    'Requirements: deterministic, self-contained, no randomness, no I/O, one clear entry '
    'function. Output ONLY the code in a single ```python block, no explanation.'
)
# Stage 2: given the harder function PLUS the seed it grew from and the topics, describe
# the problem it answers. Seeing the seed pulls the wording back toward the MBPP task
# family (similarity 0.32 -> 0.42 when the seed is shown). The solution is NOT taken from
# this JSON -- we overwrite it with stage 1's code so the GT matches what was produced.
_TWO_STEP_PROB = (
    'Here is a Python function.\n\n```python\n{code}\n```\n\n'
    'It was written as a harder follow-up to this exercise:\n\n[original exercise]\n'
    '{seed}\n\nand it was pushed in the direction of these topic(s):\n{keywords}\n\n'
    'Write the problem statement that the function above is the answer to, as if it were '
    'a coding exercise in the same series as the original: name the entry function, '
    'describe its inputs and the exact behaviour expected, and do NOT reveal the '
    'implementation. Phrase it as plainly and briefly as the original exercise.\n'
    'Return ONLY one JSON object, no prose around it, with keys:\n'
    '  "problem":  the statement shown to the solver.\n'
    '  "entry":    the entry function name.\n'
    '  "checks":   a list of 3-6 Python expressions calling the entry function on '
    'concrete inputs; each must be evaluable after running the function above. Do NOT '
    'write the expected value.\n'
    'The "solution" is already known, so do not include it.'
)



_SOLVER_SYS = {'role': 'system', 'content': 'You are an expert Python programmer.'}
_SOLVER_USER = (
    '{problem}\n\n'
    'Write the complete Python solution. Put the final code in a single ```python fenced '
    'block. Define the exact function name required by the problem.'
)


# ── sandbox (mirrors rsi_rl.run_asserts / extract_code; duplicated on purpose:
#    importing rsi_rl would run its module-level CLI.from_args() + swanlab.init) ─
_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)


def extract_code(text: str) -> str:
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


def run_asserts(code: str, setup: str, asserts: List[str], timeout: int = SANDBOX_TIMEOUT) -> bool:
    """True when every assert passes (returncode 0). Same contract as rsi_rl."""
    if not code.strip() or not asserts:
        return False
    parts = [code]
    if (setup or '').strip():
        parts.append(setup)
    parts.extend(asserts)
    rc, _ = _run_script('\n\n'.join(parts), timeout)
    return rc == 0


def build_asserts(solution: str, checks: List[str], timeout: int = SANDBOX_TIMEOUT) -> Optional[List[str]]:
    """Run the reference solution once to capture repr of each check expression,
    then form ``assert <check> == <captured>``. Sentinel-marked + returncode
    checked so a crash or a banner line can never be read as a value.

    Returns the assert list, or None if the solution crashed / produced no usable
    output (that problem is then dropped upstream).
    """
    checks = [c for c in checks if isinstance(c, str) and c.strip()][:MAX_CHECKS]
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
    asserts = []
    for i, c in enumerate(checks):
        val = captured[i]
        # The captured text is a repr, so it is a valid literal to compare against.
        asserts.append(f'assert ({c}) == ({val})')
    return asserts


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

    build_asserts emits ``assert (<check>) == (<repr>)``, but a check may itself be a
    comparison, giving ``assert (f(x) == 3) == (True)``. Reading the outer side there
    would report 'True' and make such a problem look constant-answer, so the inner
    right-hand side is used instead. An outer ``False`` pins nothing down at all and
    is reported as unknown.
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

    Requires at least two asserts with a readable expectation: a single assert is
    trivially 'constant' and one unreadable assert must not hide a constant set.
    """
    vals = [_expected_of(a) for a in asserts]
    if any(v is None for v in vals) or len(vals) < 2:
        return False
    return len(set(vals)) == 1


# ── challenger output parsing ──────────────────────────────────────────────
_JSON_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.I)


def parse_challenger(text: str, require_solution: bool = True) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of the challenger's completion.

    ``require_solution=False`` is for the two-step (V4) flow, where the harder solution
    comes from a separate call and stage 2 returns only ``problem``/``entry``/``checks``.
    """
    body = text
    idx = body.rfind('</think>')
    if idx >= 0:
        body = body[idx + len('</think>'):]
    body = _JSON_FENCE_RE.sub('', body.strip()).strip()
    # Grab the outermost {...} if there is leading/trailing prose.
    start = body.find('{')
    end = body.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(body[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    problem = obj.get('problem')
    solution = obj.get('solution')
    checks = obj.get('checks')
    if not (isinstance(problem, str) and problem.strip()
            and isinstance(checks, list) and checks):
        return None
    if require_solution:
        if not (isinstance(solution, str) and solution.strip()):
            return None
    else:
        # Stage 2 is told NOT to include a solution; if it did anyway, ignore it -- we
        # will overwrite with stage 1's code so the GT matches what actually ran.
        solution = solution if isinstance(solution, str) else ''
    # solution may still arrive fenced despite instructions.
    if solution and '```' in solution:
        solution = extract_code(solution)
    return {'problem': problem.strip(), 'solution': (solution or '').strip(),
            'entry': str(obj.get('entry') or '').strip(), 'checks': checks}


# ── sampling helpers ───────────────────────────────────────────────────────
def _completion_text(seq) -> str:
    """The assistant text of one sampled sequence (decode is fine: not training)."""
    if seq.decoded:
        return seq.decoded
    feat = seq.new_input_feature or {}
    for m in reversed(feat.get('messages', []) or []):
        if m.get('role') == 'assistant':
            return m.get('content', '') or ''
    return ''


def sample_texts(sampler, message_lists: List[List[Dict[str, Any]]],
                 sampling_params: SamplingParams) -> List[str]:
    """Sample one completion per message list; return the assistant texts."""
    if not message_lists:
        return []
    trajs = [{'messages': msgs} for msgs in message_lists]
    responses = sampler.sample(trajs, sampling_params)
    texts: List[str] = []
    for resp in responses:
        seqs = resp.sequences
        texts.append(_completion_text(seqs[0]) if seqs else '')
    return texts


def _parse_keyword_list(text: str) -> List[str]:
    """Extract a JSON array of short strings from a (possibly thinking) reply."""
    body = text
    idx = body.rfind('</think>')
    if idx >= 0:
        body = body[idx + len('</think>'):]
    start, end = body.find('['), body.rfind(']')
    if start < 0 or end <= start:
        return []
    try:
        arr = json.loads(body[start:end + 1])
    except (ValueError, TypeError):
        return []
    out: List[str] = []
    for x in arr:
        if isinstance(x, str) and x.strip() and len(x.strip()) <= 60:
            out.append(x.strip())
    return out


class KeywordStore:
    """Persistent 3-category keyword bank with usage tracking (see CATEGORIES).

    On-disk format (KEYWORD_DB, one JSON per line):
        {"category", "text", "used": bool, "used_count": int, "source": "gen"|"expand",
         "parent": <keyword or null>}
    De-duplicates case-insensitively within each category so re-runs never conflict.
    """

    def __init__(self, path: str):
        self.path = path
        self.items: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORIES}
        self._seen: Dict[str, set] = {c: set() for c in CATEGORIES}
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
            for c in CATEGORIES:
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

    def recycle(self, category: str) -> None:
        """Mark every keyword unused again (safety valve when the model is tapped out)."""
        for r in self.items[category]:
            r['used'] = False


def generate_category_keywords(sampler, category: str, n_want: int,
                               avoid: List[str], rng, nonce: int = 0) -> List[str]:
    """Ask the model for up to ``n_want`` distinct keywords in ``category``.

    ``avoid`` (already-known texts) is both injected as a soft "don't repeat" hint and
    used to filter the parsed result. ``nonce`` perturbs the prompt so refills differ.
    """
    if n_want <= 0:
        return []
    n_calls = max(KEYWORD_GEN_CALLS, SAMPLER_GPUS)  # batch must cover every DP worker
    per_call = max(1, -(-n_want // n_calls) + 4)    # ceil(n/calls) + margin
    avoid_note = ''
    if avoid:
        shown = avoid if len(avoid) <= 40 else rng.sample(avoid, 40)
        avoid_note = '\nDo NOT repeat any of these already-used topics: ' + ', '.join(shown)
    base = _KEYWORD_CAT_USER.format(k=per_call, desc=_CATEGORY_DESC[category]) + avoid_note
    msgs = [[{'role': 'system', 'content': _KEYWORD_SYS},
             {'role': 'user', 'content': f'{base}\n(batch {nonce}-{i})'}]
            for i in range(n_calls)]
    sp = SamplingParams(max_tokens=KEYWORD_MAX_TOKENS, num_samples=1, logprobs=1,
                        temperature=KEYWORD_TEMP, top_p=0.98)
    texts = sample_texts(sampler, msgs, sp)
    out: List[str] = []
    seen = {a.strip().lower() for a in avoid}
    for t in texts:
        for kw in _parse_keyword_list(t):
            key = kw.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(kw.strip())
    rng.shuffle(out)
    return out[:n_want]


def ensure_unused(store: 'KeywordStore', sampler, category: str, need: int, rng,
                  nonce: int) -> int:
    """Make ``category`` hold >= ``need`` unused keywords, generating/recycling as needed.

    Returns the next free ``nonce`` to use for the following generation call.
    """
    tries = 0
    while len(store.unused(category)) < need:
        new = generate_category_keywords(sampler, category, KEYWORDS_N,
                                         store.texts(category), rng, nonce=nonce)
        added = store.add(category, new, source='gen')
        nonce += 1
        tries += 1
        if added == 0 and tries >= KEYWORD_REFILL_TRIES:
            # Model is out of fresh distinct topics; recycle so combinations keep flowing.
            if store.items[category]:
                store.recycle(category)
                logger.info(f'[rsi_challenge] keyword category {category!r} exhausted -> recycled '
                            f'{len(store.items[category])} topics')
            break
    return nonce


def expand_hard_keywords(store: 'KeywordStore', sampler, hard, rng, nonce: int) -> int:
    """For each (category, keyword) the solver failed hardest, brainstorm same-domain
    topics and add them (source='expand') to the bank. Returns count added."""
    hard = hard[:EXPAND_MAX_KWS]
    if not hard or EXPAND_PER_KW <= 0:
        return 0
    # Batch must cover all DP workers; cycle the hard list if it is too short.
    reqs = list(hard)
    while len(reqs) < SAMPLER_GPUS:
        reqs.append(hard[len(reqs) % len(hard)])
    msgs = [[{'role': 'system', 'content': _KEYWORD_SYS},
             {'role': 'user', 'content': _KEYWORD_EXPAND_USER.format(kw=kw, m=EXPAND_PER_KW)
              + f'\n(batch {nonce}-{i})'}]
            for i, (_c, kw) in enumerate(reqs)]
    sp = SamplingParams(max_tokens=KEYWORD_MAX_TOKENS, num_samples=1, logprobs=1,
                        temperature=KEYWORD_TEMP, top_p=0.98)
    texts = sample_texts(sampler, msgs, sp)
    added = 0
    for (cat, kw), t in zip(reqs, texts):
        added += store.add(cat, _parse_keyword_list(t), source='expand', parent=kw)
    return added


def _draw_keywords(store: 'KeywordStore', sampler, rng, nonce: int) -> Tuple[List[Tuple[str, str]], int]:
    """Consume one keyword combination from the bank (arity per COMBO_ARITY).

    Split out of the propose loop so a proposal can carry keywords whether or not
    it also carries a seed problem.
    """
    if COMBO_ARITY == 'mix':
        if _ARITY_W and len(_ARITY_W) == len(CATEGORIES):
            k = rng.choices(range(1, len(CATEGORIES) + 1), weights=_ARITY_W)[0]
        else:
            k = rng.randint(1, len(CATEGORIES))
        cats = rng.sample(list(CATEGORIES), k)
    elif rng.random() < SINGLE_KW_PROB:
        cats = [rng.choice(CATEGORIES)]
    else:
        cats = list(CATEGORIES)
    picks: List[Tuple[str, str]] = []
    for c in cats:
        if not store.unused(c):
            nonce = ensure_unused(store, sampler, c, 1, rng, nonce)
        un = store.unused(c)
        if not un:
            continue
        r = rng.choice(un)
        r['used'] = True
        r['used_count'] = r.get('used_count', 0) + 1
        picks.append((c, r['text']))
    return picks, nonce


def load_seeds(path: str) -> List[Dict[str, str]]:
    """Read seed problems from a jsonl. Returns dicts with at least 'query'; may also
    have 'code' (reference solution) when the file was written by split_mbpp.py v2+.

    When the file only has 'query' (legacy format), the returned dicts have code=''.
    The two-step challenger requires 'code' to be non-empty; if all seeds lack code it
    falls back to the original single-step prompt automatically.
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


def main():
    rng = random.Random(CH_SEED)
    for p in (OUT_FLOWS, OUT_TESTS, DUMP_REJECTED):
        os.makedirs(os.path.dirname(os.path.abspath(p)) or '.', exist_ok=True)

    device_groups = [DeviceGroup(name='sampler', ranks=list(range(SAMPLER_GPUS)), device_type='GPU')]
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=SAMPLER_GPUS, groups=device_groups, lazy_collect=False)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={'gpu_memory_utilization': 0.8, 'max_model_len': MAX_MODEL_LEN},
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE, model_id=MODEL_ID, enable_thinking=True, max_length=MAX_MODEL_LEN)

    seeds = load_seeds(SEED_PATH)
    logger.info(f'[rsi_challenge] seeds loaded: {len(seeds)} from {SEED_PATH!r} '
                f'(seed_mix_prob={SEED_MIX_PROB if seeds else 0.0}, '
                f'{"seed+keywords / keywords-only mix" if seeds else "keywords-only"})')

    # ── stage 1: challenger proposes, we execute to build GT asserts ────────
    # Diversity: each proposal draws a cross-domain keyword combination from a
    # persistent 3-category bank, consuming keywords without replacement. When a seed
    # pool is given, SEED_MIX_PROB of the proposals additionally get one seed problem
    # (drawn WITH replacement) as inspiration on top of the keywords -- the keyword
    # bank is now used in BOTH modes, where it used to be skipped entirely whenever
    # seeds were present.
    store = KeywordStore(KEYWORD_DB) if KEYWORDS_N > 0 else None
    nonce = int(CH_SEED)
    if store is not None and KEYWORDS_N > 0:
        for c in CATEGORIES:
            nonce = ensure_unused(store, sampler, c, 1, rng, nonce)
        logger.info('[rsi_challenge] keyword bank: '
                    + ', '.join(f'{c}={len(store.items[c])}({len(store.unused(c))} free)'
                                for c in CATEGORIES))

    # Build the FIRST-call message for every proposal. A seeded proposal that (a) drew a
    # seed carrying reference code and (b) has TWO_STEP on becomes a two-step proposal:
    # its first call writes a HARDER solution (raw code), and a second call -- built once
    # we see that code -- turns it into a problem statement + checks. Everything else is a
    # single JSON-producing call, exactly as before.
    propose_msgs: List[List[Dict[str, Any]]] = []
    propose_kws: List[List[Tuple[str, str]]] = []  # (category, text) picked per proposal, for feedback
    propose_seeded: List[bool] = []                # whether an MBPP seed rode along, for the audit
    is_two_step: List[bool] = []                   # whether this proposal uses the V4 two-call flow
    seed_query: List[str] = []                     # seed statement carried into stage 2 (two-step only)
    kw_body: List[str] = []                        # keyword block carried into stage 2 (two-step only)
    n_seeded = 0
    n_two = 0
    for _ in range(NUM_PROPOSE):
        picks: List[Tuple[str, str]] = []
        if store is not None:
            picks, nonce = _draw_keywords(store, sampler, rng, nonce)
        body = '\n'.join(f'- {c}: {t}' for c, t in picks)
        use_seed = bool(seeds) and rng.random() < SEED_MIX_PROB
        seed = rng.choice(seeds) if use_seed else None
        two = bool(use_seed and TWO_STEP and picks and seed and seed.get('code'))
        if use_seed:
            n_seeded += 1
        if two:
            n_two += 1
            sys_msg = dict(CODE_SYSTEM)
            user = _TWO_STEP_SOL.format(seed=seed['query'], code=seed['code'], keywords=body)
        elif use_seed and picks:
            sys_msg = {'role': 'system', 'content': _CHALLENGER_SYS}
            user = _CHALLENGER_FROM_SEED_KEYWORDS.format(seed=seed['query'], keywords=body)
        elif use_seed:
            sys_msg = {'role': 'system', 'content': _CHALLENGER_SYS}
            user = _CHALLENGER_FROM_SEED.format(seed=seed['query'])
        elif picks:
            sys_msg = {'role': 'system', 'content': _CHALLENGER_SYS}
            user = _CHALLENGER_FROM_KEYWORDS.format(keywords=body)
        else:
            sys_msg = {'role': 'system', 'content': _CHALLENGER_SYS}
            user = _CHALLENGER_FROM_SCRATCH
        propose_kws.append(picks)
        propose_seeded.append(use_seed)
        is_two_step.append(two)
        seed_query.append(seed['query'] if two else '')
        kw_body.append(body if two else '')
        propose_msgs.append([sys_msg, {'role': 'user', 'content': user}])
    logger.info(f'[rsi_challenge] proposals={NUM_PROPOSE} seeded={n_seeded} two_step={n_two} '
                f'keywords_only={NUM_PROPOSE - n_seeded} (RSI_CH_SEED_MIX_PROB={SEED_MIX_PROB}, '
                f'RSI_CH_TWO_STEP={int(TWO_STEP)})')
    propose_sp = SamplingParams(max_tokens=PROPOSE_MAX_TOKENS, num_samples=1, logprobs=1,
                                temperature=PROPOSE_TEMP, top_p=0.95)
    logger.info(f'[rsi_challenge] proposing {NUM_PROPOSE} problems (T={PROPOSE_TEMP})')
    stage1_text = sample_texts(sampler, propose_msgs, propose_sp)

    # objs[i] = the parsed proposal for index i (or None if it failed). For single-step
    # proposals this is just parse_challenger(stage1). For two-step, stage 1 gave code and
    # a second batched call turns each into problem+checks; we then FORCE solution = the
    # stage-1 code so the ground truth matches what was actually produced.
    # Driven by len(stage1_text), not NUM_PROPOSE, so a short sampler return cannot
    # IndexError here (the original enumerate-based loop was naturally tolerant).
    n_prop = len(stage1_text)
    if n_prop != NUM_PROPOSE:
        logger.warning(f'[rsi_challenge] sampler returned {n_prop} texts for '
                       f'{NUM_PROPOSE} proposals; proceeding with {n_prop}')
    objs: List[Optional[Dict[str, Any]]] = [None] * n_prop
    two_codes: List[str] = [''] * n_prop
    n_no_code = 0
    stage2_idx: List[int] = []
    stage2_msgs: List[List[Dict[str, Any]]] = []
    for i in range(n_prop):
        if not is_two_step[i]:
            objs[i] = parse_challenger(stage1_text[i])
            continue
        code = extract_code(stage1_text[i])
        two_codes[i] = code
        if not code.strip():
            # Stage 1 produced no code block (usually a truncated completion): this
            # proposal dies here. Counted separately from JSON parse failures.
            n_no_code += 1
            continue
        stage2_idx.append(i)
        stage2_msgs.append([{'role': 'system', 'content': _CHALLENGER_SYS},
                            {'role': 'user', 'content': _TWO_STEP_PROB.format(
                                code=code, seed=seed_query[i], keywords=kw_body[i])}])
    stage2_text_by_idx: Dict[int, str] = {}
    if stage2_msgs:
        logger.info(f'[rsi_challenge] two-step stage 2: {len(stage2_msgs)} problem writes '
                    f'({n_no_code} stage-1 completions had no code block)')
        for i, txt in zip(stage2_idx, sample_texts(sampler, stage2_msgs, propose_sp)):
            stage2_text_by_idx[i] = txt
            obj = parse_challenger(txt, require_solution=False)
            if obj is not None:
                obj['solution'] = two_codes[i]  # GT = the harder solution stage 1 produced
            objs[i] = obj

    # For the audit, keep the completion whose JSON we parsed (stage 2 for two-step, else
    # the single call) so resp_chars / think_closed describe the statement-producing call.
    proposals_text = [stage2_text_by_idx.get(i, stage1_text[i]) if is_two_step[i]
                      else stage1_text[i] for i in range(n_prop)]

    problems: List[Dict[str, Any]] = []
    stat = {'parsed': 0, 'parse_fail': 0, 'stage1_no_code': n_no_code, 'too_long': 0,
            'gt_fail': 0, 'selfcheck_fail': 0, 'constant_answer': 0}
    outcome: List[str] = ['parse_fail'] * len(proposals_text)  # per-proposal audit label
    for i in range(n_prop):
        if is_two_step[i] and not two_codes[i].strip():
            outcome[i] = 'stage1_no_code'
    rejected = open(DUMP_REJECTED, 'w', encoding='utf-8')
    for pi in range(len(proposals_text)):
        obj = objs[pi]
        if obj is None:
            if outcome[pi] != 'stage1_no_code':
                stat['parse_fail'] += 1
            continue
        stat['parsed'] += 1
        # Reject rambling / non-problem statements early (also keeps solver input in-context).
        if len(obj['problem']) > PROBLEM_MAX_CHARS:
            stat['too_long'] += 1
            outcome[pi] = 'too_long'
            continue
        asserts = build_asserts(obj['solution'], obj['checks'])
        if not asserts:
            stat['gt_fail'] += 1
            outcome[pi] = 'gt_fail'
            rejected.write(json.dumps({'reason': 'gt_build_fail', **obj}, ensure_ascii=False) + '\n')
            continue
        # The reference solution must pass its own asserts, or the GT is noise.
        if not run_asserts(obj['solution'], '', asserts):
            stat['selfcheck_fail'] += 1
            outcome[pi] = 'selfcheck_fail'
            rejected.write(json.dumps({'reason': 'selfcheck_fail', 'asserts': asserts, **obj},
                                      ensure_ascii=False) + '\n')
            continue
        # A problem whose every assert expects the same value rewards `return <constant>`,
        # so it would train the solver to ignore the statement. Drop it before the (much
        # more expensive) solver rollouts.
        if DROP_CONSTANT_ANSWER and is_constant_answer(asserts):
            stat['constant_answer'] += 1
            outcome[pi] = 'constant_answer'
            rejected.write(json.dumps({'reason': 'constant_answer', 'asserts': asserts, **obj},
                                      ensure_ascii=False) + '\n')
            continue
        obj['asserts'] = asserts
        obj['_kw'] = propose_kws[pi] if pi < len(propose_kws) else []  # origin keywords (feedback)
        obj['_idx'] = pi
        outcome[pi] = 'usable'
        problems.append(obj)
    logger.info(f'[rsi_challenge] proposal stage: {stat}, usable problems={len(problems)}')

    # ── stage 2: solver difficulty filter (keep 0 < pass < N) ───────────────
    kept: List[Dict[str, Any]] = []
    if problems:
        solver_msgs: List[List[Dict[str, Any]]] = []
        for prob in problems:
            for _ in range(SOLVER_ROLLOUTS):
                solver_msgs.append([_SOLVER_SYS,
                                    {'role': 'user', 'content': _SOLVER_USER.format(problem=prob['problem'])}])
        solver_sp = SamplingParams(max_tokens=SOLVER_MAX_TOKENS, num_samples=1, logprobs=1,
                                   temperature=SOLVER_TEMP, top_p=0.95)
        logger.info(f'[rsi_challenge] difficulty rollout: {len(problems)} problems x '
                    f'{SOLVER_ROLLOUTS} (T={SOLVER_TEMP})')
        solver_text = sample_texts(sampler, solver_msgs, solver_sp)

        hi = SOLVER_ROLLOUTS - KEEP_MAX_PASS_MARGIN
        for pi, prob in enumerate(problems):
            n_pass = 0
            for k in range(SOLVER_ROLLOUTS):
                code = extract_code(solver_text[pi * SOLVER_ROLLOUTS + k])
                if run_asserts(code, '', prob['asserts']):
                    n_pass += 1
            prob['n_pass'] = n_pass
            if KEEP_MIN_PASS <= n_pass <= hi:
                kept.append(prob)
                outcome[prob['_idx']] = f'kept(pass={n_pass})'
            else:
                outcome[prob['_idx']] = f'dropped(pass={n_pass})'
                rejected.write(json.dumps({'reason': f'difficulty_pass={n_pass}/{SOLVER_ROLLOUTS}',
                                           'problem': prob['problem']}, ensure_ascii=False) + '\n')
    rejected.close()

    # ── per-proposal audit: attribute outcome to the keyword combination + flag
    #    truncation (thinking never closed) so combo/domain effects can be measured.
    if AUDIT_PATH:
        with open(AUDIT_PATH, 'w', encoding='utf-8') as af:
            for i, txt in enumerate(proposals_text):
                kws = propose_kws[i] if i < len(propose_kws) else []
                af.write(json.dumps({
                    'idx': i,
                    'arity': len(kws),
                    'cats': [c for c, _ in kws],
                    'kws': kws,
                    # Whether this proposal also carried an MBPP seed statement, so the
                    # keep rate of seeded vs keywords-only proposals can be compared.
                    'seeded': bool(propose_seeded[i]) if i < len(propose_seeded) else False,
                    # Whether the V4 two-call flow was used (seed carried code + TWO_STEP).
                    'two_step': bool(is_two_step[i]) if i < len(is_two_step) else False,
                    'outcome': outcome[i],
                    'resp_chars': len(txt),
                    'think_closed': '</think>' in txt,
                }, ensure_ascii=False) + '\n')
        logger.info(f'[rsi_challenge] per-proposal audit -> {AUDIT_PATH}')

    # ── feedback: expand the keywords behind the hardest problems (solver pass
    #    <= LOW_PASS_EXPAND) into more same-domain topics, then persist the bank.
    if store is not None:
        hard: List[Tuple[str, str]] = []
        seen_hard = set()
        for prob in problems:
            if prob.get('n_pass', SOLVER_ROLLOUTS) <= LOW_PASS_EXPAND:
                for c, t in prob.get('_kw', []):
                    if (c, t.lower()) not in seen_hard:
                        seen_hard.add((c, t.lower()))
                        hard.append((c, t))
        if hard:
            rng.shuffle(hard)
            added = expand_hard_keywords(store, sampler, hard, rng, nonce)
            logger.info(f'[rsi_challenge] feedback: expanded {min(len(hard), EXPAND_MAX_KWS)} '
                        f'hard keyword(s) -> +{added} new same-domain topics')
        store.save()
        logger.info('[rsi_challenge] keyword bank saved: '
                    + ', '.join(f'{c}={len(store.items[c])}' for c in CATEGORIES)
                    + f' -> {KEYWORD_DB}')

    # File order = increasing difficulty (fewer solver passes later), which the
    # fixed-pool validation in rsi_rl relies on.
    if SORT_BY_DIFFICULTY:
        kept.sort(key=lambda p: -p['n_pass'])

    # Optional even subsample to KEEP_TARGET across the difficulty-sorted list.
    if KEEP_TARGET and len(kept) > KEEP_TARGET:
        n = len(kept)
        idx = sorted({round(i * (n - 1) / (KEEP_TARGET - 1)) for i in range(KEEP_TARGET)})
        kept = [kept[j] for j in idx]
        logger.info(f'[rsi_challenge] capped to KEEP_TARGET={KEEP_TARGET} '
                    f'(evenly across difficulty), stored={len(kept)}')

    # ── write flows + tests for rsi_rl (skip prepare/refine) ────────────────
    with open(OUT_FLOWS, 'w', encoding='utf-8') as ff, open(OUT_TESTS, 'w', encoding='utf-8') as ft:
        for i, prob in enumerate(kept):
            cid = f'ch_{i:06d}'
            flow = {
                'id': cid,
                'system': CODE_SYSTEM,
                'query': {'role': 'user', 'content': prob['problem']},
                'tools': [],
                # Difficulty audit: how many of the SOLVER_ROLLOUTS attempts passed
                # (0 < n_pass < N by construction). rsi_rl ignores these extra keys;
                # kept so a persisted flow can be analyzed without re-running.
                'n_pass': prob.get('n_pass'),
                'n_rollouts': SOLVER_ROLLOUTS,
                # Origin keywords (category, text) of the cross-domain triple, for audit.
                'keywords': prob.get('_kw', []),
                'rounds': [{
                    'intent': 'solve the problem',
                    'type': 'code',
                    'tool_call': None,
                    'code': prob['solution'],     # challenger's passing solution (OPSD reads this)
                    'result': '',
                    'reward_method': 'rubric',
                }],
            }
            ff.write(json.dumps(flow, ensure_ascii=False) + '\n')
            ft.write(json.dumps({'id': cid, 'test_list': prob['asserts'], 'test_setup_code': ''},
                                ensure_ascii=False) + '\n')

    logger.info(f'[rsi_challenge] kept {len(kept)}/{len(problems)} problems -> '
                f'{OUT_FLOWS} + {OUT_TESTS}')
    if kept:
        dist: Dict[int, int] = {}
        for p in kept:
            dist[p['n_pass']] = dist.get(p['n_pass'], 0) + 1
        logger.info(f'[rsi_challenge] kept pass-count distribution (0<pass<N): '
                    f'{dict(sorted(dist.items()))}')


if __name__ == '__main__':
    main()
