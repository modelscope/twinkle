# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI step 3 — full-parameter GRPO where each round of a standard flow becomes its
own training query.

Idea (confirmed): a multi-turn standard flow is decomposed into one training
query PER key round. For a tool round i the model is shown the fixed prior key
nodes (their tool calls + results) and must roll out {reasoning + the tool call}
for round i. The reward is whether the GENERATED tool call matches the recorded
standard call — name exact + every standard-call argument key/value present in
the generated call (extra args / order ignored). No sandbox is needed because
the standard call is the reference answer. Only the reasoning ("思路") varies
across rollouts; the key node is the target.

Rounds trained: TOOL rounds are rewarded either by matching the recorded standard
call (RSI_TOOL_REWARD=match, the default) or by asking a judge model whether the
generated call means the same thing as the recorded one (RSI_TOOL_REWARD=rubric).
CODE rounds are rewarded by EXECUTION when the flow carries tests (RSI_TESTS,
keyed by record id): the generated code runs against those asserts and scores
1.0 only if all pass.

The rubric judge's system prompt and score parsing below are a verbatim copy of
output/rsi/rubric_judge.py, which was used to measure offline that rubric scoring
raises the share of groups with a non-zero advantage from 6.2% to 15.0% on
ToolACE. Keep the two in sync or that number no longer describes this run.

RL data-flow discipline (verified): train ONLY on ``sequence.new_input_feature``
and use ``sequence.logprobs`` as old_logps — never decode-then-re-encode. The
generated tool call is already parsed into ``new_input_feature['messages'][-1]``
by the template, and the reference call rides along in ``user_data``.

Structure mirrors cookbook/rl/grpo/short_math_grpo.py, but FULL-PARAMETER: no
adapter is added, so all weights are trained and CheckpointEngineManager ships the
whole model to vLLM each step. RSI-specific paths come from env vars so the standard
CLI (model/infra/rl knobs) stays identical to the reference:

    RSI_STD_FLOWS   standard_flows.jsonl from rsi_refine.py (default output/rsi/standard_flows.jsonl)
    RSI_TEMPLATE    template name, must match the model (default Template, for text-only Qwen3-4B)
    RSI_TESTS       jsonl with {id, test_list, test_setup_code} to score code rounds
                    by execution (empty = code rounds are not trained)
    RSI_TOOL_REWARD 'match' (default) or 'rubric' for tool rounds
    RSI_JUDGE_MODEL judge model name for rubric scoring (default qwen3.8-max); the
                    endpoint and key come from LLM_BACKUP_BASE_URL / LLM_BACKUP_API_KEY
    RSI_MAX_ROUNDS  keep only the first N trainable rounds (0 = all), in file order
    RSI_RUN_NAME    swanlab experiment name
    RSI_REWARD_DUMP path to append a per-sample reward audit jsonl (off when unset)

Solver learning mode (RSI step-3 subclass, RSI_SOLVER_MODE):
  * 'grpo' (default) -- on a code round whose first attempt FAILS the asserts, the
    sandbox error is injected back as a {'role':'tool'} message and the model is
    asked to continue, up to RSI_SOLVER_MAX_TURNS total turns. The whole
    multi-turn trajectory (turn-1 tokens + turn-2 tokens, the tool error bridged
    in as -100) is trained by GRPO on the final pass/fail reward. Tool rounds and
    length-stopped rollouts stay single-shot. Bridge tokens are computed in
    template space and appended verbatim -- never decode-then-re-encode.
  * 'opsd' -- single turn. A teacher forward conditioned on a PRIVILEGED extra
    system message carrying the challenger's passing reference solution
    (RSI_OPSD_TEACHER_SYS) scores the SAME student response tokens; the per-token
    teacher log-probs pull the student via OPSDLoss (no advantages, no reward).
    Teacher log-probs come from model.forward_only on the trainer (same engine +
    same weights as the student, so r = teacher - student reflects only the prompt
    context, not vLLM<->Megatron skew). The response-only extraction is
    self-checked on the first batch against the sampler's old_logps.

    RSI_SOLVER_MODE      'grpo' (default) or 'opsd'
    RSI_SOLVER_MAX_TURNS GRPO: max total turns per code rollout (default 2)
    RSI_OPSD_TEACHER_SYS OPSD: teacher-only system template, '{solution}' filled
    RSI_OPSD_REVERSE     OPSD: k3 direction (1 = KL(student||teacher), default)
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
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.processor import InputProcessor
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler

logger = get_logger()
args = CLI.from_args()

# ── RSI-specific paths (env) ───────────────────────────────────────────────
STD_FLOWS = os.environ.get('RSI_STD_FLOWS', 'output/rsi/standard_flows.jsonl')
TEMPLATE = os.environ.get('RSI_TEMPLATE', 'Template')  # base text template for Qwen3-4B (text-only)
REWARD_TOOL_RESULT = 'tool_result'  # matches rsi_refine.attach_reward_method
REWARD_RUBRIC = 'rubric'
# Tests for code rounds, keyed by the flow's id (rsi_refine passes the id through
# from the step-1 subset, which carries the dataset's own tests).
TESTS_PATH = os.environ.get('RSI_TESTS', '')
TEST_TIMEOUT = int(os.environ.get('RSI_TEST_TIMEOUT', 30))
JUDGE_WORKERS = int(os.environ.get('RSI_JUDGE_WORKERS', max(24, min(96, (os.cpu_count() or 24) // 2))))
# Tool-round scoring: 'match' compares the call literally, 'rubric' asks a judge
# model whether it means the same as the recorded call.
TOOL_REWARD = os.environ.get('RSI_TOOL_REWARD', 'match')
JUDGE_MODEL = os.environ.get('RSI_JUDGE_MODEL', 'qwen3.8-max')
JUDGE_BASE_URL = os.environ.get('LLM_BACKUP_BASE_URL', '')
JUDGE_API_KEY = os.environ.get('LLM_BACKUP_API_KEY', '')
# 16 workers is what the offline judging run used; each step needs one call per
# rollout that actually produced a tool call.
RUBRIC_WORKERS = int(os.environ.get('RSI_RUBRIC_WORKERS', 16))
RUBRIC_RETRIES = int(os.environ.get('RSI_RUBRIC_RETRIES', 3))
# Keep only the first N trainable rounds, in file order (no shuffle anywhere).
MAX_ROUNDS = int(os.environ.get('RSI_MAX_ROUNDS', 0))
RUN_NAME = os.environ.get('RSI_RUN_NAME', '')
# Optional per-sample reward audit: when set, every scored rollout is appended as
# one jsonl line (step, kind, ref/gen call, completion head, score, judge reason).
# Pure observability; the reward and training path are untouched.
REWARD_DUMP = os.environ.get('RSI_REWARD_DUMP', '')
# Raw step-1 conversations (before rsi_refine). rsi_refine's flow schema keeps
# only the FIRST user message (as `query`) plus the tool rounds, so any parameter
# the user stated in a LATER user turn is missing from a round's prompt and the
# model is asked to produce a call it cannot possibly know. When this points at
# the raw file, each round's prompt is rebuilt to splice those dropped user (and
# assistant clarification) turns back in, joined to the raw conversation by the
# first user message (unique for ~99.6% of flows); flows that cannot be joined
# fall back to the flow-only prompt. Empty reproduces the old flow-only behavior.
RAW_MESSAGES = os.environ.get('RSI_RAW_MESSAGES', '')
# Diagnostic knobs for "does reward rise on a FIXED distribution". Training reward
# on the default sequential single-epoch feed cannot answer that: the flow file is
# ordered so later rounds carry more arguments (harder), so a falling curve mixes
# difficulty with capability. Set both to hold the distribution still:
#   RSI_SHUFFLE_SEED  shuffle the rounds once with this seed (empty = file order)
#   RSI_POOL_SIZE     keep only this many rounds and repeat them until MAX_ROUNDS
#                     is filled, re-shuffling each pass (0 = no repetition)
# With a pool the same questions are seen every pass, so reward must climb unless
# the optimizer itself is at fault.
SHUFFLE_SEED = os.environ.get('RSI_SHUFFLE_SEED', '')
POOL_SIZE = int(os.environ.get('RSI_POOL_SIZE', 0))

# ── solver learning mode (RSI step-3 subclass) ─────────────────────────────
SOLVER_MODE = os.environ.get('RSI_SOLVER_MODE', 'grpo').lower()
if SOLVER_MODE not in ('grpo', 'opsd'):
    raise ValueError(f"RSI_SOLVER_MODE must be 'grpo' or 'opsd', got {SOLVER_MODE!r}")
# GRPO subclass: on a failed code round, inject the sandbox error as a tool
# message and let the model retry until it passes or the turn budget is spent.
SOLVER_MAX_TURNS = int(os.environ.get('RSI_SOLVER_MAX_TURNS', 2))
# OPSD subclass: the teacher sees one extra system message carrying the
# challenger's passing reference solution ('{solution}' is filled per sample).
OPSD_TEACHER_SYS = os.environ.get(
    'RSI_OPSD_TEACHER_SYS',
    'A correct reference solution is provided to guide you:\n'
    '```python\n{solution}\n```\n'
    'Study it, then produce your own complete, correct solution to the task above.')
# k3 direction guard, exposed so the divergence sign can be swapped without
# touching call sites (see opsd.py `reverse`).
OPSD_REVERSE = os.environ.get('RSI_OPSD_REVERSE', '1') == '1'
# Tolerance for the OPSD first-batch self-check that pins the response-logps
# frame against the sampler's known-correct old_logps (mean abs diff per token).
OPSD_SELFCHECK_TOL = float(os.environ.get('RSI_OPSD_SELFCHECK_TOL', 0.5))

# ── standard CLI knobs (same shape as the reference script) ────────────────
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.6-35B-A3B'
MODEL_GPUS = args.infra.model_gpus or 4
SAMPLER_GPUS = args.infra.sampler_gpus or 4
# KL anchor. GRPOLoss adds beta * KL(pi || ref) per response token only when BOTH
# beta > 0 AND ref_logps are passed to forward_backward (grpo.py:315), so leaving
# either at its default silently trains without any anchor.
#
# The anchor model gets its OWN gpus rather than sharing the trainer's: Megatron
# calls mpu.initialize_model_parallel() unconditionally per process
# (model/megatron/strategy/megatron.py:115), so constructing a second
# MegatronModel on the same ranks would re-init the process-global parallel
# state. REF_GPUS > 0 appends a separate device group.
KL_BETA = float(os.environ.get('RSI_KL_BETA', 0.0))
REF_GPUS = int(os.environ.get('RSI_REF_GPUS', 0))
# Stays the ORIGINAL base weights for every self-play iteration, while MODEL_ID
# advances to the previous iteration's checkpoint — that is the point of the
# anchor: it bounds the drift accumulated across iterations, not just within one.
REF_MODEL_ID = os.environ.get('RSI_REF_MODEL_ID', 'ms://Qwen/Qwen3-4B')
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS + REF_GPUS
# Which GRPO-family aggregation to use. 'GRPOLoss' normalizes each sequence by
# its OWN token count (grpo.py:132), which leaves a per-group residual
# sum_i(A_i / len_i); with group-centred advantages (sum_i A_i = 0) that residual
# is zero only if all lengths are equal, and it measured +0.16..+0.57 in
# equivalent-advantage terms across the 17 self-play iterations (passing rollouts
# were consistently the shorter ones), i.e. a standing push toward shorter output.
# 'DRGRPOLoss' divides by batch * max_completion_length, a constant, so the same
# residual becomes sum_i(A_i)/const == 0.
LOSS_NAME = os.environ.get('RSI_LOSS', 'GRPOLoss')
NUM_GENERATIONS = args.rl.num_generations or 8
MAX_NEW_TOKENS = args.sampling.max_tokens or 4096
LEARNING_RATE = args.optimizer.learning_rate or 5e-5
MAX_STEPS = args.training.max_steps or 1000
BATCH_SIZE = args.training.batch_size or 4
MINI_BATCH_SIZE = args.training.mini_batch_size or 4
MICRO_BATCH_SIZE = args.training.micro_batch_size or 1
GRADIENT_ACCUMULATION_STEPS = args.training.gradient_accumulation_steps or 1
SAVE_STEPS = args.training.save_steps or 1000
# Context window (prompt + generation). Raised above the old hard-coded 8192 so a
# larger --max-tokens cannot overflow the engine: RSI code prompts can be long,
# and on a policy collapse the model rambles to the token cap. Keep this >=
# max_tokens + longest prompt. Matches the challenger's MAX_MODEL_LEN default.
MAX_MODEL_LEN = int(os.environ.get('RSI_MAX_MODEL_LEN', 16384))
# Where the trained weights land. Kept per-iteration-configurable so a self-play
# loop can point each iteration's checkpoint at its own dir (and feed the final
# HF-format dir back to the next challenge/rl via --model-id / RSI_CH_MODEL).
SAVE_DIR = os.environ.get('RSI_SAVE_DIR', 'output')
SAVE_NAME = os.environ.get('RSI_SAVE_NAME', 'rsi-executor-final')

import swanlab
swanlab.init(project='twinkle-rsi', experiment_name=RUN_NAME or None)


# ── tool-call matching (name exact + standard-call arg subset) ─────────────
def _json_safe(o: Any) -> Any:
    """Coerce anything json.dumps cannot handle into a string, recursively.

    A malformed rollout can parse into arguments holding a bare ``...`` (Python
    Ellipsis) or other non-JSON objects; without this a single such sample makes
    json.dumps raise and takes the whole run down. Normal dicts/lists/scalars are
    returned unchanged, so well-formed calls serialize exactly as before.
    """
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(x) for x in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _as_args(a: Any) -> Dict[str, Any]:
    if isinstance(a, str):
        try:
            return json.loads(a)
        except (ValueError, TypeError):
            return {}
    return a or {}


def tool_call_matches(gen_call: Optional[Dict[str, Any]], ref_call: Dict[str, Any]) -> bool:
    """True iff name matches and every reference arg (key+value) is present."""
    if not gen_call or gen_call.get('name') != ref_call.get('name'):
        return False
    gen_args = _as_args(gen_call.get('arguments'))
    ref_args = _as_args(ref_call.get('arguments'))
    for k, v in ref_args.items():
        if k not in gen_args or gen_args[k] != v:
            return False
    return True


# ── rubric reward for tool rounds (judge model over the API) ───────────────
# Verbatim from output/rsi/rubric_judge.py, which produced the offline 6.2% -> 15.0%
# comparison; changing a word here means this run is no longer that measurement.
JUDGE_SYSTEM = """\
You are a strict tool-call equivalence judge. You will be given:
1. The user's request (what they asked for).
2. The STANDARD tool call (the reference answer: function name + arguments).
3. The MODEL's output (what the model actually produced).

Your job: decide whether the model's output is semantically equivalent to the standard call.

Rules:
- The model MUST have attempted a tool/function call. If it only gave a natural language
  answer without any call, score 0.
- Function name must match (case-insensitive, ignore spacing differences).
- Arguments: check SEMANTIC equivalence, not exact string match.
  * Search queries: "Oscars cinema drama" ≈ "Oscars newest cinema drama" (same intent) → OK
  * Numbers: "5" = 5 = 5.0 → OK
  * Coordinates/measurements that point to the same place or value → OK
  * Lists with same elements in different order → OK
  * Completely different values → NOT OK
- EXTRA arguments the model added that the standard call omits do NOT count against it,
  as long as they do not contradict the user's request. Spelling out an optional
  parameter at its default value (e.g. output="json" when json is the default) is
  fully equivalent to leaving it out → still score 1.0
- Only the arguments present in the STANDARD call have to be matched.
- If the function is correct and every standard argument is semantically equivalent → 1.0
- If the function is correct but a standard argument is partially wrong, or a required
  one is missing → 0.5
- If wrong function, no call at all, or standard arguments completely wrong → 0.0

Output ONLY a JSON object: {"score": <0.0 or 0.5 or 1.0>, "reason": "<one sentence>"}
Nothing else.
"""

_SCORE_RE = re.compile(r'"score"\s*:\s*([\d.]+)')
# The judge sometimes replies with a bare number or "Score: 0" instead of JSON. Offline
# every such reply was a genuine 0, so read it rather than throwing the sample away.
_BARE_SCORE_RE = re.compile(r'(?:score\D{0,12})?\b(0(?:\.0)?|0\.5|1(?:\.0)?)\b', re.I)

_judge_client = None


def judge_client():
    global _judge_client
    if _judge_client is None:
        from openai import OpenAI
        if not JUDGE_API_KEY:
            raise RuntimeError('RSI_TOOL_REWARD=rubric needs LLM_BACKUP_API_KEY '
                               '(and LLM_BACKUP_BASE_URL) in the environment')
        _judge_client = OpenAI(base_url=JUDGE_BASE_URL or None, api_key=JUDGE_API_KEY)
    return _judge_client


def judge_input(completion: str, gen_call: Dict[str, Any]) -> str:
    """What the judge sees as "the model's output".

    The template already lifted the call out of the raw text into a structured
    field, so the raw text cannot be recovered: the body the model wrote comes
    first, then the call that was parsed out of it.
    """
    call = {'name': gen_call.get('name'), 'arguments': _as_args(gen_call.get('arguments'))}
    return f'{completion}\n\n[parsed tool call]\n{json.dumps(_json_safe(call), ensure_ascii=False)}'


def judge_rubric(ref_call: Dict[str, Any], model_output: str) -> Tuple[Optional[float], Optional[str]]:
    """Ask the judge for one score; (None, reason) when it never answered.

    None is not zero: the caller drops the sample from its group instead of
    counting it as a miss, so a timeout cannot masquerade as a wrong answer.
    The second element is the judge's raw reply, kept only for the audit dump.
    """
    if len(model_output) > 3000:
        model_output = model_output[:1500] + '\n...[truncated]...\n' + model_output[-1500:]
    user_msg = (f'## Standard tool call (reference answer)\n```json\n'
                f'{json.dumps(ref_call, ensure_ascii=False)}\n```\n\n'
                f"## Model's full output\n```\n{model_output}\n```\n\nScore the model's output.")
    for attempt in range(RUBRIC_RETRIES):
        try:
            resp = judge_client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'system', 'content': JUDGE_SYSTEM},
                          {'role': 'user', 'content': user_msg}],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or ''
            m = (_SCORE_RE.search(text) or _BARE_SCORE_RE.match(text.strip())
                 or _BARE_SCORE_RE.search(text[:40]))
            return (float(m.group(1)) if m else None), text
        except Exception as e:  # network / rate limit / bad gateway
            if attempt == RUBRIC_RETRIES - 1:
                logger.warning(f'[rsi_rl] judge gave up after {RUBRIC_RETRIES} tries: {str(e)[:160]}')
                return None, f'error:{str(e)[:160]}'
            time.sleep(2**attempt)
    return None, None


class ToolMatchReward(Reward):
    """1.0 when the generated tool call matches the recorded standard call."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            gen_call = None
            for m in reversed(traj.get('messages', []) or []):
                if m.get('role') == 'assistant':
                    tcs = m.get('tool_calls') or []
                    if tcs:
                        gen_call = tcs[0].get('function')
                    break
            ref_call = None
            for item in (traj.get('user_data') or []):
                if item[0] == 'ref_tool_call':
                    try:
                        ref_call = json.loads(item[1])
                    except (ValueError, TypeError):
                        ref_call = None
                    break
            rewards.append(1.0 if (ref_call and tool_call_matches(gen_call, ref_call)) else 0.0)
        return rewards


# ── code-round execution reward ────────────────────────────────────────────
# Same sandbox contract as cookbook/rl/grpo/mbpp_grpo.py, which was checked
# against all 974 MBPP reference solutions (974/974 pass): the generated code,
# the setup code and the asserts are concatenated into one file and executed, so
# a bare ``assert fn(...) == x`` resolves the function by name.
_FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)


def extract_code(text: str) -> str:
    """Take the last fenced block; fall back to the whole body when unfenced."""
    idx = (text or '').rfind('</think>')
    body = text[idx + len('</think>'):] if idx >= 0 else (text or '')
    blocks = _FENCE_RE.findall(body)
    return (blocks[-1] if blocks else body).strip()


def run_asserts(code: str, setup: str, asserts: List[str], timeout: int = TEST_TIMEOUT) -> bool:
    """True when every assert passes. Thin wrapper over run_asserts_verbose."""
    return run_asserts_verbose(code, setup, asserts, timeout)[0]


def run_asserts_verbose(code: str, setup: str, asserts: List[str],
                        timeout: int = TEST_TIMEOUT) -> Tuple[bool, str]:
    """Run code+setup+asserts and return (passed, stderr_text).

    Same sandbox contract as the MBPP-verified path (start_new_session + killpg
    so a forking solution leaves no stray processes; RLIMIT_AS caps the child at
    2GB). stderr is captured (not sent to /dev/null) so the GRPO continuation can
    feed the actual traceback back to the model as a tool message. ``passed`` is
    exactly ``returncode == 0``, identical to the old bool-only behavior.
    """
    if not code.strip() or not asserts:
        return False, 'no code was produced'
    parts = [code]
    if (setup or '').strip():
        parts.append(setup)
    parts.extend(asserts)
    tmp = tempfile.mkdtemp(prefix='rsi_code_')
    try:
        with open(os.path.join(tmp, '_run.py'), 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(parts) + '\n')
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        env.pop('CUDA_VISIBLE_DEVICES', None)

        def _limit():
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))

        proc = subprocess.Popen([sys.executable, '_run.py'], cwd=tmp, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, start_new_session=True, preexec_fn=_limit)
        try:
            _, err = proc.communicate(timeout=timeout)
            return proc.returncode == 0, (err or '')
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            return False, f'execution timed out after {timeout}s (possible infinite loop)'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_tests() -> Dict[str, Dict[str, Any]]:
    """Read the tests file into {id: {asserts, setup}} (empty when not configured)."""
    if not TESTS_PATH or not os.path.exists(TESTS_PATH):
        return {}
    tests: Dict[str, Dict[str, Any]] = {}
    with open(TESTS_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            asserts = row.get('test_list') or []
            if isinstance(asserts, str):
                asserts = json.loads(asserts)
            if asserts:
                tests[str(row.get('id'))] = {'asserts': list(asserts),
                                             'setup': row.get('test_setup_code') or ''}
    return tests


class RoundReward(Reward):
    """Score each rollout by what its round is: tool call, or code execution.

    Which branch applies is carried per-sample in ``user_data``: a tool round
    rides ``ref_tool_call``, a code round rides ``code_tests``. Execution runs in
    a thread pool because every verdict is a separate subprocess; rubric judging
    runs in a thread pool because every verdict is a separate API call.

    A returned ``None`` means "never scored" (the judge never answered after its
    retries) and is not the same as 0.0 -- see ``group_advantages``.
    """

    def __init__(self):
        # Per-step counters, read by main() for logging.
        self.stats: Dict[str, int] = {}
        # Per-sample audit rows for the latest __call__, aligned by index with the
        # returned rewards. main() stamps them with the step and writes the dump.
        self.records: List[Dict[str, Any]] = []

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[Optional[float]]:
        rewards: List[Optional[float]] = [0.0] * len(trajectories)
        code_jobs: List[Tuple[int, str, Dict[str, Any]]] = []
        rubric_jobs: List[Tuple[int, Dict[str, Any], str]] = []
        n_no_call = 0
        # One audit row per sample; branch/score/reason filled in as we go.
        recs: List[Dict[str, Any]] = [{'kind': None, 'ref_call': None, 'gen_call': None,
                                       'completion': '', 'reason': None} for _ in trajectories]

        for i, traj in enumerate(trajectories):
            ud = {item[0]: item[1] for item in (traj.get('user_data') or [])}
            completion = ''
            gen_call = None
            for m in reversed(traj.get('messages', []) or []):
                if m.get('role') == 'assistant':
                    tcs = m.get('tool_calls') or []
                    if tcs:
                        gen_call = tcs[0].get('function')
                    completion = m.get('content', '') or ''
                    break
            # Full completion, never truncated: a cut tail once hid whether the
            # model emitted </think> / <tool_call>, which is exactly what the audit
            # must answer. Store the whole thing.
            recs[i]['completion'] = completion
            recs[i]['gen_call'] = gen_call

            if 'ref_tool_call' in ud:
                try:
                    ref_call = json.loads(ud['ref_tool_call'])
                except (ValueError, TypeError):
                    ref_call = None
                recs[i]['ref_call'] = ref_call
                if TOOL_REWARD != 'rubric':
                    rewards[i] = 1.0 if (ref_call and tool_call_matches(gen_call, ref_call)) else 0.0
                    recs[i]['kind'] = 'tool_match'
                elif not gen_call:
                    # No call was parsed out, so there is nothing for the judge to
                    # compare against: 0 without spending a request.
                    n_no_call += 1
                    recs[i]['kind'] = 'tool_no_call'
                elif ref_call:
                    rubric_jobs.append((i, ref_call, judge_input(completion, gen_call)))
                    recs[i]['kind'] = 'tool_rubric'
            elif 'code_tests' in ud:
                try:
                    spec = json.loads(ud['code_tests'])
                except (ValueError, TypeError):
                    continue
                code_jobs.append((i, extract_code(completion), spec))
                recs[i]['kind'] = 'code'

        if code_jobs:
            # Judge each distinct (task, code) once: identical completions are common.
            uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for _, code, spec in code_jobs:
                uniq.setdefault((str(spec.get('id')), code), spec)
            todo = list(uniq)
            with ThreadPoolExecutor(max_workers=max(1, min(JUDGE_WORKERS, len(todo)))) as ex:
                verdicts = dict(zip(todo, ex.map(
                    lambda k: run_asserts(k[1], uniq[k]['setup'], uniq[k]['asserts']), todo)))
            for i, code, spec in code_jobs:
                rewards[i] = 1.0 if verdicts.get((str(spec.get('id')), code)) else 0.0

        n_failed = 0
        if rubric_jobs:
            # One request per distinct (reference, output) pair; the judge runs at
            # temperature 0, so repeats would only cost money.
            uniq_r: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for _, ref_call, text in rubric_jobs:
                uniq_r.setdefault((json.dumps(ref_call, ensure_ascii=False), text), ref_call)
            keys = list(uniq_r)
            with ThreadPoolExecutor(max_workers=max(1, min(RUBRIC_WORKERS, len(keys)))) as ex:
                scores = dict(zip(keys, ex.map(lambda k: judge_rubric(uniq_r[k], k[1]), keys)))
            for i, ref_call, text in rubric_jobs:
                s, reason = scores.get((json.dumps(ref_call, ensure_ascii=False), text), (None, None))
                rewards[i] = s
                recs[i]['reason'] = reason
                if s is None:
                    n_failed += 1

        for i in range(len(trajectories)):
            recs[i]['score'] = rewards[i]
        self.records = recs
        self.stats = {'no_call': n_no_call, 'judged': len(rubric_jobs) - n_failed,
                      'judge_failed': n_failed, 'executed': len(code_jobs)}
        return rewards


def group_advantages(rewards: List[Optional[float]], num_generations: int,
                     scale: str = 'group') -> List[float]:
    """GRPOAdvantage, but unscored samples stay out of their group's statistics.

    Same formula as twinkle.advantage.GRPOAdvantage (subtract the group mean,
    divide by the group's unbiased std). A ``None`` reward means the judge never
    returned a verdict: it is left out of the mean and std and gets advantage 0,
    so it pushes the policy in neither direction. A group left with a single
    verdict has no baseline to compare against, so all of it gets 0.
    """
    if all(r is not None for r in rewards):
        return GRPOAdvantage()(rewards, num_generations=num_generations, scale=scale).tolist()
    import torch
    vals = torch.tensor([0.0 if r is None else r for r in rewards], dtype=torch.float32)
    mask = torch.tensor([r is not None for r in rewards], dtype=torch.float32)
    g_vals = vals.view(-1, num_generations)
    g_mask = mask.view(-1, num_generations)
    n = g_mask.sum(dim=1, keepdim=True)
    mean = (g_vals * g_mask).sum(dim=1, keepdim=True) / n.clamp(min=1)
    adv = (g_vals - mean) * g_mask
    if scale == 'group':
        var = ((g_vals - mean)**2 * g_mask).sum(dim=1, keepdim=True) / (n - 1).clamp(min=1)
        adv = adv / (var.sqrt() + 1e-8)
    elif scale == 'batch':
        adv = adv / (adv[g_mask.bool()].std() + 1e-8)
    return (adv * (n > 1).float()).view(-1).tolist()


# ── decompose standard flows into per-round training trajectories ──────────
def _openai_tool_call(call: Dict[str, Any], idx: int) -> Dict[str, Any]:
    args_ = call.get('arguments', {})
    return {
        'id': f'call_{idx}',
        'type': 'function',
        'function': {
            'name': call.get('name', ''),
            'arguments': json.dumps(args_, ensure_ascii=False) if isinstance(args_, dict) else str(args_),
        },
    }


def _render_prior_round(r: Dict[str, Any], idx: int) -> List[Dict[str, Any]]:
    """Render a completed prior round as fixed context messages."""
    result = r.get('result', '')
    if r.get('tool_call'):
        tc = _openai_tool_call(r['tool_call'], idx)
        return [
            {'role': 'assistant', 'content': '', 'tool_calls': [tc]},
            {'role': 'tool', 'content': str(result), 'tool_call_id': tc['id']},
        ]
    # Code round: no tool_call_id exists, so keep it template-agnostic.
    return [
        {'role': 'assistant', 'content': r.get('code', '') or ''},
        {'role': 'user', 'content': f'[execution result]\n{result}'},
    ]


def _load_raw_by_query(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Index the raw step-1 conversations by their first user message, the join
    key back to a flow's ``query``. Only keys that map to exactly ONE conversation
    are kept, so an ambiguous first question never pulls the wrong conversation.
    """
    if not path or not os.path.exists(path):
        return {}
    seen: Dict[str, List[Dict[str, Any]]] = {}
    dup: set = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get('messages') or []
            fu = next((str(m.get('content') or '').strip()
                       for m in msgs if m.get('role') == 'user'), '')
            if not fu:
                continue
            if fu in seen:
                dup.add(fu)
            else:
                seen[fu] = msgs
    for k in dup:
        seen.pop(k, None)
    return seen


def _locate_calls(raw_msgs: List[Dict[str, Any]],
                  rounds: List[Dict[str, Any]]) -> Optional[List[int]]:
    """For each round, the index of the raw assistant message that made its call,
    matched forward by the round's tool name (robust to ToolACE's several call
    syntaxes: ``Name(...)``, ``[Name]=>``, ``{Name}=>`` ...). A code/nameless
    round reuses the running cursor. Returns None if any named call is not found
    in order, so the caller falls back to the flow-only prompt.
    """
    locs: List[int] = []
    cur = 0
    for r in rounds:
        name = ((r.get('tool_call') or {}).get('name')) or ''
        if not name:
            locs.append(cur)
            continue
        found = -1
        for j in range(cur, len(raw_msgs)):
            m = raw_msgs[j]
            if m.get('role') == 'assistant' and name in (m.get('content') or ''):
                found = j
                break
        if found < 0:
            return None
        locs.append(found)
        cur = found  # a later parallel call may live in the same message
    return locs


def _intervening_turns(raw_msgs: List[Dict[str, Any]], lo: int, hi: int) -> List[Dict[str, Any]]:
    """User turns and assistant clarification turns in ``raw_msgs[lo+1:hi]``.

    Assistant tool-call messages (content starting with ``[``) and tool results
    are dropped here because the structured prior rounds already carry the call
    and its result; what is recovered is exactly the conversational turns
    rsi_refine did not keep.
    """
    out: List[Dict[str, Any]] = []
    for j in range(lo + 1, hi):
        m = raw_msgs[j]
        role = m.get('role')
        content = m.get('content') or ''
        if role == 'user':
            out.append({'role': 'user', 'content': content})
        elif role == 'assistant' and content.strip() and not content.lstrip().startswith('['):
            out.append({'role': 'assistant', 'content': content})
    return out


def build_round_trajectories(records: List[Dict[str, Any]],
                             tests: Optional[Dict[str, Dict[str, Any]]] = None,
                             raw_by_query: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                             recover_stats: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """One training trajectory per trainable round; prior rounds become fixed context.

    A tool round is trainable when it has a recorded call to match. A code round
    is trainable only when ``tests`` holds asserts for the record, since that is
    what its reward executes; otherwise it stays context-only.

    When ``raw_by_query`` is given, the round prompt is rebuilt from the raw
    conversation so the user turns that rsi_refine dropped (e.g. the turn that
    states the call's arguments) are spliced back in at their real positions;
    flows whose raw conversation cannot be located fall back to the flow-only
    prompt (system + first query + prior rounds).
    """
    tests = tests or {}
    raw_by_query = raw_by_query or {}
    trajs: List[Dict[str, Any]] = []
    for rec in records:
        prefix: List[Dict[str, Any]] = []
        if rec.get('system'):
            prefix.append(rec['system'])
        if rec.get('query'):
            prefix.append(rec['query'])
        tools = rec.get('tools') or []
        rounds = rec.get('rounds') or []
        rec_id = str(rec.get('id'))
        spec = tests.get(rec_id)

        # Try to recover the dropped user turns from the raw conversation.
        raw = None
        locs = None
        first_user_idx = 0
        if raw_by_query:
            q = rec.get('query') or {}
            qtext = str(q.get('content') if isinstance(q, dict) else q).strip()
            raw = raw_by_query.get(qtext)
            if raw is not None:
                locs = _locate_calls(raw, rounds)
                first_user_idx = next((j for j, m in enumerate(raw)
                                       if m.get('role') == 'user'), 0)
            if recover_stats is not None:
                key = 'recovered' if locs is not None else ('unjoined' if raw is None else 'unlocatable')
                recover_stats[key] = recover_stats.get(key, 0) + 1

        for i, r in enumerate(rounds):
            if r.get('reward_method') == REWARD_TOOL_RESULT and r.get('tool_call'):
                user_data = [('ref_tool_call', json.dumps(r['tool_call'], ensure_ascii=False))]
            elif spec and not r.get('tool_call'):
                user_data = [('code_tests', json.dumps({'id': rec_id, **spec}, ensure_ascii=False))]
                # Carry the challenger's passing solution so OPSD can build the
                # teacher's privileged prompt; harmless/unused in GRPO mode.
                if r.get('code'):
                    user_data.append(('ref_solution', r['code']))
            else:
                continue
            messages = list(prefix)
            if locs is not None:
                anchor = first_user_idx
                for j in range(i):
                    messages.extend(_intervening_turns(raw, anchor, locs[j]))
                    messages.extend(_render_prior_round(rounds[j], j))
                    anchor = locs[j]
                messages.extend(_intervening_turns(raw, anchor, locs[i]))
            else:
                for j in range(i):
                    messages.extend(_render_prior_round(rounds[j], j))
            trajs.append({'messages': messages, 'tools': tools, 'user_data': user_data})
    return trajs


def create_rsi_dataset():
    records = Dataset(DatasetMeta(dataset_id=STD_FLOWS)).dataset.to_list()
    tests = load_tests()
    raw_by_query = _load_raw_by_query(RAW_MESSAGES)
    recover_stats: Dict[str, int] = {}
    trajs = build_round_trajectories(records, tests, raw_by_query, recover_stats)
    if raw_by_query:
        rec_n = recover_stats.get('recovered', 0)
        logger.info(f'[rsi_rl] raw-turn recovery from {RAW_MESSAGES}: '
                    f"recovered={rec_n} "
                    f"unjoined={recover_stats.get('unjoined', 0)} "
                    f"unlocatable={recover_stats.get('unlocatable', 0)} "
                    f'of {len(records)} flows ({rec_n / max(len(records), 1):.1%} rebuilt)')
    else:
        logger.info('[rsi_rl] RSI_RAW_MESSAGES unset: using flow-only prompts '
                    '(dropped user turns are NOT recovered)')
    if SHUFFLE_SEED:
        random.Random(int(SHUFFLE_SEED)).shuffle(trajs)
        logger.info(f'[rsi_rl] shuffled {len(trajs)} rounds with seed {SHUFFLE_SEED} '
                    '(difficulty no longer correlates with step)')
    if POOL_SIZE and len(trajs) > POOL_SIZE:
        pool = trajs[:POOL_SIZE]
        target = MAX_ROUNDS or len(trajs)
        rng = random.Random(int(SHUFFLE_SEED) if SHUFFLE_SEED else 0)
        repeated: List[Dict[str, Any]] = []
        while len(repeated) < target:
            one_pass = list(pool)
            rng.shuffle(one_pass)
            repeated.extend(one_pass)
        trajs = repeated[:target]
        logger.info(f'[rsi_rl] fixed pool of {POOL_SIZE} rounds repeated to {len(trajs)} '
                    f'({len(trajs) / POOL_SIZE:.1f} passes): the question distribution is '
                    'now identical across steps')
    if MAX_ROUNDS and len(trajs) > MAX_ROUNDS:
        # File order, no shuffle: one optim step consumes one round, so N rounds
        # is exactly N steps of a single epoch.
        logger.info(f'[rsi_rl] keeping the first {MAX_ROUNDS} of {len(trajs)} trainable rounds')
        trajs = trajs[:MAX_ROUNDS]
    n_code = sum(1 for t in trajs if t['user_data'][0][0] == 'code_tests')
    logger.info(f'[rsi_rl] {len(records)} standard flows -> {len(trajs)} per-round queries '
                f'({len(trajs) - n_code} tool / {n_code} code); tests loaded: {len(tests)}; '
                f'tool reward: {TOOL_REWARD}'
                + (f' via {JUDGE_MODEL}' if TOOL_REWARD == 'rubric' else ''))
    if not trajs:
        raise RuntimeError(
            'no trainable rounds: tool rounds need a recorded tool_call and code rounds '
            f'need asserts via RSI_TESTS (currently {TESTS_PATH!r})')
    dataset = Dataset(DatasetMeta(data=trajs))
    # enable_thinking=True: we train the reasoning that precedes the tool call.
    dataset.set_template(TEMPLATE, model_id=MODEL_ID, max_length=MAX_MODEL_LEN,
                         truncation_strategy='delete', enable_thinking=True)
    dataset.encode(add_generation_prompt=True)
    return dataset


# ── solver-mode helpers (GRPO continuation + OPSD teacher forward) ──────────
def make_local_template():
    """A driver-side Template instance for token surgery (bridge / concat).

    The model and sampler each hold their own remote template; bridge and
    teacher-prompt construction happen on the driver, so we need a local one.
    Mirrors cookbook/rl/multi_turn/multi_turn_grpo.py's rollout_template.
    """
    import twinkle.template as _tm
    cls = getattr(_tm, TEMPLATE, None)
    if cls is None:
        raise ValueError(f'template class {TEMPLATE!r} not found in twinkle.template')
    t = cls(MODEL_ID, max_length=MAX_MODEL_LEN, enable_thinking=True)
    t.truncation_strategy = 'delete'
    return t


def _last_assistant_text(pif: Dict[str, Any]) -> str:
    for m in reversed(pif.get('messages') or []):
        if m.get('role') == 'assistant':
            return m.get('content', '') or ''
    return ''


def _format_exec_error(err: str) -> str:
    """Turn captured stderr into the tool message shown back to the model."""
    err = (err or '').strip() or 'Your code did not pass the tests (no error output captured).'
    if len(err) > 1500:
        err = err[:700] + '\n...[truncated]...\n' + err[-700:]
    return ('Your solution failed when executed against the tests:\n'
            f'{err}\n\n'
            'Fix the bug and reply with the complete corrected solution in a single '
            '```python code block.')


def _bridge_tool_message(template, pif: Dict[str, Any], tool_content: str) -> Optional[Dict[str, Any]]:
    """Append a {'role':'tool'} turn + next generation prompt as -100 bridge.

    Computed entirely in template space (render-after minus render-before), so
    history tokens stay byte-for-byte in ``input_ids`` and only the new tool turn
    is tokenized from canonical template output -- never decode-then-re-encode.
    Mirrors Template.concat_input_feature / MultiTurnRollout._extend_with_bridge.
    Returns the extended pif, or None if it would exceed the template max_length.
    """
    import copy
    tok = template.tokenizer
    messages_before = list(pif.get('messages') or [])
    messages_after = messages_before + [{'role': 'tool', 'content': tool_content}]
    et = getattr(template, 'enable_thinking', False)
    s_before = tok.apply_chat_template(messages_before, tokenize=False,
                                       add_generation_prompt=False, enable_thinking=et)
    s_after = tok.apply_chat_template(messages_after, tokenize=False,
                                      add_generation_prompt=True, enable_thinking=et)
    # SEAM: the vLLM pif ends at the assistant's closing <|im_end|> with NO trailing
    # newline (generation stops at the eos token), but the canonical render puts a
    # "\n" right after that <|im_end|>. Splitting at len(s_before) would drop that
    # "\n" and append the tool turn directly onto <|im_end|>, producing a malformed
    # "<|im_end|><|im_start|>" boundary that the trained turn-2 tokens then condition
    # on. Split right AFTER the assistant's <|im_end|> so the bridge carries the
    # "\n" + tool turn and reproduces the canonical tokenization exactly.
    marker = '<|im_end|>'
    cut = s_before.rfind(marker)
    if cut < 0:
        raise RuntimeError('tool bridge: no <|im_end|> found in the rendered history; '
                           'cannot locate the assistant turn boundary.')
    head = s_before[:cut + len(marker)]
    if not s_after.startswith(head):
        raise RuntimeError('tool bridge: chat template is not monotonic in the message list; '
                           'cannot append a tool turn as a suffix.')
    bridge_text = s_after[len(head):]
    bridge_ids = tok.encode(bridge_text, add_special_tokens=False)
    if not bridge_ids:
        raise RuntimeError('tool bridge tokenized to an empty id list')
    result = copy.deepcopy(pif)
    input_ids = list(result['input_ids'])
    labels = list(result.get('labels') or [])
    if labels:
        if len(labels) != len(input_ids):
            raise RuntimeError('tool bridge: labels/input_ids length mismatch')
        labels = labels[-1:] + labels[:-1]  # unroll to input order (mirror concat_input_feature)
    else:
        labels = [-100] * len(input_ids)
    result['input_ids'] = input_ids + bridge_ids
    result['labels'] = labels + [-100] * len(bridge_ids)
    max_len = getattr(template, 'max_length', None)
    if max_len and len(result['input_ids']) > max_len:
        return None
    new_if = template._invoke_post_pipeline([result])[0]
    result.update(new_if)
    result['messages'] = messages_after
    return result


def grpo_continue(sampler, template, expand_prompts, sampling_params):
    """GRPO rollout with error-feedback continuation for code rounds.

    Turn 1 samples every prompt. A code sample that FAILS its asserts (and did
    not stop on 'length') gets the sandbox stderr injected as a {'role':'tool'}
    message and is re-sampled, up to SOLVER_MAX_TURNS total turns. Tool rounds
    and length-stopped samples are never continued. The returned per-sample
    input feature is the full multi-turn trajectory (turn tokens trainable, tool
    bridge -100) and old_logps is the concatenation of each turn's logprobs, so
    the (#logps == #trainable labels) invariant holds for GRPO training.
    """
    resps = sampler.sample(expand_prompts, sampling_params)
    pifs: List[Dict[str, Any]] = []
    logps: List[List[float]] = []
    lens: List[int] = []
    stops: List[Optional[str]] = []
    for r in resps:
        s = r.sequences[0]
        pifs.append(s.new_input_feature)
        logps.append([lp[0][1] for lp in s.logprobs])
        lens.append(len(s.tokens))
        stops.append(s.stop_reason)

    done = [False] * len(expand_prompts)
    dm = getattr(sampler, 'device_mesh', None)
    min_batch = dm.data_world_size if dm is not None else 1
    for _turn in range(2, SOLVER_MAX_TURNS + 1):
        retry: List[int] = []
        for i, prompt in enumerate(expand_prompts):
            if done[i]:
                continue
            ud = {item[0]: item[1] for item in (prompt.get('user_data') or [])}
            if 'code_tests' not in ud or stops[i] == 'length':
                done[i] = True
                continue
            try:
                spec = json.loads(ud['code_tests'])
            except (ValueError, TypeError):
                done[i] = True
                continue
            code = extract_code(_last_assistant_text(pifs[i]))
            passed, err = run_asserts_verbose(code, spec.get('setup', ''), spec.get('asserts', []))
            if passed:
                done[i] = True
                continue
            # Bridge the tool error in. If the template can't append a tool turn as
            # a clean suffix (e.g. a malformed/cut turn-1 without a proper </think>),
            # skip continuation for THIS sample rather than crashing the whole step.
            try:
                bridged = _bridge_tool_message(template, pifs[i], _format_exec_error(err))
            except RuntimeError as e:
                logger.warning(f'[rsi_rl][grpo] skip continuation for sample {i}: {e}')
                bridged = None
            if bridged is None:
                done[i] = True
                continue
            pifs[i] = bridged
            retry.append(i)
        if not retry:
            break
        batch = [pifs[i] for i in retry]
        if len(batch) < min_batch:
            batch = batch + [batch[-1]] * (min_batch - len(batch))
        rresps = sampler.sample(batch, sampling_params)[:len(retry)]
        for j, i in enumerate(retry):
            s2 = rresps[j].sequences[0]
            pifs[i] = s2.new_input_feature
            logps[i].extend([lp[0][1] for lp in s2.logprobs])
            lens[i] += len(s2.tokens)
            stops[i] = s2.stop_reason

    # Same invariant MultiTurnRollout enforces: one logp per trainable token.
    for i, pif in enumerate(pifs):
        trainable = sum(1 for lb in (pif.get('labels') or []) if lb != -100)
        if len(logps[i]) != trainable:
            raise RuntimeError(f'GRPO continuation logps/labels misaligned for sample {i}: '
                               f'{len(logps[i])} logps vs {trainable} trainable labels')
    return pifs, logps, lens


def _teacher_pif(template, student_pif: Dict[str, Any], ref_solution: str,
                 response_tokens: List[int]) -> Dict[str, Any]:
    """Teacher input = student's query context + a privileged system message
    carrying the reference solution, then the SAME student response tokens
    concatenated verbatim (concat_input_feature, never re-encoded). Only the
    prompt differs from the student; the scored response tokens are identical,
    which is exactly what OPSD's per-token alignment requires.
    """
    msgs = list(student_pif.get('messages') or [])
    prompt_msgs = msgs[:-1] if (msgs and msgs[-1].get('role') == 'assistant') else list(msgs)
    priv = {'role': 'system', 'content': OPSD_TEACHER_SYS.format(solution=ref_solution)}
    insert_at = 1 if (prompt_msgs and prompt_msgs[0].get('role') == 'system') else 0
    teacher_msgs = prompt_msgs[:insert_at] + [priv] + prompt_msgs[insert_at:]
    prompt_pif = template.encode({'messages': teacher_msgs}, add_generation_prompt=True)
    return template.concat_input_feature(prompt_pif, list(response_tokens))


def _as_rows(out_logps):
    """Normalize forward_only's logps (a list of [mb, L] tensors OR a stacked
    [N, L] tensor, depending on DP/microbatch config) into a flat per-sample
    list of 1-D tensors, in input order.
    """
    import torch
    rows = []
    items = out_logps if isinstance(out_logps, list) else [out_logps]
    for t in items:
        if t is None:
            continue
        t = torch.as_tensor(t)
        if t.dim() == 1:
            rows.append(t)
        elif t.dim() == 2:
            rows.extend([t[i] for i in range(t.shape[0])])
        else:
            raise RuntimeError(f'unexpected forward_only logps ndim={t.dim()}')
    return rows


_OPSD_OFFSET: Optional[int] = None


def _extract_resp(row, seq_len: int, n: int, offset: int):
    valid = row[:seq_len]
    end = len(valid) - offset
    return valid[end - n:end]


def _calibrate_opsd_offset(rows, pifs, old_logps_list):
    """Pin the response-logps frame by matching a student self-forward against
    the sampler's known-correct old_logps. Tries suffix offset 0 and 1 and picks
    the one with the smallest mean|diff|; returns (offset, mean_abs_diff).
    """
    import torch
    best, best_err = None, float('inf')
    for off in (0, 1):
        errs = []
        ok = True
        for row, pif, old in zip(rows, pifs, old_logps_list):
            n = len(old)
            if n == 0:
                continue
            resp = _extract_resp(row, len(pif['input_ids']), n, off)
            if len(resp) != n:
                ok = False
                break
            errs.append((resp.float() - torch.tensor(old, dtype=torch.float32)).abs().mean().item())
        if ok and errs:
            m = sum(errs) / len(errs)
            if m < best_err:
                best_err, best = m, off
    return best, best_err


def opsd_teacher_logps(model, template, student_pifs, response_tokens_list,
                       ref_solutions, student_old_logps):
    """Per-sample response-only teacher log-probs for OPSDLoss (ragged lists).

    On the first call, calibrates the suffix offset by self-checking a student
    forward against the sampler old_logps; if it cannot align within
    OPSD_SELFCHECK_TOL it raises rather than feed a mis-framed teacher.
    """
    global _OPSD_OFFSET
    if _OPSD_OFFSET is None:
        s_out = model.forward_only(inputs=list(student_pifs), micro_batch_size=MICRO_BATCH_SIZE)
        off, err = _calibrate_opsd_offset(_as_rows(s_out.logps), student_pifs, student_old_logps)
        if off is None or err > OPSD_SELFCHECK_TOL:
            raise RuntimeError(
                f'OPSD self-check failed: forward_only response logps could not be aligned to the '
                f'sampler old_logps (best mean|diff|={err}); the response frame is off, so teacher '
                f'logps cannot be trusted. Inspect _as_rows / _extract_resp before training.')
        _OPSD_OFFSET = off
        logger.info(f'[rsi_rl][opsd] response-logps suffix offset calibrated to {off} '
                    f'(self-check mean|diff|={err:.4f} < tol {OPSD_SELFCHECK_TOL})')

    teacher_pifs = [_teacher_pif(template, sp, sol, toks)
                    for sp, toks, sol in zip(student_pifs, response_tokens_list, ref_solutions)]
    t_rows = _as_rows(model.forward_only(inputs=teacher_pifs, micro_batch_size=MICRO_BATCH_SIZE).logps)
    teacher_logps: List[List[float]] = []
    for row, tpif, toks in zip(t_rows, teacher_pifs, response_tokens_list):
        n = len(toks)
        resp = _extract_resp(row, len(tpif['input_ids']), n, _OPSD_OFFSET)
        if len(resp) != n:
            raise RuntimeError(f'OPSD teacher extraction: {len(resp)} logps for {n} response tokens')
        teacher_logps.append([float(x) for x in resp])
    return teacher_logps


def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, MODEL_GPUS + SAMPLER_GPUS)),
                    device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    ref_mesh = None
    if REF_GPUS:
        device_groups.append(
            DeviceGroup(name='ref', ranks=list(range(MODEL_GPUS + SAMPLER_GPUS, NUM_GPUS)),
                        device_type='GPU'))
        ref_mesh = DeviceMesh.from_sizes(world_size=REF_GPUS, dp_size=REF_GPUS)

    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # Full-parameter training: no adapter is added, so every weight is trained and
    # the whole model is pushed to the sampler each step.
    from twinkle.model.megatron import MegatronModel
    model = MegatronModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
                          mixed_precision='bf16', variable_seq_lengths=True)
    model.set_optimizer('default', lr=LEARNING_RATE)
    model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    if SOLVER_MODE == 'opsd':
        # On-policy self-distillation: student pulled toward a teacher that saw
        # the reference solution. No advantages / reward in the loss.
        model.set_loss('OPSDLoss', reverse=OPSD_REVERSE)
    else:
        loss_kwargs: Dict[str, Any] = {'epsilon': 0.2, 'beta': KL_BETA}
        if LOSS_NAME == 'DRGRPOLoss':
            # Must be the real generation cap: the class default is 1024
            # (grpo.py:591) and it sits in the denominator, so leaving it there
            # while generating MAX_NEW_TOKENS scales every gradient by
            # MAX_NEW_TOKENS/1024.
            loss_kwargs['max_completion_length'] = MAX_NEW_TOKENS
        model.set_loss(LOSS_NAME, **loss_kwargs)
        logger.info(f'[rsi_rl] loss={LOSS_NAME} {loss_kwargs} '
                    f'ref={"none" if not REF_GPUS else REF_MODEL_ID}')
        if KL_BETA > 0 and not REF_GPUS:
            raise RuntimeError(
                f'RSI_KL_BETA={KL_BETA} but RSI_REF_GPUS=0: the KL term needs ref_logps '
                f'(grpo.py:315 requires beta>0 AND ref_logps), so it would silently do '
                f'nothing. Set RSI_REF_GPUS (e.g. 2) or RSI_KL_BETA=0.')

    model.set_processor(InputProcessor, padding_free=True)
    model.set_template(TEMPLATE, model_id=MODEL_ID, max_length=MAX_MODEL_LEN, enable_thinking=True)
    # Observability only: approx_kl / clip_ratio / entropy per step. approx_kl at the
    # first inner step also reconciles sampler vs trainer logps, which is the check
    # for whether the full-weight sync actually landed. OPSD has no PPO ratio.
    if SOLVER_MODE != 'opsd':
        model.add_metric('GRPOMetric', is_training=True, epsilon=0.2)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': MAX_MODEL_LEN,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE, model_id=MODEL_ID, max_length=MAX_MODEL_LEN, enable_thinking=True)

    # KL anchor: frozen base weights, forward only, no optimizer. Same template /
    # processor as the trainer so the returned per-token logps line up position by
    # position with the trainer's own forward (both are Megatron forwards over the
    # identical token sequence, so no frame calibration is needed -- unlike the
    # OPSD teacher, whose prompt has a different length).
    ref_model = None
    if REF_GPUS:
        ref_model = MegatronModel(model_id=REF_MODEL_ID, device_mesh=ref_mesh, remote_group='ref',
                                  mixed_precision='bf16', variable_seq_lengths=True)
        # advantages=None on this path, so GRPOLoss short-circuits to a zero loss
        # and only outputs['logps'] is harvested (grpo.py:298).
        ref_model.set_loss('GRPOLoss', epsilon=0.2)
        ref_model.set_processor(InputProcessor, padding_free=True)
        ref_model.set_template(TEMPLATE, model_id=REF_MODEL_ID, max_length=MAX_MODEL_LEN,
                               enable_thinking=True)

    # Driver-side template for token surgery: the GRPO tool-error bridge and the
    # OPSD teacher-prompt concat both run on the driver.
    local_template = make_local_template()

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(dataset=create_rsi_dataset, batch_size=GLOBAL_BATCH_SIZE,
                            min_batch_size=GLOBAL_BATCH_SIZE, device_mesh=model_mesh, remote_group='model')

    metrics = CompletionRewardMetric()
    reward_fn = RoundReward()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95)

    optim_step = 0
    logger.info('Starting RSI per-round GRPO (full-parameter Megatron)')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        expand_prompts = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        # No LoRA, so every sync ships the full weights.
        ckpt_manager.sync_weights(merge_and_sync=True)
        sampler.reset_prefix_cache()

        all_tokens: List[List[int]] = []
        if SOLVER_MODE == 'grpo':
            # Rollout with error-feedback continuation on failed code rounds.
            all_input_data, all_old_logps, all_completion_lengths = grpo_continue(
                sampler, local_template, expand_prompts, sampling_params)
        else:
            # OPSD: single turn; also keep raw response tokens for the teacher concat.
            all_input_data, all_old_logps, all_completion_lengths = [], [], []
            for sample_response in sampler.sample(expand_prompts, sampling_params):
                for sequence in sample_response.sequences:
                    all_input_data.append(sequence.new_input_feature)
                    all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                    all_completion_lengths.append(len(sequence.tokens))
                    all_tokens.append(list(sequence.tokens))

        # Reward drives GRPO advantages; in OPSD it is observability only.
        rewards = reward_fn(all_input_data)
        scored = [r for r in rewards if r is not None]
        metrics.accumulate(completion_lengths=all_completion_lengths, rewards={'round_reward': scored})

        teacher_logps = None
        if SOLVER_MODE == 'grpo':
            advantages = group_advantages(rewards, num_generations=NUM_GENERATIONS, scale='group')
            # First group, verbatim: catches a reward/advantage misalignment (a high-reward
            # sample must not carry a negative advantage).
            logger.info(f'[group0] rewards={rewards[:NUM_GENERATIONS]} '
                        f'advantages={[round(a, 3) for a in advantages[:NUM_GENERATIONS]]} '
                        f'lens={all_completion_lengths[:NUM_GENERATIONS]}')
        else:
            # OPSD: no advantages in the loss; keep a zero list only for the audit dump.
            advantages = [0.0] * len(all_input_data)
            all_ref_solutions: List[str] = []
            for prompt in expand_prompts:
                ud = {item[0]: item[1] for item in (prompt.get('user_data') or [])}
                all_ref_solutions.append(ud.get('ref_solution', ''))
            if any(not s for s in all_ref_solutions):
                raise RuntimeError('OPSD needs a ref_solution (challenger passing solution) on '
                                   'every code round; some rounds are missing it.')
            teacher_logps = opsd_teacher_logps(
                model, local_template, all_input_data, all_tokens,
                all_ref_solutions, all_old_logps)
            logger.info(f'[group0] opsd rewards(obs)={rewards[:NUM_GENERATIONS]} '
                        f'lens={all_completion_lengths[:NUM_GENERATIONS]}')

        if REWARD_DUMP:
            # Append one audit line per rollout of this step. Reward/advantage are
            # already computed above; this only reads them, never changes them.
            with open(REWARD_DUMP, 'a', encoding='utf-8') as fdump:
                for i, rec in enumerate(reward_fn.records):
                    fdump.write(json.dumps(_json_safe({
                        'step': optim_step + 1,
                        'group': i // NUM_GENERATIONS,
                        'score': rec.get('score'),
                        'advantage': round(advantages[i], 4),
                        'len': all_completion_lengths[i],
                        'kind': rec.get('kind'),
                        'ref_call': rec.get('ref_call'),
                        'gen_call': rec.get('gen_call'),
                        'reason': rec.get('reason'),
                        'completion': rec.get('completion'),
                    }), ensure_ascii=False) + '\n')

        total = len(all_input_data)
        for mb_start in range(0, total, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total)
            if SOLVER_MODE == 'grpo':
                ref_logps = None
                if ref_model is not None:
                    # ModelOutput is a TypedDict (data_format/output.py:15), so it is a
                    # plain dict -- index it, never attribute-access it.
                    ref_out = ref_model.forward_only(
                        inputs=all_input_data[mb_start:mb_end],
                        micro_batch_size=MICRO_BATCH_SIZE)
                    ref_logps = _as_rows(ref_out['logps'])
                    if optim_step == 0 and mb_start == 0:
                        # One-time shape check: a row must cover the whole padded
                        # sequence, otherwise GRPOLoss's full-sequence branch
                        # (grpo.py:210) would reject it and we want to see the
                        # numbers rather than only the assertion.
                        logger.info(
                            f'[rsi_rl][kl] ref rows={len(ref_logps)} '
                            f'row_lens={[len(r) for r in ref_logps[:4]]} '
                            f'input_lens={[len(x["input_ids"]) for x in all_input_data[mb_start:mb_start + 4]]}')
                model.forward_backward(
                    inputs=all_input_data[mb_start:mb_end],
                    old_logps=all_old_logps[mb_start:mb_end],
                    advantages=advantages[mb_start:mb_end],
                    ref_logps=ref_logps,
                    micro_batch_size=MICRO_BATCH_SIZE,
                )
            else:
                # OPSD: teacher_logps (ragged, response-only) drives the k3 pull;
                # OPSDLoss ignores advantages / old_logps.
                model.forward_backward(
                    inputs=all_input_data[mb_start:mb_end],
                    teacher_logps=teacher_logps[mb_start:mb_end],
                    micro_batch_size=MICRO_BATCH_SIZE,
                )
            model.clip_grad_and_step()
            optim_step += 1
            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'{SAVE_NAME}-checkpoint-{optim_step}', output_dir=SAVE_DIR)

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict.update({f'train/{k}': v for k, v in reward_fn.stats.items()})
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save(SAVE_NAME, output_dir=SAVE_DIR)


if __name__ == '__main__':
    main()
