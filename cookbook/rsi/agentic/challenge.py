# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI self-play, agentic half: one trajectory is one request, start to finish.

Three resources, each a queue anyone may put a request on:

  sandbox   N microVMs. A job holds one for as long as it needs the workspace.
  vLLM      the local sampler. ``sample`` routes each request to the least busy
            worker (``enable_continous_work``), so a batch of one is a first-class
            call and 32 threads calling it concurrently is the intended use.
  API       qwen3.8-max, for the stages that must not add untrained tokens: the
            check script, the problem statement, the rubric, and the keyword bank.
            Keywords joined this list after measuring what the local model produced
            for it: 31% of ``transform`` entries named an activity on a running
            system rather than a computation, and 71% of the bank comes from an
            expand prompt that had no category rules in it at all. Iteration 9 ran
            18 refills through the API with the rules added and 1 of 105 keywords
            missed, against 24% of the bank built without them.

Nothing waits for a batch. A proposal that finishes its build hands its statement
straight to eight solver jobs and lets go of its sandbox; those eight run whenever
a slot frees up, in any order, interleaved with proposals from other groups. The
only synchronisation is the last step, deciding whether a group is worth keeping,
and that is a counter under a lock rather than a barrier.

A group is eight proposals sharing one keyword draw and one prompt -- that is what
makes it a GRPO group, since the advantage of a proposal is its reward minus the
mean over the others answering the same prompt. It is kept when at least one of its
eight produced a task the solver passes sometimes (``1 <= n_pass <= 7``); the other
seven may be anything, including failures, and they train with reward 0. From a kept
group the highest-reward in-band proposal's eight solver attempts are what the
solver side trains on, so a kept group contributes 8 proposing and 8 solving
trajectories, and eight kept groups are the 64 + 64 one training step reads.

Output (all under ``--out-dir``):

    trajs/*.npz          input_ids / labels / logprobs per trajectory
    trajs/index.jsonl    one line per trajectory: side, group, reward, full text
    groups.jsonl         one line per decided group: why kept or dropped
    tasks.jsonl          the statements and check scripts that were delivered
    keywords.jsonl       the keyword bank, carried between iterations

Run it as a Ray job (sampler only, no trainer)::

    python cookbook/rsi/agentic/challenge.py --keep-groups 8
"""
import argparse
import collections
import json
import math
import os
import queue
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle_agentic.challenger import KeywordStore, parse_check_script, parse_problem_statement
from twinkle_agentic.challenger.agentic import brittle_check_reason
from twinkle_agentic.challenger.code import split_keyword_list
from twinkle_agentic.challenger.task_bank import TaskBank
from twinkle_agentic.rollout import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402
from sandbox import close_pool, open_pool, solver_harness  # noqa: E402

logger = get_logger()

# ── Reward ─────────────────────────────────────────────────────────────────
# The proposing side's reward, unchanged from challenger/agentic.py where it was
# measured. Kept as free functions because nothing here has the state the method
# version read off self.

# Peak and width of the pass-rate gaussian. This replaced R-Zero's
# ``1 - 2*|p - 1/2|`` for two reasons measured on run_clean9's 87 in-band
# proposals: that shape was not injective (with 8 rollouts its seven in-band
# values of n_pass mapped onto four rewards, so a task 1 of 8 solvers could do and
# one 7 of 8 could do were worth the same), and its signal was smaller than its
# noise (0.280 signal over 0.246 binomial noise, against 0.347 over 0.177 here).
# A peak below one half is also the more useful target: a proposal only teaches
# the solver something when the solver mostly cannot do it yet.
PASS_RATE_TARGET = 0.2
PASS_RATE_WIDTH = 0.3

# How many phrases the keyword prompt's 'do not repeat these' line may quote.
# Measured on armA2ser: with 130 quoted the eighth refill call was still answering
# normally, with 150 it started inventing -- 'iRAPION holistic replace', 10 of 480
# phrases that run. 100 sits below where that began.
AVOID_TOTAL = 100

# How often run() looks for a stall. Only ever reached when the run has already
# gone quiet, so it costs one wakeup per interval and nothing else.
STALL_CHECK_SECONDS = 30.0


def novelty_factor(novelty: Optional[float], floor: float) -> float:
    """What a proposal's difficulty score is multiplied by for its novelty.

    ``floor + (1 - floor) * N``. ``None`` returns 1.0, not the floor: it means
    nobody judged this proposal, and charging it for a measurement that did not
    happen would make the reward depend on API uptime.
    """
    if novelty is None:
        return 1.0
    n = min(1.0, max(0.0, float(novelty)))
    return floor + (1.0 - floor) * n


def challenger_reward(n_pass: Optional[int], rollouts: int,
                      novelty: Optional[float] = None, floor: float = 1.0) -> float:
    """How close the solver came to the target pass rate, times novelty.

    ``None`` means the proposal never got as far as being solved and 0 means no
    attempt passed. Both score 0, and that floor is load-bearing rather than
    incidental: the gaussian at p=0 is 0.801, higher than the 0.607 it gives a
    proposal half the attempts solve, so without the gate the best thing a proposer
    could do is write tasks nobody can finish.
    """
    if n_pass is None or not rollouts or n_pass <= 0:
        return 0.0
    gap = n_pass / rollouts - PASS_RATE_TARGET
    difficulty = math.exp(-(gap * gap) / (2.0 * PASS_RATE_WIDTH**2))
    return difficulty * novelty_factor(novelty, floor)


# ── Arguments ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # What to collect.
    p.add_argument('--keep-groups', type=int, default=8,
                   help='stop once this many groups have been kept. 8 groups x 8 '
                        'proposals = 64 proposing trajectories, and the selected '
                        'proposal of each x 8 attempts = 64 solving ones.')
    p.add_argument('--group-size', type=int, default=8,
                   help='proposals sharing one keyword draw and one prompt: the '
                        'GRPO group on the proposing side.')
    p.add_argument('--solver-rollouts', type=int, default=8,
                   help='attempts per candidate task: the GRPO group on the solving '
                        'side, and the denominator of n_pass.')
    p.add_argument('--max-group-attempts', type=int, default=0,
                   help='give up after this many groups have been tried, kept or '
                        'not. 0 leaves the run governed by --keep-groups alone.')

    # Local model (the trainable half).
    p.add_argument('--model-id', default='ms://Qwen/Qwen3-4B')
    p.add_argument('--template', default='Template')
    p.add_argument('--sampler-gpus', type=int, default=4)
    p.add_argument('--max-model-len', type=int, default=40960)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.8)

    # The API model: check scripts, problem statements, keywords, rubric.
    p.add_argument('--api-model', default=os.environ.get('LLM_BACKUP_MODEL', ''))
    p.add_argument('--api-base', default=os.environ.get('LLM_BACKUP_BASE_URL', ''))
    p.add_argument('--api-key', default=os.environ.get('LLM_BACKUP_API_KEY', ''))
    p.add_argument('--api-concurrency', type=int, default=32,
                   help='API calls in flight. Only the rubric runs as its own job; '
                        'check and statement calls are made from inside a sandbox '
                        'job and are already capped by the slot count.')
    p.add_argument('--api-thinking-budget', type=int, default=0,
                   help='sent as extra_body on every API call when > 0. Capping the '
                        'reasoning is the one knob that moved wall-clock: 58s -> 10s '
                        'per turn at 2048 on a ~15k-character context.')

    # Building: stage 1, the part that is trained.
    p.add_argument('--propose-temp', type=float, default=1.0)
    p.add_argument('--propose-max-tokens', type=int, default=8192)
    p.add_argument('--max-turns', type=int, default=24)
    p.add_argument('--max-build-files', type=int, default=4,
                   help='appends BUILD_SIZE_CAP to the system prompt, capping how '
                        'many files one build may leave behind. 0 removes the cap. '
                        'This changes the prompt, so it changes what is trained.')
    p.add_argument('--stop-after-stuck-turns', type=int, default=2,
                   help='end the tool phase after this many turns that repeated a '
                        'call and changed nothing. 0 turns it off.')
    p.add_argument('--one-call-per-reply', action=argparse.BooleanOptionalAction,
                   default=True,
                   help="stop generation at '</tool_call>' so a reply carries exactly "
                        'one call. The stop string is kept in the output, or every '
                        'turn would train on an unclosed block.')

    # Stages 2 and 3, over the API.
    p.add_argument('--check-max-tokens', type=int, default=8192)
    p.add_argument('--check-retries', type=int, default=1,
                   help='rewrites offered to a check script that does not parse or '
                        'does not pass on the state it was written from.')
    p.add_argument('--problem-max-tokens', type=int, default=4096)
    p.add_argument('--problem-max-chars', type=int, default=8192,
                   help='a statement longer than this is thrown away: it is quoting '
                        'the workspace rather than describing the task.')

    # Solving.
    p.add_argument('--solver-temp', type=float, default=1.0)
    p.add_argument('--solver-max-tokens', type=int, default=8192)
    p.add_argument('--solver-max-turns', type=int, default=24)

    # Keywords.
    p.add_argument('--keywords-n', type=int, default=128,
                   help='how many keywords a dry category is refilled with.')
    p.add_argument('--keyword-gen-calls', type=int, default=8,
                   help='calls a refill is split over, run one at a time so each can '
                        'be told what the ones before it already said.')
    p.add_argument('--keyword-refill-tries', type=int, default=2,
                   help='refill rounds before a dry category is recycled, i.e. every '
                        'keyword in it marked unused again. Without this a bank the '
                        'model has run out of new ideas for ends the run.')
    p.add_argument('--keyword-expand', action=argparse.BooleanOptionalAction,
                   default=True,
                   help='after the last group, ask for more keywords in the domains '
                        'that produced tasks nobody solved, and write them to the '
                        'bank the next iteration reads.')
    p.add_argument('--keyword-temp', type=float, default=1.3)
    p.add_argument('--keyword-max-tokens', type=int, default=4096)

    # Novelty.
    p.add_argument('--task-bank', default='',
                   help='jsonl of statements from earlier iterations. Empty turns '
                        'novelty off, and the reward is the pass-rate gaussian alone.')
    p.add_argument('--task-bank-refs', type=int, default=5,
                   help='stored statements shown to the judge, on top of the group '
                        "'s own siblings, which are always shown.")
    # 1.0 leaves the novelty term at exactly 1.0, so a proposal is scored on its
    # pass rate alone while the rubric still runs and still writes
    # novelty_scores.jsonl. See loop.sh for the measurement that set it there.
    p.add_argument('--novelty-floor', type=float, default=1.0)
    p.add_argument('--novelty-tries', type=int, default=3,
                   help='attempts to get a verdict for a group. After the last one '
                        'the group is dropped and its pending solver jobs skipped.')

    # Sandbox.
    p.add_argument('--sandbox-slots', type=int, default=32,
                   help='microVMs, i.e. how many jobs run at once. One job owns one '
                        'slot from the workspace clear to its last check.')
    p.add_argument('--sandbox-template', default=os.environ.get('AENV_TEMPLATE', ''))
    p.add_argument('--sandbox-api-url', default=os.environ.get('AENV_API_URL', ''))
    p.add_argument('--sandbox-timeout', type=int, default=900)
    p.add_argument('--agent-config', default='cookbook/rsi/agentic/rsi_agent.yaml')
    p.add_argument('--workspace', default='/workspace')
    p.add_argument('--snapshot-max-files', type=int, default=50)
    p.add_argument('--snapshot-per-file', type=int, default=600)
    p.add_argument('--snapshot-budget', type=int, default=6000)

    # Output.
    p.add_argument('--out-dir', default='output/rsi_agentic')
    p.add_argument('--keyword-db', default='',
                   help='defaults to <out-dir>/keywords.jsonl')
    p.add_argument('--random-seed', type=int, default=0)
    args = p.parse_args()
    if not args.api_model or not args.api_base:
        raise SystemExit('[challenge] --api-model and --api-base are required '
                         '(or LLM_BACKUP_MODEL / LLM_BACKUP_BASE_URL)')
    if args.solver_rollouts < 2:
        raise SystemExit('[challenge] --solver-rollouts must be >= 2: it is both the '
                         "solver side's GRPO group size and the denominator n_pass is "
                         'judged against')
    if args.group_size < 2:
        raise SystemExit('[challenge] --group-size must be >= 2: a group of one has '
                         'no mean to subtract, so every advantage is zero')
    args.keyword_db = args.keyword_db or os.path.join(args.out_dir, 'keywords.jsonl')
    return args


# ── Resources ──────────────────────────────────────────────────────────────


def initialize_device(args) -> Tuple[Any, Any]:
    """Bring up Ray and the local vLLM sampler; returns (sampler, template).

    The template is built here as well as inside the sampler because the rollout
    encodes with it locally: one object, so the token ids the sampler continues
    from are the ids the trajectory was encoded with.
    """
    twinkle.initialize(
        mode='ray', nproc_per_node=args.sampler_gpus, lazy_collect=False,
        groups=[DeviceGroup(name='sampler', ranks=list(range(args.sampler_gpus)),
                            device_type='GPU')])
    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args={'gpu_memory_utilization': args.gpu_memory_utilization,
                     'max_model_len': args.max_model_len},
        device_mesh=DeviceMesh.from_sizes(world_size=args.sampler_gpus,
                                          dp_size=args.sampler_gpus),
        remote_group='sampler',
    )
    sampler.set_template(args.template, model_id=args.model_id, enable_thinking=True,
                         max_length=args.max_model_len)
    import twinkle.template as template_module
    template = getattr(template_module, args.template)(
        args.model_id, max_length=args.max_model_len, enable_thinking=True)
    if not getattr(type(sampler).sample, '_enable_continous_work', False):
        raise SystemExit(
            '[challenge] this sampler does not route requests one at a time '
            '(sample lacks enable_continous_work), so a batch of one would be '
            'padded to the worker count and most of every generation thrown away. '
            'The whole design here is one trajectory per request.')
    return sampler, template


def initialize_sandbox(args) -> List[Any]:
    """Boot the slots. See ``sandbox.open_pool``."""
    return open_pool(
        args.sandbox_slots,
        template=args.sandbox_template,
        api_url=args.sandbox_api_url,
        config_path=args.agent_config,
        workspace=args.workspace,
        sandbox_timeout=args.sandbox_timeout,
        snapshot_max_files=args.snapshot_max_files,
        snapshot_per_file=args.snapshot_per_file,
        snapshot_budget=args.snapshot_budget,
    )


def rollout_one(rollout: MultiTurnRollout, traj: Dict[str, Any],
                params: SamplingParams, slot) -> Optional[Dict[str, Any]]:
    """Run one trajectory: vLLM for the replies, ``slot`` for the tool calls.

    A batch of one. The sampler routes it to whichever worker is free, so this is
    called from every sandbox thread at once and the requests share vLLM's batch
    without any of them waiting for the others to be ready.
    """
    out = rollout([traj], sampling_params=params, tool_manager=slot.tool_manager)
    return out[0] if out else None


def call_one(slot, script: str) -> Tuple[int, str]:
    """Run one python script inside ``slot``; returns (exit code, output)."""
    return slot.run(script)


def api_one(api, messages: List[Dict[str, Any]], user_text: str,
            params: SamplingParams, extra_body: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Append ``user_text`` and one API reply to ``messages``; returns the reply.

    ``messages`` is the caller's private copy, never a trainable trajectory, so
    mutating it in place costs the model nothing. ``None`` means the call raised:
    the caller rejects rather than building a task on a broken conversation.

    Tools are withdrawn for these stages on purpose -- they are answers, not
    actions -- so only the text is kept and any structured ``tool_calls`` the API
    returned are dropped.
    """
    messages.append({'role': 'user', 'content': user_text})
    request = {'messages': messages}
    try:
        reply = api(request, params, extra_body=extra_body) if extra_body else api(request, params)
    except Exception as e:  # noqa: BLE001 -- one bad call must not kill the run
        logger.warning(f'[challenge] API call failed: {type(e).__name__}: {e}')
        return None
    if isinstance(reply, list):
        reply = reply[0] if reply else {}
    content = (reply.get('content') if isinstance(reply, dict) else None) or ''
    messages.append({'role': 'assistant', 'content': content})
    return content


# ── Output ─────────────────────────────────────────────────────────────────


def logprob_column(logprobs: Any) -> List[float]:
    """One float per generated token: the logprob of the token that was chosen.

    The sampler hands these over as ``List[List[Tuple[int, float]]]`` -- per
    generated token, a list of top-k ``(token_id, logprob)`` pairs with the chosen
    token first (``SampledSequence.logprobs``, data_format/sampling.py:185).
    Passing that to ``np.asarray`` directly would store an ``(N, k, 2)`` array and
    the loader would hand GRPO nested lists where it wants one float per trainable
    token -- which is a crash inside the step, or worse a silent reshape.

    A plain list of floats is accepted too, for a sampler that already flattened.
    Anything else raises rather than being coerced: a wrong ``old_logps`` makes the
    GRPO ratio wrong on the first step, and nothing downstream would say so.
    """
    out: List[float] = []
    for step in logprobs:
        if isinstance(step, (int, float)):
            out.append(float(step))
            continue
        if isinstance(step, (list, tuple)) and step:
            head = step[0]
            if isinstance(head, (list, tuple)) and len(head) >= 2:
                out.append(float(head[1]))
                continue
        raise TypeError(f'cannot read a logprob out of {step!r}; expected a float '
                        f'or a list of (token_id, logprob) pairs')
    return out


class Recorder:
    """Everything a run writes, behind one lock.

    Trajectories go to ``.npz`` for the token fields and to ``index.jsonl`` for
    everything a reader needs to interpret them. The text is written in full and
    never truncated: these files are read to check whether a reward was deserved,
    which a shortened statement cannot answer.
    """

    def __init__(self, out_dir: str):
        self.dir = out_dir
        self.traj_dir = os.path.join(out_dir, 'trajs')
        os.makedirs(self.traj_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._n = 0
        self._index = open(os.path.join(self.traj_dir, 'index.jsonl'), 'w', encoding='utf-8')
        self._groups = open(os.path.join(out_dir, 'groups.jsonl'), 'w', encoding='utf-8')
        self._tasks = open(os.path.join(out_dir, 'tasks.jsonl'), 'w', encoding='utf-8')
        # Why a build produced no task. The reason alone is not diagnosable: nine
        # empty_workspace rejections in one run all looked like the model refusing
        # to act, and the question of whether it had run out of tokens or simply
        # emitted no call could not be answered from the record, because the fields
        # that answered it were on the trajectory and were dropped.
        self._rejected = open(os.path.join(out_dir, 'rejected.jsonl'), 'w', encoding='utf-8')
        # Keyword replies, both sides in full. The one question this file exists to
        # answer -- did the model disobey the format, or does the parser reject what
        # it produced -- cannot be answered from a count. Keyword generation was
        # silently broken for whole runs when the prompt asked for one per line and
        # the parser wanted a JSON array.
        self._keywords = open(os.path.join(out_dir, 'keyword_gen.jsonl'), 'w', encoding='utf-8')
        # Every solver attempt, passed or not, with the state it left and what the
        # check said about it. A task measured at 0 of 8 has three explanations --
        # the check is wrong, the statement withholds something the check demands,
        # or the solver gave up -- and only the attempt and the workspace it left
        # tell them apart. Written for every attempt, not only for the ones that
        # end up trained on: the failures are what this file is for.
        self._attempts = open(os.path.join(out_dir, 'solver_attempts.jsonl'), 'w',
                             encoding='utf-8')
        # The rubric, all three of its dimensions. Only novelty reaches a reward;
        # usefulness and complexity are recorded so the question of whether they
        # should count can be answered from a run instead of argued.
        self._novelty = open(os.path.join(out_dir, 'novelty_scores.jsonl'), 'w',
                             encoding='utf-8')

    def trajectory(self, traj: Dict[str, Any], **fields: Any) -> None:
        """One training sample: token fields to npz, everything else to the index.

        A trajectory with no ``logprobs`` is written anyway, with the field left
        null. It is not trainable and the loader will say so -- which is the point:
        a sample silently dropped here would make the group it belongs to look like
        a different size than it was.
        """
        input_ids = np.asarray(traj.get('input_ids') or [], dtype=np.int32)
        labels = np.asarray(traj.get('labels') or [], dtype=np.int32)
        logprobs = traj.get('logprobs')
        with self._lock:
            self._n += 1
            name = f'{self._n:06d}.npz'
        arrays = {'input_ids': input_ids, 'labels': labels}
        if logprobs is not None:
            # float64, and the chosen token's column only. These are the old_logps a
            # GRPO step divides by; float32 would round them to about 7 digits, so
            # the ratio exp(logp - old_logp) would be off by roughly 1e-7 for
            # reasons that have nothing to do with the policy having changed.
            arrays['logprobs'] = np.asarray(logprob_column(logprobs), dtype=np.float64)
        # Compressed: a 24-turn agentic episode is tens of thousands of token ids,
        # and 128 of them per iteration adds up on disk.
        np.savez_compressed(os.path.join(self.traj_dir, name), **arrays)
        record = dict(fields)
        record.update({
            'npz': name,
            'n_tokens': int(input_ids.size),
            'n_trainable': int((labels != -100).sum()) if labels.size else 0,
            'has_logprobs': logprobs is not None,
            # The rollout guarantees one logprob per trainable label; recorded so a
            # loader can check it rather than trust it.
            'n_logprobs': int(arrays['logprobs'].size) if logprobs is not None else 0,
            'turns': traj.get('turns'),
            'stop_reason': traj.get('stop_reason'),
            'truncated': bool(traj.get('truncated')),
            'tool_stop': traj.get('tool_stop'),
            'messages': traj.get('messages') or [],
        })
        self._write(self._index, record)

    def group(self, record: Dict[str, Any]) -> None:
        self._write(self._groups, record)

    def task(self, record: Dict[str, Any]) -> None:
        self._write(self._tasks, record)

    def rejected(self, record: Dict[str, Any]) -> None:
        self._write(self._rejected, record)

    def keywords(self, record: Dict[str, Any]) -> None:
        self._write(self._keywords, record)

    def attempt(self, record: Dict[str, Any]) -> None:
        self._write(self._attempts, record)

    def novelty(self, record: Dict[str, Any]) -> None:
        self._write(self._novelty, record)

    def close(self) -> None:
        for handle in (self._index, self._groups, self._tasks, self._rejected,
                       self._keywords, self._attempts, self._novelty):
            handle.close()

    def _write(self, handle, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            handle.write(line + '\n')
            handle.flush()


# ── Group state ────────────────────────────────────────────────────────────


@dataclass
class Proposal:
    """One trajectory's worth of state, from the build to its solver attempts."""

    group: 'Group'
    idx: int
    outcome: str = ''                       # 'ok', or why this one produced no task
    detail: str = ''                        # what to look at when it did not
    statement: str = ''
    check: str = ''
    traj: Optional[Dict[str, Any]] = None   # the trainable build trajectory
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    passes: List[bool] = field(default_factory=list)
    novelty: Optional[float] = None
    n_solved: int = 0                       # attempts finished, not attempts passed

    @property
    def n_pass(self) -> Optional[int]:
        """How many attempts passed, or None if the task was never measured."""
        if not self.statement or self.n_solved < self.group.rollouts:
            return None
        return sum(1 for p in self.passes if p)

    def reward(self, rollouts: int, floor: float) -> float:
        return challenger_reward(self.n_pass, rollouts, self.novelty, floor)


class Group:
    """``size`` proposals sharing one keyword draw, and the counters that decide them.

    Every method that reads more than one field takes the lock, because the
    proposals resolve on different threads and in any order. Nothing here blocks:
    a thread reports what it finished and asks whether that was the last thing
    outstanding, and only the thread that gets ``True`` runs the decision.
    """

    def __init__(self, gid: int, keywords: List[Tuple[str, str]], keyword_block: str,
                 prompt: str, size: int, rollouts: int):
        self.id = gid
        self.keywords = keywords
        self.keyword_block = keyword_block
        self.prompt = prompt
        self.rollouts = rollouts
        self.proposals = [Proposal(self, i) for i in range(size)]
        self.lock = threading.Lock()
        self.n_built = 0                    # proposals whose build stage is over
        self.rubric_done = False
        self.dropped = ''                   # reason, once this group is abandoned
        self.decided = False

    @property
    def size(self) -> int:
        return len(self.proposals)

    def abandon(self, reason: str) -> bool:
        """Give up on this group. True if this call is the one that decided it.

        Jobs already queued for it check ``dropped`` and return their slot without
        doing any work, so abandoning is also how the remaining solver attempts of
        a group are cancelled.
        """
        with self.lock:
            if self.decided:
                return False
            self.dropped = reason
            self.decided = True
            return True

    def built(self, prop: Proposal) -> str:
        """Record that ``prop``'s build stage is over; returns what to do next.

        ``'rubric'`` when this was the last build and the statements are ready to
        be judged, ``'decide'`` when the group needs no judging and nothing else is
        outstanding, ``''`` when there is still work in flight.
        """
        with self.lock:
            self.n_built += 1
            if self.n_built < self.size or self.decided:
                return ''
            if any(p.statement for p in self.proposals):
                return 'rubric'
            self.rubric_done = True
            return self._ready_locked()

    def judged(self) -> str:
        with self.lock:
            self.rubric_done = True
            return self._ready_locked()

    def solved(self, prop: Proposal, attempt: Optional[Dict[str, Any]], passed: bool) -> str:
        with self.lock:
            prop.n_solved += 1
            prop.attempts.append(attempt or {})
            prop.passes.append(passed)
            return self._ready_locked()

    def _ready_locked(self) -> str:
        """``'decide'`` once every outstanding piece of this group has landed."""
        if self.decided or self.n_built < self.size or not self.rubric_done:
            return ''
        if any(p.statement and p.n_solved < self.rollouts for p in self.proposals):
            return ''
        self.decided = True
        return 'decide'

    def statements(self) -> List[Proposal]:
        return [p for p in self.proposals if p.statement]



# ── The run ────────────────────────────────────────────────────────────────


class Run:
    """One collection pass: the resources, the queue, and the three job bodies.

    Sandbox jobs go on one FIFO queue served by one thread per slot, so a slot is
    never idle while there is work. Rubric jobs go to a separate pool because they
    need no sandbox, and putting them on the same queue would let a group's
    judgement wait behind the solver attempts of another group.

    Nothing in a job waits for another job. A build enqueues its solver attempts
    and returns its slot; a group is decided by whichever thread happens to land
    the last outstanding piece. That is what keeps the pool from deadlocking on
    itself, which a build that waited for its own solvers would do at once.
    """

    def __init__(self, args, sampler, template, slots: List[Any], recorder: Recorder):
        self.args = args
        self.slots = slots
        self.rec = recorder
        self.rng = random.Random(args.random_seed or None)

        # Two rollouts over one sampler and one template: they differ only in the
        # turn budget, and a per-call override for that does not exist. The
        # trajectory-level state a rollout keeps is all local to __call__, so both
        # are called from every thread at once with a per-call tool_manager.
        self.propose_params = SamplingParams(
            max_tokens=args.propose_max_tokens, num_samples=1, logprobs=1,
            temperature=args.propose_temp, top_p=0.95,
            stop=['</tool_call>'] if args.one_call_per_reply else None,
            include_stop_str_in_output=bool(args.one_call_per_reply))
        self.solve_params = SamplingParams(
            max_tokens=args.solver_max_tokens, num_samples=1, logprobs=1,
            temperature=args.solver_temp, top_p=0.95,
            stop=['</tool_call>'] if args.one_call_per_reply else None,
            include_stop_str_in_output=bool(args.one_call_per_reply))
        self.propose_rollout = MultiTurnRollout(
            sampler, template=template, tool_manager=ToolManager(),
            max_turns=args.max_turns, stop_after_stuck_turns=args.stop_after_stuck_turns,
            sampling_params=self.propose_params)
        self.solve_rollout = MultiTurnRollout(
            sampler, template=template, tool_manager=ToolManager(),
            max_turns=args.solver_max_turns,
            stop_after_stuck_turns=args.stop_after_stuck_turns,
            sampling_params=self.solve_params)
        self.keyword_rollout = MultiTurnRollout(
            sampler, template=template, tool_manager=ToolManager(), max_turns=1,
            sampling_params=SamplingParams(max_tokens=args.keyword_max_tokens,
                                           num_samples=1, logprobs=1,
                                           temperature=args.keyword_temp, top_p=0.98))

        from twinkle_agentic.protocol.openai import OpenAI
        self.api = OpenAI(model=args.api_model, api_key=args.api_key or None,
                          base_url=args.api_base)
        self.api_extra = ({'thinking_budget': args.api_thinking_budget}
                          if args.api_thinking_budget > 0 else None)
        self.check_params = SamplingParams(max_tokens=args.check_max_tokens, num_samples=1,
                                           temperature=args.propose_temp, top_p=0.95)
        self.problem_params = SamplingParams(max_tokens=args.problem_max_tokens,
                                             num_samples=1, temperature=args.propose_temp,
                                             top_p=0.95)
        # Keeps the local path's temperature and top_p rather than the 1.0/0.95 the
        # other two API stages use. The high temperature is deliberate here -- the
        # bank is worthless if every refill returns the same phrases -- and moving
        # the model and the temperature in one step would leave no way to tell which
        # one changed the result.
        self.keyword_api_params = SamplingParams(max_tokens=args.keyword_max_tokens,
                                                 num_samples=1,
                                                 temperature=args.keyword_temp,
                                                 top_p=0.98)

        # Built once: the cap is part of the system prompt, so a build that got a
        # different one would be a different experiment.
        self.system = P.SYSTEM + (P.BUILD_SIZE_CAP.format(n=args.max_build_files)
                                  if args.max_build_files > 0 else '')
        self.store = KeywordStore(args.keyword_db, P.CATEGORIES)
        self.bank = TaskBank(args.task_bank, refs=args.task_bank_refs) if args.task_bank else None
        # ms-agent builds the solver's opening messages, and it does so through a
        # stateful agent object -- so one instance, one lock, and only for the few
        # milliseconds it takes to shape two messages.
        self.harness = solver_harness(args.agent_config)
        self.harness_lock = threading.Lock()

        self.jobs: 'queue.Queue' = queue.Queue()
        self.api_pool = ThreadPoolExecutor(max_workers=args.api_concurrency,
                                           thread_name_prefix='api')
        self.state = threading.Lock()
        self.kw_lock = threading.Lock()
        # Jobs actually being worked on right now, sandbox and API. Only used by
        # the stall check in run(): 'the queue is empty' is not 'there is nothing
        # left to do' while a thread is still inside a job that will queue more.
        self.busy = 0
        self.api_jobs = 0
        self.nonce = 0
        # (category, keyword) pairs behind tasks nobody solved. Read at the end by
        # expand_hard_keywords, which asks for more in the same domains.
        self.hard: List[Tuple[str, str]] = []
        self.kept: List[Group] = []
        self.groups: List[Group] = []
        self.n_launched = 0
        self.stop = threading.Event()
        self.counts: Dict[str, int] = {}

    # ---------------------------------------------------------------- helpers

    def bump(self, key: str, n: int = 1) -> None:
        with self.state:
            self.counts[key] = self.counts.get(key, 0) + n

    def draw_keywords(self) -> Tuple[List[Tuple[str, str]], str]:
        """One entry from each category, refilling any that has run dry.

        On its own lock, not ``state``: a refill is eight model calls and holding
        the lock every ``bump`` needs for that long would stall all 32 slots. Two
        threads drawing at once still have to take turns, or the second refill's
        prompt would not know what the first one had just said.
        """
        with self.kw_lock:
            for category in P.CATEGORIES:
                if not self.store.unused(category):
                    self.refill(category)
            picks = []
            for category in P.CATEGORIES:
                text = self.store.take(category, self.rng)
                if text is not None:
                    picks.append((category, text))
        return picks, '\n'.join(f'- {c}: {t}' for c, t in picks)

    def refill(self, category: str) -> None:
        """Ask the local model for more keywords in ``category``.

        Says so when it comes back empty. A silent no-op here is the worst outcome
        available: every proposal then falls back to a keyword-less prompt and the
        run looks normal while producing one identical prompt over and over. That
        is exactly what happened for whole runs when the prompt asked for one
        keyword per line and the parser wanted a JSON array.

        The calls run one at a time so each can be told what the ones before it
        said; each is answered by a rollout with ``max_turns=1``, which ends the
        trajectory before any tool could be dispatched -- brainstorming a list is a
        text round, and a bracketed list in a reply is exactly what a tool-calling
        rollout would try to run.
        """
        for attempt in range(1, max(1, self.args.keyword_refill_tries) + 1):
            if self.generate_keywords(category):
                return
            logger.warning(f'[challenge] keyword refill for {category!r} produced '
                           f'nothing new on try {attempt}')
        if self.store.items[category]:
            # Every keyword marked unused again. The alternative is a category that
            # can never be drawn from, which stops the run: a repeat draw is worse
            # than no run only if diversity matters more than collecting anything.
            self.store.recycle(category)
            self.store.save()
            logger.warning(f'[challenge] keyword category {category!r} exhausted -> '
                           f'recycled {len(self.store.items[category])} topics')

    def generate_keywords(self, category: str) -> bool:
        """One refill round. True when it added something the bank did not have."""
        want = self.args.keywords_n
        calls = max(1, self.args.keyword_gen_calls)
        per_call = max(1, -(-want // calls) + 4)
        known = self.store.texts(category)
        fresh: List[str] = []
        seen = {t.strip().lower() for t in known}
        for i in range(calls):
            # Newest first: the calls run one at a time so each can avoid what the
            # ones before it said, and letting older entries evict those would undo
            # it. Past the cap the oldest of this refill's phrases fall off, which
            # is also the least costly thing to drop.
            avoid = (fresh + known)[:AVOID_TOTAL]
            self.nonce += 1
            user = (P.KEYWORD_USER.format(k=per_call, desc=P.CATEGORY_DESC[category])
                    + ('\nDo NOT repeat any of these already-used topics: '
                       + ', '.join(avoid) if avoid else '')
                    + f'\n(batch {self.nonce}-{i})')
            # Asked of the API model rather than the local one. Keyword text never
            # enters a trajectory -- it is parsed into a list and thrown away -- so
            # this adds no untrained tokens, which is the rule that decides what may
            # use the API. And the bank is the single input every task downstream is
            # built from: measured over the 1344 keywords iterations 1-7 generated
            # locally at temperature 1.3, against category rules the model is shown
            # in full, 31% of transform named an activity on a running system rather
            # than a computation, 13% of domain named an operation rather than
            # material, and 24% of edge_case needed hardware the container does not
            # have. Downstream, 42-70% of statements described themselves as
            # simulating their own subject matter, which is what a keyword the
            # sandbox cannot honour turns into. A 4B policy at that temperature is
            # the wrong instrument for a constraint list this long.
            #
            # Falls back to the local model instead of giving up: an unreachable API
            # must not leave a category dry, because dry means keyword-less prompts
            # and a run that looks healthy while producing one prompt over and over
            # -- the exact failure the refill logic already guards against.
            out = None
            messages = [{'role': 'system', 'content': P.KEYWORD_SYSTEM}]
            reply = api_one(self.api, messages, user, self.keyword_api_params,
                            self.api_extra)
            via = 'api'
            if reply is None:
                traj = {'messages': [{'role': 'system', 'content': P.KEYWORD_SYSTEM},
                                     {'role': 'user', 'content': user}]}
                out = self.keyword_rollout([traj])
                reply = self._assistant_text(out[0] if out else {})
                via = 'local-fallback'
            parsed, dropped_long = split_keyword_list(reply)
            new = [k for k in parsed if k.lower() not in seen]
            for keyword in new:
                seen.add(keyword.lower())
            fresh.extend(new)
            self.rec.keywords({'category': category, 'prompt': user, 'reply': reply,
                               'parsed': parsed, 'n_parsed': len(parsed),
                               'n_new': len(new), 'via': via,
                               'dropped_long': dropped_long,
                               'n_dropped_long': len(dropped_long),
                               'stop_reason': (out[0].get('stop_reason') if out else None),
                               'truncated': bool(out[0].get('truncated')) if out else None})
            if len(fresh) >= want:
                break
        added = self.store.add(category, fresh[:want], source='gen')
        if added:
            # Written now rather than at the end of the run: a run that crashes after
            # spending eight model calls on keywords should not have to spend them
            # again, and the next iteration reads this file to know what was used.
            self.store.save()
            logger.info(f'[challenge] keywords {category!r} +{added}')
        return bool(added)

    @staticmethod
    def _assistant_text(traj: Dict[str, Any]) -> str:
        for message in reversed((traj.get('messages') if traj else []) or []):
            if message.get('role') == 'assistant':
                return message.get('content') or ''
        return ''

    def expand_hard_keywords(self) -> int:
        """More keywords in the domains that produced tasks nobody solved.

        Run once at the end, so what it adds is there for the next iteration rather
        than for the groups still in flight. One call per hard keyword, capped at
        32 of them: this is the only feedback the keyword bank gets from difficulty,
        and without it the bank drifts wherever the refill prompt happens to go.
        """
        with self.state:
            hard = list(self.hard)[:32]
        if not hard:
            return 0
        added = 0
        for i, (category, keyword) in enumerate(hard):
            self.nonce += 1
            user = (P.KEYWORD_EXPAND_USER.format(kw=keyword, m=8,
                                                 desc=P.CATEGORY_DESC[category])
                    + f'\n(batch {self.nonce}-{i})')
            # Same reasoning as the refill path: the API model, falling back to the
            # local one. This path matters more, not less -- it wrote 960 of the 1344
            # keywords the first seven iterations banked, at more than twice their
            # rule-break rate, so it is the one shaping what later iterations draw.
            out = None
            messages = [{'role': 'system', 'content': P.KEYWORD_SYSTEM}]
            reply = api_one(self.api, messages, user, self.keyword_api_params,
                            self.api_extra)
            via = 'api'
            if reply is None:
                traj = {'messages': [{'role': 'system', 'content': P.KEYWORD_SYSTEM},
                                     {'role': 'user', 'content': user}]}
                out = self.keyword_rollout([traj])
                reply = self._assistant_text(out[0] if out else {})
                via = 'local-fallback'
            parsed, dropped_long = split_keyword_list(reply)
            added += self.store.add(category, parsed, source='expand', parent=keyword)
            # The prompt goes in whole, as the refill path already does. It used to
            # record the literal string 'expand', which made this file unable to
            # answer the one question it gets asked -- whether a change to
            # KEYWORD_EXPAND_USER was live in a given iteration -- and cost an
            # afternoon to a wrong answer inferred from mtimes instead.
            #
            # ``dropped_long`` for the same reason one step further in: iteration 9
            # recorded n_parsed 0 on four of six expand calls whose replies were
            # well-formed JSON, and nothing in this file said the phrases had been
            # thrown away for length rather than never produced.
            self.rec.keywords({'category': category, 'parent': keyword, 'prompt': user,
                               'reply': reply, 'parsed': parsed, 'n_parsed': len(parsed),
                               'dropped_long': dropped_long,
                               'n_dropped_long': len(dropped_long), 'via': via})
        self.store.save()
        logger.info(f'[challenge] expanded {len(hard)} hard keyword(s) -> '
                    f'+{added} same-domain topics')
        return added

    def launch_group(self) -> Optional[Group]:
        """Draw a topic and queue its ``group_size`` builds. None once at the cap."""
        with self.state:
            if self.stop.is_set():
                return None
            if self.args.max_group_attempts and self.n_launched >= self.args.max_group_attempts:
                return None
            gid = self.n_launched
            self.n_launched += 1
        picks, block = self.draw_keywords()
        if len(picks) != len(P.CATEGORIES):
            # Every proposal's prompt is the keyword draw, so there is no honest
            # prompt to send without one. Stopping is the reportable outcome; a
            # substitute prompt would change what is being trained and say nothing.
            logger.error(f'[challenge] keyword bank gave {len(picks)} of '
                         f'{len(P.CATEGORIES)} categories; cannot build a prompt')
            return None
        prompt = P.FROM_KEYWORDS.format(keywords=block)
        group = Group(gid, picks, block, prompt, self.args.group_size,
                      self.args.solver_rollouts)
        with self.state:
            self.groups.append(group)
        for prop in group.proposals:
            self.jobs.put(lambda slot, p=prop: self.build_job(p, slot))
        logger.info(f'[challenge] group {gid} launched: {block.replace(chr(10), " | ")}')
        return group

    # ------------------------------------------------------------------ jobs

    def build_job(self, prop: Proposal, slot) -> None:
        """Stage 1-3 for one proposal, then hand its statement to eight solvers."""
        if prop.group.dropped:
            self.bump('build_skipped')
            return
        try:
            self.build(prop, slot)
        except Exception as e:  # noqa: BLE001 -- one bad build must not end the run
            logger.warning(f'[challenge] build g{prop.group.id}/{prop.idx} raised: '
                           f'{type(e).__name__}: {e}')
            prop.outcome, prop.detail = 'build_error', f'{type(e).__name__}: {e}'
        self.bump(f'build:{prop.outcome}')
        if not prop.statement:
            self.record_rejection(prop)
        if prop.statement and not prop.group.dropped:
            for _ in range(self.args.solver_rollouts):
                self.jobs.put(lambda s, p=prop: self.solve_job(p, s))
        action = prop.group.built(prop)
        if action == 'rubric':
            with self.state:
                self.api_jobs += 1
            self.api_pool.submit(self.rubric_job, prop.group)
        elif action == 'decide':
            self.decide(prop.group)

    def record_rejection(self, prop: Proposal) -> None:
        """Why this build produced no task, with enough of the episode to tell.

        How the episode ended travels with the reason. A reason on its own is not
        diagnosable: whether a model that left an empty workspace ran out of tokens
        or simply emitted no tool call is answered by stop_reason and the call
        count, not by the word 'empty_workspace'.
        """
        traj = prop.traj or {}
        messages = traj.get('messages') or []
        self.rec.rejected({
            'group_id': prop.group.id,
            'proposal_idx': prop.idx,
            'reason': prop.outcome,
            'detail': prop.detail,
            'keywords': prop.group.keywords,
            'stop_reason': traj.get('stop_reason'),
            'truncated': bool(traj.get('truncated')),
            'stuck_stop': bool(traj.get('stuck_stop')),
            'tool_stop': traj.get('tool_stop'),
            'turns': traj.get('turns'),
            'n_assistant': sum(1 for m in messages
                               if isinstance(m, dict) and m.get('role') == 'assistant'),
            'n_tool_calls': sum(len(m.get('tool_calls') or []) for m in messages
                                if isinstance(m, dict)),
            'last_assistant': self._assistant_text(traj),
            'check': prop.check,
        })

    def build(self, prop: Proposal, slot) -> None:
        """Build in the sandbox, then have the API write the check and the task.

        The build is the trainable part and runs on the local model. The two stages
        after it are appended to a *copy* of its messages and answered by the API,
        so the check script and the statement are written with the whole build
        history in view while the trajectory keeps exactly the tokens the local
        model produced.
        """
        args = self.args
        slot.clear()
        traj = {'messages': [{'role': 'system', 'content': self.system},
                             {'role': 'user', 'content': prop.group.prompt}],
                'tools': slot.schemas}
        prop.traj = rollout_one(self.propose_rollout, traj, self.propose_params, slot)
        if prop.traj is None:
            prop.outcome = 'rollout_empty'
            return
        if prop.traj.get('stop_reason') == 'length':
            # A reply cut off at the token budget never finished its thought, so
            # continuing the conversation over the API would write a check against
            # a half-written turn. The trajectory is kept and trains with reward 0;
            # what stops here are the two stages after it.
            prop.outcome = 'cut_short'
            prop.detail = f'stop_reason=length after {prop.traj.get("turns")} turn(s)'
            return

        snapshot, error = slot.snapshot()
        if error:
            # Not filed as an empty workspace: a snapshot that says "empty" when it
            # means "I could not look" produces tasks whose only true assertion is
            # that nothing happened.
            prop.outcome, prop.detail = 'snapshot_unavailable', error
            return
        if not snapshot:
            prop.outcome = 'empty_workspace'
            return

        messages = [dict(m) for m in prop.traj.get('messages') or []]
        user_text = P.CHECK_FOLLOWUP.format(final_state=snapshot)
        attempt = 0
        while True:
            attempt += 1
            reply = api_one(self.api, messages, user_text, self.check_params, self.api_extra)
            if reply is None:
                prop.outcome, prop.detail = 'api_error', 'check-script call failed'
                return
            script = parse_check_script(reply)
            if script is None:
                if attempt <= args.check_retries:
                    user_text = P.CHECK_RETRY_FOLLOWUP.format(
                        error='Could not read a check script from your reply: it was '
                              'not a fenced python code block. Do not wrap it in a '
                              'tool call and do not add prose -- return ONLY a fenced '
                              'python code block.',
                        final_state=snapshot)
                    continue
                prop.outcome, prop.detail = 'check_parse_fail', reply
                return
            # Rejected on the syntax tree before it can pass on the author's own
            # state, since passing there is exactly what hides the defect: a check
            # that pins a file's size or quotes a script's source passes for its
            # author and fails every correct reproduction.
            brittle = brittle_check_reason(script)
            exit_code, output = (1, brittle) if brittle else call_one(slot, script)
            if exit_code == 0:
                prop.check = script
                break
            after, _ = slot.snapshot()
            if attempt <= args.check_retries:
                user_text = P.CHECK_RETRY_FOLLOWUP.format(error=output,
                                                          final_state=after or snapshot)
                continue
            prop.outcome = 'check_run_fail'
            prop.detail = (f'exit {exit_code}\n{output}\n--- check script ---\n{script}'
                           f'\n--- state after check ---\n{after}')
            return

        reply = api_one(self.api, messages, P.PROBLEM_FOLLOWUP, self.problem_params,
                        self.api_extra)
        if reply is None:
            prop.outcome, prop.detail = 'api_error', 'problem-statement call failed'
            return
        statement = parse_problem_statement(reply)
        if not statement:
            prop.outcome, prop.detail = 'problem_parse_fail', reply
            return
        if len(statement) > args.problem_max_chars:
            prop.outcome = 'too_long'
            prop.detail = f'{len(statement)} chars > {args.problem_max_chars}'
            return
        prop.statement = statement
        prop.outcome = 'ok'

    def solve_job(self, prop: Proposal, slot) -> None:
        """One attempt at ``prop``'s task, scored by ``prop``'s own check script.

        A truncated attempt is a failed attempt: it left a workspace the check
        rejects, and the denominator stays at ``solver_rollouts`` so the same
        ``n_pass`` means the same thing in every group.
        """
        if prop.group.dropped:
            self.bump('solve_skipped')
            return
        attempt, passed = None, False
        exit_code, output, end_state = None, '', ''
        try:
            slot.clear()
            with self.harness_lock:
                opening = self.harness.start(prop.statement)
            if not opening.get('tools'):
                # The harness only shapes messages -- its tool list is empty on
                # purpose -- so the schemas come from the slot that will run them.
                opening['tools'] = slot.schemas
            attempt = rollout_one(self.solve_rollout, opening, self.solve_params, slot)
            if attempt is not None:
                exit_code, output = call_one(slot, prop.check)
                passed = exit_code == 0
            # Read after the check, not before: the check is allowed to write, and
            # what a reader of a failed attempt needs is the workspace the check
            # was unhappy with.
            end_state, _ = slot.snapshot()
        except Exception as e:  # noqa: BLE001 -- a lost attempt is a failed attempt
            logger.warning(f'[challenge] solve g{prop.group.id}/{prop.idx} raised: '
                           f'{type(e).__name__}: {e}')
            output = f'{type(e).__name__}: {e}'
        self.rec.attempt({
            'group_id': prop.group.id,
            'proposal_idx': prop.idx,
            'statement': prop.statement,
            'check_script': prop.check,
            'passed': passed,
            'check_exit': exit_code,
            'check_output': output,
            # A cut-off reply counts as a failed attempt and stays in the
            # denominator, so the flag travels with the record for that to be
            # checkable from the file rather than taken on trust.
            'truncated': bool((attempt or {}).get('truncated')),
            'stop_reason': (attempt or {}).get('stop_reason'),
            'turns': (attempt or {}).get('turns'),
            'messages': (attempt or {}).get('messages') or [],
            'end_state': end_state,
        })
        self.bump('solve_pass' if passed else 'solve_fail')
        if prop.group.solved(prop, attempt, passed) == 'decide':
            self.decide(prop.group)

    def rubric_job(self, group: Group) -> None:
        """Score the group's statements for novelty, all against each other.

        The siblings are the references that matter: a whole group can be scored
        identically novel against history while being eight versions of one idea,
        and GRPO subtracts the group mean, so a term identical across the group
        produces no gradient at all. That is why this waits for all eight builds
        instead of scoring each statement as it lands -- and why waiting costs
        nothing: the slots are held by other groups' jobs the whole time.

        Retried up to ``--novelty-tries`` times. If the last one still has no
        verdict for some statement, the group is dropped and its pending solver
        attempts are skipped.

        Wrapped whole, because this is the one job whose exceptions nobody would
        see: it runs on a pool whose futures are never read, so a raise in here
        left the group waiting for a verdict that never came, and the run then sat
        with an empty queue and idle slots until it was killed. Anything
        unexpected drops the group instead of stalling everything.
        """
        try:
            self._rubric(group)
        except Exception as e:  # noqa: BLE001
            logger.warning(f'[challenge] rubric for group {group.id} raised: '
                           f'{type(e).__name__}: {e}')
            if group.abandon(f'rubric_error: {type(e).__name__}: {e}'):
                self.bump('group_dropped:rubric_error')
                self.decide(group)
        finally:
            with self.state:
                self.api_jobs -= 1

    def _rubric(self, group: Group) -> None:
        if group.dropped:
            return
        props = group.statements()
        if self.bank is None:
            # No bank means no reference set, so nothing to be novel against.
            # Novelty stays None and the reward is the pass-rate gaussian alone.
            self._advance(group, group.judged())
            return
        from twinkle_agentic.verifier import DIMENSIONS, score_tasks
        texts = [p.statement for p in props]
        pending = list(range(len(props)))
        for attempt in range(1, max(1, self.args.novelty_tries) + 1):
            payload = [{
                'statement': texts[i],
                'check': props[i].check,
                'references': self.bank.references(
                    texts[i], extra=[t for j, t in enumerate(texts) if j != i]),
            } for i in pending]
            results = score_tasks(payload, workers=self.args.api_concurrency,
                                  model=self.args.api_model,
                                  extra_body=self.api_extra)
            still: List[int] = []
            for i, task, result in zip(pending, payload, results):
                score = result.scores.get('novelty')
                self.rec.novelty({
                    'group_id': group.id, 'proposal_idx': props[i].idx, 'try': attempt,
                    **{dim: result.scores.get(dim) for dim in DIMENSIONS},
                    'verdicts': result.verdicts, 'n_votes': result.n_votes,
                    'error': result.error,
                    'n_references': len(task.get('references') or ()),
                    # Full text on both sides: this file is read to check whether a
                    # score was deserved, which a shortened statement cannot answer.
                    'statement': task.get('statement') or '',
                    'references': list(task.get('references') or ()),
                })
                if score is None:
                    still.append(i)
                else:
                    props[i].novelty = float(score)
            if not still:
                break
            pending = still
            logger.warning(f'[challenge] group {group.id}: {len(still)} statement(s) '
                           f'came back without a novelty verdict (try {attempt})')
        else:
            if pending and group.abandon(f'novelty_unscored x{len(pending)}'):
                self.bump('group_dropped:novelty')
                self.decide(group)
                return
        self._advance(group, group.judged())

    # -------------------------------------------------------------- decision

    def decide(self, group: Group) -> None:
        """Keep or drop the group, then start a replacement or stop the run.

        The second half is in a ``finally`` because the first half writes files: a
        raise while writing used to take the replacement topic down with it, and
        the run then had one fewer group in flight for every failure until there
        was nothing left running and nothing left to wait for.
        """
        try:
            self._decide(group)
        except Exception as e:  # noqa: BLE001
            logger.warning(f'[challenge] deciding group {group.id} raised: '
                           f'{type(e).__name__}: {e}')
            self.bump('group_decide_error')
        finally:
            self._after_decision()

    def _decide(self, group: Group) -> None:
        rollouts = self.args.solver_rollouts
        floor = self.args.novelty_floor
        in_band = [p for p in group.proposals
                   if p.n_pass is not None and 1 <= p.n_pass <= rollouts - 1]
        chosen = max(in_band, key=lambda p: p.reward(rollouts, floor)) if in_band else None
        if chosen is not None and group.dropped:
            # An abandoned group can still have in-band proposals: its solver
            # attempts were already running when it was abandoned. Keeping it on
            # that basis would train on the very group that was judged unusable,
            # and would do it with a novelty term measured for some members and
            # missing for others.
            chosen = None
        if chosen is not None and not self._claim_keep(group):
            # The target was reached while this group was finishing. Claimed before
            # anything is written, because writing first and counting after is how
            # a run ends up with eleven groups on disk and a loader that reads a
            # different number of GRPO groups than the run reported.
            chosen = None
            self.bump('group_late')
        record = {
            'group_id': group.id,
            'kept': chosen is not None,
            'dropped': group.dropped,
            'keywords': group.keywords,
            'chosen': chosen.idx if chosen is not None else None,
            'n_in_band': len(in_band),
            'proposals': [{
                'idx': p.idx,
                'outcome': p.outcome,
                'n_pass': p.n_pass,
                'novelty': p.novelty,
                'reward': p.reward(rollouts, floor),
                'statement': p.statement,
                'check': p.check,
                'detail': p.detail,
            } for p in group.proposals],
        }
        self.rec.group(record)
        # Keyword draws behind tasks nobody solved, for expand_hard_keywords. Taken
        # from every decided group, kept or not: a task at n_pass=0 says the same
        # thing about its keywords either way.
        if any(p.statement and p.n_pass == 0 for p in group.proposals):
            with self.state:
                seen = {(c, t.lower()) for c, t in self.hard}
                for category, text in group.keywords:
                    if (category, text.lower()) not in seen:
                        self.hard.append((category, text))
        if chosen is None:
            self.bump('group_dropped' if not group.dropped else 'group_dropped_early')
            return

        # Every proposal of a kept group trains, including the ones that produced
        # no task: they are the zero-reward half of the GRPO group, and a set of
        # kept-only records has no variance to learn from.
        for prop in group.proposals:
            if prop.traj is None:
                continue
            self.rec.trajectory(
                prop.traj, side='propose', group_id=group.id, proposal_idx=prop.idx,
                reward=prop.reward(rollouts, floor), n_pass=prop.n_pass,
                novelty=prop.novelty, outcome=prop.outcome,
                keywords=group.keywords, selected=prop is chosen)
        # Only the chosen proposal's attempts. The others were measured and are
        # reported in groups.jsonl, but training on eight near-identical tasks from
        # one keyword draw is what a group of one keyword direction is meant to
        # avoid.
        for i, (attempt, passed) in enumerate(zip(chosen.attempts, chosen.passes)):
            if not attempt:
                continue
            self.rec.trajectory(attempt, side='solve', group_id=group.id,
                                proposal_idx=chosen.idx, attempt_idx=i,
                                reward=1.0 if passed else 0.0, passed=passed,
                                statement=chosen.statement)
        self.rec.task({'id': f'ag_g{group.id:04d}p{chosen.idx}',
                       'group_id': group.id, 'proposal_idx': chosen.idx,
                       'query': chosen.statement, 'check_script': chosen.check,
                       'n_pass': chosen.n_pass, 'n_rollouts': rollouts,
                       'novelty': chosen.novelty,
                       'reward': chosen.reward(rollouts, floor),
                       'keywords': group.keywords})
        if self.bank is not None:
            self.bank.add(chosen.statement, chosen.check, group_id=group.id,
                          n_pass=chosen.n_pass)
        self.bump('group_kept')

    def _claim_keep(self, group: Group) -> bool:
        """Take one of the ``--keep-groups`` slots, if there is one left.

        The slot is taken before the group's trajectories are written and released
        by nobody, so the number of groups on disk is exactly the number claimed
        even though several groups can finish at the same moment.
        """
        with self.state:
            if len(self.kept) >= self.args.keep_groups:
                return False
            self.kept.append(group)
            return True

    def _after_decision(self) -> None:
        """Stop the run if the target is met, otherwise start a replacement topic.

        Replacing one topic per decided group is what keeps the number of groups in
        flight at ``sandbox_slots / group_size`` without anything having to track
        it: the queue is fed by whatever finishes.
        """
        with self.state:
            enough = len(self.kept) >= self.args.keep_groups
        if enough:
            if not self.stop.is_set():
                logger.info(f'[challenge] {len(self.kept)} groups kept; stopping')
                self.stop.set()
            return
        if self.launch_group() is None and self._idle():
            logger.warning('[challenge] no topics left to try and nothing in flight; '
                           f'stopping with {len(self.kept)} kept group(s)')
            self.stop.set()

    def _idle(self) -> bool:
        with self.state:
            return all(g.decided for g in self.groups)

    def _advance(self, group: Group, action: str) -> None:
        if action == 'decide':
            self.decide(group)

    # ------------------------------------------------------------- the loop

    def work(self, slot) -> None:
        """One thread, one slot, jobs until the run stops."""
        while not self.stop.is_set():
            try:
                job = self.jobs.get(timeout=1.0)
            except queue.Empty:
                continue
            with self.state:
                self.busy += 1
            try:
                job(slot)
            except Exception as e:  # noqa: BLE001 -- never lose the thread
                logger.warning(f'[challenge] job on slot {slot.slot} raised: '
                               f'{type(e).__name__}: {e}')
            finally:
                with self.state:
                    self.busy -= 1
                self.jobs.task_done()

    def run(self) -> None:
        """Start one thread per slot, prime the queue, and wait for the target."""
        n_topics = max(1, len(self.slots) // self.args.group_size)
        threads = [threading.Thread(target=self.work, args=(slot,), daemon=True,
                                    name=f'slot{slot.slot}') for slot in self.slots]
        for thread in threads:
            thread.start()
        for _ in range(n_topics):
            if self.launch_group() is None:
                break
        # Waited on in slices rather than once, so a run that has stopped making
        # progress ends with the reason on stdout instead of sitting there. Two
        # consecutive idle checks, because one can catch the moment between a job
        # being taken off the queue and the counter going up.
        idle_rounds = 0
        while not self.stop.wait(STALL_CHECK_SECONDS):
            with self.state:
                quiet = self.busy == 0 and self.api_jobs == 0
                stuck = [g.id for g in self.groups if not g.decided]
            if not (quiet and self.jobs.empty()):
                idle_rounds = 0
                continue
            idle_rounds += 1
            if idle_rounds < 2:
                continue
            # Reached only when the run has gone quiet without meeting its target
            # and without deciding to stop, which is a bug rather than attrition:
            # normally either a group finishes (and launches a replacement) or
            # launch_group runs out and sets stop itself.
            logger.error(f'[challenge] nothing running and nothing queued after '
                         f'{len(self.kept)}/{self.args.keep_groups} kept groups '
                         f'and {self.n_launched} launched'
                         + (f'; group(s) {stuck} were never decided' if stuck else '')
                         + '. Stopping.')
            for group in list(self.groups):
                if group.abandon('never_decided'):
                    self.bump('group_dropped:never_decided')
                    self.decide(group)
            self.stop.set()
        for thread in threads:
            thread.join(timeout=self.args.sandbox_timeout)
        self.api_pool.shutdown(wait=True)


def collect_metrics(out_dir: str, counts: Dict[str, int], launched: int,
                    rollouts: int, wall: float) -> Dict[str, Any]:
    """What this collection produced, as numbers, for ``challenge_metrics.json``.

    Read back out of ``groups.jsonl`` rather than taken from the live objects, so
    the file cannot disagree with the audit files it sits next to, and so the same
    function can recompute the metrics for a directory that finished hours ago.

    Three sections, because they are read for different things:

    * ``scalars`` -- fixed keys, always present, every value a float or int. This
      is the set that goes to swanlab; a key appearing in one iteration and not the
      next would make a chart that means something different in each.
    * ``counts`` -- the raw bump counters, dynamic keys and all
      (``group_dropped:rubric_error`` only exists in a run where that happened).
      Kept here and not uploaded.
    * ``distributions`` -- the histograms behind the means, because a mean n_pass
      of 4 is a different collection depending on whether it came from eights and
      zeros or from fours.

    ``solve_pass_rate`` is over every solver attempt run, the number the user
    asked for as accuracy. It is not a fixed yardstick: the tasks change every
    iteration, so it moving says the pair moved, not which half.
    """
    path = os.path.join(out_dir, 'groups.jsonl')
    groups: List[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        groups.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    props = [p for g in groups for p in g.get('proposals') or []]
    with_stmt = [p for p in props if p.get('statement')]
    measured = [p for p in with_stmt if p.get('n_pass') is not None]
    in_band = [p for p in measured if 1 <= p['n_pass'] <= rollouts - 1]
    chosen = [next((p for p in g['proposals'] if p['idx'] == g.get('chosen')), None)
              for g in groups if g.get('kept')]
    chosen = [p for p in chosen if p is not None]
    novelty = [p['novelty'] for p in props if p.get('novelty') is not None]
    rewards = [p['reward'] for p in props if p.get('reward') is not None]
    passes = counts.get('solve_pass', 0)
    attempts = passes + counts.get('solve_fail', 0)
    kept = sum(1 for g in groups if g.get('kept'))

    def rate(num: float, den: float) -> float:
        return float(num) / den if den else 0.0

    scalars = {
        'groups_launched': launched,
        'groups_kept': kept,
        'groups_decided': len(groups),
        'group_keep_rate': rate(kept, len(groups)),
        'wall_seconds': round(wall, 1),
        'builds': len(props),
        'builds_with_statement': len(with_stmt),
        'build_statement_rate': rate(len(with_stmt), len(props)),
        # The accuracy: every solver attempt that ran, passed over total.
        'solve_attempts': attempts,
        'solve_pass_rate': rate(passes, attempts),
        # Of the tasks that were measured at all, how many landed in the band the
        # keep rule wants. This is the proposer's hit rate.
        'n_pass_in_band_rate': rate(len(in_band), len(measured)),
        'n_pass_mean': statistics.fmean(p['n_pass'] for p in measured) if measured else 0.0,
        'delivered_n_pass_mean':
            statistics.fmean(p['n_pass'] for p in chosen if p.get('n_pass') is not None)
            if chosen else 0.0,
        'proposer_reward_mean': statistics.fmean(rewards) if rewards else 0.0,
        'novelty_scored_rate': rate(len(novelty), len(with_stmt)),
        'novelty_mean': statistics.fmean(novelty) if novelty else 0.0,
        'novelty_zero_rate': rate(sum(1 for v in novelty if v == 0.0), len(novelty)),
    }
    return {
        'scalars': scalars,
        'counts': dict(sorted(counts.items())),
        'distributions': {
            'n_pass': {str(k): v for k, v in
                       sorted(collections.Counter(p['n_pass'] for p in measured).items())},
            'build_outcome': {k.split(':', 1)[1]: v for k, v in sorted(counts.items())
                              if k.startswith('build:')},
            'novelty': {str(round(v, 2)): n for v, n in
                        sorted(collections.Counter(novelty).items())},
        },
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recorder = Recorder(args.out_dir)
    sampler, template = initialize_device(args)
    slots = initialize_sandbox(args)
    run = Run(args, sampler, template, slots, recorder)
    started = time.time()
    try:
        run.run()
        # After the loop, not during: what it adds is for the next iteration, and
        # doing it here means a crash in collection does not also lose the bank.
        if args.keyword_expand:
            run.expand_hard_keywords()
    finally:
        rebuilds = close_pool(slots)
        recorder.close()
        if run.bank is not None:
            logger.info(f'[challenge] task bank: {run.bank.stats()}')
        run.store.save()
        if rebuilds:
            logger.warning(f'[challenge] sandboxes were rebuilt {rebuilds} time(s); '
                           f'the jobs in flight at those moments were lost')
        # Written after recorder.close(), so groups.jsonl is complete and flushed
        # before it is read back. In the finally block because a run that crashed
        # is the one whose numbers are most worth having.
        metrics = collect_metrics(args.out_dir, run.counts, run.n_launched,
                                  args.solver_rollouts, time.time() - started)
        with open(os.path.join(args.out_dir, 'challenge_metrics.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f'[challenge] {len(run.kept)}/{run.n_launched} groups kept in '
                    f'{time.time() - started:.0f}s, counts: '
                    f'{dict(sorted(run.counts.items()))}')
        logger.info(f'[challenge] metrics -> '
                    f'{os.path.join(args.out_dir, "challenge_metrics.json")}: '
                    f'{metrics["scalars"]}')


if __name__ == '__main__':
    main()
