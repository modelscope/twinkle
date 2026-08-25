# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI self-play, agentic half: generate training tasks by doing them first.

One model plays both roles, and one proposal is one conversation. It first acts
as an agent in a sandbox (multi-turn with tools), producing a trajectory and a
final workspace state; then, appended to that same conversation, it writes a
check script that verifies the end state, and finally describes the task as a
problem statement. The same model then attempts the problem multiple times, and
only problems it solves *sometimes* are kept.

The machinery lives in :mod:`twinkle_agentic.challenger`; the prompts live in
``prompts.py`` next to this file. What is here is the wiring: which model, how
many, the sandbox connection, and where the output goes.

Output format (one JSONL line per task):

    --out-flows  {id, query, check_script, n_pass, n_rollouts, keywords, seeded}

Run it as a Ray job (sampler only, no trainer)::

    python cookbook/rsi/agentic/challenge.py --keep-target 200
"""
import argparse
import json
import base64
import binascii
import hashlib
import os
import sys

import numpy as np
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams, user_data_get
from twinkle.sampler import vLLMSampler
from twinkle_agentic.challenger import AgenticChallenger, KeywordStore
from twinkle_agentic.envs import EnvTool
from twinkle_agentic.rollout import build_rollout
from twinkle_agentic.tools.tool_manager import ToolManager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode import solver_harness  # noqa: E402
from prompts import CATEGORIES, CATEGORY_DESC, agentic_prompts  # noqa: E402
from remote_tool_env import RemoteMsAgentToolEnv, tool_payload  # noqa: E402

logger = get_logger()

# Cleared through the python tool, not `rm -rf`: ms-agent's safety policy rejects
# `rm -rf` outright ("Blocked by safety rule"), and it rejects globs in write
# operations, which rules out `find -delete` too. The script asserts the
# directory really is empty, so a future policy change surfaces as a failed reset
# instead of tasks quietly inheriting the previous workspace.
#
# Module level so a test can drive the same string the run does; a copy in a test
# would keep passing after this one changed.
CLEAR_WORKSPACE = '''
import os, shutil
root = {workspace!r}
os.makedirs(root, exist_ok=True)
for name in os.listdir(root):
    path = os.path.join(root, name)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.remove(path)
leftover = os.listdir(root)
assert not leftover, 'workspace not empty after clear: %r' % (leftover,)
'''

# ── Arm B: copy the episode's input files out, and put them back later ──────
#
# The bytes travel, not a description of them: the model is not asked to write a
# script that recreates its own inputs, because a wrong one costs the whole
# episode and would fail exactly where the inputs are least ordinary.
#
# Read in slices because the executor truncates its output at roughly 8 KB. Each
# slice is base64 so any byte survives the trip, and the manifest's sha256 is
# what says the trip was faithful -- checked locally against the bytes that
# arrived, so a truncated read cannot pass as a smaller file.
INPUT_MANIFEST = '''
import hashlib, os
root = os.path.join({workspace!r}, 'input')
rows = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in {{'__pycache__', '.ipynb_checkpoints'}}]
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        with open(path, 'rb') as handle:
            body = handle.read()
        rows.append((os.path.relpath(path, {workspace!r}), len(body),
                     hashlib.sha256(body).hexdigest()))
for rel, size, digest in sorted(rows):
    print(rel, size, digest)
'''

INPUT_SLICE = '''
import base64, os
path = os.path.join({workspace!r}, {rel!r})
with open(path, 'rb') as handle:
    handle.seek({offset})
    print(base64.b64encode(handle.read({length})).decode())
'''

# What gets stored with the task and run before every attempt at it. Writes the
# captured bytes and nothing else: no cleanup, because whoever runs this has just
# cleared the workspace.
SETUP_SCRIPT_TEMPLATE = '''# Recreate the task's input files.
import base64, os, pathlib

FILES = {files!r}

for rel, payload in FILES.items():
    path = pathlib.Path(rel)
    if path.parent != pathlib.Path('.'):
        os.makedirs(path.parent, exist_ok=True)
    with open(path, 'wb') as handle:
        handle.write(base64.b64decode(payload))
'''

# The ground truth the check script is written against. A listing alone is not
# enough: three of the six rejected proposals in the first real run failed on a
# value the model recomputed from its own recollection ("Mean values mismatch")
# rather than read off the file, so the end state has to arrive as content, not
# just as names. Bounded on both axes -- 50 files, 600 bytes each, 6000 overall --
# because this goes into a prompt and a 100k artifact would push the trajectory
# it has to be read alongside out of the window.
#
# Walks the tree in python rather than shelling out to `find`: the same code then
# decides what is text, what is truncated, and what the budget was spent on,
# which a pipeline of find/head cannot report back.
#
# File bodies go out byte for byte. An earlier version printed `body.rstrip()`,
# which hid trailing newlines while the size column still counted them, so a
# check writer shown an 11-byte file whose content looked 10 characters long
# wrote `content == 'Mean: 63.9'` and the check failed against the very state it
# was written from. The listing is only ground truth if it does not tidy up.
#
# Facts *about* a file go in its header, never after its body. A note printed
# below the content is indistinguishable from content: annotated one file with a
# trailing `(no newline at end of file)` line and the next check script asserted
# the README's content ending in that sentence.
WORKSPACE_SNAPSHOT = '''
import os

root = {workspace!r}
skip = {{'.ms_agent', '__pycache__', '.ipynb_checkpoints', '.git'}}
rows = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip]
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        try:
            rows.append((os.path.relpath(path, root), os.path.getsize(path), path))
        except OSError:
            pass
rows.sort()
for rel, size, _ in rows[:{max_files}]:
    print(rel, size)

budget = {total_budget}
for rel, size, path in rows[:{max_files}]:
    if budget <= 0:
        break
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read({per_file} + 1)
    except (OSError, UnicodeDecodeError):
        continue          # binary or unreadable: the listing already names it
    if '\\x00' in text:
        continue
    body = text[:{per_file}]
    budget -= len(body)
    # The trailing-newline count is stated for every file, both ways. Saying it
    # only when it is absent made "this file ends with a newline" invisible, and
    # the check writer then compared exact bytes without one: in ex9 two of the
    # three checks that failed their own verification failed on exactly that --
    # the same reply asserted three files, guessed right on the two marked "no
    # newline at end" and wrong on the unmarked one.
    trailing = len(body) - len(body.rstrip(chr(10)))
    if len(text) > len(body):
        suffix = ' (first {per_file} bytes)'
    elif trailing == 0:
        suffix = ' (no newline at end)'
    else:
        suffix = ' (ends with %d newline character(s))' % trailing
    print()
    print('--- ' + rel + suffix + ' ---')
    print(body, end='')
    if not body.endswith(chr(10)):
        print()
'''


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model
    p.add_argument('--model-id', default='ms://Qwen/Qwen3-4B')
    p.add_argument('--template', default='Template',
                   help='template class in twinkle.template')
    p.add_argument('--sampler-gpus', type=int, default=4)
    # Challenger backend: local vLLM by default, or an OpenAI-compatible API for
    # the proposing side only (keywords + explore + check + statement). The
    # solver side of the difficulty stage still runs through the local sampler,
    # so --challenger-api is only usable with --solver-rollouts 0 (no local
    # sampler is built at all in that case). The three connection args default
    # to the LLM_BACKUP_* env vars the summarizer teacher already uses.
    p.add_argument('--challenger-api', action='store_true',
                   help='propose through an OpenAI-compatible API instead of local vLLM; '
                        'requires --solver-rollouts 0')
    p.add_argument('--challenger-api-model',
                   default=os.environ.get('LLM_BACKUP_MODEL', ''))
    p.add_argument('--challenger-api-base',
                   default=os.environ.get('LLM_BACKUP_BASE_URL', ''))
    p.add_argument('--challenger-api-key',
                   default=os.environ.get('LLM_BACKUP_API_KEY', ''))
    p.add_argument('--challenger-concurrency', type=int, default=8,
                   help='parallel API conversations (API backend only)')
    # Measured on qwen3.8-max at a ~15k-character exploration context: one turn
    # took 58s with the default (unbounded) thinking and 10s at 2048, because the
    # default spent ~5300 characters on reasoning per turn. 0 leaves the API's own
    # default in place; anything else is sent as extra_body={'thinking_budget': N}
    # on every proposing call (explore, check, statement, keyword generation).
    p.add_argument('--challenger-thinking-budget', type=int, default=0,
                   help='cap reasoning tokens per API call; 0 = leave the API default')
    # Split mode: explore on the local (trainable) vLLM model, but write the check
    # script (success judgement) and the problem statement over the API instead of
    # the local model. Only the exploration turns keep labels/logprobs and get
    # trained; the two API stages are text-only and never enter the trajectory.
    # Reuses the --challenger-api-* connection args, and is mutually exclusive with
    # --challenger-api (which sends the whole proposing side to the API).
    p.add_argument('--followup-api', action='store_true',
                   help='explore locally (trainable) but generate the check script and '
                        'problem statement over --challenger-api-* (e.g. qwen3-max); '
                        'only the exploration part is trained. Not with --challenger-api.')
    # How many episodes run at once, each in its own sandbox. An episode owns its
    # workspace from the reset until its check has run, so this is also the number
    # of sandboxes booted at startup. Default 48: episodes alternate between vLLM
    # generation (~24 concurrent sequences fit in KV cache) and sandbox execution,
    # so 48 keeps both the GPU cluster and the sandbox host saturated.
    p.add_argument('--episode-concurrency', type=int, default=48,
                   help='sandboxes to boot, and how many things run at once in both '
                        'stages: proposal episodes in flight, and solver attempts '
                        'per wave in the difficulty filter')
    # 40960, up from 32768, because one proposal is now a single conversation:
    # the tool-using turns, the check script and the problem statement all share
    # this window. Measured on ex9's three separate calls, the worst case summed
    # to about 25k tokens (12394 + 7643 + 2943 plus the appended messages), so
    # this leaves room for episodes that take more steps than ex9's 2-5.
    #
    # 40960 and not more: it is Qwen3-4B's max_position_embeddings, and vLLM
    # refuses to start above it -- 49152 was tried and rejected, since going past
    # a RoPE model's trained positions produces nan rather than longer context.
    p.add_argument('--max-model-len', type=int, default=40960)

    # Generation control
    p.add_argument('--keep-target', type=int, default=200,
                   help='how many tasks to keep; generation stops once reached')
    p.add_argument('--batch-size', type=int, default=0,
                   help='tasks per yielded batch (0 = one batch of --keep-target)')
    p.add_argument('--max-proposals-per-round', type=int, default=64,
                   help='max proposals per round (serial, so keep moderate)')
    p.add_argument('--seed-file', default='', help='seed jsonl with query field')
    p.add_argument('--seed-mix-prob', type=float, default=0.5)

    # Sampling params for the exploring stage (proposing)
    p.add_argument('--propose-temp', type=float, default=1.0)
    # 8192, not 4096. At 4096, 3 of 12 explore episodes ended on the first turn
    # with stop_reason=length and an untouched workspace: the model had written
    # 15k, 16k and 10k characters of <think>, two of them without ever closing the
    # tag, and one degenerating into a run of newlines. Nothing was dispatched, so
    # those three cost a full episode each and produced no end state to write a
    # check about. The trajectory ceiling and the engine's max_model_len are
    # 32768, well above prompt plus this.
    p.add_argument('--propose-max-tokens', type=int, default=8192)
    p.add_argument('--max-turns', type=int, default=24,
                   help='max tool-calling turns for the exploring stage')
    # One call per reply, because the calls in one reply run *concurrently*:
    # tool_manager.call_many hands a turn to Env.step_batch, which the sandbox
    # server runs through ms-agent's parallel_call_tool. The model writes them in
    # the order it means them to happen and gets none of the results, so a reply
    # that writes a file and then reads it back reads the file as it was before.
    # Measured on ex11: 13 of 36 episodes contain an observation that contradicts
    # the end state -- read_file answering FileNotFound for a file the snapshot
    # lists, glob answering with 0 files, `ls -R` missing a file written earlier
    # in the same reply -- and 3 of the 4 kept tasks are among them. One of those
    # kept tasks describes two files as "empty", which is what they were only
    # because the call that filled them had not run yet.
    #
    # It also removes the other failure of a batched reply: 6 of 36 episodes
    # spent the whole 8192-token budget on one reply holding 70 to 259 calls,
    # the tail of it the same read_file over and over, and were discarded whole.
    # A reply that can hold one call cannot do either.
    p.add_argument('--one-call-per-reply', action='store_true', default=True,
                   help='stop generation at </tool_call> so each reply carries a single '
                        'call and the model sees its result before choosing the next')
    p.add_argument('--no-one-call-per-reply', dest='one_call_per_reply',
                   action='store_false',
                   help='let a reply carry several calls, which then run concurrently')
    p.add_argument('--stop-after-stuck-turns', type=int, default=2,
                   help='end an episode after this many consecutive turns that made no '
                        'progress; 0 runs to --max-turns regardless. A turn counts as '
                        'stuck when every call in it came back an error, or every call '
                        'in it was byte-identical to one already made in the episode. '
                        'Replayed over 12 recorded episodes: errors alone would stop 1 '
                        'of 12 and save 9 of 239 calls, since the worst offenders mix a '
                        'failing call with a glob that succeeds; adding the repeat rule '
                        'stops 3 of 12 and saves 63 calls, and those 3 are exactly the '
                        'ones that spent 54, 84 and 17 calls to leave a script that '
                        'could not run.')

    # Problem statement
    p.add_argument('--problem-max-chars', type=int, default=8192)
    p.add_argument('--check-retries', type=int, default=1,
                   help='How many times a check script that fails is handed back, '
                        'with the traceback and the workspace listing, to be '
                        'rewritten. ex12 lost 36 of 72 proposals here, and 29 of '
                        'those were one assertion naming a value the model had '
                        'never read -- a row count, a nearly-right content '
                        'string, a timestamp -- on a workspace state that was '
                        'fine. 0 rejects on the first failure, as ex9-ex12 did.')
    # Budgets for the two stages appended to the episode. Separate numbers
    # because the two are not alike: writing the checks reads the whole episode
    # plus the end state and reasons at length (ex9's largest such reply was 7643
    # trainable tokens, so 4096 would cut the tail off and the proposal would be
    # discarded as unparseable), while the statement is prose and ex9's largest
    # was 2943.
    p.add_argument('--check-max-tokens', type=int, default=8192)
    p.add_argument('--problem-max-tokens', type=int, default=4096)

    # Keywords
    p.add_argument('--keywords-n', type=int, default=128,
                   help='per-category refill target; 0 disables keyword bank')
    p.add_argument('--keyword-db', default='output/rsi_agentic/keywords.jsonl')
    p.add_argument('--keyword-gen-calls', type=int, default=8)
    # How many of a refill's generating calls go out together. 1 means each is
    # told what the ones before it produced; the first round of arm measurements
    # effectively ran at 8, where the whole first refill went out with an empty
    # 'do not repeat' list and came back with synonyms of each other.
    p.add_argument('--keyword-refill-concurrency', type=int, default=1)
    p.add_argument('--keyword-refill-tries', type=int, default=2)
    p.add_argument('--keyword-temp', type=float, default=1.3)
    # 1024 measured 8 of 24 generation calls cut off at the budget with nothing
    # parseable: the model spends most of it listing candidates inside <think>,
    # rewrites the list two or three times, and the JSON array afterwards gets
    # severed mid-string. The successful calls landed just under 1024, so the cap
    # sat inside the distribution of working replies rather than beyond it.
    p.add_argument('--keyword-max-tokens', type=int, default=4096)
    p.add_argument('--single-kw-prob', type=float, default=0.1)
    p.add_argument('--combo-arity', default='triple', choices=['triple', 'mix'])
    p.add_argument('--arity-weights', default='',
                   help="'w1,w2,w3' for --combo-arity mix (empty = uniform)")

    # Difficulty filter
    # 8 attempts, keeping 2-6: with 4 attempts the band was 1-3 and ex9's
    # measured pass counts came out {0: 6, 1: 1, 4: 2} -- two-thirds of the
    # tasks landed on an end of the range where one attempt either way changes
    # the verdict. 8 costs twice the sandbox time per task and puts the kept
    # band around one third of attempts passing.
    p.add_argument('--solver-rollouts', type=int, default=8)
    p.add_argument('--solver-temp', type=float, default=1.0)
    # Same 8192 as the explore round, and for the same measured reason: at 4096,
    # 15 of 50 solver attempts ended on stop_reason=length with an untouched
    # workspace, and one task lost all four of its attempts that way and was
    # discarded as too hard. Raising it took that to 0 of 20. It has to stay in
    # step with --propose-max-tokens: a task the proposer needed room to build is
    # not solvable in less.
    p.add_argument('--solver-max-tokens', type=int, default=8192)
    p.add_argument('--solver-max-turns', type=int, default=24,
                   help='NOT WIRED: no solver_explorer is passed, so solver attempts '
                        'run through the same rollout as the proposing episodes and '
                        'obey --max-turns. Kept so the value can be set once the two '
                        'are separated; changing it alone has no effect.')
    p.add_argument('--keep-min-pass', type=int, default=2)
    p.add_argument('--keep-max-margin', type=int, default=2)

    # Sandbox
    p.add_argument('--sandbox-template', default='',
                   help='AgentENV/e2b template name (required)')
    p.add_argument('--sandbox-api-url', default='',
                   help='AgentENV server URL (or AENV_API_URL env var)')
    p.add_argument('--agent-config', default='cookbook/rsi/agentic/rsi_agent.yaml',
                   help='ms-agent yaml for the sandbox tool server')
    p.add_argument('--sandbox-timeout', type=int, default=900)
    p.add_argument('--workspace', default='/workspace',
                   help='working directory inside the sandbox')
    p.add_argument('--snapshot-max-files', type=int, default=50,
                   help='files listed in the end-state snapshot')
    p.add_argument('--snapshot-per-file', type=int, default=600,
                   help='bytes of each file shown to the check writer')
    p.add_argument('--snapshot-budget', type=int, default=6000,
                   help='total bytes of file content in the snapshot')

    # Output
    # ---- Experiment arms. Each isolates one measured failure and can be used
    # alone or stacked. Off by default, so an unflagged run is the old behaviour.
    #
    # A: ex11/ex12/ex13 statements quoted their own answer, because stage 3 asks
    # for the full end state while the solver starts empty -- the only way to say
    # what a derived file holds is to write out what was computed. Difficulty came
    # out 8/8 or 0/8. This makes the statement give input data verbatim and
    # everything derived as a rule.
    # C: apitest4's statements each wanted an 8-12 file package with a CLI, and
    # Qwen3-4B passed 0 of 96 -- 32 attempts spent the whole token budget typing
    # source, 64 declared success with the files unwritten.
    p.add_argument('--max-build-files', type=int, default=0,
                   help='arm C: cap the episode at this many files, no package, '
                        'no CLI with subcommands (0 = no cap)')
    # For measuring a configuration rather than filling a dataset: two arms are
    # only comparable when given the same number of tries.
    p.add_argument('--max-proposals-total', type=int, default=0,
                   help='stop after this many proposals regardless of keep-target '
                        '(0 = run until keep-target)')
    p.add_argument('--random-seed', type=int, default=0)
    p.add_argument('--out-flows', default='output/rsi_agentic/challenge_flows.jsonl')
    p.add_argument('--dump-rejected', default='output/rsi_agentic/challenge_rejected.jsonl')
    p.add_argument('--dump-propose-traj', default='output/rsi_agentic/propose_traj',
                   help='directory for the proposing rounds (token ids + logprobs, one npz '
                        'per attempt plus index.jsonl). Empty string turns it off; keeping '
                        'it is what leaves the door open to training the challenger itself.')
    p.add_argument('--dump-solver-attempts',
                   default='output/rsi_agentic/solver_attempts.jsonl',
                   help='one line per difficulty-stage solver attempt: the statement, the '
                        'check script, the attempt, the state it left and what the check '
                        'said. Without it a task measured 0 of 4 gives no way to tell an '
                        'impossible task from a statement that withholds what the check '
                        'demands. Empty string turns it off.')
    p.add_argument('--no-sort-by-difficulty', action='store_true')
    p.add_argument('--stage', default='all', choices=['all', 'keywords', 'explore'],
                   help="'keywords' runs step 1 only -- fill the keyword bank, draw "
                        'the combinations, write the proposal prompts they produce to '
                        '--out-flows, and exit without touching the sandbox. '
                        "'explore' adds steps 2-4: clear the workspace, run the "
                        'sandbox episode, snapshot the end state, and stop before the '
                        'check-writing round. Both exist because a stage that is '
                        'broken cannot be diagnosed from the far end of an '
                        'hours-long full run.')
    p.add_argument('--stage-proposals', type=int, default=16,
                   help='how many proposals --stage keywords or --stage explore runs')
    p.add_argument('--dump-explore', default='output/rsi_agentic/explore_episodes.jsonl',
                   help='one line per --stage explore episode: the prompt, every '
                        'message, every tool call and its observation, and the end '
                        'state the snapshot saw. Empty string turns it off.')
    p.add_argument('--dump-keyword-gen',
                   default='output/rsi_agentic/keyword_gen.jsonl',
                   help='one line per keyword-generation call: the prompt, the raw '
                        'reply, and what the parser made of it. Without it a bank that '
                        'stays empty gives no way to tell a disobedient model from a '
                        'parser that rejects valid output. Empty string turns it off.')
    return p.parse_args()


def build_env(args):
    """Create the long-lived sandbox environment."""
    template = args.sandbox_template or os.environ.get('AENV_TEMPLATE', '')
    api_url = args.sandbox_api_url or os.environ.get('AENV_API_URL', '')
    if not template:
        raise SystemExit('[challenge] --sandbox-template or AENV_TEMPLATE is required')
    if not api_url:
        raise SystemExit('[challenge] --sandbox-api-url or AENV_API_URL is required')

    env = RemoteMsAgentToolEnv(
        template=template,
        config_path=args.agent_config,
        api_url=api_url,
        workspace=args.workspace,
        sandbox_timeout=args.sandbox_timeout,
    )
    env.reset()
    return env


def main():
    args = parse_args()
    for path in (args.out_flows, args.dump_rejected, args.keyword_db,
                 args.dump_keyword_gen, args.dump_explore):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

    # Build the proposing backend: an OpenAI-compatible API, or a local vLLM
    # sampler. The API path skips twinkle.initialize entirely -- it needs no GPUs
    # -- and passes no template, since the API re-sends messages as text rather
    # than splicing token ids the way the sampler continuation does.
    use_api = args.challenger_api
    if args.followup_api:
        # Split mode needs the local sampler for exploration (that is the trainable
        # half), so it cannot run under --challenger-api, which builds no sampler.
        if use_api:
            raise SystemExit('[challenge] --followup-api and --challenger-api are mutually '
                             'exclusive: --followup-api explores on the local sampler and '
                             'sends only the check/statement stages to the API, while '
                             '--challenger-api sends the whole proposing side to the API.')
        if not args.challenger_api_model or not args.challenger_api_base:
            raise SystemExit('[challenge] --followup-api needs --challenger-api-model and '
                             '--challenger-api-base (or LLM_BACKUP_MODEL / '
                             'LLM_BACKUP_BASE_URL).')
    if use_api:
        if args.solver_rollouts:
            raise SystemExit('[challenge] --challenger-api needs --solver-rollouts 0: the '
                             'solver side still runs on the local sampler, which is not '
                             'built in API mode.')
        if not args.challenger_api_model or not args.challenger_api_base:
            raise SystemExit('[challenge] --challenger-api needs --challenger-api-model and '
                             '--challenger-api-base (or LLM_BACKUP_MODEL / LLM_BACKUP_BASE_URL).')
        from twinkle_agentic.protocol.openai import OpenAI
        backend = OpenAI(model=args.challenger_api_model,
                         api_key=args.challenger_api_key or None,
                         base_url=args.challenger_api_base)
        template = None
        logger.info(f'[challenge] proposing via API model={args.challenger_api_model} '
                    f'base={args.challenger_api_base}')
    else:
        # Initialize twinkle (sampler only, no trainer)
        twinkle.initialize(
            mode='ray', nproc_per_node=args.sampler_gpus, lazy_collect=False,
            groups=[DeviceGroup(name='sampler', ranks=list(range(args.sampler_gpus)),
                                device_type='GPU')])
        backend = vLLMSampler(
            model_id=args.model_id,
            engine_args={'gpu_memory_utilization': 0.8, 'max_model_len': args.max_model_len},
            device_mesh=DeviceMesh.from_sizes(world_size=args.sampler_gpus,
                                              dp_size=args.sampler_gpus),
            remote_group='sampler',
        )
        backend.set_template(args.template, model_id=args.model_id, enable_thinking=True,
                             max_length=args.max_model_len)

        import twinkle.template as template_module
        template = getattr(template_module, args.template)(
            args.model_id, max_length=args.max_model_len, enable_thinking=True)

    # Build sandbox environments -- one per episode slot, since an episode owns
    # its workspace from the reset until its check has run and two episodes
    # sharing a sandbox would read each other's files. Skipped for
    # --stage keywords: that stage only brainstorms and draws keywords, and
    # booting a microVM to do it would make checking step 1 depend on the one part
    # of the setup most likely to be down.
    #
    # ``envs[0]`` is also the one the serial paths use (the difficulty stage, and
    # --stage explore), so ``env`` stays a name for it.
    envs = []
    env = None
    schemas = None
    tool_manager = ToolManager()
    episode_tool_managers = None
    if args.stage != 'keywords':
        n_slots = max(1, args.episode_concurrency)
        if n_slots == 1:
            envs = [build_env(args)]
        else:
            # Booted in parallel: each is a microVM taking ~10s, and doing eight
            # of them one after another would put a minute and a half in front of
            # every run.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_slots) as pool:
                envs = list(pool.map(lambda _: build_env(args), range(n_slots)))
        env = envs[0]
        schemas = env.tool_schemas()
        # One ToolManager per sandbox: the tools carry the env they dispatch into,
        # so slot i's model turns have to go through slot i's manager.
        episode_tool_managers = [ToolManager(EnvTool.from_schemas(e, schemas)) for e in envs]
        tool_manager = episode_tool_managers[0]
        logger.info(f'[challenge] {len(envs)} sandbox(es) ready '
                    f'(episode concurrency {n_slots})')

    # Explorer: multi-turn rollout with sandbox tools
    # ``stop`` ends the reply at the end of the first tool call, and
    # ``include_stop_str_in_output`` keeps that '</tool_call>' in what the policy
    # is trained on -- vLLM drops the matched stop by default, which would train
    # every turn to end on an unclosed block.
    explore_stop = ['</tool_call>'] if (args.one_call_per_reply and not use_api) else None
    # Sent on every API call when set. Capping the reasoning is the one knob that
    # moved the wall-clock: 58s -> 10s per turn at 2048 on a ~15k-character context.
    api_extra_body = ({'thinking_budget': args.challenger_thinking_budget}
                      if (use_api and args.challenger_thinking_budget > 0) else None)
    if api_extra_body:
        logger.info(f'[challenge] thinking_budget={args.challenger_thinking_budget} '
                    f'on every API call')
    explore_params = SamplingParams(max_tokens=args.propose_max_tokens, num_samples=1,
                                    logprobs=1, temperature=args.propose_temp, top_p=0.95,
                                    stop=explore_stop,
                                    include_stop_str_in_output=bool(explore_stop))
    # One tool call per reply and stuck-turn early stop are sampler-path features:
    # the API dispatches native tool_calls (never a '</tool_call>' string) and
    # APIMultiTurnRollout takes neither kwarg.
    if use_api:
        explorer = build_rollout(
            backend, tool_manager=tool_manager, max_turns=args.max_turns,
            concurrency=args.challenger_concurrency, sampling_params=explore_params,
            extra_body=api_extra_body)
    else:
        explorer = build_rollout(
            backend, template=template, tool_manager=tool_manager,
            max_turns=args.max_turns, stop_after_stuck_turns=args.stop_after_stuck_turns,
            sampling_params=explore_params)

    # Keyword brainstorming runs through this one instead of the sandbox
    # explorer: a list is a text answer, and a bracketed list in a reply is
    # exactly what the sandbox explorer would try to dispatch as a call.
    #
    # max_turns=1 is what makes it tool-less: MultiTurnRollout ends the
    # trajectory on the turn limit before it dispatches anything, so the empty
    # ToolManager below is never consulted. It is here because the rollout
    # requires one at construction, not because these calls have tools.
    keyword_params = SamplingParams(max_tokens=args.keyword_max_tokens, num_samples=1,
                                    logprobs=1, temperature=args.keyword_temp, top_p=0.98)
    if use_api:
        keyword_explorer = build_rollout(
            backend, tool_manager=ToolManager(), max_turns=1,
            concurrency=args.challenger_concurrency, sampling_params=keyword_params,
            extra_body=api_extra_body)
    else:
        keyword_explorer = build_rollout(
            backend,
            template=template,
            tool_manager=ToolManager(),
            max_turns=1,
            sampling_params=keyword_params,
        )

    # Sandbox control functions -- use env.runner() which resolves tool names
    # (ms-agent registers tools as "server---name") and parses exit codes from
    # the marker protocol, so we don't rely on string matching.
    #
    # One runner per sandbox; ``slot`` picks which one. The challenger passes the
    # slot of the episode it is serving, so a concurrent episode never clears or
    # inspects another episode's workspace. Everything serial (the difficulty
    # stage, --stage explore) leaves it at the default and uses sandbox 0.
    #
    # Empty for --stage keywords, which has no sandbox. The functions below index
    # into it and would raise if that stage ever reached them; it returns first.
    runners = [e.runner() for e in envs]
    runner = runners[0] if runners else None

    def reset_fn(slot: int = 0):
        """Empty sandbox ``slot``'s workspace before an episode.

        Raises rather than returning: every caller depends on a clean start, and
        a silent no-op here means a task inherits the previous task's files --
        which lets a solver pass without doing anything and makes the difficulty
        numbers meaningless.

        This is also the one point where losing the sandbox costs nothing, since
        the workspace is about to be emptied regardless -- so a runtime that went
        away is rebuilt here instead of ending a run that may have hours of
        proposals behind it.
        """
        if envs[slot].ensure_ready():
            # A rebuilt sandbox starts empty, so the clear below is redundant,
            # but running it anyway keeps one path through this function. The
            # rebuild replaces the sandbox behind this env, so the runner is
            # re-fetched rather than reused.
            runners[slot] = envs[slot].runner()
            logger.warning(f'[challenge] sandbox {slot} was rebuilt before this episode')
        exit_code, output = runners[slot](
            CLEAR_WORKSPACE.format(workspace=args.workspace), 'python')
        if exit_code != 0:
            raise RuntimeError(f'workspace reset failed (exit {exit_code}): {output[-400:]}')

    def run_check_fn(script: str, slot: int = 0):
        """Run a python check script in sandbox ``slot``; returns (exit_code, output)."""
        return runners[slot](script, 'python')

    def workspace_snapshot_fn(slot: int = 0):
        """Every file the episode left behind: ``path size`` lines, then contents.

        This is the ground truth the check script is written against, so it is
        unwrapped from the tool's JSON envelope and returned as a bare listing:
        the model has to be able to read it as a directory rather than as a tool
        result, or it falls back on what it *believes* it created.

        Returns an empty string when the episode left nothing behind, and also
        when the listing could not be read at all. Both mean the same thing to
        the caller -- there is no end state to write checks about -- and neither
        may be dressed up as a plausible one: a snapshot that says "empty"
        when it means "I could not look" produces tasks whose only true
        assertion is that nothing happened.
        """
        exit_code, output = runners[slot](
            WORKSPACE_SNAPSHOT.format(workspace=args.workspace,
                                      max_files=args.snapshot_max_files,
                                      per_file=args.snapshot_per_file,
                                      total_budget=args.snapshot_budget),
            'python')
        if exit_code != 0:
            # Not fatal, but not silent either: checks written against a missing
            # end state are the failure this whole function exists to prevent.
            logger.warning(f'[challenge] workspace snapshot failed (exit {exit_code}): '
                           f'{output[-200:]}')
            return ''
        return tool_payload(output).strip()

    # Arm B. Read at most this much per call: the executor truncates its output
    # near 8 KB, and base64 grows 3 bytes into 4, so 4 KB of file is about 5.5 KB
    # of text with room left for the JSON envelope.
    SLICE_BYTES = 4096


    # The opening the solver is measured against, built by the same function the
    # eval script uses so that n_pass and pass@k are measuring one thing. Until
    # this existed the difficulty stage handed over the statement as a lone user
    # message with no system prompt: nothing said the model was in a sandbox, could
    # take many turns, or should make one call per reply, and it answered by
    # writing whole programs into a single call argument until they truncated.
    _solver_harness = solver_harness(args.agent_config) if args.solver_rollouts > 0 else None

    def solver_prompt_fn(query: str):
        return _solver_harness.start(query)

    # Keywords
    store = None
    # Arm D replaces the three topic axes with one bank of 'kind of work' phrases:
    # a proposal takes one phrase, not one entry from each of three axes.
    categories = CATEGORIES
    category_desc = CATEGORY_DESC
    if args.keywords_n > 0:
        store = KeywordStore(args.keyword_db, categories)
        logger.info('[challenge] keyword bank loaded: '
                    + ', '.join(f'{c}={len(store.items[c])}' for c in categories))

    # Seeds
    seeds = []
    if args.seed_file:
        with open(args.seed_file, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    seeds.append(json.loads(line))
        logger.info(f'[challenge] loaded {len(seeds)} seeds from {args.seed_file}')

    # Rejected log
    rejected = open(args.dump_rejected, 'w', encoding='utf-8') if args.dump_rejected else None

    def _reject(record):
        if rejected is not None:
            rejected.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            # Flushed per record: this file is the only account of why proposals
            # are being dropped, and a run is worth watching for hours before it
            # ends. Buffered, it stays empty until then.
            rejected.flush()

    propose_writer = ProposeTrajWriter(args.dump_propose_traj)

    # Solver attempts from the difficulty stage. One line per attempt, so a task
    # measured at 0 of 4 can be read rather than guessed at: the attempt, the
    # state it left, and what the check said about it.
    solver_log = (open(args.dump_solver_attempts, 'w', encoding='utf-8')
                  if args.dump_solver_attempts else None)

    def _solver_attempt(record):
        if solver_log is not None:
            solver_log.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            solver_log.flush()

    # Keyword generation, one line per call. The bank is the first step of the
    # whole pipeline and the easiest place to fail invisibly: a reply the parser
    # rejects leaves the bank empty, and every proposal downstream then runs the
    # no-keyword prompt while the run looks healthy. Whole runs went that way
    # before this existed, so prompt and reply are both kept verbatim.
    keyword_log = (open(args.dump_keyword_gen, 'w', encoding='utf-8')
                   if args.dump_keyword_gen else None)

    def _keyword_gen(record):
        if keyword_log is not None:
            keyword_log.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            keyword_log.flush()

    # Build challenger
    prompts = agentic_prompts(max_build_files=args.max_build_files)
    # Followup API: explore on the local (trainable) sampler above, but write the
    # check script and problem statement over an OpenAI-compatible API (qwen3-max).
    # Reuses the --challenger-api-* connection args; the thinking cap, when set, is
    # sent as extra_body on every check/statement call.
    followup_api = None
    followup_extra_body = None
    if args.followup_api:
        from twinkle_agentic.protocol.openai import OpenAI
        followup_api = OpenAI(model=args.challenger_api_model,
                              api_key=args.challenger_api_key or None,
                              base_url=args.challenger_api_base)
        if args.challenger_thinking_budget > 0:
            followup_extra_body = {'thinking_budget': args.challenger_thinking_budget}
        logger.info(f'[challenge] followup (check + statement) via API '
                    f'model={args.challenger_api_model} base={args.challenger_api_base}'
                    + (f' thinking_budget={args.challenger_thinking_budget}'
                       if followup_extra_body else ''))
    challenger = AgenticChallenger(
        prompts,
        explorer,
        seeds=seeds,
        keyword_store=store,
        category_desc=category_desc if store else None,
        seed_mix_prob=args.seed_mix_prob,
        reset_fn=reset_fn,
        run_check_fn=run_check_fn,
        workspace_snapshot_fn=workspace_snapshot_fn,
        # The executor's own schemas, so the rounds that may call tools advertise
        # exactly what will run -- same source as the training script uses.
        tool_schemas=schemas,
        episode_concurrency=max(1, args.episode_concurrency),
        episode_tool_managers=episode_tool_managers,
        combo_arity=args.combo_arity,
        arity_weights=[float(x) for x in args.arity_weights.split(',')] if args.arity_weights
        else None,
        single_kw_prob=args.single_kw_prob,
        keyword_refill_target=args.keywords_n,
        keyword_gen_calls=args.keyword_gen_calls,
        keyword_refill_concurrency=max(1, args.keyword_refill_concurrency),
        keyword_refill_tries=args.keyword_refill_tries,
        keyword_params=SamplingParams(max_tokens=args.keyword_max_tokens, num_samples=1,
                                      logprobs=1, temperature=args.keyword_temp, top_p=0.98),
        # Same temperature as the episode, different budgets: the only thing
        # being changed per stage is how much room the reply gets.
        check_params=SamplingParams(max_tokens=args.check_max_tokens, num_samples=1,
                                    logprobs=1, temperature=args.propose_temp, top_p=0.95),
        problem_params=SamplingParams(max_tokens=args.problem_max_tokens, num_samples=1,
                                      logprobs=1, temperature=args.propose_temp, top_p=0.95),
        followup_api=followup_api,
        followup_extra_body=followup_extra_body,
        keyword_explorer=keyword_explorer,
        min_batch=args.sampler_gpus,
        problem_max_chars=args.problem_max_chars,
        max_proposals_total=args.max_proposals_total,
        solver_prompt_fn=solver_prompt_fn if _solver_harness is not None else None,
        check_retries=args.check_retries,
        reject_sink=_reject,
        propose_sink=propose_writer.write,
        solver_sink=_solver_attempt,
        keyword_sink=_keyword_gen,
        max_proposals_per_round=args.max_proposals_per_round,
        solver_rollouts=args.solver_rollouts,
        keep_min_pass=args.keep_min_pass,
        keep_max_pass_margin=args.keep_max_margin,
        # One call per reply here too, for the same reason and to keep the two
        # sides comparable: a solver whose read-back is dispatched alongside the
        # write it is checking fails a task the proposer built cleanly, and
        # n_pass would then be measuring the dispatch, not the difficulty.
        solver_params=SamplingParams(max_tokens=args.solver_max_tokens, num_samples=1,
                                     logprobs=1, temperature=args.solver_temp, top_p=0.95,
                                     stop=explore_stop,
                                     include_stop_str_in_output=bool(explore_stop)),
        seed=args.random_seed,
    )

    # Generate
    batch_size = args.batch_size or args.keep_target
    kept = []

    # --stage keywords stops after step 1: fill the bank, draw the combinations,
    # write out the proposal prompts they produce, and exit without touching the
    # sandbox. Step 1 was broken for several runs and the failure was only
    # visible by reading what it fed the next step, so it has to be runnable on
    # its own rather than only as the first minute of an hours-long run.
    if args.stage == 'keywords':
        proposals = challenger.propose(args.stage_proposals)
        with open(args.out_flows, 'w', encoding='utf-8') as out:
            for i, proposal in enumerate(proposals):
                data = proposal.get('user_data')
                out.write(json.dumps({
                    'index': i,
                    'keywords': user_data_get(data, 'keywords', []),
                    'seeded': user_data_get(data, 'seeded', False),
                    'prompt': proposal['messages'][-1]['content'],
                }, ensure_ascii=False) + '\n')
        drawn = sum(1 for p in proposals
                    if user_data_get(p.get('user_data'), 'keywords', []))
        logger.info(f'[challenge] stage=keywords: {len(proposals)} proposals, '
                    f'{drawn} of them carry keywords')
        if store is not None:
            store.save()
            logger.info('[challenge] keyword bank saved -> ' + args.keyword_db
                        + ' (' + ', '.join(f'{c}={len(store.items[c])}'
                                           for c in categories) + ')')
        if keyword_log is not None:
            keyword_log.close()
        propose_writer.close()
        if rejected is not None:
            rejected.close()
        if solver_log is not None:
            solver_log.close()
        if env is not None:
            for e in envs:
                e.close()
        return

    # --stage explore stops after step 4: draw a proposal, clear the workspace,
    # run the sandbox episode, snapshot what it left, and stop before the
    # check-writing round. What it is for: the episode is where the run either
    # produces something worth writing a check about or leaves an empty directory,
    # and 9 of 30 proposals in run11 left an empty one for reasons the rejection
    # record could not distinguish. Everything the episode saw and did goes out
    # verbatim, so that question is answerable from the file.
    if args.stage == 'explore':
        explore_log = (open(args.dump_explore, 'w', encoding='utf-8')
                       if args.dump_explore else None)
        empty = 0
        for i, proposal in enumerate(challenger.propose(args.stage_proposals)):
            reset_fn()
            result = challenger.explore([proposal])
            episode = result[0] if result else {}
            snapshot = workspace_snapshot_fn()
            if not snapshot.strip():
                empty += 1
            messages = episode.get('messages') or []
            calls = sum(len(m.get('tool_calls') or []) for m in messages
                        if isinstance(m, dict))
            logger.info(f'[challenge] episode {i}: stop={episode.get("stop_reason")} '
                        f'truncated={bool(episode.get("truncated"))} '
                        f'stuck_stop={bool(episode.get("stuck_stop"))} '
                        f'turns={episode.get("turns")} calls={calls} '
                        f'end_state={len(snapshot)}b')
            if explore_log is not None:
                explore_log.write(json.dumps({
                    'index': i,
                    'keywords': user_data_get(proposal.get('user_data'), 'keywords', []),
                    'prompt': proposal['messages'][-1]['content'],
                    'stop_reason': episode.get('stop_reason'),
                    'truncated': bool(episode.get('truncated')),
                    'stuck_stop': bool(episode.get('stuck_stop')),
                    'turns': episode.get('turns'),
                    'n_tool_calls': calls,
                    'messages': messages,
                    'end_state': snapshot,
                }, ensure_ascii=False, default=str) + '\n')
                explore_log.flush()
        logger.info(f'[challenge] stage=explore: {args.stage_proposals} episodes, '
                    f'{empty} left an empty workspace')
        if explore_log is not None:
            explore_log.close()
        if keyword_log is not None:
            keyword_log.close()
        propose_writer.close()
        if rejected is not None:
            rejected.close()
        if solver_log is not None:
            solver_log.close()
        if store is not None:
            store.save()
        if env is not None:
            for e in envs:
                e.close()
        return

    # Appended as batches arrive, then rewritten sorted at the end. A run that
    # keeps one task every few minutes for hours cannot afford to hold them all
    # in memory only: a crash at hour three would leave nothing to train on,
    # while an unsorted partial file is a usable task set.
    with open(args.out_flows, 'w', encoding='utf-8') as partial:
        for batch in challenger(batch_size=batch_size, total=args.keep_target):
            for offset, task in enumerate(batch):
                partial.write(json.dumps(flow_record(len(kept) + offset, task),
                                         ensure_ascii=False) + '\n')
            partial.flush()
            kept.extend(batch)
            logger.info(f'[challenge] kept {len(kept)}/{args.keep_target} so far; '
                        f'stats {challenger.stats}')
    if rejected is not None:
        rejected.close()
    if solver_log is not None:
        solver_log.close()
    propose_writer.close()

    # Before the keyword log is closed: expanding the bank generates keywords,
    # and generating them writes to that log. Closing it first ended ex11 --
    # after all 4 tasks were kept and written -- with `ValueError: I/O operation
    # on closed file`, which also skipped store.save() below and every line
    # after it, so the run reported nothing about what it had produced.
    if store is not None:
        challenger.expand_hard_keywords()
        store.save()
        logger.info('[challenge] keyword bank saved -> ' + args.keyword_db)
    if keyword_log is not None:
        keyword_log.close()

    # Sort by difficulty (hardest last)
    if not args.no_sort_by_difficulty:
        kept.sort(key=lambda t: -(user_data_get(t.get('user_data'), 'n_pass', 0) or 0))

    # Write output
    write_flows(kept, args)
    logger.info(f'[challenge] wrote {len(kept)} tasks -> {args.out_flows}')
    if env.n_recoveries:
        logger.warning(f'[challenge] sandbox was rebuilt {env.n_recoveries} time(s) during '
                       f'this run; episodes in flight at those moments were lost')
    dist = {}
    for task in kept:
        n = user_data_get(task.get('user_data'), 'n_pass', 0)
        dist[n] = dist.get(n, 0) + 1
    logger.info(f'[challenge] pass-count distribution: {dict(sorted(dist.items()))}')

    for e in envs:
        e.close()


class ProposeTrajWriter:
    """Persist the proposing rounds so the challenger could be trained later.

    One ``.npz`` per proposal attempt holds the arrays, and one line per attempt
    in ``index.jsonl`` holds everything a human reads plus the outcome. Splitting
    them is what keeps this affordable: a 20-turn agentic episode is tens of
    thousands of token ids, which as JSON is an order of magnitude larger than
    the same numbers as int32.

    ``logprobs`` arrive as ``[[(token_id, logprob)]]`` and are flattened to the
    logprob column alone -- that is the shape GRPO's ``old_logps`` wants, and the
    token each one belongs to is already in ``labels``.

    Rejected attempts are written too. Their outcome is the reward's zero, and a
    dump of kept-only attempts would have nothing to contrast against.
    """

    def __init__(self, out_dir: str):
        self.dir = out_dir
        self.index = None
        self.n = 0
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        self.index = open(os.path.join(out_dir, 'index.jsonl'), 'w', encoding='utf-8')

    def write(self, record):
        if self.index is None:
            return
        trace_id = f'p{self.n:06d}'
        self.n += 1
        arrays, meta = {}, []
        for i, rnd in enumerate(record.get('rounds') or []):
            labels = rnd.get('labels') or []
            logprobs = rnd.get('logprobs') or []
            if rnd.get('input_ids'):
                arrays[f'r{i}_input_ids'] = np.asarray(rnd['input_ids'], dtype=np.int32)
            if labels:
                arrays[f'r{i}_labels'] = np.asarray(labels, dtype=np.int32)
            if logprobs:
                arrays[f'r{i}_logprobs'] = np.asarray(
                    [lp[0][1] for lp in logprobs], dtype=np.float32)
            meta.append({
                'stage': rnd.get('stage'),
                'messages': rnd.get('messages') or [],
                'n_tokens': len(rnd.get('input_ids') or []),
                'n_trainable': sum(1 for label in labels if label != -100),
                'n_logprobs': len(logprobs),
            })
        # No arrays means the explorer was text-only (an API rollout), so there
        # is nothing trainable to store -- record the attempt without an npz
        # rather than leaving thousands of empty archives behind.
        npz_name = f'{trace_id}.npz' if arrays else None
        if arrays:
            np.savez_compressed(os.path.join(self.dir, npz_name), **arrays)
        line = {
            'trace_id': trace_id,
            'npz': npz_name,
            'outcome': record.get('outcome'),
            'n_pass': record.get('n_pass'),
            'n_rollouts': record.get('n_rollouts'),
            'pass_rate': record.get('pass_rate'),
            'keywords': record.get('keywords'),
            'seeded': record.get('seeded'),
            'rounds': meta,
        }
        self.index.write(json.dumps(line, ensure_ascii=False, default=str) + '\n')
        self.index.flush()

    def close(self):
        if self.index is not None:
            self.index.close()
            logger.info(f'[challenge] wrote {self.n} propose traces -> {self.dir}')


def flow_record(index, task):
    """One task as the training script reads it back."""
    data = task.get('user_data')
    messages = task.get('messages') or []
    query = next((m['content'] for m in messages if m.get('role') == 'user'), '')
    return {
        'id': f'ag_{index:06d}',
        'query': query,
        'check_script': user_data_get(data, 'check_script', ''),
        # Arm B: run before the solver starts, to put the input files it is told
        # it already has on disk. Empty for every other arm.
        'setup_script': user_data_get(data, 'setup_script', ''),
        'n_pass': user_data_get(data, 'n_pass'),
        'n_rollouts': user_data_get(data, 'n_rollouts'),
        'keywords': user_data_get(data, 'keywords', []),
        'seeded': user_data_get(data, 'seeded', False),
    }


def write_flows(kept, args):
    """Write one flow per task, replacing whatever the run appended as it went."""
    with open(args.out_flows, 'w', encoding='utf-8') as f:
        for i, task in enumerate(kept):
            f.write(json.dumps(flow_record(i, task), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
