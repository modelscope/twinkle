# Copyright (c) ModelScope Contributors. All rights reserved.
"""RSI self-play, agentic half: generate training tasks by doing them first.

One model plays both roles. It first acts as an agent in a sandbox (multi-turn
with tools), producing a trajectory and final workspace state. Then it writes a
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
from prompts import CATEGORIES, CATEGORY_DESC, agentic_prompts  # noqa: E402
from remote_tool_env import RemoteMsAgentToolEnv  # noqa: E402

logger = get_logger()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model
    p.add_argument('--model-id', default='ms://Qwen/Qwen3-4B')
    p.add_argument('--template', default='Template',
                   help='template class in twinkle.template')
    p.add_argument('--sampler-gpus', type=int, default=4)
    p.add_argument('--max-model-len', type=int, default=32768)

    # Generation control
    p.add_argument('--keep-target', type=int, default=200,
                   help='how many tasks to keep; generation stops once reached')
    p.add_argument('--batch-size', type=int, default=0,
                   help='tasks per yielded batch (0 = one batch of --keep-target)')
    p.add_argument('--max-proposals-per-round', type=int, default=64,
                   help='max proposals per round (serial, so keep moderate)')
    p.add_argument('--seed-file', default='', help='seed jsonl with query field')
    p.add_argument('--seed-mix-prob', type=float, default=0.5)

    # Sampling params for round 1 (proposing)
    p.add_argument('--propose-temp', type=float, default=1.0)
    p.add_argument('--propose-max-tokens', type=int, default=4096)
    p.add_argument('--max-turns', type=int, default=20,
                   help='max tool-calling turns for round 1')

    # Problem statement
    p.add_argument('--problem-max-chars', type=int, default=8192)

    # Keywords
    p.add_argument('--keywords-n', type=int, default=128,
                   help='per-category refill target; 0 disables keyword bank')
    p.add_argument('--keyword-db', default='output/rsi_agentic/keywords.jsonl')
    p.add_argument('--keyword-gen-calls', type=int, default=8)
    p.add_argument('--keyword-refill-tries', type=int, default=2)
    p.add_argument('--keyword-temp', type=float, default=1.3)
    p.add_argument('--keyword-max-tokens', type=int, default=1024)
    p.add_argument('--single-kw-prob', type=float, default=0.1)
    p.add_argument('--combo-arity', default='triple', choices=['triple', 'mix'])
    p.add_argument('--arity-weights', default='',
                   help="'w1,w2,w3' for --combo-arity mix (empty = uniform)")

    # Difficulty filter
    p.add_argument('--solver-rollouts', type=int, default=4)
    p.add_argument('--solver-temp', type=float, default=1.0)
    p.add_argument('--solver-max-tokens', type=int, default=4096)
    p.add_argument('--solver-max-turns', type=int, default=20)
    p.add_argument('--keep-min-pass', type=int, default=1)
    p.add_argument('--keep-max-margin', type=int, default=1)

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

    # Output
    p.add_argument('--random-seed', type=int, default=0)
    p.add_argument('--out-flows', default='output/rsi_agentic/challenge_flows.jsonl')
    p.add_argument('--dump-rejected', default='output/rsi_agentic/challenge_rejected.jsonl')
    p.add_argument('--dump-propose-traj', default='output/rsi_agentic/propose_traj',
                   help='directory for the proposing rounds (token ids + logprobs, one npz '
                        'per attempt plus index.jsonl). Empty string turns it off; keeping '
                        'it is what leaves the door open to training the challenger itself.')
    p.add_argument('--no-sort-by-difficulty', action='store_true')
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
    for path in (args.out_flows, args.dump_rejected, args.keyword_db):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

    # Initialize twinkle (sampler only, no trainer)
    twinkle.initialize(
        mode='ray', nproc_per_node=args.sampler_gpus, lazy_collect=False,
        groups=[DeviceGroup(name='sampler', ranks=list(range(args.sampler_gpus)),
                            device_type='GPU')])
    sampler = vLLMSampler(
        model_id=args.model_id,
        engine_args={'gpu_memory_utilization': 0.8, 'max_model_len': args.max_model_len},
        device_mesh=DeviceMesh.from_sizes(world_size=args.sampler_gpus,
                                          dp_size=args.sampler_gpus),
        remote_group='sampler',
    )
    sampler.set_template(args.template, model_id=args.model_id, enable_thinking=True,
                         max_length=args.max_model_len)

    import twinkle.template as template_module
    template = getattr(template_module, args.template)(
        args.model_id, max_length=args.max_model_len, enable_thinking=True)

    # Build sandbox environment
    env = build_env(args)
    schemas = env.tool_schemas()
    tool_manager = ToolManager(EnvTool.from_schemas(env, schemas))

    # Explorer: multi-turn rollout with sandbox tools
    explorer = build_rollout(
        sampler,
        template=template,
        tool_manager=tool_manager,
        max_turns=args.max_turns,
        sampling_params=SamplingParams(max_tokens=args.propose_max_tokens, num_samples=1,
                                       logprobs=1, temperature=args.propose_temp, top_p=0.95),
    )

    # Sandbox control functions -- use env.runner() which resolves tool names
    # (ms-agent registers tools as "server---name") and parses exit codes from
    # the marker protocol, so we don't rely on string matching.
    runner = env.runner()

    # Cleared through the python tool, not `rm -rf`: ms-agent's safety policy
    # rejects `rm -rf` outright ("Blocked by safety rule"), and it rejects globs
    # in write operations, which rules out `find -delete` too. The script asserts
    # the directory really is empty so a future policy change surfaces as a
    # failed reset instead of tasks quietly inheriting the previous workspace.
    _CLEAR = '''
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

    def reset_fn():
        """Empty the sandbox workspace before an episode.

        Raises rather than returning: every caller depends on a clean start, and
        a silent no-op here means a task inherits the previous task's files --
        which lets a solver pass without doing anything and makes the difficulty
        numbers meaningless.
        """
        exit_code, output = runner(_CLEAR.format(workspace=args.workspace), 'python')
        if exit_code != 0:
            raise RuntimeError(f'workspace reset failed (exit {exit_code}): {output[-400:]}')

    def run_check_fn(script: str):
        """Run a python check script in the sandbox; returns (exit_code, output)."""
        return runner(script, 'python')

    def workspace_snapshot_fn():
        """Get a summary of the current workspace state."""
        _, output = runner(
            f'find {args.workspace} -type f -printf "%P %s\\n" 2>/dev/null | head -50',
            'shell')
        return output or '(empty)'

    # Keywords
    store = None
    if args.keywords_n > 0:
        store = KeywordStore(args.keyword_db, CATEGORIES)
        logger.info('[challenge] keyword bank loaded: '
                    + ', '.join(f'{c}={len(store.items[c])}' for c in CATEGORIES))

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

    propose_writer = ProposeTrajWriter(args.dump_propose_traj)

    # Build challenger
    prompts = agentic_prompts()
    challenger = AgenticChallenger(
        prompts,
        explorer,
        seeds=seeds,
        keyword_store=store,
        category_desc=CATEGORY_DESC if store else None,
        seed_mix_prob=args.seed_mix_prob,
        reset_fn=reset_fn,
        run_check_fn=run_check_fn,
        workspace_snapshot_fn=workspace_snapshot_fn,
        combo_arity=args.combo_arity,
        arity_weights=[float(x) for x in args.arity_weights.split(',')] if args.arity_weights
        else None,
        single_kw_prob=args.single_kw_prob,
        keyword_refill_target=args.keywords_n,
        keyword_gen_calls=args.keyword_gen_calls,
        keyword_refill_tries=args.keyword_refill_tries,
        keyword_params=SamplingParams(max_tokens=args.keyword_max_tokens, num_samples=1,
                                      logprobs=1, temperature=args.keyword_temp, top_p=0.98),
        min_batch=args.sampler_gpus,
        problem_max_chars=args.problem_max_chars,
        reject_sink=_reject,
        propose_sink=propose_writer.write,
        max_proposals_per_round=args.max_proposals_per_round,
        solver_rollouts=args.solver_rollouts,
        keep_min_pass=args.keep_min_pass,
        keep_max_pass_margin=args.keep_max_margin,
        solver_params=SamplingParams(max_tokens=args.solver_max_tokens, num_samples=1,
                                     logprobs=1, temperature=args.solver_temp, top_p=0.95),
        seed=args.random_seed,
    )

    # Generate
    batch_size = args.batch_size or args.keep_target
    kept = []
    for batch in challenger(batch_size=batch_size, total=args.keep_target):
        kept.extend(batch)
        logger.info(f'[challenge] kept {len(kept)}/{args.keep_target} so far; '
                    f'stats {challenger.stats}')
    if rejected is not None:
        rejected.close()
    propose_writer.close()

    if store is not None:
        challenger.expand_hard_keywords()
        store.save()
        logger.info('[challenge] keyword bank saved -> ' + args.keyword_db)

    # Sort by difficulty (hardest last)
    if not args.no_sort_by_difficulty:
        kept.sort(key=lambda t: -(user_data_get(t.get('user_data'), 'n_pass', 0) or 0))

    # Write output
    write_flows(kept, args)
    logger.info(f'[challenge] wrote {len(kept)} tasks -> {args.out_flows}')
    dist = {}
    for task in kept:
        n = user_data_get(task.get('user_data'), 'n_pass', 0)
        dist[n] = dist.get(n, 0) + 1
    logger.info(f'[challenge] pass-count distribution: {dict(sorted(dist.items()))}')

    env.close()


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

    def close(self):
        if self.index is not None:
            self.index.close()
            logger.info(f'[challenge] wrote {self.n} propose traces -> {self.dir}')


def write_flows(kept, args):
    """Write one flow per task."""
    with open(args.out_flows, 'w', encoding='utf-8') as f:
        for i, task in enumerate(kept):
            data = task.get('user_data')
            messages = task.get('messages') or []
            query = next((m['content'] for m in messages if m.get('role') == 'user'), '')
            flow = {
                'id': f'ag_{i:06d}',
                'query': query,
                'check_script': user_data_get(data, 'check_script', ''),
                'n_pass': user_data_get(data, 'n_pass'),
                'n_rollouts': user_data_get(data, 'n_rollouts'),
                'keywords': user_data_get(data, 'keywords', []),
                'seeded': user_data_get(data, 'seeded', False),
            }
            f.write(json.dumps(flow, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
